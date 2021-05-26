import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils

from constants import *
from models.base import *
from models.helpers import *
from models.encoder import *

class EventCorefModel(BaseModel):
    def __init__(self, configs, event_types):
        BaseModel.__init__(self, configs)
        self.event_types = sorted(event_types)

        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(configs)
        self.linear = nn.Linear(3 * self.transformer_encoder.hidden_size, configs['latent_size'])

        # Symbolic Features Encoder
        self.symbolic_encoder = SymbolicFeaturesEncoder(configs, self.event_types)

        # Feature Fusion Network
        self.fusion_network = FeatureFusionNetwork(latent_size=configs['latent_size'],
                                                   combine_strategy=configs['combine_strategy'],
                                                   nb_modules=len(self.symbolic_encoder.enabled_features))

        # Pair Scorer
        self.dropout = nn.Dropout(configs['dropout_rate'])
        self.pair_scorer = FFNNModule(input_size=self.get_pair_size(),
                                      hidden_sizes=[configs['ffnn_size']] * configs['ffnn_depth'],
                                      output_size=1,
                                      dropout=configs['dropout_rate'])

        # Move model to device
        self.to(self.device)

    def forward(self, inst, is_training):
        self.train() if is_training else self.eval()

        # Extract event_mentions and entity_mentions
        if self.configs['use_groundtruth']:
            entity_mentions = inst.entity_mentions
            event_mentions = inst.event_mentions
        else:
            entity_mentions = inst.pred_entities
            event_mentions = inst.pred_event_mentions

        # Convert to Torch Tensor
        input_ids = torch.tensor(inst.token_windows).to(self.device)
        input_masks = torch.tensor(inst.input_masks).to(self.device)
        mask_windows = torch.tensor(inst.mask_windows).to(self.device)
        num_windows, window_size = input_ids.size()

        # Apply the Transfomer encoder to get tokens features
        tokens_features = self.transformer_encoder(input_ids, input_masks, mask_windows,
                                                   num_windows, window_size, is_training).squeeze()
        num_tokens = tokens_features.size()[0]

        # Compute word_features (averaging)
        word_features = []
        word_starts_indexes = inst.word_starts_indexes
        word_ends_indexes = word_starts_indexes[1:] + [num_tokens]
        word_features = get_span_emb(tokens_features, word_starts_indexes, word_ends_indexes)
        assert(word_features.size()[0] == inst.num_words)

        # Compute entity_features
        entity_starts = [m['start'] for m in entity_mentions]
        entity_ends = [m['end'] for m in entity_mentions]
        entity_features = get_span_emb(word_features, entity_starts, entity_ends)

        # Compute trigger_features
        event_starts = [e['trigger']['start'] for e in event_mentions]
        event_ends = [e['trigger']['end'] for e in event_mentions]
        trigger_features = get_span_emb(word_features, event_starts, event_ends)

        # Compute pair_trigger_features
        pair_trigger_features = get_pair_embs(trigger_features)
        pair_trigger_features = F.relu(self.linear(pair_trigger_features))

        # Compute pair_features
        if len(self.symbolic_encoder.enabled_features) == 0:
            # Not using any additional symbolic features
            pair_features = pair_trigger_features
        else:
            # Use additional symbolic features
            pair_symbolic_features = self.symbolic_encoder(event_mentions)
            pair_features = self.fusion_network(pair_trigger_features, pair_symbolic_features)

        # Compute pair_scores
        pair_features = self.dropout(pair_features)
        pair_scores = self.pair_scorer(pair_features)

        # Compute antecedent_scores
        k = len(event_mentions)
        span_range = torch.arange(0, k).to(self.device)
        antecedent_offsets = span_range.view(-1, 1) - span_range.view(1, -1)
        antecedents_mask = antecedent_offsets >= 1 # [k, k]
        antecedent_scores = pair_scores + torch.log(antecedents_mask.float())

        # Compute antecedent_labels
        candidate_cluster_ids = self.get_cluster_ids(event_mentions, inst.coreferential_pairs)
        same_cluster_indicator = candidate_cluster_ids.unsqueeze(0) == candidate_cluster_ids.unsqueeze(1)
        same_cluster_indicator = same_cluster_indicator & antecedents_mask

        non_dummy_indicator = (candidate_cluster_ids > -1).unsqueeze(1)
        pairwise_labels = same_cluster_indicator & non_dummy_indicator
        dummy_labels = ~pairwise_labels.any(1, keepdim=True)
        antecedent_labels = torch.cat([dummy_labels, pairwise_labels], 1)

        # Compute loss
        dummy_zeros = torch.zeros([k, 1]).to(self.device)
        antecedent_scores = torch.cat([dummy_zeros, antecedent_scores], dim=1)
        gold_scores = antecedent_scores + torch.log(antecedent_labels.float())
        log_norm = logsumexp(antecedent_scores, dim = 1)
        loss = torch.sum(log_norm - logsumexp(gold_scores, dim=1))

        # loss and preds
        top_antecedents = torch.arange(0, k).to(self.device)
        top_antecedents = top_antecedents.unsqueeze(0).repeat(k, 1)
        preds = [torch.tensor(event_starts),
                 torch.tensor(event_ends),
                 top_antecedents,
                 antecedent_scores]

        return loss, preds

    def get_cluster_ids(self, event_mentions, coreferential_pairs):
        cluster_ids = [-1] * len(event_mentions)
        nb_nonsingleton_clusters = 0
        for i in range(len(event_mentions)):
            mention_i = event_mentions[i]
            loc_i = (mention_i['trigger']['start'], mention_i['trigger']['end'])
            for j in range(i-1, -1, -1):
                mention_j = event_mentions[j]
                loc_j = (mention_j['trigger']['start'], mention_j['trigger']['end'])
                if ((loc_i, loc_j)) in coreferential_pairs:
                    if cluster_ids[j] > -1:
                        cluster_ids[i] = cluster_ids[j]
                    else:
                        cluster_ids[i] = cluster_ids[j] = nb_nonsingleton_clusters
                        nb_nonsingleton_clusters += 1
        return torch.tensor(cluster_ids).to(self.device)


    def get_pair_size(self):
        return (1 + len(self.symbolic_encoder.enabled_features)) * self.configs['latent_size']

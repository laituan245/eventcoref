import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import random

from constants import *
from transformers import *
from models.helpers import *

class TransformerEncoder(nn.Module):
    def __init__(self, configs):
        super(TransformerEncoder, self).__init__()
        self.configs = configs

        # Transformer Encoder
        self.transformer = AutoModel.from_pretrained(configs['transformer'])
        self.transformer_dropout = nn.Dropout(configs['transformer_dropout_rate'])
        self.transformer.config.gradient_checkpointing  = configs['gradient_checkpointing']
        self.hidden_size = self.transformer.config.hidden_size

    def forward(self, input_ids, input_masks, mask_windows,
                num_windows, window_size, is_training,
                context_lengths = [0], token_type_ids = None):
        self.train() if is_training else self.eval()
        num_contexts = len(context_lengths)

        features = self.transformer(input_ids, input_masks, token_type_ids)[0]
        features = features.view(num_contexts, num_windows, -1, self.hidden_size)

        flattened_features = []
        for i in range(num_contexts):
            _features = features[i, :, :, :]
            _features = _features[:, context_lengths[i]:, :]
            _features = _features[:, : window_size, :]
            flattened_features.append(self.flatten(_features, mask_windows))
        flattened_features = torch.cat(flattened_features)

        return self.transformer_dropout(flattened_features)

    def flatten(self, features, mask_windows):
        num_windows, window_size, hidden_size = features.size()
        flattened_emb = torch.reshape(features, (num_windows * window_size, hidden_size))
        boolean_mask = mask_windows > 0
        boolean_mask = boolean_mask.view([num_windows * window_size])
        return flattened_emb[boolean_mask].unsqueeze(0)

class SymbolicFeaturesEncoder(nn.Module):
    def __init__(self, configs, event_types):
        super(SymbolicFeaturesEncoder, self).__init__()
        self.configs = configs
        self.feature_size = configs['feature_size']
        self.latent_size = configs['latent_size']
        self.event_types = event_types

        # Embeddings and Linear Layers
        if configs['use_typ_features']:
            self.typ_embed = nn.Embedding(len(event_types), self.feature_size)
            self.typ_linear = nn.Linear(3 * self.feature_size, self.latent_size)
        if configs['use_pol_features']:
            self.pol_embed = nn.Embedding(len(POL_TYPES), self.feature_size)
            self.pol_linear = nn.Linear(3 * self.feature_size, self.latent_size)
        if configs['use_mod_features']:
            self.mod_embed = nn.Embedding(len(MOD_TYPES), self.feature_size)
            self.mod_linear = nn.Linear(3 * self.feature_size, self.latent_size)
        if configs['use_gen_features']:
            self.gen_embed = nn.Embedding(len(GEN_TYPES), self.feature_size)
            self.gen_linear = nn.Linear(3 * self.feature_size, self.latent_size)
        if configs['use_ten_features']:
            self.ten_embed = nn.Embedding(len(TEN_TYPES), self.feature_size)
            self.ten_linear = nn.Linear(3 * self.feature_size, self.latent_size)

        # Initialize Embeddings
        for name, param in self.named_parameters():
            if (not 'transformer' in name.lower()) and 'embedding' in name.lower():
                print('Re-initialize embedding {}'.format(name))
                param.data.uniform_(-0.5, 0.5)

    def forward(self, events):
        features = []
        if self.configs['use_typ_features']: features.append(self.get_features(events, 'event_type'))
        if self.configs['use_pol_features']: features.append(self.get_features(events, 'event_polarity'))
        if self.configs['use_mod_features']: features.append(self.get_features(events, 'event_modality'))
        if self.configs['use_gen_features']: features.append(self.get_features(events, 'event_genericity'))
        if self.configs['use_ten_features']: features.append(self.get_features(events, 'event_tense'))
        return features

    def get_features(self, events, key):
        if key == 'event_type':
            embed, linear, value_types = self.typ_embed, self.typ_linear, self.event_types
            noisy_prob = self.configs['typ_noise_prob']
        if key == 'event_polarity':
            embed, linear, value_types = self.pol_embed, self.pol_linear, POL_TYPES
            noisy_prob = self.configs['pol_noise_prob']
        if key == 'event_modality':
            embed, linear, value_types = self.mod_embed, self.mod_linear, MOD_TYPES
            noisy_prob = self.configs['mod_noise_prob']
        if key == 'event_genericity':
            embed, linear, value_types = self.gen_embed, self.gen_linear, GEN_TYPES
            noisy_prob = self.configs['gen_noise_prob']
        if key == 'event_tense':
            embed, linear, value_types = self.ten_embed, self.ten_linear, TEN_TYPES
            noisy_prob = self.configs['ten_noise_prob']

        values = []
        for e in events:
            value = e[key]
            if self.training and random.uniform(0, 1) < noisy_prob and e['has_correct_trigger']:
                value = random.choice(value_types)
            values.append(value_types.index(value))

        values = torch.tensor(values).to(next(self.parameters()).device)
        latent_feats = F.relu(linear(get_pair_embs(embed(values))))
        return latent_feats

    @property
    def enabled_features(self):
        enabled_features = []
        if self.configs['use_typ_features']: enabled_features.append('event_type')
        if self.configs['use_pol_features']: enabled_features.append('event_polarity')
        if self.configs['use_mod_features']: enabled_features.append('event_modality')
        if self.configs['use_gen_features']: enabled_features.append('event_genericity')
        if self.configs['use_ten_features']: enabled_features.append('event_tense')
        return enabled_features

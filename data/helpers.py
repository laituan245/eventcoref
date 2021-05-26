import json
import random

from os.path import join
from data.base import Dataset, Document

def load_oneie_dataset(
        base_path, tokenizer,
        predictions_path=None, remove_doc_with_no_events=True,
        increase_ace_dev_set=False
    ):
    id2split, id2sents = {}, {}

    # Read ground-truth data files
    for split in ['train', 'dev', 'test']:
        path = join(base_path, '{}.oneie.json'.format(split))
        with open(path, 'r', encoding='utf-8') as r:
            for line in r:
                sent_inst = json.loads(line)
                doc_id = sent_inst['doc_id']
                id2split[doc_id] = split
                # Update id2sents
                if not doc_id in id2sents:
                    id2sents[doc_id] = []
                id2sents[doc_id].append(sent_inst)

    # Read prediction files (if available)
    predicted_attrs = None
    if predictions_path:
        sentid2graph = {}
        for split in ['train', 'dev', 'test']:
            path = join(predictions_path, '{}.json'.format(split))
            with open(path, 'r', encoding='utf-8') as r:
                for line in r:
                    sent_preds = json.loads(line)
                    sentid2graph[sent_preds['sent_id']] = sent_preds['graph']

        # Read attributes prediction files
        attrs_preds_path = join(predictions_path, 'attrs_preds.json')
        predicted_attrs = json.load(open(attrs_preds_path, 'r'))
        _predicted_attrs = {}
        for key in predicted_attrs:
            split_index = key.rfind('.(')
            doc_id = key[:split_index]
            start, end = key[split_index+2:-1].split('-')
            start, end = int(start), int(end)
            _predicted_attrs[(doc_id, start, end)] = predicted_attrs[key]
        predicted_attrs = _predicted_attrs

    # Parse documents one-by-one
    train, dev, test = [], [], []
    for doc_id in id2sents:
        words_ctx, pred_trigger_ctx, pred_entities_ctx = 0, 0, 0
        sents = id2sents[doc_id]
        sentences, event_mentions, entity_mentions, pred_graphs = [], [], [], []
        for sent_index, sent in enumerate(sents):
            sentences.append(sent['tokens'])
            # Parse entity mentions
            for entity_mention in sent['entity_mentions']:
                entity_mention['start'] += words_ctx
                entity_mention['end'] += words_ctx
                entity_mentions.append(entity_mention)
            # Parse event mentions
            for event_mention in sent['event_mentions']:
                event_mention['sent_index'] = sent_index
                event_mention['trigger']['start'] += words_ctx
                event_mention['trigger']['end'] += words_ctx
                event_mentions.append(event_mention)
            # Update pred_graphs
            if predictions_path:
                graph = sentid2graph.get(sent['sent_id'], {})
                if len(graph) > 0:
                    for entity in graph['entities']:
                        entity[0] += words_ctx
                        entity[1] += words_ctx
                    for trigger in graph['triggers']:
                        trigger[0] += words_ctx
                        trigger[1] += words_ctx
                        # Look up predicted attributes
                        if predicted_attrs:
                            lookedup_attrs = predicted_attrs[(doc_id, trigger[0], trigger[1])]
                            trigger.append(lookedup_attrs)
                    for relation in graph['relations']:
                        relation[0] += pred_entities_ctx
                        relation[1] += pred_entities_ctx
                    for role in graph['roles']:
                        role[0] += pred_trigger_ctx
                        role[1] += pred_entities_ctx
                    pred_trigger_ctx += len(graph['triggers'])
                    pred_entities_ctx += len(graph['entities'])
                pred_graphs.append(graph)
            # Update words_ctx
            words_ctx += len(sent['tokens'])
        doc = Document(doc_id, sentences, event_mentions, entity_mentions, pred_graphs)
        split = id2split[doc_id]
        if split == 'train':
            if not remove_doc_with_no_events or len(event_mentions) > 0:
                train.append(doc)
        if split == 'dev': dev.append(doc)
        if split == 'test': test.append(doc)

    if increase_ace_dev_set:
        # Randomly move 12 docs from train set to dev set
        random.seed(0)
        random.shuffle(train)
        dev = train[:12] + dev
        train = train[12:]

    # Convert to Document class
    train, dev, test = Dataset(train, tokenizer), Dataset(dev, tokenizer), Dataset(test, tokenizer)

    # Verbose
    print('Loaded {} train examples'.format(len(train)))
    print('Loaded {} dev examples'.format(len(dev)))
    print('Loaded {} test examples'.format(len(test)))

    return train, dev, test

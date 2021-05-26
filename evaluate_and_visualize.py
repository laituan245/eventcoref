import os
import math
import json
import torch
import tqdm
import pyhocon
import random

from transformers import *
from models import EventCorefModel
from scorer import evaluate
from argparse import ArgumentParser
from data import load_oneie_dataset
from utils import RunningAverage, prepare_configs, flatten
from scorer import get_predicted_antecedents

def generate_coref_preds(model, data, output_path='predictions.json'):
    predictions = {}
    for inst in data:
        doc_words = inst.words
        event_mentions = inst.event_mentions
        preds = model(inst, is_training=False)[1]
        preds = [x.cpu().data.numpy() for x in preds]
        top_antecedents, top_antecedent_scores = preds[2:]
        predicted_antecedents = get_predicted_antecedents(top_antecedents, top_antecedent_scores)

        predicted_clusters, m2cluster = [], {}
        for ix, e in enumerate(event_mentions):
            if predicted_antecedents[ix] < 0:
                cluster_id = len(predicted_clusters)
                predicted_clusters.append([e])
            else:
                antecedent_idx = predicted_antecedents[ix]
                p_e = event_mentions[antecedent_idx]
                cluster_id = m2cluster[p_e['id']]
                predicted_clusters[cluster_id].append(e)
            m2cluster[e['id']] = cluster_id
        # Update predictions
        predictions[inst.doc_id] = {}
        predictions[inst.doc_id]['words']= doc_words
        predictions[inst.doc_id]['predicted_clusters'] = predicted_clusters

    with open(output_path, 'w+') as outfile:
        json.dump(predictions, outfile)

def generate_visualizations(sample_outputs, output_path='visualization.html'):
    with open(sample_outputs) as json_file:
        data = json.load(json_file)

    with open(output_path, 'w+') as output_file:
        for doc_id in data.keys():
            doc = data[doc_id]
            doc_words = doc['words']
            clusters = doc['predicted_clusters']
            event_mentions = flatten(clusters)
            output_file.write('<b>Document {}</b><br>'.format(doc_id))
            output_file.write('{}<br><br><br>'.format(doc_to_html(doc, event_mentions)))
            for ix, cluster in enumerate(doc['predicted_clusters']):
                if len(cluster) == 1: continue
                output_file.write('<b>Cluster {}</b></br>'.format(ix+1))
                for em in cluster:
                    output_file.write('{}<br>'.format(event_mentions_to_html(doc_words, em)))
                output_file.write('<br><br>')
            output_file.write('<br><hr>')

def doc_to_html(doc, event_mentions):
    doc_words = doc['words']
    doc_words = [str(word) for word in doc_words]
    for e in event_mentions:
        t_start, t_end = e['trigger']['start'], e['trigger']['end'] - 1
        doc_words[t_start] = '<span style="color:blue">' + doc_words[t_start]
        doc_words[t_end] = doc_words[t_end] + '</span>'
    return ' '.join(doc_words)

def event_mentions_to_html(doc_words, em):
    trigger_start = em['trigger']['start']
    trigger_end = em['trigger']['end']
    context_left = ' '.join(doc_words[trigger_start-10:trigger_start])
    context_right = ' '.join(doc_words[trigger_end:trigger_end+10])
    final_str = context_left + ' <span style="color:red">' + em['trigger']['text'] + '</span> ' + context_right
    final_str = '<i>Event {} (Type {}) </i> | '.format(em['id'], em['event_type']) + final_str
    return final_str

def evaluate_and_visualize(config_name, model_path, output_path):
    # Prepare tokenizer, dataset, and model
    configs = prepare_configs(config_name, verbose=False)
    tokenizer = BertTokenizer.from_pretrained(configs['transformer'])
    train_set, dev_set, test_set = load_oneie_dataset(configs['base_dataset_path'], tokenizer)
    model = EventCorefModel(configs, train_set.event_types)

    # Reload the model and evaluate
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Evaluation on the dev set', flush=True)
    evaluate(model, dev_set, configs)['avg']
    print('Evaluation on the test set', flush=True)
    evaluate(model, test_set, configs)

    # Generate visualizations (for the test set)
    generate_coref_preds(model, test_set, '_predictions.json')
    generate_visualizations('_predictions.json', output_path)
    os.remove('_predictions.json')

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_name')
    parser.add_argument('-m', '--model_path')
    parser.add_argument('-o', '--output_path', default='visualization.html')
    args = parser.parse_args()

    # Start training
    evaluate_and_visualize(args.config_name, args.model_path, args.output_path)

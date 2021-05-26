import os
import math
import numpy as np
import tempfile
import subprocess
import torch
import re

from boltons.iterutils import pairwise, windowed
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

COREF_RESULTS_REGEX = re.compile(r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)
BLANC_RESULTS_REGEX = re.compile(r".*BLANC: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)

def evaluate(model, eval_set, configs, verbose=True):
    return evaluate_coref(model, eval_set, configs, verbose)

def evaluate_coref(model, eval_set, configs, verbose=True):
    predictions = []
    for inst in eval_set:
        # Apply the model for prediction
        with torch.no_grad():
            loss, preds = model(inst, is_training=False)
        preds = [x.cpu().data.numpy() for x in preds]
        top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = preds
        predicted_antecedents = get_predicted_antecedents(top_antecedents, top_antecedent_scores)

        predicted_clusters, m2cluster = [], {}
        for ix, (s, e) in enumerate(zip(top_span_starts, top_span_ends)):
            if predicted_antecedents[ix] < 0:
                cluster_id = len(predicted_clusters)
                predicted_clusters.append([(s, e)])
            else:
                antecedent_idx = predicted_antecedents[ix]
                p_s, p_e = top_span_starts[antecedent_idx], top_span_ends[antecedent_idx]
                cluster_id = m2cluster[(p_s, p_e)]
                predicted_clusters[cluster_id].append((s,e))
            m2cluster[(s,e)] = cluster_id
        predictions.append(m2cluster)


    with tempfile.NamedTemporaryFile(delete=False, mode='w') as gold_file:
        output_gold_conll(gold_file, eval_set.data)
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as prediction_file:
            for ix, inst in enumerate(eval_set.data):
                doc_id = inst.doc_id
                m2cluster = predictions[ix]
                cluster_labels = ['-'] * inst.num_words
                for (start, end) in m2cluster.keys():
                    c_label = m2cluster[(start, end)]
                    end = end - 1
                    if start == end:
                        cluster_labels[start] = '({})'.format(c_label)
                    else:
                        cluster_labels[start] = '({}'.format(c_label)
                        cluster_labels[end] = '{})'.format(c_label)

                # Write the doc info to output file
                prediction_file.write('#begin document ({}); part 000\n'.format(doc_id))
                for i in range(inst.num_words):
                    prediction_file.write('{} {}\n'.format(doc_id, cluster_labels[i]))
                prediction_file.write('\n')
                prediction_file.write('#end document\n')

            gold_file.flush()
            prediction_file.flush()
            print("Gold conll file: {}".format(gold_file.name))
            print("Prediction conll file: {}".format(prediction_file.name))
            metrics = ("muc", "bcub", "ceafe", "blanc", "ceafm")
            summary = { m: official_conll_eval(gold_file.name, prediction_file.name, m) for m in metrics}
            os.remove(gold_file.name)
            os.remove(prediction_file.name)

            avg = 0.0
            for metric in metrics[:-1]: avg += summary[metric]['f'] # Excluding ceafm when calculating avg
            avg /= len(metrics[:-1])
            summary['avg'] = avg

            summary_text = ''
            for metric in metrics:
                summary_text += '[{}] F1 = {} | '.format(metric, summary[metric]['f'])
            summary_text +=  'AVG = {}'.format(avg)
            print(summary_text)

            return summary


def official_conll_eval(gold_path, predicted_path, metric, official_stdout=False):
    cmd = ["reference-coreference-scorers-8.01/scorer.pl", metric, gold_path, predicted_path, "none"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    if stderr is not None:
        print(stderr)

    if official_stdout:
        print("Official result for {}".format(metric))
        print(stdout)

    regexp = COREF_RESULTS_REGEX if metric != 'blanc' else BLANC_RESULTS_REGEX
    coref_results_match = re.match(regexp, stdout)
    recall = float(coref_results_match.group(1))
    precision = float(coref_results_match.group(2))
    f1 = float(coref_results_match.group(3))
    return { "r": recall, "p": precision, "f": f1 }

def get_predicted_antecedents(antecedents, antecedent_scores):
    predicted_antecedents = []
    for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
        if index < 0: predicted_antecedents.append(-1)
        else: predicted_antecedents.append(antecedents[i, index])
    return predicted_antecedents

def output_gold_conll(gold_file, documents):
    for doc in documents:
        doc_id = doc.doc_id

        # Build cluster_labels
        eventid2label = {}
        cluster_labels = ['-'] * doc.num_words
        for e in doc.event_mentions:
            mention_id = e['id']
            event_id = mention_id[:mention_id.rfind('-')]
            if not event_id in eventid2label:
                eventid2label[event_id] = 1 + len(eventid2label)
            start_idx, end_idx = e['trigger']['start'], e['trigger']['end']-1
            if start_idx == end_idx:
                cluster_labels[start_idx] = '({})'.format(eventid2label[event_id])
            else:
                cluster_labels[start_idx] = '({}'.format(eventid2label[event_id])
                cluster_labels[end_idx] = '{})'.format(eventid2label[event_id])

        # Write the doc info to output file
        gold_file.write('#begin document ({}); part 000\n'.format(doc_id))
        for i in range(doc.num_words):
            gold_file.write('{} {}\n'.format(doc_id, cluster_labels[i]))
        gold_file.write('\n')
        gold_file.write('#end document\n')

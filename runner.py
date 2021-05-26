import os
import math
import torch
import tqdm
import random
import time

from transformers import *
from models import EventCorefModel
from utils import RunningAverage, prepare_configs, get_n_params
from scorer import evaluate
from data import load_oneie_dataset
from argparse import ArgumentParser

def train(config_name):
    # Prepare tokenizer, dataset, and model
    configs = prepare_configs(config_name)
    tokenizer = AutoTokenizer.from_pretrained(configs['transformer'])
    predictions_path = None if configs['use_groundtruth'] else configs['predictions_path']
    train_set, dev_set, test_set = \
        load_oneie_dataset(configs['base_dataset_path'], tokenizer, predictions_path,
                           increase_ace_dev_set=configs['increase_ace_dev_set'])
    model = EventCorefModel(configs, train_set.event_types)
    print('Initialized tokenier, dataset, and model')
    print('Number of parameters is {}'.format(get_n_params(model)))

    # Initialize the optimizer
    num_train_docs = len(train_set)
    epoch_steps = int(math.ceil(num_train_docs / configs['batch_size']))
    num_train_steps = int(epoch_steps * configs['epochs'])
    num_warmup_steps = int(num_train_steps * 0.1)
    optimizer = model.get_optimizer(num_warmup_steps, num_train_steps)
    print('Initialized optimizer')

    # Main training loop
    best_dev_score, iters, batch_loss = 0.0, 0, 0
    for epoch in range(configs['epochs']):
        #print('Epoch: {}'.format(epoch))
        print('\n')
        progress = tqdm.tqdm(total=epoch_steps, ncols=80,
                             desc='Train {}'.format(epoch))
        accumulated_loss = RunningAverage()

        train_indices = list(range(num_train_docs))
        random.shuffle(train_indices)
        start_train = time.time()
        for train_idx in train_indices:
            iters += 1
            inst = train_set[train_idx]
            iter_loss = model(inst, is_training=True)[0]
            iter_loss /= configs['batch_size']
            iter_loss.backward()
            batch_loss += iter_loss.data.item()
            if iters % configs['batch_size'] == 0:
                accumulated_loss.update(batch_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), configs['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                batch_loss = 0
                # Update progress bar
                progress.update(1)
                progress.set_postfix_str('Average Train Loss: {}'.format(accumulated_loss()))
        progress.close()
        print('One epoch training took {} seconds'.format(time.time() - start_train))

        # Evaluation after each epoch
        print('Evaluation on the dev set', flush=True)
        start_dev = time.time()
        dev_score = evaluate(model, dev_set, configs)['avg']
        print('Evaluation on dev set took {} seconds'.format(time.time() - start_dev))

        # Save model if it has better dev score
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            # Save the model
            save_path = os.path.join(configs['saved_path'], 'model.pt')
            torch.save({'model_state_dict': model.state_dict()}, save_path)
            print('Saved the model', flush=True)
            # Evaluation on the test set
            print('Evaluation on the test set', flush=True)
            start_test = time.time()
            evaluate(model, test_set, configs)
            print('Evaluation on test set took {} seconds'.format(time.time() - start_test))

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_name', default='basic')
    args = parser.parse_args()

    # Start training
    train(args.config_name)

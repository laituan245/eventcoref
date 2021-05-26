import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import random

from transformers import *
from math import ceil, floor

# Optimizer
class ModelOptimizer(object):
    def __init__(self, transformer_optimizer, transformer_scheduler,
                 task_optimizer, task_init_lr, max_iter):
        self.iter = 0
        self.transformer_optimizer = transformer_optimizer
        self.transformer_scheduler = transformer_scheduler

        self.task_optimizer = task_optimizer
        self.task_init_lr = task_init_lr
        self.max_iter = max_iter

    def zero_grad(self):
        self.transformer_optimizer.zero_grad()
        self.task_optimizer.zero_grad()

    def step(self):
        self.iter += 1
        self.transformer_optimizer.step()
        self.task_optimizer.step()
        self.transformer_scheduler.step()
        self.poly_lr_scheduler(self.task_optimizer, self.task_init_lr, self.iter, self.max_iter)

    @staticmethod
    def poly_lr_scheduler(optimizer, init_lr, iter, max_iter,
                          lr_decay_iter=1, power=1.0):
        """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param max_iter is number of maximum iterations
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param power is a polymomial power
        """
        if iter % lr_decay_iter or iter > max_iter:
            return optimizer

        lr = init_lr*(1 - iter/max_iter)**power
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

# BaseModel
class BaseModel(nn.Module):
    def __init__(self, configs):
        super(BaseModel, self).__init__()
        self.configs = configs
        self.device = torch.device('cuda' if torch.cuda.is_available() and not configs['no_cuda'] else 'cpu')

    def get_optimizer(self, num_warmup_steps, num_train_steps, start_iter = 0):
        # Extract transformer parameters and task-specific parameters
        transformer_params, task_params = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "transformer.encoder" in name:
                    transformer_params.append((name, param))
                else:
                    task_params.append((name, param))

        # Prepare transformer_optimizer and transformer_scheduler
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in transformer_params if not any(nd in n for nd in no_decay)], 'weight_decay': self.configs['transformer_weight_decay']},
            {'params': [p for n, p in transformer_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        transformer_optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.configs['transformer_learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-06,
        )
        transformer_scheduler = get_linear_schedule_with_warmup(transformer_optimizer,
                                                                num_warmup_steps=num_warmup_steps,
                                                                num_training_steps=num_train_steps)

        # Prepare the optimizer for task-specific parameters
        task_optimizer = optim.Adam([p for n, p in task_params], lr=self.configs['task_learning_rate'])

        # Unify transformer_optimizer and task_optimizer
        model_optimizer = ModelOptimizer(transformer_optimizer, transformer_scheduler,
                                         task_optimizer, self.configs['task_learning_rate'],
                                         num_train_steps)
        model_optimizer.iter = start_iter

        return model_optimizer

# FFNN Module
class FFNNModule(nn.Module):
    """ Generic FFNN-based Scoring Module
    """
    def __init__(self, input_size, hidden_sizes, output_size, dropout = 0.2):
        super(FFNNModule, self).__init__()
        self.layers = []

        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.ReLU(True))
            self.layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.layers.append(nn.Linear(prev_size, output_size))

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out.squeeze()

# FeatureSelectionModule
class FeatureSelectionModule(nn.Module):
    def __init__(self, latent_size, combine_strategy):
        super(FeatureSelectionModule, self).__init__()

        self.latent_size = latent_size
        self.combine_strategy = combine_strategy
        assert(combine_strategy in ['simple', 'gated'])

        if combine_strategy == 'gated':
            # Gate Computation Parameters
            self.Wu = nn.Linear(2 * latent_size, latent_size)

    def forward(self, x1, x2):
        if self.combine_strategy == 'simple':
            return x2
        if self.combine_strategy == 'gated':
            x = torch.cat([x1, x2], dim=-1)
            # Orthogonal Decomposition
            x1_dot_x2 = torch.sum(x1 * x2, dim=-1, keepdim=True)
            x1_dot_x1 = torch.sum(x1 * x1, dim=-1, keepdim=True)
            parallel = (x1_dot_x2 / x1_dot_x1) * x1
            orthogonal = x2 - parallel
            # Gates
            ug = torch.sigmoid(self.Wu(x))
            x2_prime = (1 - ug) * parallel + ug * orthogonal
            return x2_prime

# FeatureFusionNetwork
class FeatureFusionNetwork(nn.Module):
    def __init__(self, latent_size, combine_strategy, nb_modules):
        super(FeatureFusionNetwork, self).__init__()

        self.latent_size = latent_size
        self.combine_strategy = combine_strategy
        self.nb_modules = nb_modules

        modules = []
        for _ in range(nb_modules):
            modules.append(FeatureSelectionModule(latent_size, combine_strategy))
        self.fusion_modules = nn.ModuleList(modules)

    def forward(self, c, xs):
        features = [c]
        for module, x in zip(self.fusion_modules, xs):
            features.append(module(c, x))
        return torch.cat(features, dim=-1)

    @property
    def output_size(self):
        return (self.nb_modules + 1) * self.latent_size

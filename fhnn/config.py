import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'threshold': (0.3598, 'PLV matrix to binary adjacency matrix threshold'),
        'sbj-index': (0, 'subject index for single FHNN training on subject PLV matrix; ranges from 0 to NUM_SBJS - 1'),
        'lr': (0.01, 'learning rate'),
        'dropout': (0.0, 'dropout probability'),
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (150, 'maximum number of epochs to train for'),
        # 'epochs': (150, 'maximum number of epochs to train for'),
        # 'epochs': (300, 'maximum number of epochs to train for'),
        # 'epochs': (500, 'maximum number of epochs to train for'),
        # 'epochs': (1000, 'maximum number of epochs to train for'),
        # 'epochs': (5000, 'maximum number of epochs to train for'),
        'weight-decay': (0., 'l2 regularization strength'),
        'optimizer': ('radam', 'which optimizer to use, can be any of [rsgd, radam]'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (100, 'patience for early stopping'),
        'seed': (1234, 'seed for training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'save': (0, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'sweep-c': (0, ''),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        # 'lr-reduce-freq': (20, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        # 'gamma': (0.9, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (100, 'do not early stop before min-epochs')
    },
    'model_config': {
        'task': ('nc', 'which tasks to train on, can be any of [lp, nc]'),
        'model': ('GCN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN, HyboNet]'),
        'dim': (128, 'embedding dimension'),
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall, Lorentz]'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'margin': (2., 'margin of MarginLoss'),
        'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('None', 'which activation function to use (or None for no activation)'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'double-precision': ('0', 'whether to use double precision'),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not')
    },
    'data_config': {
        'dataset': ('cora', 'which dataset to use'),
        'val-prop': (0.05, 'proportion of validation edges for link prediction'),
        'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'use-feats': (1, 'whether to use node features or not'),
        # 'normalize-feats': (0, 'whether to normalize input node features'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        # 'normalize-adj': (0, 'whether to row-normalize the adjacency matrix'),
        'split-seed': (1234, 'seed for data splits (train/test/val)'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)

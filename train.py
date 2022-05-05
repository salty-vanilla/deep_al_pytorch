import numpy as np
import sys
import argparse
sys.path.append('./')
sys.path.append('./distil')
import torch
from torch.utils.data import Subset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase, MLFlowLogger, TensorBoardLogger
from omegaconf import DictConfig
import hydra

from distil.utils.models.resnet import ResNet18
from distil.active_learning_strategies import GLISTER, BADGE, EntropySampling, RandomSampling, LeastConfidenceSampling, \
                                        MarginSampling, CoreSet, AdversarialBIM, AdversarialDeepFool, KMeansSampling, \
                                        BALDDropout, FASS
from distil.utils.models.simple_net import TwoLayerNet
from distil.utils.train_help_module import TrainModule
from distil.utils.config_helper import read_config_file
from distil.utils.utils import LabeledToUnlabeledDataset
import time
import pickle

from dataset import Cifar10


np.random.seed(42)


def get_strategy(
    strategy_name: str,
    train_dataset: torch.utils.data.Subset,
    unlabeled_dataset: torch.utils.data.Subset,
    net: torch.nn.Module,
    n_classes: int,
    strategy_args: dict
):
    if strategy_name == 'badge':
        strategy = BADGE(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, n_classes, strategy_args)
    elif strategy_name == 'glister':
        strategy = GLISTER(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, n_classes, strategy_args,validation_dataset=None,\
                typeOf='Diversity',lam=10)
    elif strategy_name == 'entropy_sampling':
        strategy = EntropySampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, n_classes, strategy_args)
    elif strategy_name == 'margin_sampling':
        strategy = MarginSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, n_classes, strategy_args)
    elif strategy_name == 'least_confidence':
        strategy = LeastConfidenceSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, n_classes, strategy_args)
    elif strategy_name == 'coreset':
        strategy = CoreSet(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, n_classes, strategy_args)
    elif strategy_name == 'fass':
        strategy = FASS(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, n_classes, strategy_args)
    elif strategy_name == 'random_sampling':
        strategy = RandomSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, n_classes, strategy_args)
    elif strategy_name == 'bald_dropout':
        strategy = BALDDropout(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, n_classes, strategy_args)
    elif strategy_name == 'adversarial_bim':
        strategy = AdversarialBIM(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, n_classes, strategy_args)
    elif strategy_name == 'kmeans_sampling':
        strategy = KMeansSampling(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, n_classes, strategy_args)
    elif strategy_name == 'adversarial_deepfool':
        strategy = AdversarialDeepFool(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset), net, n_classes, strategy_args)
    else:
        raise IOError('Enter Valid Strategy!')
    return strategy


def get_logger(
    logger_name: str,
    logger_args: dict
) -> LightningLoggerBase:
    if logger_name == 'mlflow':
        logger = MLFlowLogger(
            logger_args.experiment_name,
            tracking_uri=logger_args.uri
        )
    elif logger_name == 'tensorboard':
        logger = TensorBoardLogger(
            './'
        )
    else:
        raise ValueError
    return logger

@hydra.main(config_path='.', config_name='config')
def main(cfg: DictConfig):
    model = ResNet18(num_classes=cfg.model.target_classes, channels=cfg.model.channels)

    full_train_dataset, test_dataset = Cifar10(cfg.dataset.download_path)()
    n_samples = len(full_train_dataset)

    start_idxs = np.random.choice(n_samples, size=cfg.active_learning.initial_points, replace=False)
    train_dataset = Subset(full_train_dataset, start_idxs)
    unlabeled_dataset = Subset(full_train_dataset, list(set(range(len(full_train_dataset))) -  set(start_idxs)))

    strategy = get_strategy(
        cfg.active_learning.strategy,
        train_dataset,
        unlabeled_dataset,
        model,
        cfg.model.target_classes,
        cfg.active_learning.strategy_args
    )

    logger = get_logger(cfg.logger.name, cfg.logger.args)

    if cfg.logger.name == 'mlflow':
        logger.log_hyperparams(cfg)

    for round_dix in range(cfg.active_learning.rounds):
        print(f'Round: {round_dix+1}/{cfg.active_learning.rounds}')
        if cfg.logger.name == 'mlflow':
            logger.finalize('RUNNING')

        if round_dix > 0:
            print('selecting ... ', end='')
            idx = strategy.select(cfg.active_learning.budget)
            train_dataset = ConcatDataset([train_dataset, Subset(unlabeled_dataset, idx)])
            remain_idx = list(set(range(len(unlabeled_dataset))) - set(idx))
            unlabeled_dataset = Subset(unlabeled_dataset, remain_idx)
            strategy.update_data(train_dataset, LabeledToUnlabeledDataset(unlabeled_dataset))
            print('Done')

        trainer = pl.Trainer(
            logger=[logger],
            max_epochs=cfg.train_parameters.n_epochs,
            devices=1,
            accelerator='gpu',
            enable_checkpointing=False
        )

        train_module = TrainModule(model, optimizer='adam', lr=1e-3)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_parameters.batch_size, shuffle=True)
        trainer.fit(train_module, train_dataloader)
        strategy.update_model(model)

        trainer.save_checkpoint(f'round-{round_dix}.ckpt')
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.train_parameters.batch_size)
        trainer.test(train_module, test_dataloader)

if __name__ == '__main__':
    main()
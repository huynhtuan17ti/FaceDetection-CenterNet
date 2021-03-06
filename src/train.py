from os import confstr
from dataset_factory.data_helper import prepare_loader
from trainer.cnn_trainer import Trainer
import yaml

if __name__ == '__main__':
    config = yaml.safe_load(open('config.yaml'))
    train_loader, valid_loader = prepare_loader(config)
    cnn = Trainer(config, data_loaders=[train_loader, valid_loader])
    cnn.train()
    cnn.save_last_model()
from dataset_factory.custom_dataset import FaceDataset
from torch.utils.data import DataLoader
from .transform import Transform

def prepare_loader(config):
    train_ds = FaceDataset(config, config['train_path'], Transform(), mode = 'train')
    valid_ds = FaceDataset(config, config['valid_path'], Transform(), mode = 'valid')
    print('TRAIN:', len(train_ds))
    print('VALID:', len(valid_ds))

    train_loader = DataLoader(train_ds, batch_size=config['train_batch'], collate_fn=train_ds.collate_fn, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=config['valid_batch'], collate_fn=valid_ds.collate_fn)

    return train_loader, valid_loader
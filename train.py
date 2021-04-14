from dataset_factory.custom_dataset import FaceDataset
from losses.loss import CenterLoss
from config import Config
#from models.custom_resnet50.centernet import CenterNet
from models.resnet_dcn.get_model import create_model, load_model
from torch.utils.data import DataLoader
from scheduler.lr_scheduler import *
from tqdm import tqdm
import torch

def prepare_loader(cfg):
    train_ds = FaceDataset(cfg.train_path)
    train_loader = DataLoader(train_ds, batch_size=cfg.train_batch, collate_fn=train_ds.collate_fn, shuffle=True)

    valid_ds = FaceDataset(cfg.valid_path, mode='valid')
    valid_loader = DataLoader(valid_ds, batch_size=cfg.valid_batch, collate_fn=valid_ds.collate_fn)

    return train_loader, valid_loader

def train_one_epoch(epoch, net, train_loader, loss_func, optimizer):
    net.train()
    total_loss = 0
    total_hm_loss = 0
    total_reg_loss = 0

    pbar = tqdm(enumerate(train_loader), total = len(train_loader))
    for step, (gt) in pbar:
        gt = [i.cuda() if isinstance(i, torch.Tensor) else i for i in gt]

        optimizer.zero_grad()

        pred = net(gt[0])
        losses = loss_func(pred, gt)
        hm_loss, reg_loss = losses
        loss = sum(losses)

        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_hm_loss += hm_loss.item()
        total_reg_loss += reg_loss.item()
        description = f'epoch {epoch} || Hm loss: {total_hm_loss/(step+1):.6f} | Reg loss: {total_reg_loss/(step+1):.6} | Total loss: {total_loss/(step+1):.6f}'
        pbar.set_description(description)

def valid_one_epoch(epoch, net, valid_loader, loss_func):
    net.eval()
    total_loss = 0
    total_hm_loss = 0
    total_reg_loss = 0

    pbar = tqdm(enumerate(valid_loader), total = len(valid_loader))
    for step, (gt) in pbar:
        gt = [i.cuda() if isinstance(i, torch.Tensor) else i for i in gt]

        pred = net(gt[0])
        losses = loss_func(pred, gt)
        hm_loss, reg_loss = losses
        loss = sum(losses)

        total_loss += loss.item()
        total_hm_loss += hm_loss.item()
        total_reg_loss += reg_loss.item()
        description = f'epoch {epoch} || Hm loss: {total_hm_loss/(step+1):.6} | Reg loss: {total_reg_loss/(step+1):.6} | Total loss: {total_loss/(step+1):.6f}'
        pbar.set_description(description)

    return total_loss / (step + 1)

if __name__ == '__main__':
    cfg = Config()
    train_loader, valid_loader = prepare_loader(cfg)

    if cfg.custom_net:
        net = None # CenterNet(cfg).cuda()
    else:
        net = create_model(cfg.model_name, cfg.heads, cfg.head_conv).cuda()
        if cfg.pretrained:
            net = load_model(net, cfg.pretrained_path + cfg.pretrained_model)
            for param in net.parameters():
                param.requires_grad = True
            print('Load pretrained model successfully!')

    loss_func = CenterLoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr = cfg.lr, weight_decay=1e-6)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, 
                                                     gamma=cfg.gamma, last_epoch=-1)

    #net.load_state_dict(torch.load(cfg.save_path + '/' + 'resnet50.pth'))
    print('Start trainning ...')
    best_loss = 10**9
    for epoch in range(cfg.epochs):
        train_one_epoch(epoch, net, train_loader, loss_func, optimizer)
        with torch.no_grad():
            valid_loss = valid_one_epoch(epoch, net, valid_loader, loss_func)
            if best_loss > valid_loss:
                print('Save best loss at epoch {}!'.format(epoch))
                best_loss = valid_loss
                torch.save(net.state_dict(), cfg.save_path + '/' + cfg.model_name + '.pth')
        lr_schedule.step()
    print('Best loss: ', best_loss)

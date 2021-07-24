import os
import torch
from torch import nn
from losses.loss import CenterLoss
from tqdm import tqdm
from models.resnet_dcn.get_model import create_model, load_model 

class Trainer:
    def __init__(self, config, data_loaders = None):
        self.config = config
        self.device = torch.device('cpu' if config['gpu'] < 0 else 'cuda:%s' % config['gpu'])
        self.net = create_model(config['model_name'], config['CNN']['heads'], config['CNN']['head_conv']).to(self.device)
        if config['CNN']['pretrained']:
            model_path = os.path.join(config['CNN']['pretrained_path'], config['CNN']['pretrained_model'])
            self.net = load_model(self.net, model_path)
            for param in self.net.parameters():
                param.requires_grad = True
            print('Load pretrained model successfully!')
        
        cfg_opt = config['Optimizer']
        self.CenterLoss = CenterLoss(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = cfg_opt['lr'], 
                                            weight_decay = cfg_opt['weight_decay'])
        self.lr_schedule = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=cfg_opt['milestones'], 
                                                        gamma=cfg_opt['gamma'], last_epoch=-1)
        self.best_loss = 10**9
        
        if data_loaders != None:
            self.train_loader = data_loaders[0]
            self.valid_loader = data_loaders[1]
    
    def save_model(self):
        save_model_path = os.path.join(self.config['save_path'], self.config['model_name'] + '.pth')
        torch.save(self.net.state_dict(), save_model_path)

    def train(self):
        self.net.train()
        for epoch in range(self.config['Optimizer']['epochs']):
            total_loss = 0
            total_hm_loss = 0
            total_reg_loss = 0
            pbar = tqdm(enumerate(self.train_loader), total = len(self.train_loader))
            for step, (gt) in pbar:
                gt = [i.to(self.device) if isinstance(i, torch.Tensor) else i for i in gt]

                self.optimizer.zero_grad()

                pred = self.net(gt[0])
                losses = self.CenterLoss(pred, gt)
                hm_loss, reg_loss = losses
                loss = sum(losses)
                
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_hm_loss += hm_loss.item()
                total_reg_loss += reg_loss.item()
                description = f'epoch {epoch} || Hm loss: {total_hm_loss/(step+1):.4f} | Reg loss: {total_reg_loss/(step+1):.4} | Total loss: {total_loss/(step+1):.4f}'
                pbar.set_description(description)

            self.lr_schedule.step()
            valid_loss = self.validate(epoch, self.valid_loader)
            if self.best_loss > valid_loss:
                print('Save best loss at epoch {}!'.format(epoch))
                self.best_loss = valid_loss
                self.save_model()

    def validate(self, epoch, loader):
        self.net.eval()
        total_loss = 0
        total_hm_loss = 0
        total_reg_loss = 0

        with torch.no_grad():
            pbar = tqdm(enumerate(loader), total = len(loader))
            for step, (gt) in pbar:
                gt = [i.to(self.device) if isinstance(i, torch.Tensor) else i for i in gt]

                pred = self.net(gt[0])
                losses = self.CenterLoss(pred, gt)
                hm_loss, reg_loss = losses
                loss = sum(losses)

                total_loss += loss.item()
                total_hm_loss += hm_loss.item()
                total_reg_loss += reg_loss.item()
                description = f'epoch {epoch} || Hm loss: {total_hm_loss/(step+1):.4} | Reg loss: {total_reg_loss/(step+1):.4} | Total loss: {total_loss/(step+1):.4f}'
                pbar.set_description(description)

        return total_loss / (step + 1)

        
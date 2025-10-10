import torch
from dataReader import SurfDataset
from torch.utils.data import DataLoader, Dataset
import network
import torch.optim as optim
from torch_ema import ExponentialMovingAverage
import time

class Trainer(object):
    def __init__(self,opt):
        self.device = opt['device']
        self.max_epochs = opt['max_epochs']
        self.batch = opt['batch']
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.surfdata = SurfDataset(self.device, 'surf/trainTerm1.dat')
        self.dataloader = DataLoader(
            self.surfdata, batch_size=self.batch, shuffle=True, num_workers=4)
        print('data load done. dataloader num_workers=', self.dataloader.num_workers)
        self.surfmodel = network.SurfNetwork(
            device, self.surfdata.maxFeatId, self.surfdata.featLevelCnt, 4)
        self.surfmodel.to(self.device)
        self.optimizer = optim.Adam(
            self.surfmodel.parameters(), lr=0.001, weight_decay=5e-4)  # naive adam
        self.scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / 30000, 1))
        self.lr_scheduler = self.scheduler(self.optimizer)
        self.ema = ExponentialMovingAverage(
            self.surfmodel.parameters(), decay=0.95)
        self.criterion = torch.nn.MSELoss(reduction='none')
        pass
    def train(self):
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self.train_one_epoch()
            # if self.epoch % 10 == 0:
            #     self.save_checkpoint()

    def evaluate(self, loader, name=None):
        self.evaluate_one_epoch(loader, name)

    def train_one_epoch(self):
        total_loss = 0
        self.local_step = 0
        self.surfmodel.train()
        for rgb, dirEncode, featId in self.dataloader:
            start_time = time.time()
            self.local_step += 1
            self.global_step += 1
            self.optimizer.zero_grad()
            B, N, _ = featId.shape
            sigma, color = self.surfmodel(featId, dirEncode)
            # MSE loss
            # [B, N, 3] --> [B, N]
            loss = sigma*self.criterion(rgb.unsqueeze(
                1).tile(1, N, 1), color).mean(-1)
            loss.sum().backward()
            self.optimizer.step()
            if self.global_step % 2 == 0:
                showLoss = loss.detach().sum()/B
                end_time = time.time()                
                print(f"batch={B}, lr={self.lr_scheduler.get_lr()}, loss={showLoss},  cost={(end_time - start_time)/B}s")
                self.save_checkpoint()

    def save_checkpoint(self):
        torch.save(self.surfmodel.state_dict(), f'surf/{self.global_step}_{self.local_step}.save.pt')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = {'device': device, 'max_epochs': 100, 'batch': 512000}
    trainer = Trainer(opt)
    trainer.train()



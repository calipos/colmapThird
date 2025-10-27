import torch
from dataReader import SurfDataset
from torch.utils.data import DataLoader, Dataset
import network
import torch.optim as optim
from torch_ema import ExponentialMovingAverage
import time
import os
import numpy as np
class Trainer(object):
    def __init__(self,opt):
        self.device = opt['device']
        self.dataPath = opt['dataPath']
        self.max_epochs = opt['max_epochs']
        self.batch = opt['batch']
        self.dataload_num_workers = opt['dataload_num_workers']
        self.gridLevelCnt = opt['gridLevelCnt']
        self.eachGridFeatDim = opt['eachGridFeatDim']
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.surfdata = SurfDataset(self.device, self.dataPath)
        self.dataloader = DataLoader(
            self.surfdata, batch_size=self.batch, shuffle=True, num_workers=self.dataload_num_workers)
        print('data load done. dataloader num_workers=', self.dataloader.num_workers)
        self.surfmodel = network.SurfNetwork(
            device, self.surfdata.maxFeatId, self.surfdata.featLevelCnt, self.eachGridFeatDim)
        self.surfmodel.to(self.device)
        self.optimizer = optim.Adam(
            self.surfmodel.parameters(), lr=0.1, weight_decay=5e-4)  # naive adam
        self.scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / 5, 1))
        self.lr_scheduler = self.scheduler(self.optimizer)
        self.ema = ExponentialMovingAverage(
            self.surfmodel.parameters(), decay=0.95)
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.criterionSigma = torch.nn.MSELoss(reduction='mean')
        pass
    def train(self):
        # if os.path.exists('surf/160_20.save.pt'):
        #     self.surfmodel.load_state_dict(torch.load('surf/160_20.save.pt'))
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self.train_one_epoch()
            self.save_checkpoint()

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
            # sigma, color, max_indices = self.surfmodel(featId, dirEncode)
            sigma, colors = self.surfmodel(featId, dirEncode)
            # MSE loss
            # [B, N, 3] --> [B, N]
            #loss = sigma*self.criterion(rgb.unsqueeze(1).tile(1, N, 1), color).mean(-1) #softmax
            loss0 = (torch.log(sigma)*sigma).sum()
            loss1 = self.criterion(rgb, colors).sum()
            loss = -0.001*loss0+loss1
            loss.backward()
            self.optimizer.step()
            if self.global_step % 2 == 0:
                showLoss = loss.detach().sum()/B
                end_time = time.time()
                print(f"batch={B}, lr={self.lr_scheduler.get_last_lr()}, loss={showLoss},  cost={(end_time - start_time)/B}s")


    def save_checkpoint(self):
        torch.save(self.surfmodel.state_dict(),f'surf/{self.epoch}.pt')
def surfPtsConstruct(opt,trainDataPath,modelPath,outPath):
    device = torch.device('cpu')
    dataload_num_workers = opt['dataload_num_workers']
    if not os.path.exists(trainDataPath):
        print('not found : ', trainDataPath)
    if not os.path.exists(modelPath):
        print('not found : ', modelPath)
    surfdata = SurfDataset(device, trainDataPath)
    constructionloader = DataLoader(
        surfdata, 256, shuffle=False, num_workers=dataload_num_workers)
    surfmodel = network.SurfNetwork(
        device, surfdata.maxFeatId, surfdata.featLevelCnt, opt['eachGridFeatDim'])
    surfmodel.load_state_dict(torch.load(modelPath))
    surfmodel.to(device)
    maxValueKey = max(surfdata.featIdToPosEncode,
                      key=surfdata.featIdToPosEncode.get)
    quickQuery = torch.zeros([surfdata.featIdToPosEncode[maxValueKey]],dtype=torch.int32)
    for kv in surfdata.featIdToPosEncode:
        quickQuery[kv] = surfdata.featIdToPosEncode[kv]



    ptsId = set()
    with torch.no_grad():
        for rgb, dirEncode, featId in constructionloader:
            B, N, _ = featId.shape

            sigma, colors = surfmodel(featId, dirEncode)
            print(sigma)
            # featLevel0Ids = featId[torch.arange(B), max_indices, 0]
            # featLevel0Ids = featLevel0Ids.to(torch.device('cpu'))
            # ptsId.update(quickQuery[featLevel0Ids].numpy())    
               
            # featId_1 = featId.reshape(B*N, -1)
            # ptsId.update(quickQuery[featId.reshape(B*N, -1)[:, 0]].numpy())
            # ptsId.update(quickQuery[featId.reshape(B*N, -1).reshape(-1)].numpy())
            continue

    pts = np.zeros([len(ptsId),3])
    for i, posEncode in enumerate(ptsId):
        pts[i, 0], pts[i, 1], pts[i, 2] = surfdata.posEncodeToXyz(posEncode)
    np.savetxt(outPath, pts, delimiter=' ')
if __name__ == '__main__':
    # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    opt = {'device': device, 
        'dataPath': 'surf/trainTerm1.dat',
        'max_epochs': 100, 
        'batch': 512000, 
        'dataload_num_workers': 4, 
        'gridLevelCnt':2,
        'eachGridFeatDim':3}
    # trainer = Trainer(opt)
    # trainer.train()
    surfPtsConstruct(opt, 'surf/trainTerm1.dat','surf/15.pt', 'surf/asd.pts')



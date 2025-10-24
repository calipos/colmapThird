import numpy as np
import struct
import torch
from torch.utils.data import DataLoader, Dataset


class SurfDataset(Dataset):
    def __init__(self, device, dataPath,type='train', n_test=10):
        super().__init__()
        self.dataPath = dataPath
        self.device = device
        self.type = type  # train, val, test
        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = -1
        with open(dataPath, 'rb') as file:
            self.dataType = struct.unpack('i', file.read(4))[0]
            self.num_rays = struct.unpack('i', file.read(4))[0]
            print('num_rays = ', self.num_rays)
            self.featLevelCnt = struct.unpack('i', file.read(4))[0]
            self.potentialGridCnt = struct.unpack('i', file.read(4))[0]
            cnt = self.featLevelCnt*self.potentialGridCnt
            self.sphereHarmonicSize = struct.unpack('i', file.read(4))[0]
            rgbData = np.zeros(
                [self.num_rays, 3], dtype=np.float32)
            dirData = np.zeros(
                [self.num_rays, self.sphereHarmonicSize], dtype=np.float32)
            featsId = np.zeros(
                [self.num_rays*self.potentialGridCnt, self.featLevelCnt], dtype=np.int32)
            viewIds = np.zeros(self.num_rays, dtype=np.int32)
            imgPosXYs = np.zeros([self.num_rays, 2], dtype=np.int32)
            t0Pos = np.zeros([self.num_rays, 3], dtype=np.float32)
            if self.dataType == 1:
                print('dataType == 1')
                for i in range(self.num_rays):
                    rgbData[i, :] = np.array(struct.unpack('3f', file.read(12)))
                    dirData[i, :] = np.array(
                        struct.unpack('16f', file.read(4*16)))
                    featsId[self.potentialGridCnt*i:self.potentialGridCnt*i+self.potentialGridCnt, :] = np.array(
                        struct.unpack(str(cnt)+'I', file.read(4*cnt)), dtype=np.int32).reshape(self.potentialGridCnt, self.featLevelCnt)
                self.maxFeatId = np.max(featsId, axis=0)+1
                # b = [np.sum(self.maxFeatId[:n+1]) for n in range(len(self.maxFeatId))]
                # self.maxFeatId = np.array(b)
                self.maxFeatId = torch.cumsum(
                    torch.from_numpy(self.maxFeatId), axis=0)
                featsId = featsId + np.pad(self.maxFeatId[:-1], (1, 0),'constant', constant_values=(0))
                self.rgbData = torch.from_numpy(rgbData)
                self.dirData = torch.from_numpy(dirData)
                self.featsId = torch.from_numpy(featsId)
                self.rgbData = self.rgbData.to(self.device)
                self.dirData = self.dirData.to(self.device)
                self.featsId = self.featsId.to(self.device)
                del rgbData
                del dirData
                del featsId
                featIdToPosEncodeVectSize = struct.unpack('i', file.read(4))[0]
                featIdToPosEncodeVect = struct.unpack(
                    str(featIdToPosEncodeVectSize)+'i', file.read(4*featIdToPosEncodeVectSize))
                self.featIdToPosEncode = {}
                for i in range(featIdToPosEncodeVectSize//2):
                    i2=i*2
                    self.featIdToPosEncode[featIdToPosEncodeVect[i2]
                                               ] = featIdToPosEncodeVect[i2+1]
                self.resolutionX = struct.unpack('i', file.read(4))[0]
                self.resolutionY = struct.unpack('i', file.read(4))[0]
                self.resolutionZ = struct.unpack('i', file.read(4))[0]
                self.resolutionXY = self.resolutionX*self.resolutionY
                self.xStart = struct.unpack('f', file.read(4))[0]
                self.yStart = struct.unpack('f', file.read(4))[0]
                self.zStart = struct.unpack('f', file.read(4))[0]
                self.resolutionUnit = struct.unpack('f', file.read(4))[0]
            elif self.dataType == 2:
                print('dataType == 2')
                rgbData = np.array(struct.unpack(
                    str(3*self.num_rays)+'f', file.read(4*3*self.num_rays)), dtype=np.float32).reshape(-1, 3)
                dirData = np.array(struct.unpack(
                    str(16*self.num_rays)+'f', file.read(4*16*self.num_rays)), dtype=np.float32).reshape(-1, 16)
                featsId = np.array(struct.unpack(
                    str(self.potentialGridCnt*self.featLevelCnt*self.num_rays)+'I', file.read(4*self.potentialGridCnt*self.featLevelCnt*self.num_rays)), dtype=np.int32).reshape(self.num_rays*self.potentialGridCnt, self.featLevelCnt)
                viewIds = np.array(struct.unpack(
                    str(self.num_rays)+'i', file.read(4*self.num_rays)), dtype=np.int32)
                imgPosXYs = np.array(struct.unpack(
                    str(2*self.num_rays)+'i', file.read(4*2*self.num_rays)), dtype=np.int32).reshape(-1, 2)
                t0xyz = np.array(struct.unpack(
                    str(3*self.num_rays)+'f', file.read(4*3*self.num_rays)), dtype=np.float32).reshape(-1, 3)
                viewCnt = struct.unpack('i', file.read(4))[0]
                viewDistance = np.array([viewCnt])
                for v in range(viewCnt):
                    viewDistHeight = struct.unpack('i', file.read(4))[0]
                    viewDistWidth = struct.unpack('i', file.read(4))[0]
                    if len(viewDistance.shape)==1:
                        viewDistance = np.zeros(
                            [viewCnt, viewDistHeight, viewDistWidth], dtype=np.float32)
                    viewDistance[v, ...] = np.array(
                        struct.unpack(str(viewDistHeight*viewDistWidth)+'f', file.read(viewDistHeight*viewDistWidth*4)), dtype=np.float32).reshape(viewDistHeight, viewDistWidth)


                self.maxFeatId = np.max(featsId, axis=0)+1
                b = [np.sum(self.maxFeatId[:n+1])
                     for n in range(len(self.maxFeatId))]
                self.maxFeatId = np.array(b)
                featsId = featsId + \
                    np.pad(self.maxFeatId[:-1], (1, 0),
                           'constant', constant_values=(0))
                self.rgbData = torch.from_numpy(rgbData)
                self.dirData = torch.from_numpy(dirData)
                self.featsId = torch.from_numpy(featsId)
                self.viewIds = torch.from_numpy(viewIds)
                self.imgPosXYs = torch.from_numpy(imgPosXYs)
                self.t0xyz = torch.from_numpy(t0xyz)
                self.viewDistance = torch.from_numpy(viewDistance)

                self.rgbData = self.rgbData.to(self.device)
                self.dirData = self.dirData.to(self.device)
                self.featsId = self.featsId.to(self.device)
                self.viewIds = self.viewIds.to(self.device)
                self.imgPosXYs = self.imgPosXYs.to(self.device)
                self.t0xyz = self.t0xyz.to(self.device)
                self.viewDistance = self.viewDistance.to(self.device)
                del rgbData
                del dirData
                del featsId
                del viewIds
                del imgPosXYs
                del t0xyz
                del viewDistance
                featIdToPosEncodeVectSize = struct.unpack('i', file.read(4))[0]
                featIdToPosEncodeVect = struct.unpack(
                    str(featIdToPosEncodeVectSize)+'i', file.read(4*featIdToPosEncodeVectSize))
                self.featIdToPosEncode = {}
                for i in range(featIdToPosEncodeVectSize//2):
                    i2 = i*2
                    self.featIdToPosEncode[featIdToPosEncodeVect[i2]
                                           ] = featIdToPosEncodeVect[i2+1]
                self.resolutionX = struct.unpack('i', file.read(4))[0]
                self.resolutionY = struct.unpack('i', file.read(4))[0]
                self.resolutionZ = struct.unpack('i', file.read(4))[0]
                self.resolutionXY = self.resolutionX*self.resolutionY
                self.xStart = struct.unpack('f', file.read(4))[0]
                self.yStart = struct.unpack('f', file.read(4))[0]
                self.zStart = struct.unpack('f', file.read(4))[0]
                self.resolutionUnit = struct.unpack('f', file.read(4))[0]
        
    def posEncodeToXyz(self,posEncode):
        Z = posEncode//self.resolutionXY
        Y = posEncode % self.resolutionXY
        X = Y % self.resolutionX
        Y = Y // self.resolutionX
        return self.xStart+X*self.resolutionUnit, self.yStart+Y*self.resolutionUnit, self.zStart+Z*self.resolutionUnit

    def __len__(self):
        return self.rgbData.shape[0]

    def __getitem__(self, idx):
        return self.rgbData[idx, :], self.dirData[idx, :], self.featsId[self.potentialGridCnt*idx:self.potentialGridCnt*idx+self.potentialGridCnt, :],


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__=='__main__':
    surfdata = SurfDataset(device, 'surf/trainTerm1.dat')
    dataloader = DataLoader(surfdata, batch_size=3, shuffle=True)
    for batch in dataloader:
        print(batch)
        break

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import struct




class Bfm2019:
    def __init__(self,
                 bfm_folder='./BFM',
                 defaultFaceFile='model2019_bfm(47439p94464f).h5',
                 defaultHeadFile='model2019_fullHead(58203p116160f).h5',
                 defaultFaceLmIdxFile='index_mp468_from_model2019_47439p.npy'):
        self.bfm_folder=bfm_folder
        faceH5Path = os.path.join(bfm_folder, defaultFaceFile)
        headH5Path = os.path.join(bfm_folder, defaultHeadFile)
        faceLmIdxFile = os.path.join(bfm_folder, defaultFaceLmIdxFile)
        if not os.path.isfile(faceH5Path):
            print("not exists : ",faceH5Path)
            return
        if not os.path.isfile(headH5Path):
            print("not exists : ",headH5Path)
            return         
        if not os.path.isfile(faceLmIdxFile):
            print("not exists : ", faceLmIdxFile)
            return            
        with h5py.File(faceH5Path, 'r') as h5_file:
            shape_points = h5_file['shape/representer/points'][:]
            shape_cells = h5_file['shape/representer/cells'][:]
            shape_mean = h5_file['shape/model/mean'][:]
            shape_pcaBasis = h5_file['shape/model/pcaBasis'][:]
            expression_points = h5_file['expression/representer/points'][:]
            expression_cells = h5_file['expression/representer/cells'][:]
            expression_mean = h5_file['expression/model/mean'][:]
            expression_pcaBasis = h5_file['expression/model/pcaBasis'][:]

        randomShape = (shape_pcaBasis@np.random.rand(199,1)+shape_mean.reshape(-1,1)).reshape(47439,3)
        randomExpression = (expression_pcaBasis@np.random.rand(100, 1) +
                            expression_mean.reshape(-1, 1)).reshape(47439, 3)
        np.savetxt('randomShape.txt', randomShape, delimiter=' ')
        np.savetxt('randomExpression.txt', randomExpression, delimiter=' ')
        np.savetxt('randomFace.txt', randomExpression +
                   randomShape, delimiter=' ')

        # mean face shape. [3*N,1]
        self.mean_shape = (shape_mean+expression_mean).astype(np.float32)
        # identity basis. [3*N,80]
        self.id_base = shape_pcaBasis.astype(np.float32)
        # expression basis. [3*N,64]
        self.exp_base = expression_pcaBasis.astype(np.float32) 
        self.face_tri = np.array(shape_cells.T, dtype=np.int64)
        # vertex indices for 68 landmarks. starts from 0. [68,1]
        self.keypoints = np.load(faceLmIdxFile).astype(np.int64)
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value).to(device))

    def getMediapipe486BfmBase(self,):
        mediapipe486Len = len(self.keypoints)  # 只有480  里没有6个点的对应
        id_486part = np.zeros([mediapipe486Len*3, 199])
        exp_486part = np.zeros([mediapipe486Len*3, 100])
        mean_486part = np.zeros([mediapipe486Len*3, 1])
        for i in range(mediapipe486Len):
            I = int(self.keypoints[i])
            if I < 0:
                continue
            id_486part[3*i, ...] = self.id_base[3*I, ...]
            id_486part[3*i+1, ...] = self.id_base[3*I+1, ...]
            id_486part[3*i+2, ...] = self.id_base[3*I+2, ...]

            exp_486part[3*i, ...] = self.exp_base[3*I, ...]
            exp_486part[3*i+1, ...] = self.exp_base[3*I+1, ...]
            exp_486part[3*i+2, ...] = self.exp_base[3*I+2, ...]

            mean_486part[3*i, ...] = self.mean_shape[3*I, ...]
            mean_486part[3*i+1, ...] = self.mean_shape[3*I+1, ...]
            mean_486part[3*i+2, ...] = self.mean_shape[3*I+2, ...]
        return id_486part, exp_486part, mean_486part


if __name__ == '__main__':
    bfmDateFile = 'RegisterImgGui/123.bin'
    with open(bfmDateFile, 'rb') as f: 
        typeEncode = struct.unpack('<i', f.read(4))[0]
        rows = struct.unpack('<i', f.read(4))[0]
        cols = struct.unpack('<i', f.read(4))[0]
        if 1==typeEncode:
        
        # 读取后续 8 字节作为两个 float（小端序）
        float1, float2 = struct.unpack('<ff', f.read(8))
        print(f"浮点数: {float1}, {float2}")



    with h5py.File('models/model2019_face12.h5', 'r') as f:
        print("文件中的所有对象 (路径):") 
    facemodel2019 = Bfm2019('Deep3d/BFM')

    facemodel = ParametricFaceModel(facemodel2019.bfm_folder)
    shape_base = facemodel.id_base.reshape(-1,3,80)
    expression_base = facemodel.exp_base.reshape(-1,3,64)
    mean_base = facemodel.mean_shape.reshape(-1,3)
    shape_weight = np.zeros([80,1])
    expression_weight = np.zeros([64,1])
    vts = (shape_base@shape_weight + expression_base@expression_weight).reshape(-1,3)+mean_base
    vts=vts*100
    save.saveObj("bfm09.obj",vts,facemodel.face_tri)
    print()
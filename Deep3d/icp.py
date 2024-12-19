import numpy as np
import torch
import roma
def getRst(data2, data1):
    '''
    data2 = s*(R@data1)+t
    '''
    # 去中心
    data1_mean = np.mean(data1, axis=1)
    data2_mean = np.mean(data2, axis=1)
    data1_ = data1 - data1_mean.reshape(3,1)
    data2_ = data2 - data2_mean.reshape(3, 1)
    scale1 = np.mean(np.linalg.norm(data1_, axis=0))
    scale2 = np.mean(np.linalg.norm(data2_, axis=0))
    s = scale2/scale1
    data1_ = data1_*s
    # 协方差
    H = (np.dot(data1_, data2_.T)).T
    U, S, Vt = np.linalg.svd(H)
    # 旋转矩阵
    R = U.dot(Vt)
    if np.linalg.det(R) < 0:
        R[2, :] *= -1
    # 平移矩阵
    T = data2_mean - R.dot(data1_mean)*s

    return R,s, T


def getRt(data2, data1):
    '''
    data2 = R@data1+t
    '''
    # 去中心
    data1_mean = np.mean(data1, axis=1)
    data2_mean = np.mean(data2, axis=1)
    data1_ = data1 - data1_mean.reshape(3, 1)
    data2_ = data2 - data2_mean.reshape(3, 1)

    # 协方差
    H = (np.dot(data1_, data2_.T)).T
    U, S, Vt = np.linalg.svd(H)
    # 旋转矩阵
    R = U.dot(Vt)
    if np.linalg.det(R) < 0:
        R[2, :] *= -1
    # 平移矩阵
    T = data2_mean - R.dot(data1_mean)

    return R, T
if __name__ == '__main__':
    a = np.random.rand(3,4)
    r= np.random.rand(1,3)
    t= np.random.rand(3,1)
    R = roma.rotvec_to_rotmat(torch.tensor(r) ).numpy().astype(np.float32)
    b=(R@a).reshape(3,-1)+t
    print(R,t)
    print(getRst(b,a))
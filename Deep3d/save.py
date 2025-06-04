import numpy as np
import torch

def saveFacePts(facePts,path):
    if isinstance(facePts, torch.Tensor):
        if facePts.requires_grad:
            facePtsNp = facePts.detach().numpy()
        else:
            facePtsNp = facePts.numpy()
    else: facePtsNp = facePts
    if facePtsNp.ndim==2:
        facePtsNp=facePtsNp.reshape([1,-1,3])
    if facePtsNp.ndim==3:
        headCnt = facePtsNp.shape[0] 
        np.savetxt(path, facePtsNp[0], fmt='%.18e', delimiter=' ') #保存为2位小数的浮点数，用逗号分隔
        for i in range(1,headCnt):
            np.savetxt(path+str(i)+'.pts', facePtsNp[i], fmt='%.18e', delimiter=' ') #保存为2位小数的浮点数，用逗号分隔
def ptsAddIndex(facePts,path):    	
    with open(path, 'w') as f:
        n=facePts.shape[0]
        for i in range(n):
            f.write('{:d} {:.6f} {:.6f} {:.6f}\n'.format(
                i+1, facePts[i, 0], facePts[i, 1], facePts[i, 2]))
def saveColorFacePts(facePts,face_texture,path):
    if isinstance(facePts, torch.Tensor):
        facePtsNp = facePts.numpy()
        face_textureNp = face_texture.numpy()
    else: 
        facePtsNp = facePts
        face_textureNp = face_texture
    if facePtsNp.ndim==2:
        facePtsNp=facePtsNp.reshape([1,-1,3])
    if face_textureNp.ndim==2:
        face_textureNp=face_textureNp.reshape([1,-1,3])        
    if facePtsNp.ndim==3:
        headCnt = facePtsNp.shape[0]  
        np.savetxt(path, np.concatenate([facePtsNp[0], face_textureNp[0]],axis=1), fmt='%.18e', delimiter=' ') #保存为2位小数的浮点数，用逗号分隔
        for i in range(1,headCnt):
            np.savetxt(path+str(i)+'.pts', np.concatenate([facePtsNp[i], face_textureNp[i]],axis=1), fmt='%.18e', delimiter=' ') #保存为2位小数的浮点数，用逗号分隔

 

def saveObj(filepath,verts,faces):
    thefile = open(filepath, 'w')
    for item in verts:
        thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2])) 
    for item in faces:
        thefile.write("f {0} {1} {2}\n".format(item[0]+1,item[1]+1,item[2]+1))  
    thefile.close()            
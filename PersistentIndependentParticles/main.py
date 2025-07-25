import time
import numpy as np
import io
import os
from PIL import Image
import cv2
import saverloader
import imageio.v2 as imageio
from nets.pips import Pips
import utils.improc
from utils.improc import ColorMap2d
import random
import glob
import torch
import torch.nn.functional as F  

device='cuda'  #cpu #cuda

def draw_circ_on_images_py(rgbPaths, traj, vis, linewidth=1, show_dots=False, cmap='coolwarm', maxdist=None):
 
    if device=='cuda' :traj=traj.cpu().numpy()
    B=1 
    T=8
    S = len(rgbPaths) 
    assert(S>0)
    firstImg = imageio.imread(rgbPaths[0])
    H = firstImg.shape[0]
    W = firstImg.shape[1]
    C = firstImg.shape[2]
    assert(C==3)
    S1, N, D = traj[0].shape
    assert(D==2)
    assert(S1==S)
    
  
    bremm = ColorMap2d()
    rgbs = []
    for fn in rgbPaths:
        im = imageio.imread(fn)
        im = im.astype(np.uint8)
        assert(im.shape[0]==H)
        assert(im.shape[1]==W)
        assert(im.shape[2]==3)
        rgbs.append(im)
    
    
 
    traj_ = traj[0,0,:].astype(np.float32)
    traj_[:,0] /= float(W)
    traj_[:,1] /= float(H)
    color = bremm(traj_)
    # print('color', color)
    color = (color*255).astype(np.uint8)  
    color = color.astype(np.int32) 
    print('color', color)
    traj = traj.astype(np.int32) 
    x = np.clip(traj[0,0,:,0], 0, W-1).astype(np.int32) 
    y = np.clip(traj[0,0,:,1], 0, H-1).astype(np.int32) 
    color_ = rgbs[0][y,x]
    for s in range(S):
        for n in range(N):
            #cv2.circle(rgbs[s], (traj[0,s,n,0], traj[0,s,n,1]), linewidth*4, color[n].tolist(), -1)
            cv2.circle(rgbs[s], (traj[0,s,n,0], traj[0,s,n,1]), linewidth*4, [0,255,0], -1)
            #vis_color = int(np.squeeze(vis[s])*255)
            #vis_color = (vis_color,vis_color,vis_color)
            #cv2.circle(rgbs[s], (traj[s,0], traj[s,1]), linewidth*2, vis_color, -1)            
    return rgbs

def run_model2(model, rgbPaths, xy0):
    B=1 
    T=8
    S = len(rgbPaths) 
    assert(S>0)
    firstImg = imageio.imread(rgbPaths[0])
    H = firstImg.shape[0]
    W = firstImg.shape[1]
    C = firstImg.shape[2]
    assert(C==3)
    

    
    B1,N,dim1 = xy0.shape
    assert(dim1==2)
    assert(B1==B)

    cur_frame = 0
    done = False
    traj_e = torch.zeros((B, S, N,2), dtype=torch.float32, device=device)
    traj_e[:,0] = xy0 # B,N,2 
    visTotal = torch.zeros((B, S, N), dtype=torch.float32, device=device)
    feat_init = None
    while not done:
        end_frame = cur_frame + T
        if end_frame>S:end_frame=S
        S_local = end_frame-cur_frame
        print("cur_frame:end_frame =",cur_frame,":",end_frame)
        rgbs = []
        for s in range(cur_frame,end_frame):
            fn = filenames[s]
            im = imageio.imread(fn)
            im = im.astype(np.uint8)
            rgbs.append(torch.from_numpy(im).permute(2,0,1))
        rgbs = torch.stack(rgbs, dim=0).unsqueeze(0) # 1, T, C, H, W
        if device=='cpu':            rgbs = rgbs.float() # B, T, C, H, W
        else:rgbs = rgbs.cuda().float() # B, T, C, H, W  
        rgbs_ = rgbs.reshape(B*S_local, C, H, W)
        H_, W_ = H, W  
        rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
        H, W = H_, W_
        rgbs = rgbs_.reshape(B, S_local, C, H, W)
        
        rgb_seq = rgbs[:,0:S_local]
        if S_local!=T:
            rgb_seq = torch.cat([rgb_seq, rgb_seq[:,-1].unsqueeze(1).repeat(1,T-S_local,1,1,1)], dim=1)
    

        outs = model(traj_e[:,cur_frame], rgb_seq, iters=6, feat_init=feat_init, return_feat=True)
        preds = outs[0]
        vis = outs[2] # B, T, 1
        feat_init = outs[3]

        vis = torch.sigmoid(vis) # visibility confidence
        xys = preds[-1].reshape(1, 8,N, 2)
        traj_e[:,cur_frame:end_frame] = xys[:,:S_local]
        visTotal[:,cur_frame:end_frame]=vis [:,:S_local]
        cur_frame = cur_frame + 8-1
        if cur_frame >= S:
            done = True 
    rgbs = draw_circ_on_images_py(rgbPaths,traj_e,None)
    #imageio.mimsave("test.gif",rgbs,fps=12)
    for s in range(S):        cv2.imwrite("%d.jpg" % (s),rgbs[s])
    np.savetxt('vis.txt',visTotal.reshape(-1,N).cpu().numpy(),fmt='%0.5f')
    return  traj_e,visTotal


if __name__ == '__main__':    
    
    exp_name = '00' # (exp_name is used for logging notes that correspond to different runs)
    init_dir = './reference_model'

    ## choose hyps
    B = 1
    S = 450
    xy0=torch.tensor([[384,350],[266,304]])
    xy0=xy0.reshape(B,-1,2)
    N = xy0.shape[1]  # number of points to track

    filenames = glob.glob('../viewer/*.jpg')
    filenames = sorted(filenames) 
    ## autogen a name
    model_name = "%02d_%d_%d" % (B, S, N)
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H%M%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)


    model = Pips(stride=8)
    if device != 'cpu' :model=model.cuda()
    parameters = list(model.parameters())
    if init_dir:
        _ = saverloader.load(init_dir, model)
    global_step = 0
    model.eval()

        

    try:
        with torch.no_grad():
            trajs_e = run_model2(model, filenames, xy0)

        
    except FileNotFoundError as e:
        print('error', e)
            

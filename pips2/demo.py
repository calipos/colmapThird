import time
import numpy as np
import saverloader
from nets.pips2 import Pips
from nets.pips2 import Pips_BasicEncoder
import utils.improc
from utils.basic import print_, print_stats
import torch
import torch.nn.functional as F
import sys
import cv2
import imageio.v2 as imageio
from pathlib import Path
import onnx
import onnxruntime
import onnx_graphsurgeon as gs
checkmodel = False
inferShapes = False
ir_version = 10
def read_mp4(fn):
    vidcap = cv2.VideoCapture(fn)
    frames = []
    while(vidcap.isOpened()):
        ret, frame = vidcap.read()
        if ret == False:
            break
        frames.append(frame)
    vidcap.release()
    return frames
class ColorMap2d:
    def __init__(self, filename=None):
        self._img = cv2.imread('models/bremm.png')
        
        self._height = self._img.shape[0]
        self._width = self._img.shape[1]

    def __call__(self, X):
        assert len(X.shape) == 2
        output = np.zeros((X.shape[0], 3))
        for i in range(X.shape[0]):
            x, y = X[i, :]
            xp = int(self._width * x)
            yp = int(self._height * y)
            output[i, :] = self._img[yp, xp]
        return output
def draw_circ_on_images_py(rgbs, traj, vis, linewidth=1, show_dots=False, cmap='coolwarm', maxdist=None):
    if isinstance(rgbs, torch.Tensor):
        rgbs = rgbs.numpy()
    traj=traj.cpu().numpy()
    if isinstance(traj, torch.Tensor):
        traj = traj.numpy()
    B=1 
    T=8
    rgbs=rgbs.squeeze()
    rgbs=np.transpose(rgbs,[0,2,3,1])
    S = len(rgbs) 
    assert(S>0)
    firstImg = rgbs[0]
    print(firstImg.shape)
    H = firstImg.shape[0]
    W = firstImg.shape[1]
    C = firstImg.shape[2]
    assert(C==3)
    S1, N, D = traj[0].shape
    assert(D==2)
    assert(S1==S)
    
  
    bremm = ColorMap2d()
    rgbsUint8 = []
    for fn in rgbs:       
        im = fn.astype(np.uint8)
        assert(im.shape[0]==H)
        assert(im.shape[1]==W)
        assert(im.shape[2]==3)
        rgbsUint8.append(im)
    
    
 
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
    color_ = rgbsUint8[0][y,x]
    for s in range(S):
        for n in range(N):
            #cv2.circle(rgbsUint8[s], (traj[0,s,n,0], traj[0,s,n,1]), linewidth*4, color[n].tolist(), -1)
            cv2.circle(rgbsUint8[s], (traj[0,s,n,0], traj[0,s,n,1]), linewidth*4, [0,255,0], -1)
            #vis_color = int(np.squeeze(vis[s])*255)
            #vis_color = (vis_color,vis_color,vis_color)
            #cv2.circle(rgbsUint8[s], (traj[s,0], traj[s,1]), linewidth*2, vis_color, -1)            
    return rgbsUint8

def run_model(model, rgbs, S_max=128, N=64, iters=16):
    rgbs = rgbs.cpu().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    assert(B==1)

    # pick N points to track; we'll use a uniform grid
    N_ = np.sqrt(N).round().astype(np.int32)
    grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device='cpu')
    grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16)
    grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16)
    #[1,1024,2]
    xy0 = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2
    _, S, C, H, W = rgbs.shape

    # zero-vel init [1,48,1024,2]
    trajs_e = xy0.unsqueeze(1).repeat(1,S,1,1)


    iter_start_time = time.time()
    
    preds, preds_anim, _, _ = model(trajs_e, rgbs, iters=iters, feat_init=None, beautify=True)
    trajs_e = preds[-1]

    iter_time = time.time()-iter_start_time
    print('inference time: %.2f seconds (%.1f fps)' % (iter_time, S/iter_time))

    return trajs_e


def main(
        filename='./models/camel.mp4',
        S=48, # seqlen
        N=1024, # number of points per clip
        stride=8, # spatial stride of the model
        timestride=1, # temporal stride of the model
        iters=16, # inference steps of the model
        image_size=(512,896), # input resolution
        max_iters=4, # number of clips to run
        init_dir='./models',
        device_ids=[0],
):

    # the idea in this file is to run the model on a demo video,
    # and return some visualizations
    
    exp_name = 'de00' # copy from dev repo

    print('filename', filename)
    name = Path(filename).stem
    print('name', name)
    
    rgbs = read_mp4(filename)
    rgbs = np.stack(rgbs, axis=0) # S,H,W,3
    rgbs = rgbs[:,:,:,::-1].copy() # BGR->RGB
    rgbs = rgbs[::timestride]
    S_here,H,W,C = rgbs.shape
    print('rgbs', rgbs.shape)

    # autogen a name
    model_name = "%s_%d_%d_%s" % (name, S, N, exp_name)
    import datetime
    model_date = datetime.datetime.now().strftime('%H_%M_%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    log_dir = 'logs_demo'
    print()

    global_step = 0

    model = Pips(stride=8).cpu()
    parameters = list(model.parameters())
    if init_dir:
        _ = saverloader.load(init_dir, model)
    global_step = 0
    model.eval()

    idx = list(range(0, max(S_here-S,1), S))
    if max_iters:
        idx = idx[:max_iters]
    
    for si in idx:
        global_step += 1
        
        iter_start_time = time.time()



        rgb_seq = rgbs[si:si+S]
        rgb_seq = torch.from_numpy(rgb_seq).permute(0,3,1,2).to(torch.float32) # S,3,H,W
        rgb_seq = F.interpolate(rgb_seq, image_size, mode='bilinear').unsqueeze(0) # 1,S,3,H,W
        
        with torch.no_grad():
            trajs_e = run_model(model, rgb_seq, S_max=S, N=N, iters=iters)

        print(rgbs.shape)
        rgbs = draw_circ_on_images_py(rgb_seq,trajs_e,None) 
        for s in range(S):        cv2.imwrite("%d.jpg" % (s),rgbs[s])
        return
        iter_time = time.time()-iter_start_time
        
        print('%s; step %06d/%d; itime %.2f' % (
            model_name, global_step, max_iters, iter_time))
def fix_baseEncoder_shape():
    model = onnx.load('models/pips2_base_opencv.onnx')
    graph = gs.import_onnx(model)
    for node in graph.nodes:
        if node.op == 'Resize':
            value = np.array([1, 2, 3, 4]).astype(np.int64)
            inputNames = node.name+"_shape"
            constValue = gs.Constant(inputNames, value)
            find=False
            for i in range(1,len(node.inputs)):
                if node.inputs[i].name.find('Concat')>0:
                    find=True
                    node.inputs[i] = constValue
            assert find
        if node.name == '/fnet/conv1/Conv':
            node.name = node.name+'_needSqueeze'
    graph.cleanup()
    graph.toposort()
    new_mode = gs.export_onnx(graph)
    new_mode.ir_version = ir_version
    onnx.save(new_mode, 'models/pips2_base_opencv.onnx')
def export_baseEncoder():           
    init_dir = './models'
    model = Pips_BasicEncoder(stride=8).cpu()
    parameters = list(model.parameters())
    if init_dir:
        _ = saverloader.load(init_dir, model)
    global_step = 0
    model.eval()
    dummy_input = torch.randn(8,3, 256, 256)
    input_names = ["rgbs"]        # 定义onnx 输入节点名称
    output_names = ["fmaps"]      # 定义onnx 输出节点名称
    onnx_path = "models/pips2_base_opencv.onnx"
    torch.onnx.export(
        model,
        (dummy_input),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=11,  # 确保兼容性
        dynamic_axes={'rgbs':  {2: "height", 3: 'width'},
                      'fmaps': {2: 'height', 3: 'width'}}
    )


    model = onnx.load('models/pips2_base_opencv.onnx')
    # model.graph.output.extend(
    #     [onnx.ValueInfoProto(name='/fnet/Concat_4_output_0')])
    # model.graph.output.extend(
    #     [onnx.ValueInfoProto(name='/fnet/Concat_5_output_0')])
    # model.graph.output.extend(
    #     [onnx.ValueInfoProto(name='/fnet/Concat_6_output_0')])
    # model.graph.output.extend(
    #     [onnx.ValueInfoProto(name='/fnet/Concat_7_output_0')])
    # model.graph.output.extend(
    #     [onnx.ValueInfoProto(name='/fnet/Resize_output_0')])
    # model.graph.output.extend(
    #     [onnx.ValueInfoProto(name='/fnet/Resize_1_output_0')])
    
    onnx.save(model, 'models/pips2_base_opencv.onnx')
    images = np.ones([8,3, 1024, 1024]).astype(np.float32)
    session = onnxruntime.InferenceSession(
        "models/pips2_base_opencv.onnx", providers=onnxruntime.get_available_providers())

    datain = {}
    datain['rgbs'] = images
    netOut = session.run(
        None, datain)

    for i in range(len(netOut)):
        print(netOut[i].shape)
        print(netOut[i])
        print(" ")

    # fix_baseEncoder_shape()
    return

if __name__ == '__main__':
    # main()
    export_baseEncoder()

import time
import numpy as np
import saverloader
from nets.pips2 import Pips
from nets.pips2 import Pips_BasicEncoder, Pips_CorrBlock2, Pips_DeltaBlock2
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
from onnx import helper
checkmodel = False
inferShapes = False
ir_version = 10
debug_self_data = True
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
    if debug_self_data:
        xy0 = torch.Tensor(
        [[[12.5, 1.2], [13.3, 45.1], [23.3, 15.1]]]).type(torch.float32)
    else:
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
    
    if not debug_self_data: 
        rgbs = read_mp4(filename)
        rgbs = np.stack(rgbs, axis=0) # S,H,W,3
        rgbs = rgbs[:,:,:,::-1].copy() # BGR->RGB
        rgbs = rgbs[::timestride]
        S_here,H,W,C = rgbs.shape
    else:
        imgPaths = ["D:/repo/colmapThird/data2/a/00000.jpg",
                    "D:/repo/colmapThird/data2/a/00001.jpg",
                    "D:/repo/colmapThird/data2/a/00002.jpg",
                    "D:/repo/colmapThird/data2/a/00003.jpg",
                    "D:/repo/colmapThird/data2/a/00004.jpg",
                    "D:/repo/colmapThird/data2/a/00005.jpg",
                    "D:/repo/colmapThird/data2/a/00006.jpg",
                    "D:/repo/colmapThird/data2/a/00007.jpg"]
        rgbs = []
        for i in range(len(imgPaths)):
            img = cv2.imread(imgPaths[i])
            rgbs.append(img)
        rgbs = np.stack(rgbs, axis=0)  # S,H,W,3
        # rgbs = rgbs[:, :, :, ::-1].copy()  # BGR->RGB
        rgbs = rgbs[::timestride]
        S_here, H, W, C = rgbs.shape
        image_size = (H, W)






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
        if node.name == '/fnet/Resize':
            value = np.array([8, 64, 128, 128]).astype(np.int64)
            inputNames = node.name+"_shape"
            constValue = gs.Constant(inputNames, value)
            find = False
            for i in range(1, len(node.inputs)):
                if node.inputs[i].name.find('Concat') > 0:
                    find = True
                    node.inputs[i] = constValue
            assert find
        if node.name == '/fnet/Resize_1':
            value = np.array([8, 96, 128, 128]).astype(np.int64)
            inputNames = node.name+"_shape"
            constValue = gs.Constant(inputNames, value)
            find = False
            for i in range(1, len(node.inputs)):
                if node.inputs[i].name.find('Concat') > 0:
                    find = True
                    node.inputs[i] = constValue
            assert find
        if node.name == '/fnet/Resize_2':
            value = np.array([8, 128, 128, 128]).astype(np.int64)
            inputNames = node.name+"_shape"
            constValue = gs.Constant(inputNames, value)
            find = False
            for i in range(1, len(node.inputs)):
                if node.inputs[i].name.find('Concat') > 0:
                    find = True
                    node.inputs[i] = constValue
            assert find
        if node.name == '/fnet/Resize_3':
            value = np.array([8, 128, 128, 128]).astype(np.int64)
            inputNames = node.name+"_shape"
            constValue = gs.Constant(inputNames, value)
            find = False
            for i in range(1, len(node.inputs)):
                if node.inputs[i].name.find('Concat') > 0:
                    find = True
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
    batch=1
    init_dir = './models'
    model = Pips_BasicEncoder(stride=8).cpu()
    if init_dir:
        _ = saverloader.load(init_dir, model)
    model.eval()
    dummy_input = torch.randn(batch, 3, 1024, 1024)
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


    # img = cv2.imread('D:/repo/colmapThird/data2/a/00000.jpg')
    # images = np.expand_dims(img.transpose(
    #     [2, 0, 1]), axis=0).astype(np.float32)
    images = np.ones([batch,3, 1024, 1024]).astype(np.float32)
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

    fix_baseEncoder_shape()
    return


def export_CorrBlock():
    batch = 1
    init_dir = './models'
    model = Pips_CorrBlock2(stride=8).cpu()
    if init_dir:
        _ = saverloader.load(init_dir, model)
    model.eval()
    dummy_fmaps = torch.randn(8*128, 120, 67)
    dummy_feats = torch.randn(8,3, 128)
    input_names = ["fmaps", "feats"]        # 定义onnx 输入节点名称
    output_names = ["corrs0", "corrs1", "corrs2", "corrs3"]      # 定义onnx 输出节点名称
    onnx_path = "models/pips2_corrBlock_opencv.onnx"
    torch.onnx.export(
        model,
        (dummy_fmaps, dummy_feats),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=11,  # 确保兼容性
        dynamic_axes={'fmaps':  {0:"seq",1: "height", 2: 'width'},
                      'feats': {0: "seq",1:"ptCnt"}}
    )


    return
    img = cv2.imread('D:/repo/colmapThird/data2/a/00000.jpg')
    images = np.expand_dims(img.transpose(
        [2, 0, 1]), axis=0).astype(np.float32)
    # images = np.ones([batch,3, 1024, 1024]).astype(np.float32)
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

    fix_baseEncoder_shape()
    return

def export_DeltaBlock():
    init_dir = './models'
    stride = 8
    hidden_dim = 256
    latent_dim = latent_dim = 128
    corr_levels = 4
    corr_radius = 3
    model = Pips_DeltaBlock2(8).cpu()
    if init_dir:
        _ = saverloader.load(init_dir, model)
    model.eval()
    dummy_deltaIn = torch.randn(3, 718, 8)
    input_names = ["deltaIn"]        # 定义onnx 输入节点名称
    output_names = ["delta"]      # 定义onnx 输出节点名称
    onnx_path = "models/pips2_deltaBlock_opencv.onnx"
    torch.onnx.export(
        model,
        (dummy_deltaIn),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=11,  # 确保兼容性
        dynamic_axes={'deltaIn':  {
            0: "controlPtCnt",  2: 'seq'}}
    )


    return

 

def test_bilinearOp():
    # v_y0_x0 = helper.make_tensor_value_info(
    #     'v_y0_x0', onnx.TensorProto.FLOAT, [32, 4])
    # v_y0_x1 = helper.make_tensor_value_info(
    #     'v_y0_x1', onnx.TensorProto.FLOAT, [32, 4])
    # v_y1_x0 = helper.make_tensor_value_info(
    #     'v_y1_x0', onnx.TensorProto.FLOAT, [32, 4])
    # v_y1_x1 = helper.make_tensor_value_info(
    #     'v_y1_x1', onnx.TensorProto.FLOAT, [32, 4])
    # w_y0_x0 = helper.make_tensor_value_info(
    #     'w_y0_x0', onnx.TensorProto.FLOAT, [4, 1])
    # w_y0_x1 = helper.make_tensor_value_info(
    #     'w_y0_x1', onnx.TensorProto.FLOAT, [4, 1])
    # w_y1_x0 = helper.make_tensor_value_info(
    #     'w_y1_x0', onnx.TensorProto.FLOAT, [4, 1])
    # w_y1_x1 = helper.make_tensor_value_info(
    #     'w_y1_x1', onnx.TensorProto.FLOAT, [4, 1])
    # output = helper.make_tensor_value_info(
    #     'output', onnx.TensorProto.FLOAT, [32, 1]) 
    # m1 = onnx.helper.make_node(
    #     op_type='MatMul',
    #     inputs=['v_y0_x0', 'w_y0_x0'],
    #     outputs=['m1'],
    #     name='m1')
    # m2 = onnx.helper.make_node(
    #     op_type='MatMul',
    #     inputs=['v_y0_x1', 'w_y0_x1'],
    #     outputs=['m2'],
    #     name='m2')
    # m3 = onnx.helper.make_node(
    #     op_type='MatMul',
    #     inputs=['v_y1_x0', 'w_y1_x0'],
    #     outputs=['m3'],
    #     name='m3')
    # m4 = onnx.helper.make_node(
    #     op_type='MatMul',
    #     inputs=['v_y1_x1', 'w_y1_x1'],
    #     outputs=['m4'],
    #     name='m4')
    # a1 = onnx.helper.make_node(
    #     op_type='Add',
    #     inputs=['m1', 'm2'],
    #     outputs=['a1'],
    #     name='a1')
    # a2 = onnx.helper.make_node(
    #     op_type='Add',
    #     inputs=['m3', 'm4'],
    #     outputs=['a2'],
    #     name='a2')
    # a3 = onnx.helper.make_node(
    #     op_type='Add',
    #     inputs=['a1', 'a2'],
    #     outputs=['output'],
    #     name='output')
    # graph = onnx.helper.make_graph(
    #     [m1,m2,m3,m4,a1,a2,a3],
    #     'TwoLayerFC',
    #     [v_y0_x0, v_y0_x1, v_y1_x0, v_y1_x1,w_y0_x0, w_y0_x1, w_y1_x0, w_y1_x1],
    #     [output]
    # )
    # model = helper.make_model(graph, producer_name='onnx-example')
    # model = onnx.shape_inference.infer_shapes(model)
    # onnx.checker.check_model(model)
    # model.ir_version = 10
    # model.opset_import[0].version = 21
    # onnx.save(model, 'test.onnx')

    shape=[8,128,64,64]
    inputData = np.array(
        [x % 200-100 for x in range(np.cumprod(shape)[-1])]).astype(np.float32).reshape(shape)
    inputData = torch.Tensor(inputData)
    if 3 == len(inputData.shape):
        inputData = inputData.unsqueeze(0)
    xy0 = torch.Tensor([[[12.5, 1.2], [13.3, 45.1], [23.3, 15.1]]]).type(torch.float32)
    coords = xy0.unsqueeze(1).repeat(1, 8, 1, 1)
    feat1 = utils.samp.bilinear_sample2d(
        inputData, coords[:, 0, :, 0], coords[:, 0, :, 1]).permute(0, 2, 1)


    print(feat1) 
    print(feat1.shape)

if __name__ == '__main__':
    print('opencv 的onnx 似乎要快一点,但是动态的shape总是调不好,ncnn直接操作param更方便')
    print('先运行export_baseEncoder()生成models/pips2_base_opencv.onnx,再运行ncnn的onnx_pips2')
    # test_bilinearOp()
    # main()
    # export_baseEncoder()
    # export_CorrBlock()
    export_DeltaBlock()

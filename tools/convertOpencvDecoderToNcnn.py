from typing import Any, Sequence
import os
from typing import Any, Sequence
import copy
import numpy as np
import onnx
import sys
from onnx import helper
from onnx import shape_inference
from onnx import TensorProto
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import onnx_graphsurgeon as gs
import onnxruntime
import matplotlib.pyplot as plt
checkmodel = False
inferShapes = False
point_coords = np.array(
    [[[10., 10.], [500., 400.], [200., 600.], [100., 300.], [200., 300.], [0, 0]]]).astype(np.float32)
point_labels = np.array([[1, 1, 1, 1, -1, 1]]).astype(np.float32)
ScatterND_1_output_0 = np.concatenate((point_coords, np.array(
    [[[0., 0.]]]).astype(np.float32)), axis=1)/1024.
Unsqueeze_8_output_0 = np.expand_dims(np.concatenate(
    (point_labels, np.array([[-1]]).astype(np.float32)), axis=1), 2)
high_res_feats_0 = np.ones([1, 32, 256, 256]).astype(np.float32)
high_res_feats_1 = np.ones([1, 64, 128, 128]).astype(np.float32)
image_embed = np.ones([1, 256, 64, 64]).astype(np.float32)
mask_input = np.zeros((1, 1, 1024//4, 1024//4), dtype=np.float32)
has_mask_input = np.array([1], dtype=np.float32)
orig_im_size = np.array([1080, 1920], dtype=np.int32)
shared_input = ['/Unsqueeze_8_output_0', '/ScatterND_1_output_0']
shared_out = [
    '/Concat_10_output_0'
]

targetParamPath = 'models/ncnn_decoder.onnx'


def cut_subgraph(origin_graph_path, input_node_name_list, output_node_name_list, sub_graph_path):
    graph = gs.import_onnx(onnx.load(origin_graph_path))
    tensors = graph.tensors()
    graph.inputs = []
    graph.outputs = []
    for input_node_name in input_node_name_list:
        graph.inputs.append(tensors[input_node_name])
    for output_node_name in output_node_name_list:
        graph.outputs.append(tensors[output_node_name])
    graph.cleanup()
    new_mode = gs.export_onnx(graph)
    new_mode.ir_version = 10
    onnx.save(new_mode, sub_graph_path)

def test_forward():
    cut_subgraph('models/opencv_decoder.onnx',
                 shared_input,
                 shared_out,
                 targetParamPath)
    model = onnx.load(targetParamPath)
    if inferShapes:
        model = onnx.shape_inference.infer_shapes(model)
    if checkmodel:
        onnx.checker.check_model(model)
    session = onnxruntime.InferenceSession(
        targetParamPath, providers=onnxruntime.get_available_providers())
    datain = {}
    if 'image_embed' in shared_input:
        datain['image_embed'] = image_embed
    if 'high_res_feats_0' in shared_input:
        datain['high_res_feats_0'] = high_res_feats_0
    if 'high_res_feats_1' in shared_input:
        datain['high_res_feats_1'] = high_res_feats_1
    if '/ScatterND_1_output_0' in shared_input:
        datain['/ScatterND_1_output_0'] = ScatterND_1_output_0
    if '/Unsqueeze_8_output_0' in shared_input:
        datain['/Unsqueeze_8_output_0'] = Unsqueeze_8_output_0
    if 'mask_input' in shared_input:
        datain['mask_input'] = mask_input
    if 'has_mask_input' in shared_input:
        datain['has_mask_input'] = has_mask_input
    if 'orig_im_size' in shared_input:
        datain['orig_im_size'] = orig_im_size
    pointCoords = session.run(
        None, datain)

    for i in range(len(shared_out)):
        print(shared_out[i])
        print(pointCoords[i].shape)
        print(pointCoords[i])
        print(" ")
    return pointCoords

if __name__ == '__main__':
    # test_matmul()
    # test_slice()
    # test_conv()
    # test_maxpool()
    # exit(0)
    onnxParamPath = 'models/opencv_decoder.onnx'
    if os.path.exists(onnxParamPath):
        a = test_forward()
        # b = convertOpencvOnnxToNcnn()
        # for i in range(len(a)):
        #     assert a[i].size == b[i].size
        #     diff = a[i].astype(np.double).reshape(-1) - \
        #         b[i].astype(np.double).reshape(-1)
        #     print(np.max(np.abs(diff)), np.mean(diff), np.std(diff))
        #     plt.plot(diff)
        #     plt.savefig(str(i)+'.png')
    else:
        print("need run sam2_onnx_adaptor first!!")

import os
import numpy as np
import onnx
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
checkmodel = False
inferShapes = False

shared_input = [
    'image',
]
shared_out = [
    '/image_encoder/trunk/blocks.0/attn/MatMul_1_output_0',
]
targetParamPath = 'models/ncnn_encoder.onnx'
image = np.ones([3, 1024, 1024]).astype(np.float32)
shape_set=[]
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
    # sys.stdout = open('convert_sam2_decoder_point_label.txt', 'w')
    cut_subgraph('models/opencv_encoder.onnx',
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
    if 'image' in shared_input:
        datain['image'] = image.reshape([1,3,1024,1024])
    pointCoords = session.run(
        None, datain)

    for i in range(len(shared_out)):
        print(shared_out[i])
        print(pointCoords[i].shape)
        print(pointCoords[i])
        print(" ")
    return

def shapeToStr(shape):
    assert len(shape) > 0
    shapeStr="shape_"
    for i in range(len(shape)):
        shapeStr = shapeStr+str(shape[i])
        shapeStr = shapeStr+'_'
    return shapeStr
def registerShapeNode(shapes):
    model = onnx.load(targetParamPath)
    for init in model.graph.initializer:
        if init.name.startswith("shape_"):
            shape_set.append(init.name)
    for shape in shapes:
        assert len(shape)>0
        shapeStr = shapeToStr(shape)
        if shapeStr not in shape_set:
            constShape = onnx.numpy_helper.from_array(
                np.array(shape).astype(np.int64), name=shapeStr)
            model.graph.initializer.append(constShape)
            shape_set.append(shapeStr)
    onnx.save(model, targetParamPath)
    
def modifyReshapeLayer(layerNamesAndtargetShape):
    newshapes=[]
    for shape in layerNamesAndtargetShape.keys():
        newshapes.append(layerNamesAndtargetShape[shape])
    registerShapeNode(newshapes)
    model = onnx.load(targetParamPath)
    modified=[]
    while True:
        if len(modified) == len(layerNamesAndtargetShape):
            break
        for i, node in enumerate(model.graph.node):
            if node.name in layerNamesAndtargetShape.keys() and node.name not in modified:
                layerInput = node.input
                layerOutput = node.output
                assert len(layerInput) == 2
                assert len(layerOutput) == 1                

                new_reshape_node = onnx.helper.make_node(
                    op_type="Reshape",
                    inputs=[node.input[0], shapeToStr(layerNamesAndtargetShape[node.name])],
                    outputs=node.output,
                    name=node.name
                )
                model.graph.node.remove(model.graph.node[i])
                model.graph.node.insert(i, new_reshape_node)
                modified.append(node.name)
                break
    onnx.save(model, targetParamPath)

def modifyTransposeLayer(layerNamesAndtargetPermute):
    model = onnx.load(targetParamPath)
    modified = []
    while True:
        if len(modified) == len(layerNamesAndtargetPermute):
            break
        for i, node in enumerate(model.graph.node):
            if node.name in layerNamesAndtargetPermute.keys() and node.name not in modified:
                layerInput = node.input
                layerOutput = node.output
                assert len(layerInput) == 1
                assert len(layerOutput) == 1
                new_transpose_node = onnx.helper.make_node(
                    op_type="Transpose",
                    inputs=node.input,
                    outputs=node.output,
                    name=node.name,
                    perm=layerNamesAndtargetPermute[node.name]
                )
                model.graph.node.remove(model.graph.node[i])
                model.graph.node.insert(i, new_transpose_node)
                modified.append(node.name)
                break
    onnx.save(model, targetParamPath)

def modifySplitLayer(layerNamesAndtargetSplit):
    newshapes = []
    for shape in layerNamesAndtargetSplit.keys():
        newshapes.append(layerNamesAndtargetSplit[shape]['split'])
    registerShapeNode(newshapes)
    model = onnx.load(targetParamPath)
    modified = []
    while True:
        if len(modified) == len(layerNamesAndtargetSplit):
            break
        for i, node in enumerate(model.graph.node):
            if node.name in layerNamesAndtargetSplit.keys() and node.name not in modified:
                layerInput = node.input
                assert len(layerInput) == 2
                new_split_node = onnx.helper.make_node(
                    op_type="Split",
                    inputs=[node.input[0], shapeToStr(
                        layerNamesAndtargetSplit[node.name]['split'])],
                    outputs=node.output,
                    name=node.name,
                    axis=layerNamesAndtargetSplit[shape]['axis']
                )
                model.graph.node.remove(model.graph.node[i])
                model.graph.node.insert(i, new_split_node)
                modified.append(node.name)
                break
    onnx.save(model, targetParamPath)

def deleteSqueezeLayer(layernames):
    model = onnx.load(targetParamPath)
    prevNodeName=[]
    nextNodeName=[]
    for deletingLayer in layernames:
        findNode=False
        for node in model.graph.node:
            if node.name == deletingLayer:
                findNode=True
                layerInput = node.input
                layerOutput = node.output
                assert len(layerInput) == 2
                assert len(layerOutput) == 1
                prevNodeName.append(layerInput[0])
                nextNodeName.append(layerOutput[0])
                break
        assert findNode
    modified = []
    while True:
        if len(modified) == len(layernames):
            break
        for i, node in enumerate(model.graph.node):
            if node.name in prevNodeName and node.name not in modified:
                layerInput = node.input
                layerOutput = node.output


                model.graph.node.remove(model.graph.node[i])
                # model.graph.node.insert(i, new_split_node)
                modified.append(node.name)
                break
    onnx.save(model, targetParamPath)
    

def convertOpencvOnnxToNcnn():
    model = onnx.load('models/opencv_encoder.onnx')
    print(len(model.graph.input))
    cut_subgraph('models/opencv_encoder.onnx',
                 shared_input,
                 shared_out,
                 targetParamPath)
    model = onnx.load(targetParamPath)
    print(model.graph.input[0].name)

# --------------------------------
    deleteSqueezeLayer(['/image_encoder/trunk/blocks.0/attn/Squeeze_2'])
# --------------------------------
    reshapeAndtargetShape = {}
    reshapeAndtargetShape['/image_encoder/trunk/blocks.0/Reshape']=[32,8,32,1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.0/Reshape_1'] = [65536, 144]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.0/attn/Reshape'] = [1024,64, 6,72]
    modifyReshapeLayer(reshapeAndtargetShape) 
    model = onnx.load(targetParamPath)
# --------------------------------
    transposeAndtargetShape = {}
    transposeAndtargetShape['/image_encoder/trunk/blocks.0/Transpose'] = [0, 2, 1, 3]
    modifyTransposeLayer(transposeAndtargetShape)
    model = onnx.load(targetParamPath)
# --------------------------------
    splitAndAxis = {}
    splitAndAxis['/image_encoder/trunk/blocks.0/attn/Split'] = {'axis': 2, 'split':[2, 2, 2]}
    modifySplitLayer(splitAndAxis)
    model = onnx.load(targetParamPath)
# --------------------------------
    if inferShapes:
        model = onnx.shape_inference.infer_shapes(model)
    if checkmodel:
        onnx.checker.check_model(model)
    onnx.save(model, targetParamPath)


    session = onnxruntime.InferenceSession(
        targetParamPath, providers=onnxruntime.get_available_providers())
    datain = {}
    if 'image' in shared_input:
        datain['image'] = image.reshape([1,3,1024,1024])
    pointCoords = session.run(
        None, datain)

    for i in range(len(shared_out)):
        print(shared_out[i])
        print(pointCoords[i].shape)
        print(pointCoords[i])
        print(" ")


if __name__=='__main__':
    onnxParamPath='models/opencv_encoder.onnx'
    if os.path.exists(onnxParamPath):
        test_forward()
        convertOpencvOnnxToNcnn()
    else:
        print("need run sam2_onnx_adaptor first!!")

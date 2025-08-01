import os
import copy
import numpy as np
import onnx
import sys
from onnx import helper
from onnx import shape_inference
from onnx import TensorProto
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils import data
import onnx_graphsurgeon as gs
import onnxruntime
import matplotlib.pyplot as plt
import cv2
checkmodel = False
inferShapes = False
ir_version = 10
# shared_input = ['/image_encoder/trunk/Add_1_output_0']
shared_out = [ 

    # '/image_encoder/trunk/blocks.33/attn/Transpose_3_output_0',
    # '/image_encoder/trunk/Transpose_4_output_0',

    # '/Transpose_1_output_0'
 
    'high_res_feats_0',
    'high_res_feats_1', 
    '/Transpose_1_output_0'  # == 'image_embed',   -->need reshape (256,64,64)

]

targetParamPath = 'models/ncnn_encoder.onnx'
encoderBeginnngPath = 'models/ncnn_encoder_beginnng.onnx'
image0 = np.ones([3, 1024, 1024]).astype(np.float32)
image1 = np.ones([3, 1024, 1024]).astype(np.float32)
# image0 = np.random.rand(3, 1024, 1024).astype(np.float32) 
data = np.random.rand(256* 256, 144).astype(np.float32) 
temp_input0 = data.reshape([1, 256, 256, 144]).astype(np.float32)
temp_input1 = data.reshape([256* 256, 144]).astype(np.float32)

shape_set=[]
const_set={}


def cnnOutFigure(input_size, kernelsize, stride, padding):
    if isinstance(input_size, int):
        input_size = (input_size,)
    if isinstance(kernelsize, int):
        kernelsize = (kernelsize,) * len(input_size)
    if isinstance(stride, int):
        stride = (stride,) * len(input_size)
    if isinstance(padding, int):
        padding = (padding,) * len(input_size)

    output = tuple((i - k + 2 * p) // s + 1 for i, k, s, p in zip(input_size, kernelsize, stride, padding))
    return output
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
    graph.toposort()
    new_mode = gs.export_onnx(graph)
    new_mode.ir_version = ir_version
    onnx.save(new_mode, sub_graph_path)
def onnx_datatype_to_npType(data_type):
    if data_type == 1:
        return np.float32
    elif data_type == 2:
        return np.uint8
    elif data_type == 3:
        return np.int8
    elif data_type == 4:
        return np.uint16
    elif data_type == 5:
        return np.int16
    elif data_type == 6:
        return np.int32
    elif data_type == 7:
        return np.int64
    elif data_type == 8:
        return np.string_
    elif data_type == 9:
        return np.bool_
    elif data_type == 10:
        return np.float16
    elif data_type == 11:
        return np.float64
    elif data_type == 12:
        return np.uint32
    elif data_type == 14:
        return np.uint64
    else:
        raise TypeError("don't support data type")


def shape2tuple(shape):
    return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)


def printNet(model):
    print('\n\n   #########################################  \n\n')
    # print(model.graph.input)

    # in a more nicely format
    for i, obj in enumerate(model.graph.input):
        print("** inputs ** %d  name=%r dtype=%r shape=%r" % (
            i, obj.name, obj.type.tensor_type.elem_type,
            shape2tuple(obj.type.tensor_type.shape)))

    # the list of outputs
    # print('** outputs **')
    # print(model.graph.output)

    # in a more nicely format
    for obj in model.graph.output:
        print("** outputs ** name=%r dtype=%r shape=%r" % (
            obj.name, obj.type.tensor_type.elem_type,
            shape2tuple(obj.type.tensor_type.shape)))

    # the list of nodes
    # print('** nodes **')
    # print(model.graph.node)

    # in a more nicely format
    for i, _ode in enumerate(model.graph.node):
        node = model.graph.node[i]
        print("** nodes %d **  name=%r type=%r input=%r output=%r" % (i,
                                                                      node.name, node.op_type, node.input, node.output))
        # if i>200:break
    # return
    initializer = model.graph.initializer
    # name_lists = ["/image_encoder/neck/position_encoding/Constant_28_output_0",
    #               '/image_encoder/neck/position_encoding/Unsqueeze_8_output_0']#encoder check
    # name_lists = ['/Constant_2_output_0',
    #               'onnx::Unsqueeze_916', '/Constant_5_output_0']#decoder point label
    # name_lists = ['onnx::Expand_2540',
    #               '/Where_6_output_0']  # decoder position embedding  onnx::MatMul_2551
    name_lists = ['mask_decoder.transformer.layers.0.norm1.weight']  # onehot
    for i in range(len(initializer)):
        print('** initializer ** ', i, '-', initializer[i].name)
        if initializer[i].name in name_lists:
            print(i, '-', initializer[i].name)
            print('shape dim= ', *initializer[i].dims)
            dtype = initializer[i].data_type
            params = np.frombuffer(
                initializer[i].raw_data, dtype=onnx_datatype_to_npType(dtype))
            print('data = ', params)

def test_total_forward():
    
    image = cv2.imread("D:/repo/colmap-third/a.bmp")
    img_height, img_width = image.shape[:2]
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (1024, 1024))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_img = (input_img / 255.0 - mean) / std
    input_img = input_img.transpose(2, 0, 1)
    imgInput = input_img[np.newaxis, :, :, :].astype(np.float32)
    imgInput = np.ones([1,3,1024,1024]).astype(np.float32)
    # model0 = onnx.load(encoderBeginnngPath)
    # model1 = onnx.load(targetParamPath)
    session = onnxruntime.InferenceSession(
        encoderBeginnngPath, providers=onnxruntime.get_available_providers())
    datain = {}
    datain['image'] = imgInput 
    beginningOut = session.run(None, datain)
 
    session2 = onnxruntime.InferenceSession(
        targetParamPath, providers=onnxruntime.get_available_providers())
    datain2 = {}
    datain2['/image_encoder/trunk/Add_1_output_0'] = beginningOut[0] 
    featOut = session2.run(None, datain2)

    session3 = onnxruntime.InferenceSession(
        'models/opencv_encoder.onnx', providers=onnxruntime.get_available_providers())
    datain3 = {}
    datain3['image'] = imgInput 
    featOut2 = session3.run(None, datain3)

    for i in range(3):
            assert featOut[i].size==featOut2[i].size
            diff = featOut[i].astype(np.double).reshape(-1)-featOut2[i].astype(np.double).reshape(-1)
            print(np.max(np.abs(diff)),np.mean(diff),np.std(diff))
            plt.plot(diff)  
            plt.savefig(str(i)+'.png')
    
    return 

    # sys.stdout = open('test_forward.txt', 'w')
    # model = onnx.load(targetParamPath)
    # graph = gs.import_onnx(model)
    # print(graph.nodes)
    # sys.stdout = sys.stdout
    return pointCoords

def test_forward():
    # sys.stdout = open('test_forward.txt', 'w')
    cut_subgraph('models/opencv_encoder.onnx',
                 ['image'],
                 shared_out,
                 'models/temp.onnx')
    model = onnx.load(targetParamPath)
    if inferShapes:
        model = onnx.shape_inference.infer_shapes(model)
    if checkmodel:
        onnx.checker.check_model(model)
    session = onnxruntime.InferenceSession(
        'models/temp.onnx', providers=onnxruntime.get_available_providers())
    datain = {}
    datain['image'] = image0.reshape([1,3,1024,1024])
    pointCoords = session.run(
        None, datain)

    for i in range(len(shared_out)):
        # print(pointCoords[i])
        print(shared_out[i])
        print(pointCoords[i].shape)
        print(" ") 


    # sys.stdout = open('test_forward.txt', 'w')
    # model = onnx.load(targetParamPath)
    # graph = gs.import_onnx(model)
    # print(graph.nodes)
    # sys.stdout = sys.stdout
    return pointCoords

def shapeToStr(shape):
    assert len(shape) > 0
    shapeStr="shape_"
    for i in range(len(shape)):
        shapeStr = shapeStr+str(shape[i])
        shapeStr = shapeStr+'_'
    return shapeStr


def registerShapeNode(shapes, paramPath=None):
    if paramPath is not None:
        model = onnx.load(paramPath)
    else :
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
    if paramPath is not None:
        onnx.save(model, paramPath)
    else:
        onnx.save(model, targetParamPath)


def modifyBinaryOperationConst(layerNames, paramPath=None):
    if paramPath is not None:
        model = onnx.load(paramPath)
    else:
        model = onnx.load(targetParamPath)
    graph = gs.import_onnx(model)
    for name in layerNames:
        pick = [node for node in graph.nodes if node.name == name]
        if len(pick) == 0:
            continue
        hitByName=False
        for i in range(len(pick[0].inputs)):
            if pick[0].inputs[i].name == '/image_encoder/trunk/blocks.0/attn/Sqrt_1_output_0':
                hitByName=True
                value = np.array(layerNames[name]['value']).astype(pick[0].inputs[i].dtype).reshape(pick[0].inputs[i].shape)
                inputNames = ''
                for j in range(len(pick[0].inputs)):
                    inputNames += pick[0].inputs[j].name
                    inputNames += '_'
                hashName = 'const_'+str(hash(inputNames))
                constValue = gs.Constant(hashName, value)
                pick[0].inputs[i] = constValue
            elif pick[0].inputs[i].name == layerNames[name]['constName']:
                hitByName=True
                value = np.array(layerNames[name]['value']).astype(pick[0].inputs[i].dtype).reshape(pick[0].inputs[i].shape)
                inputNames = ''
                for j in range(len(pick[0].inputs)):
                    inputNames += pick[0].inputs[j].name
                    inputNames += '_'
                hashName = 'const_'+str(hash(inputNames))
                constValue = gs.Constant(hashName, value)
                pick[0].inputs[i] = constValue                
        assert hitByName,pick[0]
    graph.cleanup()
    graph.toposort()
    new_mode = gs.export_onnx(graph)
    new_mode.ir_version = ir_version
    if paramPath is not None:
        onnx.save(new_mode, paramPath)
    else:
        onnx.save(new_mode, targetParamPath)


def modifyReshapeLayer(layerNamesAndtargetShape, paramPath=None):
    newshapes = []
    for shape in layerNamesAndtargetShape.keys():
        newshapes.append(layerNamesAndtargetShape[shape])
    registerShapeNode(newshapes, paramPath)
    if paramPath is not None:
        model = onnx.load(paramPath)
    else:
        model = onnx.load(targetParamPath)
    modified = []
    loopDone = False
    while True:
        if len(modified) == len(layerNamesAndtargetShape) or loopDone:
            break
        loopDone=False
        for i, node in enumerate(model.graph.node):
            if node.name in layerNamesAndtargetShape.keys() and node.name not in modified:
                # if node.name == '/image_encoder/trunk/blocks.22/Reshape_1':
                #     print(1)
                layerInput = node.input
                layerOutput = node.output
                assert len(layerInput) == 2,node
                assert len(layerOutput) == 1, node
                new_reshape_node = onnx.helper.make_node(
                    op_type="Reshape",
                    inputs=[node.input[0], shapeToStr(
                        layerNamesAndtargetShape[node.name])],
                    outputs=node.output,
                    name=node.name,
                    allowzero=0
                )
                model.graph.node.remove(model.graph.node[i])
                model.graph.node.insert(i, new_reshape_node)
                modified.append(node.name)
                break
            if i == len(model.graph.node)-1:
                loopDone = True
    if paramPath is not None:
        onnx.save(model, paramPath)
    else:
        onnx.save(model, targetParamPath)
    

def modifyTransposeLayer(layerNamesAndtargetPermute, paramPath=None):
    if paramPath is not None:
        model = onnx.load(paramPath)
    else:
        model = onnx.load(targetParamPath)
    modified = []
    loopDone = False
    while True:
        if len(modified) == len(layerNamesAndtargetPermute) or loopDone:
            break
        loopDone = False
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
            if i == len(model.graph.node)-1:
                loopDone = True
    if paramPath is not None:
        onnx.save(model, paramPath)
    else:
        onnx.save(model, targetParamPath)


def reshapeAddNode(paramPath=None):
    if paramPath is not None:
        model = onnx.load(paramPath)
    else:
        model = onnx.load(targetParamPath)
    graph = gs.import_onnx(model)
    pick = [node for node in graph.nodes if node.name == '/image_encoder/trunk/Add_1']
    if len(pick)==1:
        pick[0].inputs[0].shape = [256*256,144]
        value = np.array(pick[0].inputs[1].values).astype(
            pick[0].inputs[1].dtype).reshape(256*256,144)    
        hashName = 'const_'+str(hash(pick[0].name))
        constValue = gs.Constant(hashName, value)
        pick[0].inputs[1] = constValue
        pick[0].outputs[0].shape = [256*256,144]

    pick = [node for node in graph.nodes if node.name =='/Add']
    if len(pick) == 1:
        pick[0].inputs[0].shape = [256]
        value = np.array(pick[0].inputs[1].values).astype(
            pick[0].inputs[1].dtype).reshape(256)
        hashName = 'const_'+str(hash(pick[0].name))
        constValue = gs.Constant(hashName, value)
        pick[0].inputs[1] = constValue
        pick[0].outputs[0].shape = [256*256, 144]


    graph.cleanup()
    graph.toposort()
    new_mode = gs.export_onnx(graph)
    new_mode.ir_version = ir_version

    if paramPath is not None:
        onnx.save(new_mode, paramPath)
    else:
        onnx.save(new_mode, targetParamPath)

def insertSpecifiedReshape(layerPairs,paramPath=None):
    prevNodeIndex=[] 
    newReshapeNodes={}
    newshapes = []
    for specifiedPair in layerPairs.keys():
        newshapes.append(layerPairs[specifiedPair]['targetShape'])
    registerShapeNode(newshapes, paramPath)
    if paramPath is not None:
        model = onnx.load(paramPath)
    else:
        model = onnx.load(targetParamPath)
    for i, node in enumerate(model.graph.node):
        if node.name in layerPairs.keys():
            prevNodeIndex.append(i)
            insert_Reshape_node = onnx.helper.make_node(
                op_type='Reshape',
                inputs=[node.output[0],
                        shapeToStr(layerPairs[node.name]['targetShape'])],
                outputs=[node.name+'_plus_reshape_out'],
                name=node.name+'_plus_reshape',
                allowzero= 0)
            newReshapeNodes[i] = insert_Reshape_node

    prevNodeIndex.sort(reverse=True)
    for i in prevNodeIndex:
        model.graph.node.insert(i+1, newReshapeNodes[i])

    if paramPath is not None:
        onnx.save(model, paramPath)
        model = onnx.load(paramPath)
    else:
        onnx.save(model, targetParamPath)
        model = onnx.load(targetParamPath)
    # printNet(model)
    graph = gs.import_onnx(model)
    for specifiedPair in layerPairs.keys():
        pickFirst = [node for node in graph.nodes if node.name ==specifiedPair]
        if 0 == len(pickFirst):
            continue
        assert len(pickFirst) == 1        
        assert len(pickFirst[0].outputs) == 1
        oldOutPutName = pickFirst[0].outputs[0].name
        if len(pickFirst) == 0:
            continue
        for nextNodeName in layerPairs[specifiedPair]['nextNodes']:
            pickSecond = [
                node for node in graph.nodes if node.name == nextNodeName]
            if len(pickSecond) == 0:
                continue
            for j in range(len(pickSecond[0].inputs)):
                if pickSecond[0].inputs[j].name == oldOutPutName:
                    insertReshapeName = pickSecond[0].i(j).name+'_plus_reshape'
                    insertReshape = [
                        node for node in graph.nodes if node.name == insertReshapeName]
                    insertReshape[0].outputs[0].dtype = insertReshape[0].inputs[0].dtype
                    insertReshape[0].outputs[0].shape = insertReshape[0].inputs[1].values.tolist()
                    pickSecond[0].inputs[j] = insertReshape[0].outputs[0]

    graph.cleanup()
    graph.toposort()
    new_mode = gs.export_onnx(graph)
    new_mode.ir_version = ir_version
    shape_set.clear()  # for gs cleanup
    for init in new_mode.graph.initializer:  # for gs cleanup
        if init.name.startswith("shape_"):  # for gs cleanup
            shape_set.append(init.name)  # for gs cleanup
    if paramPath is not None:
        onnx.save(new_mode, paramPath)
    else:
        onnx.save(new_mode, targetParamPath)

def insertSpecifiedTranspose(layerPairs,paramPath=None):
    prevNodeIndex = []
    newTransposeNodes = {}
    if paramPath is not None:
        model = onnx.load(paramPath)
    else:
        model = onnx.load(targetParamPath)
    for i, node in enumerate(model.graph.node):
        if node.name in layerPairs.keys():
            prevNodeIndex.append(i)
            insert_Transpose_node = onnx.helper.make_node(
                op_type='Transpose',
                inputs=[node.output[0]],
                outputs=[node.name+'_plus_transpose'],
                name=node.name+'_plus_transpose',
                perm=layerPairs[node.name]['targetPerm'])
            newTransposeNodes[i] = insert_Transpose_node

    prevNodeIndex.sort(reverse=True)
    for i in prevNodeIndex:
        model.graph.node.insert(i+1, newTransposeNodes[i])

    if paramPath is not None:
        onnx.save(model, paramPath)
        model = onnx.load(paramPath)
    else:
        onnx.save(model, targetParamPath)
        model = onnx.load(targetParamPath)
    # printNet(model)
    graph = gs.import_onnx(model)
    for specifiedPair in layerPairs.keys():
        pickFirst = [
            node for node in graph.nodes if node.name == specifiedPair]
        if len(pickFirst) == 0:
            continue
        pickSecond = [node for node in graph.nodes if node.name ==
                      layerPairs[specifiedPair]['second']]
        if len(pickSecond) == 0:
            continue
        insertReshapeName = pickSecond[0].i(0).name+'_plus_transpose'
        insertReshape = [
            node for node in graph.nodes if node.name == insertReshapeName]
        if len(insertReshape) == 0:
            continue
        pickSecond[0].inputs[0] = insertReshape[0].outputs[0]

    graph.cleanup()
    graph.toposort()
    new_mode = gs.export_onnx(graph)
    new_mode.ir_version = ir_version
    if paramPath is not None:
        onnx.save(new_mode, paramPath)
    else:
        onnx.save(new_mode, targetParamPath)

def modifySplitLayer(layerNamesAndtargetSplit,paramPath=None):
    newshapes = []
    for shape in layerNamesAndtargetSplit.keys():
        newshapes.append(layerNamesAndtargetSplit[shape]['split'])
    registerShapeNode(newshapes,paramPath)
    if paramPath is not None:
        model = onnx.load(paramPath)
    else:
        model = onnx.load(targetParamPath)
    modified = []
    loopDone = False
    while True:
        if len(modified) == len(layerNamesAndtargetSplit) or loopDone:
            break
        loopDone = False
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
                    axis=layerNamesAndtargetSplit[node.name]['axis']
                )
                model.graph.node.remove(model.graph.node[i])
                model.graph.node.insert(i, new_split_node)
                modified.append(node.name)
                break
            if i == len(model.graph.node)-1:
                loopDone = True
    if paramPath is not None:
        onnx.save(model, paramPath)
    else:
        onnx.save(model, targetParamPath)

def deleteLayer(layernames,paramPath=None):
    if paramPath is not None:
        model = onnx.load(paramPath)
    else:
        model = onnx.load(targetParamPath)
    graph = gs.import_onnx(model)
    for name in layernames:
        pick = [node for node in graph.nodes if node.name == name]
        if len(pick) != 1:
            print(name)
            assert len(pick) == 1
        remove_node = pick[0]
        if remove_node.op == 'Squeeze':
            output_node = remove_node.o(0)
            output_node.inputs[0] = remove_node.inputs[0]
            remove_node.outputs.clear()
            graph.cleanup()
        elif remove_node.op == 'Mul':
            output_node = remove_node.o(0)
            for i in range(len(output_node.inputs)):
                if output_node.inputs[i].name == remove_node.outputs[0].name:
                    output_node.inputs[i] = remove_node.inputs[0]
            remove_node.outputs.clear()
            graph.cleanup()
        elif remove_node.name == '/image_encoder/trunk/Add_1':
            output_node0 = remove_node.o(0)
            output_node1 = remove_node.o(1)
            output_node0.inputs[0] = remove_node.inputs[0]
            output_node1.inputs[0] = remove_node.inputs[0]
            remove_node.outputs.clear()
            graph.cleanup()
        else:
            assert False,"not support yet"

    graph.cleanup()
    graph.toposort()
    new_mode = gs.export_onnx(graph)
    new_mode.ir_version = ir_version
    if paramPath is not None:
        onnx.save(new_mode, paramPath)
    else:
        onnx.save(new_mode, targetParamPath)
def ncnnShapeSqueezeFlag(layerNames,paramPath=None):
    if paramPath is not None:
        model = onnx.load(paramPath)
    else:
        model = onnx.load(targetParamPath)
    
    graph = gs.import_onnx(model)
    for node in graph.nodes:
        if node.name in layerNames:
            node.name = node.name+'_needSqueeze'
    graph.cleanup()

    graph.cleanup()
    graph.toposort()
    new_mode = gs.export_onnx(graph)
    new_mode.ir_version = ir_version

    if paramPath is not None:
        onnx.save(new_mode, paramPath)
    else:
        onnx.save(new_mode, targetParamPath)
def refreshOutputShape(paramPath=None):
    if paramPath is not None:
        model = onnx.load(paramPath)
    else:
        model = onnx.load(targetParamPath)
    # printNet(model)
    graph = gs.import_onnx(model)
    for node in graph.nodes:
        # if node.name == '/image_encoder/trunk/blocks.21/Add_3':
        #     break
        # print('refresh layer shape: ',node.name)
        if node.op == 'Reshape' :
            assert len(node.inputs)==2
            dataIdx=-1
            shapeIdx=-1
            for i in range(len(node.inputs)):
                if node.inputs[i].dtype == 'int64':
                    shapeIdx=i
                    dataIdx = 1-i
            assert dataIdx==0 and shapeIdx==1
            if np.sum(np.array(node.inputs[shapeIdx].shape) == -1) :
                continue
            try:
                node.outputs[0].shape = node.inputs[shapeIdx].values.tolist()
                node.outputs[0].dtype = node.inputs[dataIdx].dtype
            except:
                print(node)
                assert False
        elif node.op == 'Transpose':
            try:
                inputShapeSum = sum(node.inputs[0].shape)
                inputShape = node.inputs[0].shape
                perm = node.attrs['perm']
                node.outputs[0].shape = [inputShape[i] for i in perm]
            except:
                print(node)
                assert False
        elif node.op == 'Resize':
            node.outputs[0].shape[0] =1
        elif node.op == 'LayerNormalization':
            node.outputs[0].shape = node.inputs[0].shape
        elif node.op == 'Conv':
            assert len(node.inputs)==3
            outShape = cnnOutFigure(node.inputs[0].shape[-2:],node.inputs[1].shape[-2:],node.attrs['strides'],node.attrs['pads'][-2:])
            shapeGuess= [node.inputs[0].shape[0], node.inputs[1].shape[0],outShape[0],outShape[1]]
            node.outputs[0].shape =shapeGuess[4-len(node.inputs[0].shape):]
        elif node.op == 'MatMul':
            if np.sum(np.array(node.inputs[0].shape) == -1) != 0:
                continue
            if np.sum(np.array(node.inputs[1].shape) == -1) != 0:
                continue
            try:
                A = np.ones(node.inputs[0].shape)
                B = np.ones(node.inputs[1].shape)
                C = A@B
                node.outputs[0].shape = C.shape
            except:
                print(node)
                assert False
        elif node.op == 'Softmax':
            node.outputs[0].shape = node.inputs[0].shape
        elif node.op == 'Mul':
            if np.sum(np.array(node.inputs[0].shape) == -1) != 0:
                continue
            if np.sum(np.array(node.inputs[1].shape) == -1) != 0:
                continue
            try:
                A = np.ones(node.inputs[0].shape)
                B = np.ones(node.inputs[1].shape)
                C = A*B
            except:
                print(node)
                assert False
            node.outputs[0].shape = C.shape
        elif node.op == 'Erf':
            node.outputs[0].shape = node.inputs[0].shape
        elif node.op == 'Add' :
            if np.sum(np.array(node.inputs[0].shape) == -1) != 0:
                continue
            if np.sum(np.array(node.inputs[1].shape) == -1) != 0:
                continue
            try:
                A = np.ones(node.inputs[0].shape)
                B = np.ones(node.inputs[1].shape)
                C = A+B
                for j in range(len(node.outputs)):
                    node.outputs[j].shape = C.shape
            except:
                print(node)
                assert False
        elif node.op == 'Div':
            if np.sum(np.array(node.inputs[0].shape) == -1) != 0:
                continue
            if np.sum(np.array(node.inputs[1].shape) == -1) != 0:
                continue
            try:
                A = np.ones(node.inputs[0].shape)
                B = np.ones(node.inputs[1].shape)
                C = A/B
                node.outputs[0].shape = C.shape
            except:
                print(node)
                assert False
        elif node.op == 'Split':
            axis = node.attrs['axis']
            splitArray = node.inputs[1].values.tolist()
            srcShape = node.inputs[0].shape
            try:
                np.sum(splitArray) == node.inputs[0].shape[axis]
            except:
                print(node)
                assert False
            for i in range(len(node.outputs)):
                node.outputs[i].shape = copy.deepcopy(srcShape)
                node.outputs[i].shape[axis] = splitArray[i]

        elif node.op == 'MaxPool':
            try:
                node.outputs[0].shape = copy.deepcopy(node.inputs[0].shape)
                node.outputs[0].shape[-2] = node.outputs[0].shape[-2]//2
                node.outputs[0].shape[-1] = node.outputs[0].shape[-1]//2
            except:
                print(node)
                assert False
    graph.cleanup()
    graph.toposort()
    new_mode = gs.export_onnx(graph)
    new_mode.ir_version = ir_version
    if paramPath is not None:
        onnx.save(new_mode, paramPath)
    else:
        onnx.save(new_mode, targetParamPath)


def convertOpencvOnnxToNcnn():
    model = onnx.load('models/opencv_encoder.onnx')
    print(len(model.graph.input))
    cut_subgraph('models/opencv_encoder.onnx',
                 ['image'],
                #  ['/Transpose_1_output_0'],
                 shared_out,
                 targetParamPath)
    model = onnx.load(targetParamPath)

    # if model.graph.input[0].name == shared_input[0] and shared_input[0] == '/image_encoder/trunk/Add_1_output_0':
    #     newInput = helper.make_tensor_value_info(shared_input[0], TensorProto.FLOAT, [256*256,144])
    #     model.graph.input.clear()
    #     model.graph.input.append(newInput)

    onnx.save(model, targetParamPath)

# --------------------------------
    reshapeAddNode()
# --------------------------------
    insertReshape = {}    
    insertReshape['/image_encoder/trunk/patch_embed/proj/Conv'] = {
        'nextNodes': ['/image_encoder/trunk/patch_embed/Transpose'], 'targetShape': [144,65536]}
    insertReshape['/image_encoder/trunk/blocks.1/Add_3'] = {
        'nextNodes': ['/image_encoder/trunk/Transpose_1'], 'targetShape': [1, 256, 256, 144]}
    insertReshape['/image_encoder/trunk/blocks.2/attn/Transpose']= {
        'nextNodes': ['/image_encoder/trunk/blocks.2/attn/pool/MaxPool'], 'targetShape': [294912,1, 8,8]}
    insertReshape['/image_encoder/trunk/blocks.2/attn/pool/MaxPool']= {
        'nextNodes': ['/image_encoder/trunk/blocks.2/attn/Transpose_1'], 'targetShape': [1024,288, 4,4]}
    insertReshape['/image_encoder/trunk/blocks.2/proj/Add']= {
        'nextNodes': ['/image_encoder/trunk/blocks.2/Transpose'], 'targetShape': [1, 256, 256, 288]}
    insertReshape['/image_encoder/trunk/blocks.2/pool/MaxPool']= {
        'nextNodes': ['/image_encoder/trunk/blocks.2/Transpose_1'], 'targetShape': [1,288,128,128 ]}
    insertReshape['/image_encoder/trunk/blocks.2/Transpose_1']= {
        'nextNodes': ['/image_encoder/trunk/blocks.2/Add_4'], 'targetShape': [128*128, 288]}
    insertReshape['/image_encoder/trunk/blocks.7/Add_3'] = {
        'nextNodes': ['/image_encoder/trunk/Transpose_2'], 'targetShape': [1, 128,128, 288]}
    insertReshape['/image_encoder/trunk/blocks.8/attn/Transpose'] = {
        'nextNodes': ['/image_encoder/trunk/blocks.8/attn/pool/MaxPool'], 'targetShape': [589824, 1, 4, 4]}
    insertReshape['/image_encoder/trunk/blocks.8/attn/pool/MaxPool'] = {
        'nextNodes': ['/image_encoder/trunk/blocks.8/attn/Transpose_1'], 'targetShape': [1024, 576, 2,2]}
    insertReshape['/image_encoder/trunk/blocks.8/proj/Add'] = {
        'nextNodes': ['/image_encoder/trunk/blocks.8/Transpose'], 'targetShape': [1, 128, 128, 576]}
    insertReshape['/image_encoder/trunk/blocks.8/pool/MaxPool'] = {
        'nextNodes': ['/image_encoder/trunk/blocks.8/Transpose_1'], 'targetShape': [1,576, 64,64]}
    insertReshape['/image_encoder/trunk/blocks.8/Transpose_1'] = {
        'nextNodes': ['/image_encoder/trunk/blocks.8/Add_4'], 'targetShape': [4096, 576]}
    insertReshape['/image_encoder/trunk/blocks.44/attn/Transpose']= {
        'nextNodes': ['/image_encoder/trunk/blocks.44/attn/pool/MaxPool'], 'targetShape': [18432,1,16,16]}
    insertReshape['/image_encoder/trunk/blocks.44/attn/pool/MaxPool']= {
        'nextNodes': ['/image_encoder/trunk/blocks.44/attn/Transpose_1'], 'targetShape': [16,1152,8,8]}
    insertReshape['/image_encoder/trunk/blocks.44/proj/Add']= {
        'nextNodes': ['/image_encoder/trunk/blocks.44/Transpose'], 'targetShape': [1,64,64,1152]}
    insertReshape['/image_encoder/trunk/blocks.44/Transpose']= {
        'nextNodes': ['/image_encoder/trunk/blocks.44/pool/MaxPool'], 'targetShape': [1152,1,64,64]}
    insertReshape['/image_encoder/trunk/blocks.44/pool/MaxPool']= {
        'nextNodes': ['/image_encoder/trunk/blocks.44/Transpose_1'], 'targetShape': [1,1152,32,32]}
    insertReshape['/image_encoder/trunk/blocks.44/Transpose_1']= {
        'nextNodes': ['/image_encoder/trunk/blocks.44/Add_4'], 'targetShape': [1024,1152]}    
    insertReshape['/image_encoder/trunk/blocks.47/Add_3']= {
        'nextNodes': ['/image_encoder/trunk/Transpose_4'], 'targetShape': [1,32,32,1152]}
    insertReshape['/image_encoder/trunk/blocks.43/Add_1']= {
        'nextNodes': ['/image_encoder/trunk/Transpose_3'], 'targetShape': [1, 64, 64, 576]}
    insertSpecifiedReshape(insertReshape)


# --------------------------------
    binaryOperationConst = {}    
    binaryOperationConst['/image_encoder/trunk/blocks.0/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.0/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.1/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.1/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.2/attn/Mul_2'] = {
        'constName': '/image_encoder/trunk/blocks.2/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.3/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.3/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.4/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.4/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.5/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.5/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.6/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.6/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.7/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.7/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.8/attn/Mul_2'] = {
        'constName': '/image_encoder/trunk/blocks.8/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.9/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.9/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.10/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.10/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.11/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.11/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.12/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.12/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.13/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.13/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.14/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.14/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.15/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.15/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.16/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.16/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.17/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.17/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.18/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.18/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.19/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.19/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.20/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.20/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.21/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.21/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.22/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.22/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.23/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.23/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.24/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.24/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.25/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.25/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}    
    binaryOperationConst['/image_encoder/trunk/blocks.26/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.26/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.27/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.27/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}    
    binaryOperationConst['/image_encoder/trunk/blocks.28/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.28/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.29/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.29/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.30/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.30/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.31/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.31/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.32/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.32/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.33/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.33/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.34/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.34/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.35/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.35/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.36/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.36/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.37/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.37/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.38/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.38/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.39/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.39/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.40/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.40/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.41/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.41/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.42/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.42/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.43/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.43/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.44/attn/Mul_2'] = {
        'constName': '/image_encoder/trunk/blocks.44/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.45/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.45/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.46/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.46/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}
    binaryOperationConst['/image_encoder/trunk/blocks.47/attn/Mul_1'] = {
        'constName': '/image_encoder/trunk/blocks.47/attn/Sqrt_1_output_0', 'value': [0.11785113019775792073]}


    modifyBinaryOperationConst(binaryOperationConst)
# --------------------------------
    ncnnShapeSqueezeFlag([
        '/image_encoder/neck/convs.3/conv/Conv', 
        '/image_encoder/neck/convs.2/conv/Conv',
        '/image_encoder/neck/convs.1/conv/Conv', 
        '/image_encoder/neck/convs.0/conv/Conv'])
# --------------------------------
    deleteLayer([
                 '/image_encoder/trunk/blocks.0/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.0/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.0/attn/Squeeze', 
                 '/image_encoder/trunk/blocks.0/attn/Mul_2',
                 '/image_encoder/trunk/blocks.1/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.1/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.1/attn/Squeeze',
                 '/image_encoder/trunk/blocks.1/attn/Mul_2',
                 '/image_encoder/trunk/blocks.2/attn/Squeeze',
                 '/image_encoder/trunk/blocks.2/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.2/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.2/attn/Mul_3',
                 '/image_encoder/trunk/blocks.3/attn/Squeeze',
                 '/image_encoder/trunk/blocks.3/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.3/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.3/attn/Mul_2',
                 '/image_encoder/trunk/blocks.4/attn/Squeeze',
                 '/image_encoder/trunk/blocks.4/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.4/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.4/attn/Mul_2',
                 '/image_encoder/trunk/blocks.5/attn/Squeeze',
                 '/image_encoder/trunk/blocks.5/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.5/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.5/attn/Mul_2',
                 '/image_encoder/trunk/blocks.6/attn/Squeeze',
                 '/image_encoder/trunk/blocks.6/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.6/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.6/attn/Mul_2',
                 '/image_encoder/trunk/blocks.7/attn/Squeeze',
                 '/image_encoder/trunk/blocks.7/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.7/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.7/attn/Mul_2',
                 '/image_encoder/trunk/blocks.8/attn/Squeeze',
                 '/image_encoder/trunk/blocks.8/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.8/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.8/attn/Mul_3', 
                 '/image_encoder/trunk/blocks.9/attn/Squeeze',
                 '/image_encoder/trunk/blocks.9/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.9/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.9/attn/Mul_2',
                 '/image_encoder/trunk/blocks.10/attn/Squeeze',
                 '/image_encoder/trunk/blocks.10/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.10/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.10/attn/Mul_2',
                 '/image_encoder/trunk/blocks.11/attn/Squeeze',
                 '/image_encoder/trunk/blocks.11/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.11/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.11/attn/Mul_2',
                 '/image_encoder/trunk/blocks.12/attn/Squeeze',
                 '/image_encoder/trunk/blocks.12/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.12/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.12/attn/Mul_2',
                 '/image_encoder/trunk/blocks.13/attn/Squeeze',
                 '/image_encoder/trunk/blocks.13/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.13/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.13/attn/Mul_2',
                 '/image_encoder/trunk/blocks.14/attn/Squeeze',
                 '/image_encoder/trunk/blocks.14/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.14/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.14/attn/Mul_2',
                 '/image_encoder/trunk/blocks.15/attn/Squeeze',
                 '/image_encoder/trunk/blocks.15/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.15/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.15/attn/Mul_2',
                 '/image_encoder/trunk/blocks.16/attn/Squeeze',
                 '/image_encoder/trunk/blocks.16/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.16/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.16/attn/Mul_2',
                 '/image_encoder/trunk/blocks.17/attn/Squeeze',
                 '/image_encoder/trunk/blocks.17/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.17/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.17/attn/Mul_2',
                 '/image_encoder/trunk/blocks.18/attn/Squeeze',
                 '/image_encoder/trunk/blocks.18/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.18/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.18/attn/Mul_2',      
                 '/image_encoder/trunk/blocks.19/attn/Squeeze',
                 '/image_encoder/trunk/blocks.19/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.19/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.19/attn/Mul_2',
                 '/image_encoder/trunk/blocks.20/attn/Squeeze',
                 '/image_encoder/trunk/blocks.20/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.20/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.20/attn/Mul_2',
                 '/image_encoder/trunk/blocks.21/attn/Squeeze',
                 '/image_encoder/trunk/blocks.21/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.21/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.21/attn/Mul_2',
                 '/image_encoder/trunk/blocks.22/attn/Squeeze',
                 '/image_encoder/trunk/blocks.22/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.22/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.22/attn/Mul_2',
                 '/image_encoder/trunk/blocks.23/attn/Squeeze',
                 '/image_encoder/trunk/blocks.23/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.23/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.23/attn/Mul_2',
                 '/image_encoder/trunk/blocks.24/attn/Squeeze',
                 '/image_encoder/trunk/blocks.24/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.24/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.24/attn/Mul_2',
                 '/image_encoder/trunk/blocks.25/attn/Squeeze',
                 '/image_encoder/trunk/blocks.25/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.25/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.25/attn/Mul_2',
                 '/image_encoder/trunk/blocks.26/attn/Squeeze',
                 '/image_encoder/trunk/blocks.26/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.26/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.26/attn/Mul_2',
                 '/image_encoder/trunk/blocks.27/attn/Squeeze',
                 '/image_encoder/trunk/blocks.27/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.27/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.27/attn/Mul_2',
                 '/image_encoder/trunk/blocks.28/attn/Squeeze',
                 '/image_encoder/trunk/blocks.28/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.28/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.28/attn/Mul_2',
                 '/image_encoder/trunk/blocks.29/attn/Squeeze',
                 '/image_encoder/trunk/blocks.29/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.29/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.29/attn/Mul_2',
                 '/image_encoder/trunk/blocks.30/attn/Squeeze',
                 '/image_encoder/trunk/blocks.30/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.30/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.30/attn/Mul_2',
                 '/image_encoder/trunk/blocks.31/attn/Squeeze',
                 '/image_encoder/trunk/blocks.31/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.31/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.31/attn/Mul_2',
                 '/image_encoder/trunk/blocks.32/attn/Squeeze',
                 '/image_encoder/trunk/blocks.32/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.32/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.32/attn/Mul_2',
                 '/image_encoder/trunk/blocks.33/attn/Squeeze',
                 '/image_encoder/trunk/blocks.33/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.33/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.33/attn/Mul_2',
                 '/image_encoder/trunk/blocks.34/attn/Squeeze',
                 '/image_encoder/trunk/blocks.34/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.34/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.34/attn/Mul_2',
                 '/image_encoder/trunk/blocks.35/attn/Squeeze',
                 '/image_encoder/trunk/blocks.35/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.35/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.35/attn/Mul_2',
                 '/image_encoder/trunk/blocks.36/attn/Squeeze',
                 '/image_encoder/trunk/blocks.36/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.36/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.36/attn/Mul_2',
                 '/image_encoder/trunk/blocks.37/attn/Squeeze',
                 '/image_encoder/trunk/blocks.37/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.37/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.37/attn/Mul_2',
                 '/image_encoder/trunk/blocks.38/attn/Squeeze',
                 '/image_encoder/trunk/blocks.38/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.38/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.38/attn/Mul_2',
                 '/image_encoder/trunk/blocks.39/attn/Squeeze',
                 '/image_encoder/trunk/blocks.39/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.39/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.39/attn/Mul_2',
                 '/image_encoder/trunk/blocks.40/attn/Squeeze',
                 '/image_encoder/trunk/blocks.40/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.40/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.40/attn/Mul_2',
                 '/image_encoder/trunk/blocks.41/attn/Squeeze',
                 '/image_encoder/trunk/blocks.41/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.41/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.41/attn/Mul_2',
                 '/image_encoder/trunk/blocks.42/attn/Squeeze',
                 '/image_encoder/trunk/blocks.42/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.42/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.42/attn/Mul_2',
                 '/image_encoder/trunk/blocks.43/attn/Squeeze',
                 '/image_encoder/trunk/blocks.43/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.43/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.43/attn/Mul_2',
                 '/image_encoder/trunk/blocks.44/attn/Squeeze',
                 '/image_encoder/trunk/blocks.44/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.44/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.44/attn/Mul_3',
                 '/image_encoder/trunk/blocks.45/attn/Squeeze',
                 '/image_encoder/trunk/blocks.45/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.45/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.45/attn/Mul_2',
                 '/image_encoder/trunk/blocks.46/attn/Squeeze',
                 '/image_encoder/trunk/blocks.46/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.46/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.46/attn/Mul_2',
                 '/image_encoder/trunk/blocks.47/attn/Squeeze',
                 '/image_encoder/trunk/blocks.47/attn/Squeeze_1',
                 '/image_encoder/trunk/blocks.47/attn/Squeeze_2',
                 '/image_encoder/trunk/blocks.47/attn/Mul_2'
                 ])
# --------------------------------
    reshapeAndtargetShape = {}
    reshapeAndtargetShape['/image_encoder/trunk/blocks.0/Reshape']=[32,8,32,1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.0/Reshape_1'] = [65536, 144]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.0/attn/Reshape'] = [1024,64, 6,72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.0/attn/Reshape_1'] = [65536, 144]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.0/Reshape_2'] = [32,32,  8, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.0/Reshape_3'] = [65536, 144]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.1/Reshape'] = [32, 8, 32, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.1/Reshape_1'] = [65536, 144]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.1/attn/Reshape'] = [1024, 64, 6, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.1/attn/Reshape_1'] = [65536, 144]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.1/Reshape_2'] = [32, 32,  8, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.1/Reshape_3'] = [65536, 144]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.2/Reshape'] = [32, 8, 32, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.2/Reshape_1'] = [65536, 144]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.2/attn/Reshape']=[1024,64,12,72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.2/attn/Reshape_1']=[1024,8,8,288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.2/attn/Reshape_2'] = [1024, 16,4, 72]
    reshapeAndtargetShape['/Reshape_5']=[1,32,256,256]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.2/attn/Reshape_3']=[16384,288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.2/Reshape_2']=[32, 32,  4, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.2/Reshape_3']=[128*128,288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.3/Reshape']=[32, 4, 32, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.3/Reshape_1']=[16384,288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.3/attn/Reshape']=[1024,16,12,72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.3/attn/Reshape_1']=[16384,288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.3/Reshape_2']=[32, 32,  4, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.3/Reshape_3']=[16384,288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.4/Reshape']=[32, 4, 32, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.4/Reshape_1']=[16384,288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.4/attn/Reshape']=[1024,16,12,72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.4/attn/Reshape_1']=[16384,288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.4/Reshape_2']=[32, 32,  4, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.4/Reshape_3'] = [16384, 288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.5/Reshape'] = [32, 4, 32, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.5/Reshape_1'] = [16384, 288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.5/attn/Reshape'] = [1024, 16, 12, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.5/attn/Reshape_1'] = [16384, 288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.5/Reshape_2'] = [32, 32,  4, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.5/Reshape_3'] = [16384, 288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.6/Reshape'] = [32, 4, 32, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.6/Reshape_1'] = [16384, 288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.6/attn/Reshape'] = [1024, 16, 12, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.6/attn/Reshape_1'] = [16384, 288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.6/Reshape_2'] = [32, 32,  4, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.6/Reshape_3'] = [16384, 288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.7/Reshape'] = [32, 4, 32, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.7/Reshape_1'] = [16384, 288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.7/attn/Reshape'] = [1024, 16, 12, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.7/attn/Reshape_1'] = [16384, 288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.7/Reshape_2'] = [32, 32,  4, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.7/Reshape_3'] = [16384, 288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.8/Reshape'] = [32, 4, 32, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.8/Reshape_1'] = [16384, 288]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.8/attn/Reshape'] = [1024, 16, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.8/attn/Reshape_1'] = [1024,4,4,576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.8/attn/Reshape_3'] = [4096,576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.8/Reshape_2'] = [32, 32,  2, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.8/Reshape_3'] = [4096, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.9/Reshape']=[4,16,4,9216]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.9/Reshape_1'] = [4096, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.9/attn/Reshape']=[16,256,24,72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.9/attn/Reshape_1'] = [4096, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.9/Reshape_2'] = [4, 4, 16, 9216]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.9/Reshape_3'] = [4096,576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.10/Reshape']=[4,16,4,16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.10/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.10/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.10/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.10/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.10/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.11/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.11/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.11/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.11/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.11/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.11/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.12/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.12/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.12/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.12/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.12/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.12/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.13/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.13/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.13/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.13/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.13/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.13/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.14/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.14/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.14/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.14/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.14/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.14/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.15/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.15/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.15/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.15/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.15/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.15/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.16/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.16/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.16/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.16/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.16/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.16/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.17/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.17/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.17/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.17/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.17/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.17/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.18/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.18/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.18/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.18/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.18/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.18/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.19/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.19/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.19/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.19/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.19/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.19/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.20/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.20/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.20/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.20/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.20/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.20/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.21/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.21/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.21/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.21/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.21/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.21/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.22/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.22/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.22/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.22/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.22/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.22/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.23/attn/Reshape'] = [1, 4096, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.23/attn/Reshape_1'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.24/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.24/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.24/attn/Reshape']  =[ 16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.24/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.24/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.24/Reshape_3'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.25/Reshape']= [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.25/Reshape_1']= [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.25/attn/Reshape']=[ 16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.25/attn/Reshape_1']= [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.25/Reshape_2']= [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.25/Reshape_3']= [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.26/Reshape']= [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.26/Reshape_1']= [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.26/attn/Reshape']=[ 16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.26/attn/Reshape_1']= [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.26/Reshape_2']= [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.26/Reshape_3']= [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.27/Reshape']= [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.27/Reshape_1']= [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.27/attn/Reshape']=[ 16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.27/attn/Reshape_1']= [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.27/Reshape_2']= [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.27/Reshape_3']= [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.28/Reshape']= [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.28/Reshape_1']= [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.28/attn/Reshape']=[ 16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.28/attn/Reshape_1']= [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.28/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.28/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.29/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.29/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.29/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.29/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.29/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.29/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.30/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.30/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.30/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.30/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.30/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.30/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.31/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.31/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.31/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.31/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.31/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.31/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.32/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.32/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.32/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.32/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.32/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.32/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.33/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.33/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.33/attn/Reshape'] = [1, 4096, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.33/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.33/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.33/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.34/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.34/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.34/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.34/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.34/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.34/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.35/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.35/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.35/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.35/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.35/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.35/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.36/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.36/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.36/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.36/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.36/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.36/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.37/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.37/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.37/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.37/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.37/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.37/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.38/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.38/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.38/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.38/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.38/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.38/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.39/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.39/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.39/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.39/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.39/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.39/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.40/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.40/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.40/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.40/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.40/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.40/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.41/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.41/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.41/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.41/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.41/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.41/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.42/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.42/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.42/attn/Reshape'] = [16, 256, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.42/attn/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.42/Reshape_2'] = [4, 4, 16, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.42/Reshape_3'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.43/attn/Reshape'] = [1, 4096, 24, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.43/attn/Reshape_1'] = [64*64, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.44/Reshape'] = [4, 16, 4, 16*576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.44/Reshape_1'] = [16*16*16, 576]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.44/attn/Reshape'] = [16, 256, 48, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.44/attn/Reshape_1'] = [16,16,16, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.44/attn/Reshape_2'] = [16, 64, 16, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.44/attn/Reshape_3'] = [1024, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.44/Reshape_2'] =[4, 4, 8, 8* 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.44/Reshape_3'] = [1024, 1152] 
    reshapeAndtargetShape['/image_encoder/trunk/blocks.45/Reshape']= [4, 8, 4, 8*1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.45/Reshape_1']= [4* 8* 4* 8,1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.45/attn/Reshape']=[16, 64, 3* 16, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.45/attn/Reshape_1']=[1024, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.45/Reshape_2']=[ 4, 4, 8, 8* 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.45/Reshape_3']=[ 4* 8* 4* 8,1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.46/Reshape']= [4, 8, 4, 8*1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.46/Reshape_1']= [4* 8* 4* 8,1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.46/attn/Reshape']=[16, 64, 3* 16, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.46/attn/Reshape_1']=[1024, 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.46/Reshape_2']=[ 4, 4, 8, 8* 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.46/Reshape_3']=[ 4* 8* 4* 8,1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.47/Reshape']= [4, 8, 4, 8*1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.47/Reshape_1']= [ 4* 8* 4* 8,1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.47/attn/Reshape']=[16, 64, 3* 16, 72]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.47/attn/Reshape_1']= [ 16*8*8,1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.47/Reshape_2']=[ 4, 4, 8, 8* 1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.47/Reshape_3']= [ 16*8*8,1152]
    reshapeAndtargetShape['/Reshape_2'] = [256,4096]
    modifyReshapeLayer(reshapeAndtargetShape)
# --------------------------------
    transposeAndtargetShape = {}    
    transposeAndtargetShape['/image_encoder/trunk/patch_embed/Transpose'] = [1, 0]
    transposeAndtargetShape['/image_encoder/trunk/blocks.0/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.0/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.1/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.1/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.2/Transpose_2'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.2/Transpose_3'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.2/Transpose'] = [3,0,1,2]
    transposeAndtargetShape['/image_encoder/trunk/blocks.2/Transpose_1'] = [0,2,3,1]
    transposeAndtargetShape['/image_encoder/trunk/blocks.3/Transpose']= [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.3/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.4/Transpose']= [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.4/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.5/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.5/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.6/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.6/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.7/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.7/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.8/Transpose_2'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.8/Transpose_3'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.8/Transpose'] = [3, 0, 1, 2]
    transposeAndtargetShape['/image_encoder/trunk/blocks.9/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.9/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.10/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.10/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.11/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.11/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.12/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.12/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.13/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.13/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.14/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.14/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.15/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.15/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.16/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.16/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.17/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.17/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.18/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.18/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.19/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.19/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.20/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.20/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.21/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.21/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.22/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.22/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.24/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.24/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.25/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.25/Transpose_1']= [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.26/Transpose']= [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.26/Transpose_1']= [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.27/Transpose']= [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.27/Transpose_1']= [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.28/Transpose']= [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.28/Transpose_1']= [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.29/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.29/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.30/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.30/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.31/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.31/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.32/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.32/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.33/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.33/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.34/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.34/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.35/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.35/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.36/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.36/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.37/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.37/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.38/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.38/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.39/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.39/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.40/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.40/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.41/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.41/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.42/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.42/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.44/Transpose_2'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.44/Transpose_3'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.45/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.45/Transpose_1']= [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.46/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.46/Transpose_1']= [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.47/Transpose'] = [0, 2, 1, 3]
    transposeAndtargetShape['/image_encoder/trunk/blocks.47/Transpose_1'] = [0, 2, 1, 3]
    transposeAndtargetShape['/Transpose'] = [1,0]
    transposeAndtargetShape['/Transpose_1'] = [1,0]
    modifyTransposeLayer(transposeAndtargetShape)
# --------------------------------
    splitAndAxis = {}
    splitAndAxis['/image_encoder/trunk/blocks.0/attn/Split'] = {'axis': 2, 'split':[2, 2, 2]}
    splitAndAxis['/image_encoder/trunk/blocks.1/attn/Split'] = {
        'axis': 2, 'split': [2, 2, 2]}
    splitAndAxis['/image_encoder/trunk/blocks.2/attn/Split'] = {
        'axis': 2, 'split': [4, 4, 4]}
    splitAndAxis['/image_encoder/trunk/blocks.3/attn/Split'] = {
        'axis': 2, 'split': [4, 4, 4]}
    splitAndAxis['/image_encoder/trunk/blocks.4/attn/Split'] = {
        'axis': 2, 'split': [4, 4, 4]}
    splitAndAxis['/image_encoder/trunk/blocks.5/attn/Split'] = {
        'axis': 2, 'split': [4, 4, 4]}
    splitAndAxis['/image_encoder/trunk/blocks.6/attn/Split'] = {
        'axis': 2, 'split': [4, 4, 4]}
    splitAndAxis['/image_encoder/trunk/blocks.7/attn/Split'] = {
        'axis': 2, 'split': [4, 4, 4]}
    splitAndAxis['/image_encoder/trunk/blocks.8/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.9/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.10/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.11/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.12/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.13/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.14/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.15/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.16/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.17/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.18/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.19/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.20/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.21/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.22/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.23/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.24/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.25/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.26/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.27/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.28/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.29/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.30/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.31/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.32/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.33/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.34/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.35/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.36/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.37/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.38/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.39/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.40/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.41/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.42/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.43/attn/Split'] = {
        'axis': 2, 'split': [8, 8, 8]}
    splitAndAxis['/image_encoder/trunk/blocks.44/attn/Split'] = {
        'axis': 2, 'split': [16, 16, 16]}
    splitAndAxis['/image_encoder/trunk/blocks.45/attn/Split'] = {
        'axis': 2, 'split': [16, 16, 16]}
    splitAndAxis['/image_encoder/trunk/blocks.46/attn/Split'] = {
        'axis': 2, 'split': [16, 16, 16]}
    splitAndAxis['/image_encoder/trunk/blocks.47/attn/Split'] = {
        'axis': 2, 'split': [16, 16, 16]}
    modifySplitLayer(splitAndAxis)
# --------------------------------
    refreshOutputShape()

    model = onnx.load(targetParamPath)
    graph = gs.import_onnx(model)
    graph.cleanup()
    graph.toposort()
    new_mode = gs.export_onnx(graph)
    new_mode.ir_version = ir_version
    onnx.save(new_mode, targetParamPath)
    model = onnx.load(targetParamPath)
    if inferShapes:
        model = onnx.shape_inference.infer_shapes(model)
    if checkmodel:
        onnx.checker.check_model(model)
    onnx.save(model, targetParamPath)


    session = onnxruntime.InferenceSession(
        targetParamPath, providers=onnxruntime.get_available_providers())
    datain = {}
    datain['image'] =  image0.reshape([1,3,1024,1024]) 
    pointCoords = session.run(
        None, datain) 

    for i in range(len(shared_out)):
        print(shared_out[i])
        print(pointCoords[i].shape) 
        print(pointCoords[i]) 
        print(" ")
    return pointCoords


def test_matmul():
    # sys.stdout = open('convert_sam2_decoder_point_label.txt', 'w')
    input = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [2, 3, 4])
    output = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [2, 3, 4])
    w0 = onnx.numpy_helper.from_array(np.random.rand(
        2,4, 5).astype(np.float32), name='w0') 
    w1 = onnx.numpy_helper.from_array(np.random.rand(
        5, 3).astype(np.float32), name='w1') 
    beforeMatMul = onnx.numpy_helper.from_array(
        np.array([-1,  5]).astype(np.int64), name='beforeMatMul')  
    afterMatMul = onnx.numpy_helper.from_array(
        np.array([-1,  3, 3]).astype(np.int64), name='afterMatMul')  
    layer0 = onnx.helper.make_node(
        op_type='MatMul',
        inputs=['input', 'w0'],
        outputs=['input1'],
        name='input1')
    beforeMatMulLayer = onnx.helper.make_node(
        op_type='Reshape',
        inputs=['input1', 'beforeMatMul'],
        outputs=['input1Reshape'],
        name='input1Reshape')
    layer1 = onnx.helper.make_node(
        op_type='MatMul',
        inputs=['input1Reshape', 'w1'],
        outputs=['hide'],
        name='hide')
    afterMatMulLayer = onnx.helper.make_node(
        op_type='Reshape',
        inputs=['hide', 'afterMatMul'],
        outputs=['hide2'],
        name='hide2')
    layer2 = onnx.helper.make_node(
        op_type='MatMul',
        inputs=['hide2', 'input1'],
        outputs=['output'],
        name='output')
    graph = onnx.helper.make_graph(
        [layer0,beforeMatMulLayer, layer1, afterMatMulLayer,  layer2],
        'TwoLayerFC',
        [input],
        [output],
        initializer=[w0,w1, beforeMatMul, afterMatMul]
    )
    model = helper.make_model(graph, producer_name='onnx-example')
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    model.ir_version = 10
    model.opset_import[0].version = 21
    onnx.save(model, 'test.onnx')

    model = onnx.load('test.onnx')
    # onnx.checker.check_model(model)
    # printNet(model)
    session = onnxruntime.InferenceSession(
        'test.onnx', providers=onnxruntime.get_available_providers())
    coordPts = np.array([x for x in range(24)]).astype(
        np.float32).reshape(2, 3, 4)
    out = session.run(None, {'input': coordPts})
    print(out[0])
    print(out[0].shape)
def test_slice():
    # sys.stdout = open('convert_sam2_decoder_point_label.txt', 'w')
    input = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [12, 72])
    w1 = onnx.numpy_helper.from_array(np.ones(
        [72, 36]).astype(np.float32), name='w1')
    output = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [12, 36])
    output1 = helper.make_tensor_value_info(
        'output1', TensorProto.FLOAT, [2, 2, 36])
    output2 = helper.make_tensor_value_info(
        'output2', TensorProto.FLOAT, [2, 2, 36])
    output3 = helper.make_tensor_value_info(
        'output3', TensorProto.FLOAT, [2, 2, 36])
    shape_2_2_2 = onnx.numpy_helper.from_array(
        np.array([2, 2, 2]).astype(np.int64), name='shape_2_2_2')  # 1
    shape_2_6_36 = onnx.numpy_helper.from_array(
        np.array([2, 6, 36]).astype(np.int64), name='shape_2_6_36')  # 1

    layer1 = onnx.helper.make_node(
        op_type='MatMul',
        inputs=['input', 'w1'],
        outputs=['hide'],
        name='hide')
    reshape_node = onnx.helper.make_node(
        op_type='Reshape',
        inputs=['hide', 'shape_2_6_36'],
        outputs=['inputshape'],
        name='inputshape'
    )
    split_node = onnx.helper.make_node(
        op_type='Split',
        inputs=['inputshape', 'shape_2_2_2'],
        outputs=['output1', 'output2', 'output3'],
        name='Split',
        axis=1
    )
    graph = onnx.helper.make_graph(
        [layer1,reshape_node, split_node],
        'TwoLayerFC',
        [input],
        [output1, output2, output3],
        initializer=[shape_2_2_2, shape_2_6_36,w1]
    )
    model = helper.make_model(graph, producer_name='onnx-example')
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    model.ir_version = 10
    model.opset_import[0].version = 21
    onnx.save(model, 'test.onnx')
    session = onnxruntime.InferenceSession(
        'test.onnx', providers=onnxruntime.get_available_providers())
    indata = np.array([x for x in range(2*6*72)]).astype(
        np.float32).reshape(12, 72)
    out = session.run(None, {'input': indata})
    print(out[0])
    print(out[0].shape)
def test_conv():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 10, 10])    
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 12, 8, 8])
    weights = helper.make_tensor('weights', TensorProto.FLOAT, [12, 3, 3,3], np.ones([12, 3, 3,3]).astype(np.float32))
    bias = helper.make_tensor('bias', TensorProto.FLOAT, [12], np.ones(12).astype(np.float32))
    axes = helper.make_tensor('axes', TensorProto.INT64, [1], np.array([0], dtype=np.int64))
    outShape=cnnOutFigure([10,10],[3,3],[1,1],[0,0])
    kernel_shape = [3,3]  # 卷积核的形状
    strides = [1,1]  # 步幅
    pads = [0,0,0,0]  # 填充
    dilations = [1, 1]  # 膨胀率
    group = 1  # 常规卷积，不分组

    squeezeNode = helper.make_node(
            op_type='Squeeze',
            inputs=['input','axes'],
            outputs=['inputSqueeze'],
            name='squeezeNode'
        )

    conv = helper.make_node(op_type='Conv',
                        inputs=['input', 'weights', 'bias'],
                        outputs=['output'],
                        name='Conv_needSqueeze',
                        kernel_shape=kernel_shape,
                        strides=strides,
                        pads=pads,
                        dilations=dilations,
                        group=group
                        )
    graph = helper.make_graph(
        nodes=[conv],
        name="test_graph",
        inputs=[input],
        outputs=[output], 
        initializer=[weights, bias],
    )
    model = helper.make_model(graph, producer_name='onnx-example')
    model.ir_version = 10
    model.opset_import[0].version = 21
    onnx.save(model, 'test.onnx')    
    session = onnxruntime.InferenceSession('test.onnx', providers=onnxruntime.get_available_providers())
    indata = np.ones([3,10, 10]).astype(
        np.float32).reshape(1,3,10, 10)
    out = session.run(None, {'input': indata})
    print(out[0])
    print(out[0].shape)

    return model
def test_maxpool():
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [ 3,1, 8, 8])    
    output = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [3, 1, 4, 4])
    maxpool_node = onnx.helper.make_node(
        "MaxPool",
        inputs=["input"],
        outputs=["output"],
        ceil_mode=0,
        dilations=[1,1],
        kernel_shape=[2, 2],
        pads=[0,0,0,0],
        strides=[2,2]
    ) 
    graph = helper.make_graph(
        nodes=[maxpool_node],
        name="test_graph",
        inputs=[input],
        outputs=[output]
    )
    model = helper.make_model(graph, producer_name='onnx-example')
    model.ir_version = 10
    model.opset_import[0].version = 21
    onnx.save(model, 'test.onnx')    
    session = onnxruntime.InferenceSession('test.onnx', providers=onnxruntime.get_available_providers())
    indata = np.array([x for x in range(3*8*8)]).astype(
        np.float32).reshape(3, 1, 8, 8)
    out = session.run(None, {'input': indata})
    print(out[0])
    print(out[0].shape)




if __name__=='__main__':

    print(
        '需要先通过sam2_onnx_adaptor.py 把sam2_hiera_large_encoder.onnx转成opencv_encoder.onnx,'
        '这一步去除掉里面ncnn不支持的gather,cast等算子',
        '再通过当前py把opencv_encoder.onnx转成ncnn_encoder.onnx,修改成ncnn支持的通道特征',
        '最后通过colmapThird/ncnn-20250503/tools/onnx/onnx2ncnn这个工程,转成ncnn需要的模型文件')
    exit(0)

    # test_matmul()
    # test_slice()
    # test_conv()
    # test_maxpool()
    # exit(0)
    onnxParamPath = 'models/opencv_encoder.onnx'
    if os.path.exists(onnxParamPath):
        # test_total_forward()
        # exit(0))

        a = test_forward()
        netOut = convertOpencvOnnxToNcnn()
        for i in range(len(a)):
            assert a[i].size==netOut[i].size,print(a[i].shape,netOut[i].shape)
            diff = a[i].astype(np.float64).reshape(-1)-netOut[i].astype(np.float64).reshape(-1)
            print(shared_out[i],'\t',np.max(np.abs(diff)),np.mean(diff),np.std(diff))
            plt.plot(diff)  
            plt.savefig(str(i)+'.png')
    else: 
        print("need run sam2_onnx_adaptor first!!")

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
checkmodel = True
inferShapes = True

shared_input = [
    'image',
]
shared_out = [
    '/image_encoder/trunk/patch_embed/Transpose_output_0'
    # '/image_encoder/trunk/blocks.0/attn/MatMul_output_0',
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
    return pointCoords

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
    newshapes = []
    for shape in layerNamesAndtargetShape.keys():
        newshapes.append(layerNamesAndtargetShape[shape])
    registerShapeNode(newshapes)
    model = onnx.load(targetParamPath)
    modified = []
    loopDone = False
    while True:
        if len(modified) == len(layerNamesAndtargetShape) or loopDone:
            break
        loopDone=False
        for i, node in enumerate(model.graph.node):
            if node.name in layerNamesAndtargetShape.keys() and node.name not in modified:
                layerInput = node.input
                layerOutput = node.output
                assert len(layerInput) == 2
                assert len(layerOutput) == 1
                new_reshape_node = onnx.helper.make_node(
                    op_type="Reshape",
                    inputs=[node.input[0], shapeToStr(
                        layerNamesAndtargetShape[node.name])],
                    outputs=node.output,
                    name=node.name
                )
                model.graph.node.remove(model.graph.node[i])
                model.graph.node.insert(i, new_reshape_node)
                modified.append(node.name)
                break
            if i == len(model.graph.node)-1:
                loopDone = True
    onnx.save(model, targetParamPath)

def modifyTransposeLayer(layerNamesAndtargetPermute):
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
    onnx.save(model, targetParamPath)

def insertSpecifiedReshape(layerPairs):
    prevNodeIndex=[] 
    newReshapeNodes={}
    newshapes = []
    for specifiedPair in layerPairs.keys():
        newshapes.append(layerPairs[specifiedPair]['targetShape'])
    registerShapeNode(newshapes)
    model = onnx.load(targetParamPath)
    for i, node in enumerate(model.graph.node):
        if node.name in layerPairs.keys():
            prevNodeIndex.append(i)
            insert_Reshape_node = onnx.helper.make_node(
                op_type='Reshape',
                inputs=[node.output[0],
                        shapeToStr(layerPairs[node.name]['targetShape'])],
                outputs=[node.name+'_plus_reshape'],
                name=node.name+'_plus_reshape')
            newReshapeNodes[i] = insert_Reshape_node

    prevNodeIndex.sort(reverse=True)
    for i in prevNodeIndex:
        model.graph.node.insert(i+1, newReshapeNodes[i])
    # printNet(model)
    onnx.save(model, targetParamPath)

    model = onnx.load(targetParamPath)
    # printNet(model)
    graph = gs.import_onnx(model)
    for specifiedPair in layerPairs.keys():
        pickFirst = [node for node in graph.nodes if node.name ==specifiedPair]
        if len(pickFirst) == 0:
            continue
        pickSecond = [node for node in graph.nodes if node.name ==
                      layerPairs[specifiedPair]['second']]
        if len(pickSecond) == 0:
            continue
        insertReshapeName = pickSecond[0].i(0).name+'_plus_reshape'
        insertReshape = [
            node for node in graph.nodes if node.name == insertReshapeName]
        if len(insertReshape) == 0:
            continue
        pickSecond[0].inputs[0] = insertReshape[0].outputs[0]
    graph.cleanup()
    new_mode = gs.export_onnx(graph)
    new_mode.ir_version = 10
    onnx.save(new_mode, targetParamPath)


def modifySplitLayer(layerNamesAndtargetSplit):
    newshapes = []
    for shape in layerNamesAndtargetSplit.keys():
        newshapes.append(layerNamesAndtargetSplit[shape]['split'])
    registerShapeNode(newshapes)
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
                    axis=layerNamesAndtargetSplit[shape]['axis']
                )
                model.graph.node.remove(model.graph.node[i])
                model.graph.node.insert(i, new_split_node)
                modified.append(node.name)
                break
            if i == len(model.graph.node)-1:
                loopDone = True
    onnx.save(model, targetParamPath)

def modifyMatmulToGemm(layerNames):
    model = onnx.load(targetParamPath)
    graph = gs.import_onnx(model)
    for name in layerNames:
        pick = [node for node in graph.nodes if node.name == name]
        if len(pick) == 0:
            continue
        matmul_node = pick[0]
        if matmul_node.op == 'MatMul':
            matmul_node.op = 'Gemm'
            graph.cleanup()
        else:
            assert False, "not support yet"

    new_mode = gs.export_onnx(graph)
    new_mode.ir_version = 10
    onnx.save(new_mode, targetParamPath)
def deleteLayer(layernames):
    model = onnx.load(targetParamPath)
    graph = gs.import_onnx(model)

    for name in layernames:
        pick = [node for node in graph.nodes if node.name == name]
        if len(pick)==0:
            continue
        remove_node = pick[0]
        if remove_node.op == 'Squeeze':
            output_node = remove_node.o(0)
            output_node.inputs[0] = remove_node.inputs[0]
            remove_node.outputs.clear()
            graph.cleanup()
        else:
            assert False,"not support yet"
    new_mode = gs.export_onnx(graph)
    new_mode.ir_version = 10
    onnx.save(new_mode, targetParamPath)
    

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
    insertReshape={}
    insertReshape['/image_encoder/trunk/patch_embed/proj/Conv']={ 'second': '/image_encoder/trunk/patch_embed/Transpose','targetShape':[1,144,256,256]}
    insertReshape['/image_encoder/trunk/blocks.0/attn/Transpose'] = {
        'second': '/image_encoder/trunk/blocks.0/attn/Mul_1', 'targetShape': [2048,64,72]}
    insertReshape['/image_encoder/trunk/blocks.0/attn/Transpose_2'] = {
        'second': '/image_encoder/trunk/blocks.0/attn/Mul_2', 'targetShape': [2048, 72, 64]}
    insertSpecifiedReshape(insertReshape)
# --------------------------------
    # modifyMatmulToGemm(['/image_encoder/trunk/blocks.0/attn/MatMul'])
# --------------------------------
    deleteLayer(['/image_encoder/trunk/blocks.0/attn/Squeeze_2',
                '/image_encoder/trunk/blocks.0/attn/Squeeze_1', 
                '/image_encoder/trunk/blocks.0/attn/Squeeze'])
# --------------------------------
    reshapeAndtargetShape = {}
    reshapeAndtargetShape['/image_encoder/trunk/blocks.0/Reshape']=[32,8,32,1152]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.0/Reshape_1'] = [65536, 144]
    reshapeAndtargetShape['/image_encoder/trunk/blocks.0/attn/Reshape'] = [1024,64, 6,72]
    modifyReshapeLayer(reshapeAndtargetShape) 
# --------------------------------
    transposeAndtargetShape = {}
    transposeAndtargetShape['/image_encoder/trunk/blocks.0/Transpose'] = [0, 2, 1, 3]
    modifyTransposeLayer(transposeAndtargetShape)
# --------------------------------
    splitAndAxis = {}
    splitAndAxis['/image_encoder/trunk/blocks.0/attn/Split'] = {'axis': 2, 'split':[2, 2, 2]}
    modifySplitLayer(splitAndAxis)
# --------------------------------

    model = onnx.load(targetParamPath)
    graph = gs.import_onnx(model)
    graph.cleanup()
    new_mode = gs.export_onnx(graph)
    new_mode.ir_version = 10
    onnx.save(new_mode, targetParamPath)
    model = onnx.load(targetParamPath)
    if inferShapes:
        model = onnx.shape_inference.infer_shapes(model)
    if checkmodel:
        onnx.checker.check_model(model)
    onnx.save(model, targetParamPath)

    printNet(model)
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
def test_matmul0():
    # sys.stdout = open('convert_sam2_decoder_point_label.txt', 'w')
    input = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [2, 3, 4])
    output = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [2, 3, 4])
    w1 = onnx.numpy_helper.from_array(np.random.rand(
        4, 3).astype(np.float32), name='w1')  # [2,16]
    beforeMatMul = onnx.numpy_helper.from_array(
        np.array([-1,  4]).astype(np.int64), name='beforeMatMul')  # 1
    afterMatMul = onnx.numpy_helper.from_array(
        np.array([-1,  3, 3]).astype(np.int64), name='afterMatMul')  # 1
    beforeMatMulLayer = onnx.helper.make_node(
        op_type='Reshape',
        inputs=['input', 'beforeMatMul'],
        outputs=['inputReshape'],
        name='inputReshape')
    layer1 = onnx.helper.make_node(
        op_type='MatMul',
        inputs=['input', 'w1'],
        outputs=['hide'],
        name='hide')
    afterMatMulLayer = onnx.helper.make_node(
        op_type='Reshape',
        inputs=['hide', 'afterMatMul'],
        outputs=['hide2'],
        name='hide2')
    layer2 = onnx.helper.make_node(
        op_type='MatMul',
        inputs=['hide', 'input'],
        outputs=['output'],
        name='output')
    graph = onnx.helper.make_graph(
        [layer1,   layer2],
        'TwoLayerFC',
        [input],
        [output],
        initializer=[w1]
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

if __name__=='__main__':
    # test_matmul()
    # exit(0)
    onnxParamPath='models/opencv_encoder.onnx'
    if os.path.exists(onnxParamPath):
        a = test_forward()
        b = convertOpencvOnnxToNcnn()
        for i in range(len(a)):
            print(np.max(np.abs(a[i].reshape([-1])-b[i].reshape([-1]))))
    else: 
        print("need run sam2_onnx_adaptor first!!")

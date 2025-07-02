import sys
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
import os
# ['image_embed', '/ScatterND_1_output_0','/Unsqueeze_8_output_0',  'mask_input', 'has_mask_input']
shared_input = ['image_embed',   'mask_input', 'has_mask_input']
shared_out = [
    '/transformer/layers.0/cross_attn_token_to_image/MatMul_1_output_0']
class testNet(torch.nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.constant1 = torch.rand([1,1,1, 128])

    def forward(self, input):
        return input/self.constant1

def run_Shape_Inference_model():
    # Preprocessing: create a model with two nodes, Y"s shape is unknown
    node1 = helper.make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])
    node2 = helper.make_node("Transpose", ["Y"], ["Z"], perm=[1, 0, 2])
    graph = helper.make_graph(
        [node1, node2],
        "two-transposes",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, (2, 3, 4))],
        [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (2, 3, 4))],
    )
    original_model = helper.make_model(graph, producer_name="onnx-examples")
    # Check the model and print Y"s shape information
    onnx.checker.check_model(original_model)
    print(f"Before shape inference, the shape info of Y is:\n{original_model.graph.value_info}")
    # Apply shape inference on the model
    inferred_model = shape_inference.infer_shapes(original_model)
    # Check the model and print Y"s shape information
    onnx.checker.check_model(inferred_model)
    print(f"After shape inference, the shape info of Y is:\n{inferred_model.graph.value_info}")
def save_test_onnx_model():
    testNetIns = testNet()
    torch_input = torch.zeros(1, 64, 64, 1)
    onnx_program = torch.onnx.export(testNetIns,  torch_input, "testNet.onnx", verbose=False,
                                    opset_version=17,
                                    input_names=['inputs'],
                                    output_names=['outputs'],
                                    do_constant_folding=False,)


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
    for i,obj in enumerate (model.graph.input):
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
    for i,_ode in enumerate(model.graph.node):
        node = model.graph.node[i]
        print("** nodes %d **  name=%r type=%r input=%r output=%r" % (i,
            node.name, node.op_type, node.input, node.output))
        if i>200:break
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
        print('** initializer ** ',i, '-', initializer[i].name)
        if i == 398:  # encoder check
            dtype = initializer[i].data_type
            print(*initializer[i].dims)
            params = np.frombuffer(initializer[i].raw_data, dtype=onnx_datatype_to_npType(dtype))
        if initializer[i].name in name_lists:
            print(i, '-', initializer[i].name)
            print('shape dim= ',*initializer[i].dims)
            dtype = initializer[i].data_type
            params = np.frombuffer(
                initializer[i].raw_data, dtype=onnx_datatype_to_npType(dtype))
            print('data = ',params)


def convert_sam2_hiera_large_encoder_to_opencvOnnx():
    sys.stdout = open('convert_sam2_encoder_to_opencvOnnx.txt', 'w')
    model = onnx.load('models/sam2_hiera_large_encoder.onnx')
    printNet(model)
    print(model.graph.initializer[618].name)
    print(model.graph.initializer[618].raw_data)
    dtype = model.graph.initializer[618].data_type
    params = np.frombuffer(
        model.graph.initializer[618].raw_data, dtype=onnx_datatype_to_npType(dtype))
    params.reshape([1, 1, 1, 128])
    new_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=[model.graph.initializer[618].name],
        value=onnx.helper.make_tensor(
            'value', onnx.TensorProto.FLOAT, [1, 1, 1, 128], params.reshape([1, 1, 1, 128]))
    )
    model.graph.initializer.remove(model.graph.initializer[618])
    model.graph.node.insert(618, new_node)
    onnx.checker.check_model(model)
    onnx.save(model, 'models/opencv_encoder.onnx')



def test_forward():
    # sys.stdout = open('convert_sam2_decoder_point_label.txt', 'w')
    point_coords = np.array(
        [[[10., 10.], [500., 400.], [200., 600.], [100., 300.], [200., 300.],[0,0]]]).astype(np.float32)
    point_labels = np.array([[1, 1,1,1,-1,1]]).astype(np.float32)
    ScatterND_1_output_0 = np.concatenate((point_coords, np.array(
            [[[0., 0.]]]).astype(np.float32)), axis=1)/1024.
    Unsqueeze_8_output_0 = np.expand_dims(np.concatenate(
            (point_labels, np.array([[-1]]).astype(np.float32)), axis=1), 2)       
    high_res_feats_0 = np.ones([1, 32, 256, 256]).astype(np.float32)
    high_res_feats_1 = np.ones([1, 64, 128, 128]).astype(np.float32)
    image_embed = np.ones([1, 256, 64, 64]).astype(np.float32)
    mask_input = np.zeros((1, 1, 1024//4,1024//4), dtype=np.float32)
    has_mask_input = np.array([0], dtype=np.float32)
    original_size = np.array([1080,1920], dtype=np.int32)
    cut_subgraph('models/decoder.onnx',
                 shared_input,
                 shared_out,
                 'decoderBody2.onnx')
    model = onnx.load('decoderBody2.onnx')
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    session = onnxruntime.InferenceSession('decoderBody2.onnx', providers=onnxruntime.get_available_providers())
    datain={}
    if 'image_embed' in shared_input:
        datain['image_embed'] = image_embed
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

    print(pointCoords[0].shape)
    print(pointCoords[0])
    # print(pointCoords[1].shape)
    # print(pointCoords[1])
    return
def convert_sam2_decoder_point_label():
    checkmodel=True
    inferShapes = True
    inputPointSize=-1# 6+1
    inputPointSizePlus6=-1# 6+inputPointSize
    sys.stdout = open('convert_sam2_decoder_point_label.txt', 'w')
    model = onnx.load('models/decoder.onnx')
    point_coords = np.array(
        [[[10., 10.], [500., 400.], [200., 600.], [100., 300.], [200., 300.],[0,0]]]).astype(np.float32)
    point_labels = np.array([[1, 1,1,1,-1,1]]).astype(np.float32)
### anglysis the point coord in ##################################################################
    if False:
        cut_subgraph('models/decoder.onnx',
                    ['point_coords'], ['/ScatterND_1_output_0'], 'pointCoordsIn.onnx')
        model = onnx.load('pointCoordsIn.onnx')
        if checkmodel:onnx.checker.check_model(model)
        printNet(model)
        session = onnxruntime.InferenceSession(
            'pointCoordsIn.onnx', providers=onnxruntime.get_available_providers())
        pointCoords = session.run(
            None, {"point_coords": point_coords})
        print(pointCoords[0].shape)
        print(pointCoords[0])
        # np.savetxt('outputs.txt', pointCoords[0].squeeze().transpose(1, 0))
### anglysis the point label in ##################################################################
    if False:
        cut_subgraph('models/decoder.onnx',
                    ['point_labels'], ['/Unsqueeze_8_output_0'], 'pointLabelsIn.onnx')
        model = onnx.load('pointLabelsIn.onnx')
        if checkmodel:onnx.checker.check_model(model)
        printNet(model)
        session = onnxruntime.InferenceSession(
            'pointLabelsIn.onnx', providers=onnxruntime.get_available_providers())
        pointLabels = session.run(
            None, {"point_labels": point_labels})
        print(pointLabels[0].shape)
        print(pointLabels[0])

### position embeding ##################################################################
    if False:
        cut_subgraph('models/decoder.onnx',
                     ['/ScatterND_1_output_0', '/Unsqueeze_8_output_0'],
                     ['/transformer/layers.0/self_attn/q_proj/Add_output_0',
                        '/transformer/layers.0/self_attn/k_proj/Add_output_0',
                        '/transformer/layers.0/self_attn/v_proj/Add_output_0'
                        ], 'positionEmbeding.onnx')
        model = onnx.load('positionEmbeding.onnx')
        if checkmodel:onnx.checker.check_model(model)

        model.graph.node.remove(model.graph.node[39])
        model.graph.node.remove(model.graph.node[38])
        model.graph.node.remove(model.graph.node[37])
        model.graph.node.remove(model.graph.node[36])
        model.graph.node.remove(model.graph.node[35])
        model.graph.node.remove(model.graph.node[34])
        model.graph.node.remove(model.graph.node[33])
        model.graph.node.remove(model.graph.node[32])

        model.graph.initializer.remove(model.graph.initializer[16])
        model.graph.initializer.remove(model.graph.initializer[15])
        model.graph.initializer.remove(model.graph.initializer[14])
        model.graph.initializer.remove(model.graph.initializer[13])
        model.graph.initializer.remove(model.graph.initializer[12])
        ConcatNode = onnx.helper.make_node(
            op_type = 'Concat',
            inputs = ['onnx::Expand_2540','/Add_9_output_0'],
            outputs = ['/Concat_10_output_0'],
            name = '/Concat_10',
            axis = 1)
        model.graph.node.insert(32,ConcatNode)
        if checkmodel: onnx.checker.check_model(model)
        onnx.save(model, 'positionEmbeding.onnx')
        printNet(model)
        print("positionEmbeding")
        # return
        ScatterND_1_output_0 = np.concatenate((point_coords, np.array(
            [[[0., 0.]]]).astype(np.float32)), axis=1)/1024.
        Unsqueeze_8_output_0 = np.expand_dims(np.concatenate(
            (point_labels, np.array([[-1]]).astype(np.float32)), axis=1), 2)
        session = onnxruntime.InferenceSession(
            'positionEmbeding.onnx', providers=onnxruntime.get_available_providers())
        positionEmbeding = session.run(
            None, {"/ScatterND_1_output_0": ScatterND_1_output_0, "/Unsqueeze_8_output_0": Unsqueeze_8_output_0})
        print(positionEmbeding[0].shape)
        print(positionEmbeding[0])
        Concat_10_output_0 = positionEmbeding[0]
        return

### position embeding 222 ##################################################################
    if False:
        cut_subgraph('models/decoder.onnx',
                     ['/ScatterND_1_output_0', '/Unsqueeze_8_output_0'],
                     ['/transformer/layers.0/self_attn/Concat_output_0',
                      '/transformer/layers.0/self_attn/Reshape_output_0',
                      '/transformer/layers.0/self_attn/Reshape_1_output_0',
                      '/transformer/layers.0/self_attn/Reshape_2_output_0'
                      ], 'positionEmbeding.onnx')
        model = onnx.load('positionEmbeding.onnx')
        if checkmodel:onnx.checker.check_model(model)
        ScatterND_1_output_0 = np.concatenate((point_coords, np.array(
            [[[0., 0.]]]).astype(np.float32)), axis=1)/1024.
        Unsqueeze_8_output_0 = np.expand_dims(np.concatenate(
            (point_labels, np.array([[-1]]).astype(np.float32)), axis=1), 2)
        session = onnxruntime.InferenceSession(
            'positionEmbeding.onnx', providers=onnxruntime.get_available_providers())
        positionEmbeding = session.run(
            None, {"/ScatterND_1_output_0": ScatterND_1_output_0, "/Unsqueeze_8_output_0": Unsqueeze_8_output_0})
        print(positionEmbeding[0])
        print(positionEmbeding[0].shape)
        print(positionEmbeding[1].shape)
        print(positionEmbeding[2].shape)




        model.graph.node.remove(model.graph.node[39])
        model.graph.node.remove(model.graph.node[38])
        model.graph.node.remove(model.graph.node[37])
        model.graph.node.remove(model.graph.node[36])
        model.graph.node.remove(model.graph.node[35])
        model.graph.node.remove(model.graph.node[34])
        model.graph.node.remove(model.graph.node[33])
        model.graph.node.remove(model.graph.node[32])

        model.graph.initializer.remove(model.graph.initializer[16])
        model.graph.initializer.remove(model.graph.initializer[15])
        model.graph.initializer.remove(model.graph.initializer[14])
        model.graph.initializer.remove(model.graph.initializer[13])
        model.graph.initializer.remove(model.graph.initializer[12])
        ConcatNode = onnx.helper.make_node(
            op_type='Concat',
            inputs=['onnx::Expand_2540', '/Add_9_output_0'],
            outputs=['/Concat_10_output_0'],
            name='/Concat_10',
            axis=1)
        model.graph.node.insert(32, ConcatNode)
        if checkmodel: onnx.checker.check_model(model)
        onnx.save(model, 'positionEmbeding.onnx')
        printNet(model)
        print("positionEmbeding")
        # return
        ScatterND_1_output_0 = np.concatenate((point_coords, np.array(
            [[[0., 0.]]]).astype(np.float32)), axis=1)/1024.
        Unsqueeze_8_output_0 = np.expand_dims(np.concatenate(
            (point_labels, np.array([[-1]]).astype(np.float32)), axis=1), 2)
        session = onnxruntime.InferenceSession(
            'positionEmbeding.onnx', providers=onnxruntime.get_available_providers())
        positionEmbeding = session.run(
            None, {"/ScatterND_1_output_0": ScatterND_1_output_0, "/Unsqueeze_8_output_0": Unsqueeze_8_output_0})
        print(positionEmbeding[0].shape)
        print(positionEmbeding[0]) 
        return


### decoderBody ##################################################################
    if True:
        cut_subgraph('models/decoder.onnx', ['high_res_feats_0', 'high_res_feats_1', 'image_embed', '/ScatterND_1_output_0',
                     '/Unsqueeze_8_output_0', 'mask_input', 'has_mask_input', 'orig_im_size'], ['masks', 'iou_predictions'], 'decoderBody.onnx')
        model = onnx.load('decoderBody.onnx')
        if checkmodel: onnx.checker.check_model(model)

        pointSize = np.array([point_coords.shape[1]], dtype=np.float32)
        high_res_feats_0 = np.ones([1, 32, 256, 256]).astype(np.float32)
        high_res_feats_1 = np.ones([1, 64, 128, 128]).astype(np.float32)
        image_embed = np.ones([1, 256, 64, 64]).astype(np.float32)
        ScatterND_1_output_0 = np.concatenate((point_coords, np.array(
            [[[0., 0.]]]).astype(np.float32)), axis=1)/1024.
        Unsqueeze_8_output_0 = np.expand_dims(np.concatenate(
            (point_labels, np.array([[-1]]).astype(np.float32)), axis=1), 2)
        mask_input = np.zeros((1, 1, 1024//4,1024//4), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)
        original_size = np.array([1080,1920], dtype=np.int32)

        for index, eachNode in enumerate(model.graph.input):
            now_name = model.graph.input[index].name
            if now_name == '/ScatterND_1_output_0':
                model.graph.input.remove(model.graph.input[index])
                ScatterND_1_output_0_node = helper.make_tensor_value_info(
                    '/ScatterND_1_output_0', TensorProto.FLOAT, [1, inputPointSize, 2])
                model.graph.input.insert(index, ScatterND_1_output_0_node)
                print('change the ScatterND_1_output_0 input node')
            if now_name == '/Unsqueeze_8_output_0':
                model.graph.input.remove(model.graph.input[index])
                Unsqueeze_8_output_0_node = helper.make_tensor_value_info(
                    '/Unsqueeze_8_output_0', TensorProto.FLOAT, [1, inputPointSize, 1])
                model.graph.input.insert(index, Unsqueeze_8_output_0_node)
                print('change the Unsqueeze_8_output_0 input node')
                break

# *************************************
        nodeBaseShift=0
        inputArrayPlus6 = helper.make_tensor_value_info(
            'inputArrayPlus6', TensorProto.FLOAT, [1, inputPointSizePlus6, 1])
        model.graph.input.append(inputArrayPlus6)
        # new_input = helper.make_tensor_value_info(
        #     "pointSize", onnx.TensorProto.FLOAT, [1])
        # model.graph.input.append(inputArrayPlus6)
        # constNumbers = onnx.numpy_helper.from_array(
        #     np.arange(0+7, 256+7, 1, np.int64), name='constNumbers')  # [1,256]
        # model.graph.initializer.append(constNumbers)  # [1,256]
        # innerPointCast = onnx.helper.make_node(
        #     op_type='Cast',
        #     inputs=['pointSize'],
        #     outputs=['innerPointCast'],
        #     name='innerPointCast',
        #     to=7)
        # model.graph.node.insert(nodeBaseShift, innerPointCast)
        # nodeBaseShift+=1
        # innerPointCastPick = onnx.helper.make_node(
        #     op_type='Gather',
        #     inputs=['constNumbers', 'innerPointCast'],
        #     outputs=['innerPointCastPick'],
        #     name='innerPointCastPick')
        # model.graph.node.insert(nodeBaseShift, innerPointCastPick)
        # nodeBaseShift += 1

        # innerPointCntBaseValue = np.array([7], dtype=np.int64)  # 7
        # innerPointCntBase = onnx.numpy_helper.from_array(
        #     innerPointCntBaseValue, name='innerPointCntBase')  # 7
        # model.graph.initializer.append(innerPointCntBase)  # 7
        # innerPointCntAdd7 = onnx.helper.make_node(
        #     op_type='Add',
        #     inputs=['innerPointCntBase', 'innerPointCastPick'],
        #     outputs=['innerPointCntAdd7'],
        #     name='innerPointCntAdd7')
        # model.graph.node.insert(2, innerPointCntAdd7)
        # nodeBaseShift += 1

        neg1 = onnx.numpy_helper.from_array(
            np.array([-1]).astype(np.float32), name='neg1')  # -1 float
        model.graph.initializer.append(neg1)  # 1
        const1 = onnx.numpy_helper.from_array(
            np.array([1]).astype(np.int64), name='const1')  # 1
        model.graph.initializer.append(const1)  # 1
        const8 = onnx.numpy_helper.from_array(
            np.array([8]).astype(np.int64), name='const8')  # 8
        model.graph.initializer.append(const8)  # 8
        const32 = onnx.numpy_helper.from_array(
            np.array([32]).astype(np.int64), name='const32')  # 32
        model.graph.initializer.append(const32)  # 32
        const16 = onnx.numpy_helper.from_array(
            np.array([16]).astype(np.int64), name='const16')  # 16
        model.graph.initializer.append(const16)  # 16
        const256 = onnx.numpy_helper.from_array(
            np.array([256]).astype(np.int64), name='const256')  # 256
        model.graph.initializer.append(const256)  # 128
        const128 = onnx.numpy_helper.from_array(
            np.array([128]).astype(np.int64), name='const128')  # 128
        model.graph.initializer.append(const128)  # 128
        shape_1_4096_8_16 = onnx.numpy_helper.from_array(
            np.array([1,4096,8,16]).astype(np.int64), name='shape_1_4096_8_16')  # 128
        model.graph.initializer.append(shape_1_4096_8_16)  # 128
        constOnes256f = onnx.numpy_helper.from_array(
            np.ones([1, 1, 256]).astype(np.float32), name='constOnes256f')  # constOnes256f
        model.graph.initializer.append(constOnes256f)  # constOnes256f
        array6 = onnx.numpy_helper.from_array(
            np.ones([1, 6, 2]).astype(np.float32), name='array6')  # array6
        model.graph.initializer.append(array6)  # 6
        sqrt32inv = onnx.numpy_helper.from_array(
            np.array([1/np.sqrt(32)]).astype(np.float32), name='sqrt32inv')  # sqrt32inv
        model.graph.initializer.append(sqrt32inv)  # sqrt32inv

        pointInShapeLayer = onnx.helper.make_node(
            op_type='Shape',
            inputs=['/ScatterND_1_output_0'],
            outputs=['pointInShape'],
            name='pointInShapeLayer')
        pointCntLayer = onnx.helper.make_node(
            op_type='Gather',
            inputs=['pointInShape', 'const1'],
            outputs=['pointCnt'],
            name='pointCntLayer')
        shape_1_p_8_32 = onnx.helper.make_node(
            op_type='Concat',
            inputs=['const1', 'pointCnt', 'const8', 'const32'],
            outputs=['shape_1_p_8_32'],
            name='shape_1_p_8_32',
            axis=0)
        
 
        arrayPlus6ShapeNode = onnx.helper.make_node(
            op_type='Shape',
            inputs=['inputArrayPlus6'],
            outputs=['arrayPlus6Shape'],
            name='arrayPlus6Shape')
        arrayPlus6CntNode = onnx.helper.make_node(
            op_type='Gather',
            inputs=['arrayPlus6Shape', 'const1'],
            outputs=['arrayPlus6Cnt'],
            name='arrayPlus6Cnt')
        shape_1_p6_8_32 = onnx.helper.make_node(
            op_type='Concat',
            inputs=['const1', 'arrayPlus6Cnt', 'const8', 'const32'],
            outputs=['shape_1_p6_8_32'],
            name='shape_1_p6_8_32',
            axis=0)
        shape_1_p6_256 = onnx.helper.make_node(
            op_type='Concat',
            inputs=['const1', 'arrayPlus6Cnt', 'const256'],
            outputs=['shape_1_p6_256'],
            name='shape_1_p6_256',
            axis=0)
        shape_1_p6_128 = onnx.helper.make_node(
            op_type='Concat',
            inputs=['const1', 'arrayPlus6Cnt', 'const128'],
            outputs=['shape_1_p6_128'],
            name='shape_1_p6_128',
            axis=0)
        shape_1_p6_8_16 = onnx.helper.make_node(
            op_type='Concat',
            inputs=['const1', 'arrayPlus6Cnt', 'const8', 'const16'],
            outputs=['shape_1_p6_8_16'],
            name='shape_1_p6_8_16',
            axis=0)
        shape_1_8_p6_p6 = onnx.helper.make_node(
            op_type='Concat',
            inputs=['const1',  'const8', 'arrayPlus6Cnt', 'arrayPlus6Cnt'],
            outputs=['shape_1_8_p6_p6'],
            name='shape_1_8_p6_p6',
            axis=0)

        # shape_1_p_8_32_Node
        model.graph.node.insert(nodeBaseShift, pointInShapeLayer)
        nodeBaseShift += 1
        # shape_1_p_8_32_Node
        model.graph.node.insert(nodeBaseShift, pointCntLayer)
        nodeBaseShift += 1
        # shape_1_p_8_32_Node
        model.graph.node.insert(nodeBaseShift, shape_1_p_8_32)
        nodeBaseShift += 1  
        # shape_1_p6_8_32_Node
        model.graph.node.insert(nodeBaseShift, arrayPlus6ShapeNode)
        nodeBaseShift += 1
        # shape_1_p6_8_32_Node
        model.graph.node.insert(nodeBaseShift, arrayPlus6CntNode)
        nodeBaseShift += 1
        # shape_1_p6_8_32_Node
        model.graph.node.insert(nodeBaseShift, shape_1_p6_8_32)
        nodeBaseShift += 1
        # shape_1_p_256_Node
        model.graph.node.insert(nodeBaseShift, shape_1_p6_256)
        nodeBaseShift += 1
        # shape_1_p6_128_Node
        model.graph.node.insert(nodeBaseShift, shape_1_p6_128)
        nodeBaseShift += 1
        # shape_1_p6_8_16_Node
        model.graph.node.insert(nodeBaseShift, shape_1_p6_8_16)
        nodeBaseShift += 1
        # shape_1_8_p6_p6_Node
        model.graph.node.insert(nodeBaseShift, shape_1_8_p6_p6)
        nodeBaseShift += 1
        

        removelist = ['/Shape_18',
                      '/Expand_8']
        removeIndexlist = []
        for i, node in enumerate(model.graph.node):
            if node.name in removelist:
                removeIndexlist.append(i)
        removeIndexlist.sort(reverse=True)
        for removeI in removeIndexlist:
            model.graph.node.remove(model.graph.node[removeI])
        Expand_8_output_0_node = onnx.helper.make_node(
            op_type='MatMul',
            inputs=['/Unsqueeze_8_output_0', 'constOnes256f'],
            outputs=['/Expand_8_output_0'],
            name='/Expand_8_output_0_node')
        model.graph.node.insert(
            removeIndexlist[-1], Expand_8_output_0_node)
        if inferShapes:
            model = onnx.shape_inference.infer_shapes(model)
        if checkmodel:
            onnx.checker.check_model(model)


        removelist = ['/Not','/Cast_4']
        removeIndexlist = []
        for i, node in enumerate(model.graph.node):
            if node.name in removelist:
                removeIndexlist.append(i)
        removeIndexlist.sort(reverse=True)
        for removeI in removeIndexlist:
            model.graph.node.remove(model.graph.node[removeI])
        gt_neg1_node = onnx.helper.make_node(
            op_type='Greater',
            inputs=['/Expand_8_output_0', 'neg1'],
            outputs=['/gt_neg1'],
            name='/gt_neg1')
        gt_neg1_cast = onnx.helper.make_node(
            op_type='Cast',
            inputs=['/gt_neg1'],
            outputs=['/Cast_4_output_0'],
            name='/gt_neg1_cast',
            to=1)
        model.graph.node.insert(removeIndexlist[-1], gt_neg1_cast)
        model.graph.node.insert(removeIndexlist[-1], gt_neg1_node)
        if inferShapes:
            model = onnx.shape_inference.infer_shapes(model)
        if checkmodel:
            onnx.checker.check_model(model)

        removelist = ['/Shape_19', '/Gather_10','/Unsqueeze_10', '/Concat_9', '/Equal_12', '/Where_6', '/Expand_9']
        removeIndexlist = []
        for i, node in enumerate(model.graph.node):
            if node.name in removelist:
                removeIndexlist.append(i)
        removeIndexlist.sort(reverse=True)
        for removeI in removeIndexlist:
            model.graph.node.remove(model.graph.node[removeI])
        dtype = model.graph.initializer[31].data_type
        params = np.frombuffer(
            model.graph.initializer[31].raw_data, dtype=onnx_datatype_to_npType(dtype))
        Expand_9_output_0_node = onnx.helper.make_node(
            op_type='Constant',
            inputs=[],
            outputs=['/Expand_9_output_0'],
            name='/Expand_9_output_0',
            value=onnx.helper.make_tensor(
                'value', onnx.TensorProto.FLOAT, [
                    1, 6, 256], params.reshape([1, 6, 256])))
        model.graph.node.insert(removeIndexlist[-1], Expand_9_output_0_node)
        if inferShapes:
            model = onnx.shape_inference.infer_shapes(model)
        if checkmodel:
            onnx.checker.check_model(model)

        removelist = ['/transformer/layers.0/self_attn/Shape', 
                      '/transformer/layers.0/self_attn/Gather', 
                      '/transformer/layers.0/self_attn/Gather_1',
                      '/transformer/layers.0/self_attn/Unsqueeze', 
                      '/transformer/layers.0/self_attn/Unsqueeze_1', 
                      '/transformer/layers.0/self_attn/Concat', 
                      '/transformer/layers.0/self_attn/Reshape']
        removeIndexlist = []
        for i, node in enumerate(model.graph.node):
            if node.name in removelist:
                removeIndexlist.append(i)
        removeIndexlist.sort(reverse=True)
        for removeI in removeIndexlist:
            model.graph.node.remove(model.graph.node[removeI])
        transformer_layers0_selfattn_Reshape_node = onnx.helper.make_node(
            op_type='Reshape',
            inputs=['/transformer/layers.0/self_attn/q_proj/Add_output_0',
                    'shape_1_p6_8_32'],
            outputs=['/transformer/layers.0/self_attn/Reshape_output_0'],
            name='transformer_layers0_selfattn_Reshape_node')
        model.graph.node.insert(
            removeIndexlist[-1], transformer_layers0_selfattn_Reshape_node)
        if inferShapes:
            model = onnx.shape_inference.infer_shapes(model)
        if checkmodel:
            onnx.checker.check_model(model)





        removelist = ['/transformer/layers.0/self_attn/Shape_3',
                      '/transformer/layers.0/self_attn/Gather_3',
                      '/transformer/layers.0/self_attn/Gather_4',
                      '/transformer/layers.0/self_attn/Unsqueeze_3',
                      '/transformer/layers.0/self_attn/Unsqueeze_4',
                      '/transformer/layers.0/self_attn/Concat_1',
                      '/transformer/layers.0/self_attn/Reshape_1']
        removeIndexlist = []
        for i, node in enumerate(model.graph.node):
            if node.name in removelist:
                removeIndexlist.append(i)
        removeIndexlist.sort(reverse=True)
        for removeI in removeIndexlist:
            model.graph.node.remove(model.graph.node[removeI])
        transformer_layers0_selfattn_Reshape1_node = onnx.helper.make_node(
            op_type='Reshape',
            inputs=['/transformer/layers.0/self_attn/k_proj/Add_output_0',
                    'shape_1_p6_8_32'],
            outputs=['/transformer/layers.0/self_attn/Reshape_1_output_0'],
            name='transformer_layers0_selfattn_Reshape1_node')
        model.graph.node.insert(
            removeIndexlist[-1], transformer_layers0_selfattn_Reshape1_node)
        if inferShapes:
            model = onnx.shape_inference.infer_shapes(model)
        if checkmodel:
            onnx.checker.check_model(model)

        removelist = ['/transformer/layers.0/self_attn/Shape_6',
                      '/transformer/layers.0/self_attn/Gather_6',
                      '/transformer/layers.0/self_attn/Gather_7',
                      '/transformer/layers.0/self_attn/Unsqueeze_6',
                      '/transformer/layers.0/self_attn/Unsqueeze_7',
                      '/transformer/layers.0/self_attn/Concat_2',
                      '/transformer/layers.0/self_attn/Reshape_2']
        removeIndexlist = []
        for i, node in enumerate(model.graph.node):
            if node.name in removelist:
                removeIndexlist.append(i)
        removeIndexlist.sort(reverse=True)
        for removeI in removeIndexlist:
            model.graph.node.remove(model.graph.node[removeI])
        transformer_layers0_selfattn_Reshape2_node = onnx.helper.make_node(
            op_type='Reshape',
            inputs=['/transformer/layers.0/self_attn/v_proj/Add_output_0',
                    'shape_1_p6_8_32'],
            outputs=['/transformer/layers.0/self_attn/Reshape_2_output_0'],
            name='transformer_layers0_selfattn_Reshape2_node')
        model.graph.node.insert(
            removeIndexlist[-1], transformer_layers0_selfattn_Reshape2_node)
        if inferShapes:
            model = onnx.shape_inference.infer_shapes(model)
        if checkmodel:
            onnx.checker.check_model(model)


        removelist = ['/transformer/layers.0/self_attn/Shape_10',
                      '/transformer/layers.0/self_attn/Gather_9',
                      '/transformer/layers.0/self_attn/Gather_10',
                      '/transformer/layers.0/self_attn/Gather_11',
                      '/transformer/layers.0/self_attn/Gather_12',
                      '/transformer/layers.0/self_attn/Mul_2',
                      '/transformer/layers.0/self_attn/Unsqueeze_9',
                      '/transformer/layers.0/self_attn/Unsqueeze_10',
                      '/transformer/layers.0/self_attn/Unsqueeze_11',
                      '/transformer/layers.0/self_attn/Concat_3',
                      '/transformer/layers.0/self_attn/Reshape_3']
        removeIndexlist = []
        for i, node in enumerate(model.graph.node):
            if node.name in removelist:
                removeIndexlist.append(i)
        removeIndexlist.sort(reverse=True)
        for removeI in removeIndexlist:
            model.graph.node.remove(model.graph.node[removeI])
        transformer_layers0_selfattn_Reshape3_node = onnx.helper.make_node(
            op_type='Reshape',
            inputs=['/transformer/layers.0/self_attn/Transpose_3_output_0',
                    'shape_1_p6_256'],
            outputs=['/transformer/layers.0/self_attn/Reshape_3_output_0'],
            name='transformer_layers0_selfattn_Reshape3_node')
        model.graph.node.insert(
            removeIndexlist[-1]+1, transformer_layers0_selfattn_Reshape3_node)
        if inferShapes:
            model = onnx.shape_inference.infer_shapes(model)
        if checkmodel:
            onnx.checker.check_model(model)
        
        
        

        removelist = ['/transformer/layers.0/cross_attn_token_to_image/Shape',
                      '/transformer/layers.0/cross_attn_token_to_image/Gather',
                      '/transformer/layers.0/cross_attn_token_to_image/Gather_1',
                      '/transformer/layers.0/cross_attn_token_to_image/Unsqueeze',
                      '/transformer/layers.0/cross_attn_token_to_image/Unsqueeze_1',
                      '/transformer/layers.0/cross_attn_token_to_image/Concat',
                      '/transformer/layers.0/cross_attn_token_to_image/Reshape']
        removeIndexlist = []
        for i, node in enumerate(model.graph.node):
            if node.name in removelist:
                removeIndexlist.append(i)
        removeIndexlist.sort(reverse=True)
        for removeI in removeIndexlist:
            model.graph.node.remove(model.graph.node[removeI])
        transformer_layers0_cross_attn_Reshape_node = onnx.helper.make_node(
            op_type='Reshape',
            inputs=['/transformer/layers.0/cross_attn_token_to_image/q_proj/Add_output_0',
                    'shape_1_p6_8_16'],
            outputs=[
                '/transformer/layers.0/cross_attn_token_to_image/Reshape_output_0'],
            name='transformer_layers0_cross_attn_Reshape_node')
        model.graph.node.insert(
            removeIndexlist[-1], transformer_layers0_cross_attn_Reshape_node)

        removelist = ['/transformer/layers.0/cross_attn_token_to_image/Shape_6',
                      '/transformer/layers.0/cross_attn_token_to_image/Gather_6',
                      '/transformer/layers.0/cross_attn_token_to_image/Gather_7',
                      '/transformer/layers.0/cross_attn_token_to_image/Unsqueeze_6',
                      '/transformer/layers.0/cross_attn_token_to_image/Unsqueeze_7',
                      '/transformer/layers.0/cross_attn_token_to_image/Concat_2',
                      '/transformer/layers.0/cross_attn_token_to_image/Reshape_2']
        removeIndexlist = []
        for i, node in enumerate(model.graph.node):
            if node.name in removelist:
                removeIndexlist.append(i)
        removeIndexlist.sort(reverse=True)
        for removeI in removeIndexlist:
            model.graph.node.remove(model.graph.node[removeI])
        transformer_layers0_cross_attn_Reshape2_node = onnx.helper.make_node(
            op_type='Reshape',
            inputs=['/transformer/layers.0/cross_attn_token_to_image/v_proj/Add_output_0',
                    'shape_1_4096_8_16'],
            outputs=[
                '/transformer/layers.0/cross_attn_token_to_image/Reshape_2_output_0'],
            name='transformer_layers0_cross_attn_Reshape2_node')
        model.graph.node.insert(
            removeIndexlist[-1], transformer_layers0_cross_attn_Reshape2_node)



        if inferShapes:model = onnx.shape_inference.infer_shapes(model)
        if checkmodel:  onnx.checker.check_model(model)
        onnx.save(model, 'decoderBody2.onnx')
        shared_input_temp = shared_input
        if '/Unsqueeze_8_output_0' in shared_input_temp:
            shared_input_temp = shared_input_temp+['inputArrayPlus6']
        cut_subgraph('decoderBody2.onnx', 
                     shared_input_temp,
                     shared_out,
                     'decoderBody2.onnx')
        model = onnx.load('decoderBody2.onnx')
  
        if inferShapes: model = onnx.shape_inference.infer_shapes(model)
        if checkmodel:  onnx.checker.check_model(model)

        datain = {}
        if 'inputArrayPlus6' in shared_input_temp:
            datain['inputArrayPlus6'] = np.concatenate((Unsqueeze_8_output_0, np.ones(
                [1, 6, 1]).astype(Unsqueeze_8_output_0.dtype)), axis=1)
        if 'image_embed' in shared_input_temp:
            datain['image_embed'] = image_embed
        if '/ScatterND_1_output_0' in shared_input_temp:
            datain['/ScatterND_1_output_0'] = ScatterND_1_output_0
        if '/Unsqueeze_8_output_0' in shared_input_temp:
            datain['/Unsqueeze_8_output_0'] = Unsqueeze_8_output_0            
        if 'mask_input' in shared_input_temp:
            datain['mask_input'] = mask_input
        if 'has_mask_input' in shared_input_temp:
            datain['has_mask_input'] = has_mask_input
        if 'orig_im_size' in shared_input_temp:
            datain['orig_im_size'] = orig_im_size
        session = onnxruntime.InferenceSession(
            'decoderBody2.onnx', providers=onnxruntime.get_available_providers())
        pointCoords = session.run(
            None, datain)
        print(pointCoords[0])
        print(pointCoords[0].shape)
        # print(pointCoords[1])
        # print(pointCoords[1].shape)
        return
        print(pointCoords[0])


        shape0_initializer = helper.make_tensor(
            '/Constant_shape_0', onnx.TensorProto.INT64, [1], np.zeros(1).astype(np.int64))
        model.graph.initializer.append(shape0_initializer)
        # '/transformer/Concat_1'
        model.graph.node.remove(model.graph.node[54])
        transformer_Concat_1 = onnx.helper.make_node(
            "Concat",
            inputs=['/transformer/Slice_1_output_0', '/Constant_shape_0'],
            name='/transformer/Concat_1',
            outputs=['/transformer/Concat_1_output_0'],
            axis=0
        ) 
        model.graph.node.insert(54, transformer_Concat_1)


        printNet(model)
        model.graph.node.remove(model.graph.node[46])  # '/Reshape_9'
        model.graph.node.remove(model.graph.node[45])  # '/Concat_13'
        model.graph.node.remove(model.graph.node[44])  # '/Slice_4'
        model.graph.node.remove(model.graph.node[43])  # '/Shape_23'
        model.graph.node.remove(model.graph.node[42])  # '/Tile'
        model.graph.node.remove(model.graph.node[41])  # '/reshape_8'
        model.graph.node.remove(model.graph.node[40])  # '/OneHot'
        model.graph.node.remove(model.graph.node[39])  # '/Concat_11'
        model.graph.node.remove(model.graph.node[38])  # '/Reshape_7'
        model.graph.node.remove(model.graph.node[37])  # '/Gather_11'
        model.graph.node.remove(model.graph.node[36])  # '/Shape_21'


        printNet(model)

        dtype = model.graph.initializer[20].data_type
        params = np.frombuffer(
            model.graph.initializer[20].raw_data, dtype=onnx_datatype_to_npType(dtype))
        params.reshape([1, 1, 256, 64, 64])
        Reshape_9_output_0 = onnx.helper.make_node(
            "Constant",
            inputs=[],
            name='/Reshape_9_output_0',
            outputs=['/Reshape_9_output_0'],
            value=onnx.helper.make_tensor(
                'value', onnx.TensorProto.FLOAT, [
                    1, 256, 64, 64], params.reshape([1, 256, 64, 64]))
        )
        model.graph.node.insert(36, Reshape_9_output_0)





        # model.graph.initializer.remove(
        #     model.graph.initializer[19])  # '/Shape_21'
        # new_initializer = helper.make_tensor('/Constant_21_output_0', onnx.TensorProto.INT64, [1], np.ones(1).astype(np.int64))
        # model.graph.initializer.insert(19, new_initializer)

        printNet(model)
        onnx.checker.check_model(model)
        onnx.save(model, 'decoderBody2.onnx')
 
 


        cut_subgraph('decoderBody2.onnx',
                     [], ['/Tile_output_0', '/Concat_13_output_0'], 'decoderBody3.onnx')
        model = onnx.load('decoderBody3.onnx')
        onnx.checker.check_model(model)
        printNet(model)
        session = onnxruntime.InferenceSession(
            'decoderBody3.onnx', providers=onnxruntime.get_available_providers())
        pointCoords = session.run(
            None, {})
        print(pointCoords[0].shape)
        print(pointCoords[0])


        Concat_10_output_0_idx = 3 # /Concat_10_output_0  :4 
        now_name = model.graph.input[Concat_10_output_0_idx].name
        new_input = helper.make_tensor_value_info('/Concat_10_output_0a',
                       model.graph.input[Concat_10_output_0_idx].type.tensor_type.elem_type,
                       [0,256]) #修改模型输出维度
        model.graph.input.remove(model.graph.input[Concat_10_output_0_idx]) #删除旧节点，
        model.graph.input.insert(Concat_10_output_0_idx,new_input)      #插入新节点,可以保证之前的顺序


        new_shape = np.array([1,-1,256], dtype='int64')

        shape_tensor = onnx.helper.make_tensor('/Concat_10_output_0_shape',TensorProto.INT64,new_shape.shape, new_shape)
        shape_node = helper.make_node("Constant",[],['/Concat_10_output_0_shape'],name='/Concat_10_output_0_shape',value=shape_tensor)

        reshape_node = onnx.helper.make_node(
                'Reshape',
                inputs=['/Concat_10_output_0a', '/Concat_10_output_0_shape'],
                outputs=['/Concat_10_output_0'],
                name='/Concat_10_output_0_reshape'
            )
        model.graph.node.insert(36, reshape_node)
        model.graph.node.insert(36, shape_node)

        onnx.save(model,'decoderBody2.onnx')
        
        printNet(model)
        print()
        # cut_subgraph('models/decoder.onnx',['mask_input'], ['/mask_downscaling/mask_downscaling.0/Conv_output_0'], 'decoderBody.onnx')
        # model = onnx.load('decoderBody.onnx')
        # # onnx.checker.check_model(model)
        # printNet(model)
  

def creat_simple_net():
    sys.stdout = open('convert_sam2_decoder_point_label.txt', 'w')
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1,-1,2])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, -1,256])
    w1 = onnx.numpy_helper.from_array(np.random.rand(2, 256).astype(np.float32), name='w1')  # [2,256]
    layer1 = onnx.helper.make_node(
        op_type='MatMul',
        inputs=['input','w1'],
        outputs=['output'],
        name='layer1')

    graph = onnx.helper.make_graph(
        [layer1],
        'TwoLayerFC',
        [input],
        [output],
        initializer=[w1]
    )
    model = helper.make_model(graph, producer_name='onnx-example')
    onnx.checker.check_model(model)
    printNet(model)


def test_dynamic_reshape():
    # sys.stdout = open('convert_sam2_decoder_point_label.txt', 'w')
    input = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [ -1, 2])
    output = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [ -1, 256])
    concatShape = helper.make_tensor_value_info(
        'concatShape', TensorProto.INT64, [3])
    outputReshape = helper.make_tensor_value_info(
        'outputReshape', TensorProto.FLOAT, [-1,8,32])
    w1 = onnx.numpy_helper.from_array(np.random.rand(
        2, 256).astype(np.float32), name='w1')  # [2,256]
    const1 = onnx.numpy_helper.from_array(
        np.array([1]).astype(np.int64), name='const1')  # 1
    const_1 = onnx.numpy_helper.from_array(
        np.array([3]).astype(np.int64), name='const_1')  # -1
    const8 = onnx.numpy_helper.from_array(
        np.array([8]).astype(np.int64), name='const8')  # 8
    const32 = onnx.numpy_helper.from_array(
        np.array([32]).astype(np.int64), name='const32')  # 32
    layer1 = onnx.helper.make_node(
        op_type='MatMul',
        inputs=['input', 'w1'],
        outputs=['output'],
        name='layer1')
    concatShapeLayer = onnx.helper.make_node(
        op_type='Concat',
        inputs=['const_1', 'const8', 'const32'],
        outputs=['concatShape'],
        name='concatShapeLayer',
        axis=0)
    reshapeLayer = onnx.helper.make_node(
        op_type='Reshape',
        inputs=['output', 'concatShape'],
        outputs=['outputReshape'],
        name='reshapeLayerLayer')
    graph = onnx.helper.make_graph(
        [layer1, concatShapeLayer, reshapeLayer],
        'TwoLayerFC',
        [input],
        [outputReshape],
        initializer=[w1, const1, const_1, const8, const32]
    )
    model = helper.make_model(graph, producer_name='onnx-example')
    # model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    model.ir_version = 10
    model.opset_import[0].version = 21
    onnx.save(model, 'test.onnx')



    model = onnx.load('test.onnx')
    onnx.checker.check_model(model)
    printNet(model)
    session = onnxruntime.InferenceSession('test.onnx', providers=onnxruntime.get_available_providers())
    coordPts = np.array([1,2,3,4,5,6]).astype(np.float32).reshape( 3, 2)
    out = session.run(None, {'input': coordPts})
    print(out[0].shape)
    print(out[0])

if __name__=='__main__':
    # test_dynamic_reshape()
    test_forward()
    convert_sam2_decoder_point_label()
    exit()

# convert_sam2_hiera_large_encoder_to_opencvOnnx()


exit()
print(onnx.helper.printable_graph(model.graph))
print(model)



# 检查模型并打印
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))

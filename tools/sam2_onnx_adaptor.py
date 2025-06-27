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
    #               '/Where_6_output_0']  # decoder position embedding
    name_lists = ['/Constant_2_output_0',
                  '/Constant_5_output_0',
                  '/Constant_72_output_0',
                  '/Unsqueeze_11_output_0',
                  '/Constant_21_output_0']  # onehot
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
    point_coords = np.array(
        [[[10., 10.], [500., 400.], [200., 600.], [100., 300.], [200., 300.],[0,0]]]).astype(np.float32)
    point_labels = np.array([[1, 1,1,1,-1,1]]).astype(np.float32)
    ScatterND_1_output_0 = np.concatenate((point_coords, np.array(
            [[[0., 0.]]]).astype(np.float32)), axis=1)/1024.
    Unsqueeze_8_output_0 = np.expand_dims(np.concatenate(
            (point_labels, np.array([[-1]]).astype(np.float32)), axis=1), 2)        
    cut_subgraph('decoderBody.onnx', 
                 ['/ScatterND_1_output_0', '/Unsqueeze_8_output_0'],
                 ['/transformer/layers.0/self_attn/Softmax_output_0', '/transformer/layers.0/self_attn/Transpose_1_output_0'], 
                 'decoderBody2.onnx')
    session = onnxruntime.InferenceSession('decoderBody2.onnx', providers=onnxruntime.get_available_providers())
    pointCoords = session.run(
        None, {
            #    'high_res_feats_0': high_res_feats_0,
            #    'high_res_feats_1': high_res_feats_1,
            #    'image_embed': image_embed,
                '/ScatterND_1_output_0': ScatterND_1_output_0,
                '/Unsqueeze_8_output_0': Unsqueeze_8_output_0,
            #    'mask_input': mask_input,
            #    'has_mask_input': has_mask_input,
            #    'orig_im_size': original_size
                })

    print(pointCoords[0].shape)
    print(pointCoords[0])
    return
def convert_sam2_decoder_point_label():
    checkmodel=True
    inferShapes = True
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
                    '/ScatterND_1_output_0', TensorProto.FLOAT, [1, -1, 2])
                model.graph.input.insert(index, ScatterND_1_output_0_node)
                print('change the ScatterND_1_output_0 input node')
            if now_name == '/Unsqueeze_8_output_0':
                model.graph.input.remove(model.graph.input[index])
                Unsqueeze_8_output_0_node = helper.make_tensor_value_info(
                    '/Unsqueeze_8_output_0', TensorProto.FLOAT, [1, -1, 1])
                model.graph.input.insert(index, Unsqueeze_8_output_0_node)
                print('change the Unsqueeze_8_output_0 input node')
                break

# *************************************
        nodeBaseShift=0
        inputArrayPlus6 = helper.make_tensor_value_info(
            'inputArrayPlus6', TensorProto.FLOAT, [1, -1, 1]) 
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




        const1 = onnx.numpy_helper.from_array(
            np.array([1]).astype(np.int64), name='const1')  # 1
        model.graph.initializer.append(const1)  # 1
        const_1 = onnx.numpy_helper.from_array(
            np.array([-1]).astype(np.int64), name='const_1')  # -1
        model.graph.initializer.append(const_1)  # -1
        const8 = onnx.numpy_helper.from_array(
            np.array([8]).astype(np.int64), name='const8')  # 8
        model.graph.initializer.append(const8)  # 8
        const32 = onnx.numpy_helper.from_array(
            np.array([32]).astype(np.int64), name='const32')  # 32
        model.graph.initializer.append(const32)  # 32
        const256 = onnx.numpy_helper.from_array(
            np.array([256]).astype(np.int64), name='const256')  # 256
        model.graph.initializer.append(const256)  # 32
        constOnes256f = onnx.numpy_helper.from_array(
            np.ones([1, 1, 256]).astype(np.float32), name='constOnes256f')  # constOnes256f
        model.graph.initializer.append(constOnes256f)  # constOnes256f
        array6 = onnx.numpy_helper.from_array(
            np.ones([1, 6, 2]).astype(np.float32), name='array6')  # array6
        model.graph.initializer.append(array6)  # 6
        ones13 = onnx.numpy_helper.from_array(
            np.ones([1, 8, 13, 1]).astype(np.float32), name='ones13')  # ones13
        model.graph.initializer.append(ones13)  # ones13

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

        model.graph.node.remove(model.graph.node[8+nodeBaseShift])  # not
        model.graph.node.remove(model.graph.node[7+nodeBaseShift])  # not
        Expand_8_output_0_node = onnx.helper.make_node(
            op_type='MatMul',
            inputs=['/Unsqueeze_8_output_0', 'constOnes256f'],
            outputs=['/Expand_8_output_0'],
            name='/Expand_8_output_0_node')
        model.graph.node.insert(7+nodeBaseShift, Expand_8_output_0_node)  # not


        value = np.array([-1], dtype=np.float32)  # not
        neg1 = onnx.numpy_helper.from_array(value, name='neg1')  # not
        model.graph.initializer.append(neg1)  # not
        model.graph.node.remove(model.graph.node[10+nodeBaseShift])  # not
        model.graph.node.remove(model.graph.node[9+nodeBaseShift])  # not
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
        model.graph.node.insert(9+nodeBaseShift, gt_neg1_node)  # not
        model.graph.node.insert(10+nodeBaseShift, gt_neg1_cast)  # not


  

        model.graph.node.remove(model.graph.node[72+nodeBaseShift])  # expand9
        model.graph.node.remove(model.graph.node[71+nodeBaseShift])  # expand9
        model.graph.node.remove(model.graph.node[70+nodeBaseShift])  # expand9
        model.graph.node.remove(model.graph.node[69+nodeBaseShift])  # expand9
        model.graph.node.remove(model.graph.node[68+nodeBaseShift])  # expand9
        model.graph.node.remove(model.graph.node[67+nodeBaseShift])  # expand9
        model.graph.node.remove(model.graph.node[66+nodeBaseShift])  # expand9
        
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
                    1, 6,256], params.reshape([1, 6,256])))
        model.graph.node.insert(
            64+nodeBaseShift, Expand_9_output_0_node)  # expand9

        # /transformer/Reshape
        model.graph.node.remove(model.graph.node[88+nodeBaseShift])
        # /transformer/Reshape
        model.graph.node.remove(model.graph.node[87+nodeBaseShift])
        # /transformer/Reshape
        model.graph.node.remove(model.graph.node[86+nodeBaseShift])
        model.graph.node.remove(model.graph.node[85+nodeBaseShift])  # /transformer/Reshape
        model.graph.node.remove(
            model.graph.node[79+nodeBaseShift])  # Reshape_9
        model.graph.node.remove(
            model.graph.node[78+nodeBaseShift])  # Concat_13
        model.graph.node.remove(model.graph.node[77+nodeBaseShift])  # Slice_4
        model.graph.node.remove(model.graph.node[76+nodeBaseShift])  # Shape_23
        model.graph.node.remove(model.graph.node[75+nodeBaseShift])  # Tile
        model.graph.node.remove(model.graph.node[74+nodeBaseShift])  # Reshape_8
        model.graph.node.remove(model.graph.node[73+nodeBaseShift])  # onehot
        model.graph.node.remove(model.graph.node[72+nodeBaseShift])  # Concat_11
        model.graph.node.remove(model.graph.node[71+nodeBaseShift])  # Reshape_7
        model.graph.node.remove(model.graph.node[70+nodeBaseShift])  # Gather_11
        model.graph.node.remove(model.graph.node[69+nodeBaseShift])  # Shape_21
        value = np.array([1, 256, 64*64], dtype=np.int64)
        transformer_Reshape_output_0_shape = onnx.numpy_helper.from_array(
            value, name='transformer_Reshape_output_0_shape')
        model.graph.initializer.append(
            transformer_Reshape_output_0_shape)
        Reshape_9_output_0_node = onnx.helper.make_node(
            op_type='Reshape',
            inputs=['/Unsqueeze_11_output_0',
                    'transformer_Reshape_output_0_shape'],
            outputs=['/transformer/Reshape_1_output_0'],
            name='/transformer/Reshape_1')
        model.graph.node.insert(69+nodeBaseShift, Reshape_9_output_0_node)

        model.graph.node.remove(
            model.graph.node[73+nodeBaseShift])  # /Shape_24
        model.graph.node.remove(
            model.graph.node[72+nodeBaseShift])  # /transformer/Slice
        # /transformer/Concat
        model.graph.node.remove(model.graph.node[71+nodeBaseShift])
        # /transformer/Reshape
        model.graph.node.remove(model.graph.node[70+nodeBaseShift])
        transformer_Reshape_node = onnx.helper.make_node(
            op_type='Reshape',
            inputs=['/Add_11_output_0',
                    'transformer_Reshape_output_0_shape'],
            outputs=['/transformer/Reshape_output_0'],
            name='/transformer/Reshape')
        model.graph.node.insert(70+nodeBaseShift, transformer_Reshape_node)
        if checkmodel:  onnx.checker.check_model(model)

        model.graph.node.remove(
            model.graph.node[85+nodeBaseShift])  # 0self_attn/Reshape
        model.graph.node.remove(
            model.graph.node[84+nodeBaseShift])  # 0self_attn/Concat
        model.graph.node.remove(model.graph.node[83+nodeBaseShift])  # 0self_attn/Unsqueeze_1
        # 0self_attn/Unsqueeze
        model.graph.node.remove(model.graph.node[82+nodeBaseShift])
        model.graph.node.remove(model.graph.node[81+nodeBaseShift])  # 0self_attn/Gather_1
        model.graph.node.remove(model.graph.node[80+nodeBaseShift])  # 0self_attn/Gather
        model.graph.node.remove(
            model.graph.node[79+nodeBaseShift])  # 0self_attn/Shape

        transformer_layers0_selfattn_Reshape_node = onnx.helper.make_node(
            op_type='Reshape',
            inputs=['/transformer/layers.0/self_attn/q_proj/Add_output_0',
                    'shape_1_p6_8_32'],
            outputs=['/transformer/layers.0/self_attn/Reshape_output_0'],
            name='/transformer/layers.0/self_attn/Reshape')
        model.graph.node.insert(
            79+nodeBaseShift, transformer_layers0_selfattn_Reshape_node)
        if inferShapes:model = onnx.shape_inference.infer_shapes(model)
        if checkmodel:  onnx.checker.check_model(model)

        model.graph.node.remove(
            model.graph.node[87+nodeBaseShift])  # 0self_attn/Reshape
        model.graph.node.remove(
            model.graph.node[86+nodeBaseShift])  # 0self_attn/Concat
        model.graph.node.remove(model.graph.node[85+nodeBaseShift])  # 0self_attn/Unsqueeze_1
        model.graph.node.remove(model.graph.node[84+nodeBaseShift])  # 0self_attn/Unsqueeze
        # 0self_attn/Gather_1
        model.graph.node.remove(model.graph.node[83+nodeBaseShift])
        model.graph.node.remove(
            model.graph.node[82+nodeBaseShift])  # 0self_attn/Gather
        model.graph.node.remove(model.graph.node[81+nodeBaseShift])  # 0self_attn/Shape
        transformer_layers0_selfattn_Reshape1_node = onnx.helper.make_node(
            op_type='Reshape',
            inputs=['/transformer/layers.0/self_attn/k_proj/Add_output_0',
                    'shape_1_p6_8_32'],
            outputs=['/transformer/layers.0/self_attn/Reshape_1_output_0'],
            name='/transformer/layers.0/self_attn/Reshape_1')
        model.graph.node.insert(
            81+nodeBaseShift, transformer_layers0_selfattn_Reshape1_node)
        if inferShapes:model = onnx.shape_inference.infer_shapes(model)
        if checkmodel:  onnx.checker.check_model(model)

        model.graph.node.remove(
            model.graph.node[88+nodeBaseShift])  # 0self_attn/Reshape
        model.graph.node.remove(
            model.graph.node[87+nodeBaseShift])  # 0self_attn/Concat
        model.graph.node.remove(model.graph.node[86+nodeBaseShift])  # 0self_attn/Unsqueeze_1
        # 0self_attn/Unsqueeze
        model.graph.node.remove(model.graph.node[85+nodeBaseShift])
        model.graph.node.remove(model.graph.node[84+nodeBaseShift])  # 0self_attn/Gather_1
        model.graph.node.remove(model.graph.node[83+nodeBaseShift])  # 0self_attn/Gather
        model.graph.node.remove(
            model.graph.node[82+nodeBaseShift])  # 0self_attn/Shape
        transformer_layers0_selfattn_Reshape2_node = onnx.helper.make_node(
            op_type='Reshape',
            inputs=['/transformer/layers.0/self_attn/v_proj/Add_output_0',
                    'shape_1_p6_8_32'],
            outputs=['/transformer/layers.0/self_attn/Reshape_2_output_0'],
            name='/transformer/layers.0/self_attn/Reshape_2')
        model.graph.node.insert(
            82+nodeBaseShift, transformer_layers0_selfattn_Reshape2_node)
        if inferShapes:model = onnx.shape_inference.infer_shapes(model)
        if checkmodel:  onnx.checker.check_model(model)

        model.graph.node.remove(model.graph.node[90+nodeBaseShift])  # Sqrt_1
        model.graph.node.remove(model.graph.node[88+nodeBaseShift])  # Div_3
        model.graph.node.remove(model.graph.node[87+nodeBaseShift])  # Sqrt
        model.graph.node.remove(model.graph.node[86+nodeBaseShift])  # Cast_6
        model.graph.node.remove(model.graph.node[85+nodeBaseShift])  # Slice
        model.graph.node.remove(model.graph.node[84+nodeBaseShift])  # Shape_9
        transformer_layers0_selfattn_Sqrt_1_node = onnx.helper.make_node(
            op_type='Constant',
            inputs=[],
            outputs=['/transformer/layers.0/self_attn/Sqrt_1_output_0'],
            name='/transformer/layers.0/self_attn/Sqrt_1',
            value=onnx.helper.make_tensor('value', onnx.TensorProto.FLOAT, [1], np.array([np.sqrt(1/np.sqrt(32.))])))
        model.graph.node.insert(
            84+nodeBaseShift, transformer_layers0_selfattn_Sqrt_1_node)



        model.graph.node.remove(model.graph.node[102+nodeBaseShift])#Reshape_3
        model.graph.node.remove(
            model.graph.node[101+nodeBaseShift])  # Concat_3
        model.graph.node.remove(
            model.graph.node[100+nodeBaseShift])  # Unsqueeze_11
        model.graph.node.remove(
            model.graph.node[99+nodeBaseShift])  # Unsqueeze_10
        model.graph.node.remove(
            model.graph.node[98+nodeBaseShift])  # Unsqueeze_9
        model.graph.node.remove(model.graph.node[97+nodeBaseShift])  # Mul_2
        model.graph.node.remove(
            model.graph.node[95+nodeBaseShift])  # Gather_12
        model.graph.node.remove(model.graph.node[94+nodeBaseShift]) #Gather_11
        model.graph.node.remove(model.graph.node[93+nodeBaseShift])#Gather_10
        model.graph.node.remove(model.graph.node[92+nodeBaseShift])  # Gather_9
        model.graph.node.remove(model.graph.node[91+nodeBaseShift]) #Shape_10
        transformer_layers0_selfattn_Reshape_3_node = onnx.helper.make_node(
            op_type='Reshape',
            inputs=['/transformer/layers.0/self_attn/Transpose_3_output_0',
                    'shape_1_p6_256'],
            outputs=['/transformer/layers.0/self_attn/Reshape_3_output_0'],
            name='/transformer/layers.0/self_attn/Reshape_3')
        model.graph.node.insert(
            92+nodeBaseShift, transformer_layers0_selfattn_Reshape_3_node)
        if inferShapes:
            model = onnx.shape_inference.infer_shapes(model)
        if checkmodel:
            onnx.checker.check_model(model)



        # model.graph.node.remove(model.graph.node[89+nodeBaseShift])  # Softmax
        # MatMul_output_0_exp_node = onnx.helper.make_node(
        #     op_type='Exp',
        #     inputs=['/transformer/layers.0/self_attn/MatMul_output_0'],
        #     outputs=['/transformer/layers.0/self_attn/MatMul_output_0_exp'],
        #     name='MatMul_output_0_exp_node')
        # MatMul_output_0_exp_sum_node = onnx.helper.make_node(
        #     op_type='MatMul',
        #     inputs=['/transformer/layers.0/self_attn/MatMul_output_0_exp', 'ones13'],
        #     outputs=['/transformer/layers.0/self_attn/MatMul_output_0_exp_sum'],
        #     name='MatMul_output_0_exp_sum_node')
        # transformer_layers0_selfattn_Softmax_node = onnx.helper.make_node(
        #     op_type='Div',
        #     inputs=['/transformer/layers.0/self_attn/MatMul_output_0_exp',
        #             '/transformer/layers.0/self_attn/MatMul_output_0_exp_sum'],
        #     outputs=['/transformer/layers.0/self_attn/Softmax_output_0'],
        #     name='transformer_layers0_selfattn_Softmax_node')
        # model.graph.node.insert(
        #     89+nodeBaseShift, transformer_layers0_selfattn_Softmax_node)
        # model.graph.node.insert(
        #     89+nodeBaseShift, MatMul_output_0_exp_sum_node)
        # model.graph.node.insert(
        #     89+nodeBaseShift, MatMul_output_0_exp_node)




        if inferShapes:model = onnx.shape_inference.infer_shapes(model)
        if checkmodel:  onnx.checker.check_model(model)
        onnx.save(model, 'decoderBody2.onnx')
        cut_subgraph('decoderBody2.onnx', 
                    ['image_embed', '/ScatterND_1_output_0', '/Unsqueeze_8_output_0', 'inputArrayPlus6', 'mask_input', 'has_mask_input'],
                     #['/transformer/layers.0/self_attn/Reshape_output_0','/transformer/layers.0/self_attn/Reshape_1_output_0'],
                     ['/transformer/layers.0/cross_attn_token_to_image/MatMul_1_output_0',
                        '/transformer/layers.0/cross_attn_token_to_image/Transpose_3_output_0',
                        '/transformer/layers.0/cross_attn_token_to_image/Concat_3_output_0',
                        '/transformer/layers.0/cross_attn_token_to_image/Shape_10_output_0'],
                     'decoderBody2.onnx')
        model = onnx.load('decoderBody2.onnx')
        printNet(model)
  

        model = onnx.load('decoderBody2.onnx')
        if inferShapes: model = onnx.shape_inference.infer_shapes(model)
        if checkmodel:  onnx.checker.check_model(model)
        printNet(model)


        # cut_subgraph('decoderBody2.onnx', ['high_res_feats_0', 'high_res_feats_1', 'image_embed', '/ScatterND_1_output_0',
        #              '/Unsqueeze_8_output_0', 'mask_input', 'has_mask_input', 'orig_im_size'],
        #              ['/transformer/layers.0/cross_attn_token_to_image/Softmax_output_0', '/transformer/layers.0/cross_attn_token_to_image/Transpose_1_output_0','/transformer/layers.0/cross_attn_token_to_image/MatMul_1_output_0', 'masks', 'iou_predictions'], 'decoderBody2.onnx')

        session = onnxruntime.InferenceSession(
            'decoderBody2.onnx', providers=onnxruntime.get_available_providers())
        pointCoords = session.run(
            None, {
                'inputArrayPlus6': np.concatenate((Unsqueeze_8_output_0, np.ones([1, 6, 1]).astype(Unsqueeze_8_output_0.dtype)), axis=1),
                #    'high_res_feats_0': high_res_feats_0,
                #    'high_res_feats_1': high_res_feats_1,
                   'image_embed': image_embed,
                   '/ScatterND_1_output_0': ScatterND_1_output_0,
                   '/Unsqueeze_8_output_0': Unsqueeze_8_output_0,
                   'mask_input': mask_input,
                   'has_mask_input': has_mask_input,
                #    'orig_im_size': original_size
                   })

        print(pointCoords[0])
        print(pointCoords[0].shape)
        print(pointCoords[1])
        print(pointCoords[1].shape)
        print(pointCoords[2])
        print(pointCoords[2].shape)
        print(pointCoords[3])
        print(pointCoords[3].shape)
        # print(pointCoords[4])
        # print(pointCoords[4].shape)
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


def test_dynamic_reshape():
    # input = helper.make_tensor_value_info(
    #     'input', TensorProto.FLOAT, [1, -1, 2])
    # output = helper.make_tensor_value_info(
    #     'output', TensorProto.FLOAT, [1, -1, 256])
    # outputReshape = helper.make_tensor_value_info(
    #     'outputReshape', TensorProto.FLOAT, [1,-1,8,32])
    # w1 = onnx.numpy_helper.from_array(np.random.rand(
    #     2, 256).astype(np.float32), name='w1')  # [2,256]
    # const1 = onnx.numpy_helper.from_array(
    #     np.array([1]).astype(np.int64), name='const1')  # [2,256]
    # const8 = onnx.numpy_helper.from_array(
    #     np.array([8]).astype(np.int64), name='const8')  # [2,256]
    # const32 = onnx.numpy_helper.from_array(
    #     np.array([32]).astype(np.int64), name='const32')  # [2,256]
    # layer1 = onnx.helper.make_node(
    #     op_type='MatMul',
    #     inputs=['input', 'w1'],
    #     outputs=['output'],
    #     name='layer1')
    # MulShapeLayer = onnx.helper.make_node(
    #     op_type='Shape',
    #     inputs=['output'],
    #     outputs=['outShape'],
    #     name='MulShapeLayer')
    # getPtsCntLayer = onnx.helper.make_node(
    #     op_type='Gather',
    #     inputs=['outShape', 'const1'],
    #     outputs=['PtsCnt'],
    #     name='getPtsCntLayer')
    # concatShapeLayer = onnx.helper.make_node(
    #     op_type='Concat',
    #     inputs=['const1', 'PtsCnt', 'const8', 'const32'],
    #     outputs=['concatShape'],
    #     name='concatShapeLayer',
    #     axis=0)
    # reshapeLayer = onnx.helper.make_node(
    #     op_type='Reshape',
    #     inputs=['output', 'concatShape'],
    #     outputs=['outputReshape'],
    #     name='reshapeLayerLayer')
    # graph = onnx.helper.make_graph(
    #     [layer1, MulShapeLayer, getPtsCntLayer, concatShapeLayer, reshapeLayer],
    #     'TwoLayerFC',
    #     [input],
    #     [output, outputReshape],
    #     initializer=[w1, const1,const8,const32]
    # )
    # model = helper.make_model(graph, producer_name='onnx-example')
    # model = onnx.shape_inference.infer_shapes(model)
    # onnx.checker.check_model(model)
    # model.ir_version = 10
    # model.opset_import[0].version = 21
    # printNet(model)
    # onnx.save(model, 'test.onnx')



    model = onnx.load('test.onnx')
    onnx.checker.check_model(model)
    printNet(model)
    session = onnxruntime.InferenceSession('test.onnx', providers=onnxruntime.get_available_providers())
    coordPts = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32).reshape(1, 3, 2)
    out = session.run(None, {'input': coordPts})
    print(out[0].shape)
    print(out[0])

convert_sam2_decoder_point_label()
# test_forward()
exit()
# test_dynamic_reshape()
# convert_sam2_hiera_large_encoder_to_opencvOnnx()


exit()
print(onnx.helper.printable_graph(model.graph))
print(model)



# 检查模型并打印
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))

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
    return
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


def convert_sam2_decoder_point_label():
    # sys.stdout = open('convert_sam2_decoder_point_label.txt', 'w')
    model = onnx.load('models/decoder.onnx')
    point_coords = np.array(
        [[[10., 10.], [500., 400.], [200., 600.], [100., 300.], [200., 300.],[0,0]]]).astype(np.float32)
    point_labels = np.array([[1, 1,1,1,-1,1]]).astype(np.float32)
### anglysis the point coord in ##################################################################
    if False:
        cut_subgraph('models/decoder.onnx',
                    ['point_coords'], ['/ScatterND_1_output_0'], 'pointCoordsIn.onnx')
        model = onnx.load('pointCoordsIn.onnx')
        onnx.checker.check_model(model)
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
        onnx.checker.check_model(model)
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
        onnx.checker.check_model(model)

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
        # onnx.checker.check_model(model)
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
        onnx.checker.check_model(model)
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
        # onnx.checker.check_model(model)
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
        onnx.checker.check_model(model)
        # printNet(model)
 
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

        # session = onnxruntime.InferenceSession(
        #     'decoderBody.onnx', providers=onnxruntime.get_available_providers())
        # pointCoords = session.run(
        #     None, {'high_res_feats_0': high_res_feats_0, 
        #            'high_res_feats_1': high_res_feats_1,
        #            'image_embed': image_embed,
        #            '/ScatterND_1_output_0': ScatterND_1_output_0,
        #            '/Unsqueeze_8_output_0': Unsqueeze_8_output_0,
        #            'mask_input': mask_input,
        #            'has_mask_input': has_mask_input,
        #            'orig_im_size': original_size})


# *************************************
        value = np.array([-1], dtype=np.float32)  # not
        neg1 = onnx.numpy_helper.from_array(value, name='neg1')  # not
        model.graph.initializer.append(neg1)  # not
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
        model.graph.node.insert(10, gt_neg1_node)  # not
        model.graph.node.insert(11, gt_neg1_cast)  # not
        model.graph.node.remove(model.graph.node[13])  # not
        model.graph.node.remove(model.graph.node[12])  # not

        model.graph.node.remove(model.graph.node[73])   #expand9
        model.graph.node.remove(model.graph.node[72])  # expand9
        model.graph.node.remove(model.graph.node[71])  # expand9
        model.graph.node.remove(model.graph.node[70])   #expand9
        model.graph.node.remove(model.graph.node[69])   #expand9
        model.graph.node.remove(model.graph.node[68])   #expand9
        model.graph.node.remove(model.graph.node[67])  # expand9
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
        model.graph.node.insert(67, Expand_9_output_0_node)  # expand9

        # printNet(model)
        # return

        model.graph.node.remove(model.graph.node[89])  # /transformer/Reshape
        model.graph.node.remove(model.graph.node[88])  # /transformer/Reshape
        model.graph.node.remove(model.graph.node[87])  # /transformer/Reshape
        model.graph.node.remove(model.graph.node[86])  # /transformer/Reshape
        model.graph.node.remove(model.graph.node[80])  # Reshape_9
        model.graph.node.remove(model.graph.node[79])  # Concat_13
        model.graph.node.remove(model.graph.node[78])  # Slice_4
        model.graph.node.remove(model.graph.node[77])  # Shape_23
        model.graph.node.remove(model.graph.node[76])  # Tile
        model.graph.node.remove(model.graph.node[75])  # Reshape_8
        model.graph.node.remove(model.graph.node[74])  # onehot
        model.graph.node.remove(model.graph.node[73])  # Concat_11
        model.graph.node.remove(model.graph.node[72])  # Reshape_7
        model.graph.node.remove(model.graph.node[71])  # Gather_11
        model.graph.node.remove(model.graph.node[70])  # Shape_21
        value = np.array([1,256,64*64], dtype=np.int64) 
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
        model.graph.node.insert(70, Reshape_9_output_0_node)  

        model.graph.node.remove(model.graph.node[74])  # /transformer/Reshape
        model.graph.node.remove(model.graph.node[73])  # /Shape_24
        model.graph.node.remove(model.graph.node[72])  # /transformer/Slice
        model.graph.node.remove(model.graph.node[71])  # /transformer/Concat
        transformer_Reshape_node = onnx.helper.make_node(
            op_type='Reshape',
            inputs=['/Add_11_output_0',
                    'transformer_Reshape_output_0_shape'],
            outputs=['/transformer/Reshape_output_0'],
            name='/transformer/Reshape')
        model.graph.node.insert(71, transformer_Reshape_node)
        onnx.checker.check_model(model)

        model.graph.node.remove(model.graph.node[86])  # 0self_attn/Reshape
        model.graph.node.remove(model.graph.node[85])  # 0self_attn/Concat        
        model.graph.node.remove(model.graph.node[84])  # 0self_attn/Unsqueeze_1
        model.graph.node.remove(model.graph.node[83])  # 0self_attn/Unsqueeze
        model.graph.node.remove(model.graph.node[82])  # 0self_attn/Gather_1
        model.graph.node.remove(model.graph.node[81])  # 0self_attn/Gather
        model.graph.node.remove(model.graph.node[80])  # 0self_attn/Shape
        selfattn_Reshape_output_0_shape_data = np.array([1, -1, 8, 32], dtype=np.int64)
        selfattn_Reshape_output_0_shape = onnx.numpy_helper.from_array(
            selfattn_Reshape_output_0_shape_data, name='selfattn_Reshape_output_0_shape')
        model.graph.initializer.append(selfattn_Reshape_output_0_shape)
        transformer_layers0_selfattn_Reshape_node = onnx.helper.make_node(
            op_type='Reshape',
            inputs=['/transformer/layers.0/self_attn/q_proj/Add_output_0',
                    'selfattn_Reshape_output_0_shape'],
            outputs=['/transformer/layers.0/self_attn/Reshape_output_0'],
            name='/transformer/layers.0/self_attn/Reshape')
        model.graph.node.insert(80, transformer_layers0_selfattn_Reshape_node)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)

        model.graph.node.remove(model.graph.node[95])  # 0self_attn/Reshape
        model.graph.node.remove(model.graph.node[94])  # 0self_attn/Concat
        model.graph.node.remove(model.graph.node[93])  # 0self_attn/Unsqueeze_1
        model.graph.node.remove(model.graph.node[92])  # 0self_attn/Unsqueeze
        model.graph.node.remove(model.graph.node[91])  # 0self_attn/Gather_1
        model.graph.node.remove(model.graph.node[90])  # 0self_attn/Gather
        model.graph.node.remove(model.graph.node[89])  # 0self_attn/Shape
        selfattn_Reshape2_output_0_shape_data = np.array(
            [1, -1, 8, 32], dtype=np.int64)
        selfattn_Reshape2_output_0_shape = onnx.numpy_helper.from_array(
            selfattn_Reshape2_output_0_shape_data, name='selfattn_Reshape2_output_0_shape')
        model.graph.initializer.append(selfattn_Reshape2_output_0_shape)
        transformer_layers0_selfattn_Reshape2_node = onnx.helper.make_node(
            op_type='Reshape',
            inputs=['/transformer/layers.0/self_attn/v_proj/Add_output_0',
                    'selfattn_Reshape2_output_0_shape'],
            outputs=['/transformer/layers.0/self_attn/Reshape_2_output_0'],
            name='/transformer/layers.0/self_attn/Reshape_2')
        model.graph.node.insert(89, transformer_layers0_selfattn_Reshape2_node)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)


        printNet(model)
        onnx.save(model, 'decoderBody2.onnx')
       
        cut_subgraph('decoderBody2.onnx', [ '/ScatterND_1_output_0',
                     '/Unsqueeze_8_output_0'],
                     ['/transformer/layers.0/self_attn/Concat_1_output_0','/transformer/layers.0/self_attn/Softmax_output_0', '/transformer/layers.0/self_attn/Transpose_1_output_0'], 'decoderBody2.onnx')

        
        # return
        
        # cut_subgraph('decoderBody2.onnx', ['high_res_feats_0', 'high_res_feats_1', 'image_embed', '/ScatterND_1_output_0',
        #              '/Unsqueeze_8_output_0', 'mask_input', 'has_mask_input', 'orig_im_size'],
        #              ['/transformer/layers.0/cross_attn_token_to_image/Softmax_output_0', '/transformer/layers.0/cross_attn_token_to_image/Transpose_1_output_0','/transformer/layers.0/cross_attn_token_to_image/MatMul_1_output_0', 'masks', 'iou_predictions'], 'decoderBody2.onnx')

        session = onnxruntime.InferenceSession(
            'decoderBody2.onnx', providers=onnxruntime.get_available_providers())
        pointCoords = session.run(
            None, {
                   #'high_res_feats_0': high_res_feats_0,
                   #'high_res_feats_1': high_res_feats_1,
                #    'image_embed': image_embed,
                   '/ScatterND_1_output_0': ScatterND_1_output_0,
                   '/Unsqueeze_8_output_0': Unsqueeze_8_output_0,
                #    'mask_input': mask_input,
                #    'has_mask_input': has_mask_input
                   #'orig_im_size': original_size
                   })

        print(pointCoords[0].shape)
        print(pointCoords[0])
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
  





# [[[4.8828125e-04 4.8828125e-04]
#   [1.8745117e+00 1.0541992e+00]
#   [0.0000000e+00 0.0000000e+00]]]

    # session = onnxruntime.InferenceSession(
    #     'pointLabelsIn.onnx', providers=onnxruntime.get_available_providers())
    # pointLabels = session.run(
    #     None, {"point_labels": point_labels})
    # print(pointLabels[0].shape)
    # print(pointLabels[0])


# cut_subgraph('encoder.onnx', ['image'], ['/image_encoder/neck/position_encoding/Unsqueeze_8_output_0'], 'sub.onnx')
# save_test_onnx_model()

# sys.stdout = open('log.txt', 'w')
# model = onnx.load('models/decoder.onnx')
# model = onnx.load('models/decoder.onnx')
# printNet(model)
# session = onnxruntime.InferenceSession('testNet.onnx', providers=onnxruntime.get_available_providers())
# inputs = np.ones([1, 64, 64, 1]).astype(np.float32)
# outputs = session.run(None, {"inputs": inputs})


# convert_sam2_hiera_large_encoder_to_opencvOnnx()
convert_sam2_decoder_point_label()

exit()
print(onnx.helper.printable_graph(model.graph))
print(model)



# 检查模型并打印
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))

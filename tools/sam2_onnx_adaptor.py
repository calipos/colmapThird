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


class testNet(torch.nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.constant1 = torch.rand([1,1,1, 128])

    def forward(self, input):
        return input/self.constant1


    def test(self, outPath, hdrCollectionRoot, modelPath):
        outPath = "test"
        outPath = os.path.join(os.getcwd(), outPath)
        deleteDirs(outPath)
        os.makedirs(outPath)
        collections = get_files(hdrCollectionRoot)

        HdrNetIns = HdrNet3(EXP_CNT, HIST_CNT)
        HdrNetIns.load_state_dict(torch.load(modelPath))
        HdrNetIns.eval()
        s = [x for x in range(0, 256, int(256/HIST_CNT))]
        s = np.array(s).reshape((HIST_CNT, -1))
        for testData in collections.keys():
            dirName = testData.split('_')
            dirName = dirName[0]+'_'+dirName[1]
            picDir = os.path.join(hdrCollectionRoot, dirName)
            for filepath, dirnames, filenames in os.walk(picDir):
                break
            picCnt = collections[testData].shape[1]

            thisClusterStamp = testData[len(dirName)+1:]
            picCluster = []
            for d in filenames:
                if (d.startswith(thisClusterStamp)):
                    split1 = d.split('_')
                    exp = float(split1[2])
                    if split1[3] == '-1':
                        sensity = 1
                    else:
                        sensity = float(split1[3][:-4])
                    picCluster.append((d, exp*sensity))
            sorted(picCluster, key=lambda exp: exp[1])

            hist = collections[testData][:, [1, 7]].astype(np.float32)
            imgMean = (np.sum(hist*s, axis=0))/255
            assert (np.max(imgMean) < 1+1e-7)
            # idxF = np.eye(EXP_CNT)[expLevel].reshape(-1,EXP_CNT).astype(np.float32)
            histFeat = np.concatenate(
                [hist, imgMean.reshape(1, -1)], axis=0).transpose()
            histFeat = histFeat.reshape(1, 2, -1)
            output = HdrNetIns(torch.from_numpy(histFeat.copy()).float())
            pickOut = int(torch.round(output))
            if pickOut < 0:
                pickOut = 0
            if pickOut > 8:
                pickOut = 8
            savePath = testData+'-'+str(pickOut)+'.jpg'
            savePath = os.path.join(outPath, savePath)
            oldPath = os.path.join(picDir, picCluster[pickOut][0])
            mycopyfile(oldPath, savePath)


def create_model():
    # 使用ONNX helper functions定义两个输入张量'a'和'b'以及一个权重张量'w'
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [1, 3])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [1, 3])
    w1 = helper.make_tensor_value_info('w1', TensorProto.FLOAT, [3, 3])
    w2 = helper.make_tensor_value_info('w2', TensorProto.FLOAT, [3, 2])

    # 为了使事情简单，我们来创建一些初始权重数据
    # 但是在实际的模型中，应该在训练过程中来更新这些权重值
    weights1 = helper.make_tensor('w1', TensorProto.FLOAT, [3, 3], [
                                  1, 2, 3, 4, 5, 6, 7, 8, 9])
    weights2 = helper.make_tensor('w2', TensorProto.FLOAT, [
                                  3, 2], [1, 2, 3, 4, 5, 6])

    fc1 = helper.make_node(
        'Gemm',
        inputs=['a', 'w1', 'b'],
        outputs=['h1'],
        alpha=1.0,
        beta=1.0,
        transB=1
    )

    fc2 = helper.make_node(
        'Gemm',
        inputs=['h1', 'w2', 'b'],
        outputs=['y'],
        alpha=1.0,
        beta=1.0,
        transB=1
    )

    graph = helper.make_graph(
        [fc1, fc2],
        'TwoLayerFC',
        [a, b, w1, w2],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2])],
        value_info=[helper.make_tensor_value_info(
            'h1', TensorProto.FLOAT, [1, 3])],
        initializer=[weights1, weights2]
    )

    model = helper.make_model(graph, producer_name='onnx-example')
    # The serialization
    with open("linear_regression.onnx", "wb") as f:
        f.write(model.SerializeToString())
    return model
def create_model2():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])
    pads = helper.make_tensor_value_info("pads", TensorProto.INT64, [8])  # pads is INT64
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [5, 4])

    # Create Pad node with 'value' attribute (not input)
    node_def = helper.make_node(
        "Pad",
        inputs=["X", "pads"],  # Inputs: X and pads (INT64)
        outputs=["Y"],
        mode="constant",       # Attribute for padding mode
        value=0.0              # Attribute for fill value
    )

    # Build graph and model
    graph_def = helper.make_graph(
        [node_def],
        "test-model",
        [X, pads],
        [Y],
    )
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-example",
        opset_imports=[helper.make_opsetid("", 11)]  # OPSET 11 required
    )

    # Validate the model
    onnx.checker.check_model(model_def)
    print("Model is valid!")
def create_model3():
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None])
    node1 = helper.make_node('MatMul', ['X', 'A'], ['XA'])
    node2 = helper.make_node('Add', ['XA', 'B'], ['Y'])
    graph = helper.make_graph([node1, node2], 'lr', [X, A, B], [Y])
    onnx_model = helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    # The serialization
    with open("linear_regression.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    # display
    print(onnx_model)

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
    else:
        raise TypeError("don't support data type")
def shape2tuple(shape):
    return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)
def printNet(model):
    # print('** inputs **')
    # print(model.graph.input)

    # in a more nicely format
    for obj in model.graph.input:
        print("** inputs **  name=%r dtype=%r shape=%r" % (
            obj.name, obj.type.tensor_type.elem_type,
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

    initializer = model.graph.initializer
    name_lists = ["/image_encoder/neck/position_encoding/Constant_28_output_0",
                  '/image_encoder/neck/position_encoding/Unsqueeze_8_output_0']
    for i in range(len(initializer)):
        print(i, '-', initializer[i].name)
        if i==398:
            dtype = initializer[i].data_type
            print(*initializer[i].dims)
            params = np.frombuffer(initializer[i].raw_data, dtype=onnx_datatype_to_npType(dtype))
        if initializer[i].name in name_lists:
            print(i, '-', initializer[i].name, "\t", end="")
            print('shape = ',*initializer[i].dims)
            dtype = initializer[i].data_type
            params = np.frombuffer(
                initializer[i].raw_data, dtype=onnx_datatype_to_npType(dtype))
            print(params, end="\n")


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

# cut_subgraph('encoder.onnx', ['image'], ['/image_encoder/neck/position_encoding/Unsqueeze_8_output_0'], 'sub.onnx')
# save_test_onnx_model()
# model = onnx.load('encoder.onnx')
# session = onnxruntime.InferenceSession('testNet.onnx', providers=onnxruntime.get_available_providers())
# inputs = np.ones([1, 64, 64, 1]).astype(np.float32)
# outputs = session.run(None, {"inputs": inputs})


convert_sam2_hiera_large_encoder_to_opencvOnnx()


exit()
print(onnx.helper.printable_graph(model.graph))
print(model)



# 检查模型并打印
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))

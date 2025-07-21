import numpy as np
import onnx

targetParamPath = 'models/opencv_encoder.onnx'

if __name__ == '__main__':
    model = onnx.load(targetParamPath)
    for node in model.graph.node:
        if node.op_type == "Softmax":
            print(node.name)
            print(node.op_type)
            print(node.attribute)


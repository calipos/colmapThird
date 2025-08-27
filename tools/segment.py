import os

import onnx.helper
from sam2 import SAM2Image, draw_masks
import onnx
import cv2
import numpy as np
import netron
import logging
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    filename='test.log',
                    filemode='w')
def initSAM2():
    encoder_model_path = "models/sam2_hiera_large_encoder.onnx"
    decoder_model_path = "models/decoder.onnx"
    if not os.path.exists(encoder_model_path):
        print("encoder_model_path not found:", encoder_model_path)
        return -1
    if not os.path.exists(decoder_model_path):
        print("decoder_model_path not found:", decoder_model_path)
        return -1
    try:
        sam2 = SAM2Image(encoder_model_path, decoder_model_path)
        return sam2
    except:
        return -1
    

def segFaceBaseLandmark(sam2model, imgPath, landmarks): 
    img = cv2.imread(imgPath)
    sam2model.set_image(img)
    landmarkCenter = np.mean(landmarks, axis=0)
    faceRange = landmarks-landmarkCenter
    faceInner = landmarkCenter+0.3*faceRange
    faceOuter = landmarkCenter+1.3*faceRange

    faceInner = np.round(faceInner).astype(np.int32)
    faceOuter = np.round(faceOuter).astype(np.int32)

    # imgCopy = np.ones(img.shape, np.uint8)*255
    # lmk = np.round(faceInner).astype(np.int32)
    # for i in range(lmk.shape[0]):
    #     cv2.circle(imgCopy, lmk[i], 1, (255), 1, cv2.LINE_AA)
    # lmk = np.round(faceOuter).astype(np.int32)
    # for i in range(lmk.shape[0]):
    #     cv2.circle(imgCopy, lmk[i], 3, (155), 1, cv2.LINE_AA)
    # cv2.imwrite('c.jpg', imgCopy)

    faceLabel=0
    sam2model.add_point(
        (landmarkCenter[0], landmarkCenter[1]), True, faceLabel)
    masks = sam2model.get_masks()
    for i in range(faceInner.shape[0]):
        if masks[faceLabel][faceInner[i][1], faceInner[i][0]] == 0:
            print('extra add True..')
            sam2model.add_point(
                (faceInner[i][0], faceInner[i][1]), True, faceLabel)
            masks = sam2model.get_masks()

    # for i in range(faceOuter.shape[0]):
    #     if masks[faceLabel][faceOuter[i][1], faceOuter[i][0]] == 1:
    #         print('extra add False..')
    #         sam2model.add_point(
    #             (faceOuter[i][0], faceOuter[i][1]), False, faceLabel)
    #         masks = sam2model.get_masks()

    masked_img = draw_masks(img, masks)
    pos1 = imgPath.rfind('\\')
    pos2 = imgPath.rfind('/')
    if pos1>pos2:
        pos2=pos1
    else:
        pos1=pos2
    parent = imgPath[:pos1+1]
    filename = imgPath[pos1+1:]
    cv2.imwrite(parent+'blender_'+filename, masked_img) 

    return masks[faceLabel]


def segFaceBaseAnchor(sam2model, imgPath, anchor):
    img = cv2.imread(imgPath)
    sam2model.set_image(img)  

    faceLabel = 0
    sam2model.add_point(
        (anchor[0], anchor[1]), True, faceLabel)
    masks = sam2model.get_masks()
    masked_img = draw_masks(img, masks)
    pos1 = imgPath.rfind('\\')
    pos2 = imgPath.rfind('/')
    if pos1 > pos2:
        pos2 = pos1
    else:
        pos1 = pos2
    parent = imgPath[:pos1+1]
    filename = imgPath[pos1+1:]
    cv2.imwrite(parent+'blender_'+filename, masked_img)

    return masks[faceLabel]


def onnx_datatype_to_npType(data_type):
    if data_type == 1:
        return np.float32
    elif data_type == 7:
        return np.int64
    else:
        print('data_type=',data_type)
        raise TypeError("don't support data type")


def parser_initializer(initializer):
    name = initializer.name
    logging.info(f"initializer name: {name}")

    dims = initializer.dims
    shape = [x for x in dims]
    logging.info(f"initializer with shape:{shape}")

    dtype = initializer.data_type
    if dtype == 7:
        print()
    logging.info(f"initializer with type: {onnx_datatype_to_npType(dtype)} ")

    # print tenth buffer
    weights = np.frombuffer(initializer.raw_data,
                            dtype=onnx_datatype_to_npType(dtype))
    logging.info(f"initializer first 10 wights:{weights[:10]}")


def parser_tensor(tensor, use='normal'):
    name = tensor.name
    logging.info(f"{use} tensor name: {name}")

    data_type = tensor.type.tensor_type.elem_type
    logging.info(f"{use} tensor data type: {data_type}")

    dims = tensor.type.tensor_type.shape.dim
    shape = []
    for i, dim in enumerate(dims):
        shape.append(dim.dim_value)
    logging.info(f"{use} tensor with shape:{shape} ")


def parser_node(node):
    def attri_value(attri):
        if attri.type == 1:
            return attri.i
        elif attri.type == 7:
            return list(attri.ints)

    name = node.name
    logging.info(f"node name:{name}")

    opType = node.op_type
    logging.info(f"node op type:{opType}")

    inputs = list(node.input)
    logging.info(f"node with {len(inputs)} inputs:{inputs}")

    outputs = list(node.output)
    logging.info(f"node with {len(outputs)} outputs:{outputs}")

    attributes = node.attribute
    for attri in attributes:
        name = attri.name
        value = attri_value(attri)
        logging.info(f"{name} with value:{value}")


def parser_info(onnx_model):
    ir_version = onnx_model.ir_version
    producer_name = onnx_model.producer_name
    producer_version = onnx_model.producer_version
    for info in [ir_version, producer_name, producer_version]:
        logging.info("onnx model with info:{}".format(info))


def parser_inputs(onnx_graph):
    inputs = onnx_graph.input
    for input in inputs:
        parser_tensor(input, 'input')


def parser_outputs(onnx_graph):
    outputs = onnx_graph.output
    for output in outputs:
        parser_tensor(output, 'output')


def parser_graph_initializers(onnx_graph):
    initializers = onnx_graph.initializer
    print(len(initializers))
    for initializer in initializers:
        if initializer.name.endswith('Constant_28_output_0'):
            oldname = initializer.name
            copy = initializer
            copy.name='Constant_28_output_0_copy'
            initializers.append(copy)
            initializers[-1].name=oldname
            print(initializers[-1].name)
            break
    initializers = onnx_graph.initializer
    print(len(initializers))
    print()
    for initializer in initializers:
        parser_initializer(initializer)


def parser_graph_nodes(onnx_graph):
    nodes = onnx_graph.node
    for node in nodes:
        parser_node(node)
        t = 1
def onnx_parser():
    # model_path ='modified_1.onnx'
    model_path = "models/sam2_hiera_large_encoder.onnx"
    model = onnx.load(model_path)

    # 0. parser_info(model)

    graph = model.graph
 



    # 1. parser_inputs(graph)

    # 2. parser_outputs(graph)

    # 3.
    parser_graph_initializers(graph)
    print(len(graph.initializer))
    onnx.save_model(model,'1.onnx')
    # 4.
    parser_graph_nodes(graph)
if __name__ == '__main__':

    encoder_model_path = "models/sam2_hiera_large_encoder.onnx"
    decoder_model_path = "models/decoder.onnx"
    # netron.start(encoder_model_path)    



    img = cv2.imread(
        'D:/repo/colmapThird/a.bmp')
    # img = img.astype(np.float32)
    # img -= 0.5
    # img *= 2.
    # img = img.transpose(2, 0, 1)

    # Initialize models
    sam2 = SAM2Image(encoder_model_path, decoder_model_path)

    # Set image
    sam2.set_image(img)






    # point_coords = [np.array([[673, 284]]), np.array([[819, 586], [1064,635]]), np.array([[870, 496]]),
    #                 np.array([[1611,243]])]
    # point_labels = [np.array([1]), np.array([1, 1]), np.array([1]), np.array([1])]
    point_coords = [np.array([[1197, 429], [1355, 441]])]
    point_labels = [np.array([1,0])]

    for label_id, (point_coord, point_label) in enumerate(zip(point_coords, point_labels)):
        for i in range(point_label.shape[0]):
            sam2.add_point((point_coord[i][0], point_coord[i][1]), point_label[i], label_id)

        masks = sam2.get_masks()

        # Draw masks
        masked_img = draw_masks(img, masks)

        cv2.imshow("masked_img", masked_img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)

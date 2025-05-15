import os
from sam2 import SAM2Image, draw_masks
import cv2
import numpy as np
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

    # masked_img = draw_masks(img, masks)
    # cv2.imwrite('c.jpg', masked_img) 

    return masks[faceLabel]
if __name__ == '__main__':
    encoder_model_path = "models/sam2_hiera_large_encoder.onnx"
    decoder_model_path = "models/decoder.onnx"

    img = cv2.imread(
        'D:/ucl360/UCL360Calib/CameraCalibGui/pro37/out/1_-90.bmp')
    # img = img.astype(np.float32)
    # img -= 0.5
    # img *= 2.
    # img = img.transpose(2, 0, 1)

    # Initialize models
    sam2 = SAM2Image(encoder_model_path, decoder_model_path)

    # Set image
    sam2.set_image(img)






    point_coords = [np.array([[420, 440]]), np.array([[360, 275], [370, 210]]), np.array([[810, 440]]),
                    np.array([[920, 314]])]
    point_labels = [np.array([1]), np.array([1, 1]), np.array([1]), np.array([1])]

    for label_id, (point_coord, point_label) in enumerate(zip(point_coords, point_labels)):
        for i in range(point_label.shape[0]):
            sam2.add_point(
                (point_coord[i][0], point_coord[i][1]), point_label[i], label_id)

        masks = sam2.get_masks()

        # Draw masks
        masked_img = draw_masks(img, masks)

        cv2.imshow("masked_img", masked_img)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)

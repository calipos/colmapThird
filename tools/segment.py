from sam2 import SAM2Image, draw_masks
import cv2
import numpy as np

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

    # Add points
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

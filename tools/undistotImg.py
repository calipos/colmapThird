import numpy as np
import cv2


def undist(srcPath, newPath, camera):
    if camera.cameraType == 'SIMPLE_RADIAL':
        camera_matrix = camera.intr
        dist_coeffs = np.array(
            [camera.disto[0], 0, 0, 0, 0], dtype=np.float32)
        oriImg = cv2.imread(srcPath)
        undistortImg = cv2.undistort(oriImg, camera_matrix, dist_coeffs)
        cv2.imwrite(newPath, undistortImg)
    else:
        assert(False),'not support'
if __name__ == '__main__':
    camera_matrix = np.array([[1712.6196232455209, 0, 360],
                              [0, 1712.6196232455209, 640],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array(
        [-0.95465118780250591, 0, 0, 0, 0], dtype=np.float32)
    oriImg = cv2.imread('D:/repo/colmapThird/data/b/00005.jpg')
    h = oriImg.shape[0]
    w = oriImg.shape[1]
    u = (w-camera_matrix[0, 2])/camera_matrix[0, 0]
    v = (h-camera_matrix[1, 2])/camera_matrix[1, 1]
    for iter in range(5):
        u2 = u * u
        v2 = v * v
        r2 = u2 + v2
        radial = dist_coeffs[0] * r2
        du = u * radial
        dv = v * radial
        u=u+du
        v=v+dv
    print(u*camera_matrix[0, 0]+camera_matrix[0, 2],v*camera_matrix[1, 1]+camera_matrix[1, 2])
    undistortImg = cv2.undistort(
        oriImg, camera_matrix, dist_coeffs, camera_matrix)
    cv2.imwrite('D:/repo/colmapThird/data/b/00005b.jpg', undistortImg)

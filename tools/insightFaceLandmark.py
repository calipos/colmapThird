import cv2
import numpy as np
import os
import numpy as np
import cv2
import onnx
import onnxruntime
import pickle
from skimage import transform as trans
import landmarkShapeType
import writeLabelme
import os.path as osp

import numpy as np
import onnxruntime
from numpy.linalg import norm

def cropImg(img,factor):
    maxSize = factor*25
    height = img.shape[0]
    width = img.shape[1]
    minSize = np.min([height, width])
    scale=1
    if minSize > maxSize:
        scale = maxSize/minSize
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    height = img.shape[0]
    width = img.shape[1]
    fixedHeight = height//factor*factor
    fixedWidth = width//factor*factor
    if fixedHeight == height and fixedWidth == width:
        return img,1
    else:
        return img[:fixedHeight, :fixedWidth, :], scale

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data,
                                M, (output_size, output_size),
                                borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    # print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)




class FacialDetect:
    def __init__(self, model_file=None, det_thresh=0.5, nms_thresh=0.4,session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session 
        self.model = onnx.load(self.model_file)
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        self.det_thresh = det_thresh 
        self.nms_thresh = nms_thresh
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        print(len(self.output_names))
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs)==9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs)==10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs)==15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True


        self.input_mean = 127.5
        self.input_std = 128.0

    def forward(self, img, threshold=0.5):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0/self.input_std, input_size,
                                     (self.input_mean, self.input_mean, self.input_mean), swapRB=True)          
        pred = self.session.run(self.output_names, {self.input_name: blob})
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc


        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = pred[idx]
            bbox_preds = pred[idx+fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = pred[idx+fmc*2] * stride
            height = input_height // stride
            width = input_width // stride
            if height==0 or width==0:
                break
            K = height * width
            key = (height, width, stride)

            anchor_centers = np.stack(
                np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            # print(anchor_centers.shape)

            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self._num_anchors > 1:
                anchor_centers = np.stack(
                    [anchor_centers]*self._num_anchors, axis=1).reshape((-1, 2))

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                # kpss = kps_preds
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
    def detect(self, img, max_num=0, metric='max'):
        scores_list, bboxes_list, kpss_list = self.forward(
            img, self.det_thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list)
        if self.use_kps:
            kpss = np.vstack(kpss_list) 
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                              det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

class Landmark:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        #92==88  34==38
        self.allMarkIdx = [x for x in range(106)]
        self.eyeAndNoseIdx = [33]+[x for x in range(35,38)]+[x for x in range(39,52)]+[x for x in range(72,88)]+[x for x in range(89,92)]+[x for x in range(93,106)]
        self.faceContourIdx = [33]+[x for x in range(35,38)]+[x for x in range(39,52)]+[x for x in range(72,88)]+[x for x in range(89,92)]+[x for x in range(93,106)]
        self.EyeMouthBorder = [35, 93, 52, 61,71,53,72,43,103]
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            # print(nid, node.name)
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
            if nid < 3 and node.name == 'bn_data':
                find_sub = True
                find_mul = True
        if find_sub and find_mul:
            # mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 128.0
        self.input_mean = input_mean
        self.input_std = input_std
        # print('input mean and std:', model_file, self.input_mean, self.input_std)
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names) == 1
        output_shape = outputs[0].shape
        self.require_pose = False
        # print('init output_shape:', output_shape)
        if output_shape[1] == 3309:
            self.lmk_dim = 3
            self.lmk_num = 68
            with open('C:/Users/Administrator/.insightface/models/buffalo_l/meanshape_68.pkl', 'rb') as f:
                self.mean_lmk = pickle.load(f)
            self.require_pose = True
        else:
            self.lmk_dim = 2
            self.lmk_num = output_shape[1]//self.lmk_dim
        self.taskname = 'landmark_%dd_%d' % (self.lmk_dim, self.lmk_num)
        print(self.taskname)

    def get(self, img, face):
        bbox = face['bbox']
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0] / (max(w, h)*1.5)
        # print('param:', img.shape, bbox, center, self.input_size, _scale, rotate)
        aimg, M = transform(
            img, center, self.input_size[0], _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])
        # assert input_size==self.input_size
        blob = cv2.dnn.blobFromImage(aimg, 1.0/self.input_std, input_size,
                                     (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        pred = self.session.run(
            self.output_names, {self.input_name: blob})[0][0]
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num*-1:, :]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (self.input_size[0] // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (self.input_size[0] // 2)

        IM = cv2.invertAffineTransform(M)
        pred = trans_points(pred, IM)
        face[self.taskname] = pred
        if self.require_pose:
            P = transform.estimate_affine_matrix_3d23d(self.mean_lmk, pred)
            s, R, t = transform.P2sRt(P)
            rx, ry, rz = transform.matrix2angle(R)
            pose = np.array([rx, ry, rz], dtype=np.float32)
            face['pose'] = pose  # pitch, yaw, roll
        return pred


class InsightFaceFinder:
    def __init__(self, faceParamPath, landmarkParamPath):
        onnxruntime.set_default_logger_severity(3)
        if not os.path.exists(faceParamPath):
            print("not found ", faceParamPath)
            return None
        if not os.path.exists(landmarkParamPath):
            print("not found ", landmarkParamPath)
            return None
        self.faceParamPath = faceParamPath
        self.landmarkParamPath = landmarkParamPath
        self.detector = FacialDetect(faceParamPath)
        self.landmarkFinder = Landmark(landmarkParamPath)

    def proc(self, imgPath, landmarkShapeType_=landmarkShapeType.LandmarkShapeType.EyeAndNoise,writeJson=True):
        imgOri = cv2.imread(imgPath)
        img, scale = cropImg(imgOri, self.detector._feat_stride_fpn[-1])
        det, kpss = self.detector.detect(img, 1)
        borderFactor=-1
        if len(det) == 0:
            borderSizeHeight = int(imgOri.shape[0]*0.8)
            borderSizeWidth = int(imgOri.shape[1]*0.8)
            borderImgOri = cv2.copyMakeBorder(imgOri,0,borderSizeHeight,0,borderSizeWidth,cv2.BORDER_CONSTANT,value=(255,255,255))            
            borderFactor=borderImgOri.shape[1]/imgOri.shape[1]
            imgOri = cv2.resize(borderImgOri, (imgOri.shape[1], imgOri.shape[0]))        
        img, scale = cropImg(imgOri, self.detector._feat_stride_fpn[-1])
        det, kpss = self.detector.detect(img, 1)
        if len(det) == 0:
            return None
        landmark = self.landmarkFinder.get(
            img, {'bbox': det[0], 'kps': kpss[0]})
        frontLandmarks2d = landmark/scale
        if borderFactor>1:
            frontLandmarks2d=frontLandmarks2d*borderFactor
        if landmarkShapeType_ == landmarkShapeType.LandmarkShapeType.EyeAndNoise:
            frontLandmarks2d = frontLandmarks2d[self.landmarkFinder.eyeAndNoseIdx]
        elif landmarkShapeType_ == landmarkShapeType.LandmarkShapeType.EyeMouthBorder:
            frontLandmarks2d = frontLandmarks2d[self.landmarkFinder.EyeMouthBorder]
        elif landmarkShapeType_ == landmarkShapeType.LandmarkShapeType.Contour:
            hull = cv2.convexHull(frontLandmarks2d)
            frontLandmarks2d = hull.squeeze()
        else:
            frontLandmarks2d = frontLandmarks2d
        imgDir, imgPath = os.path.split(imgPath)
        base = os.path.splitext(imgPath)[0]
        if writeJson==True:
            jsonPath = f"{base}.{'json'}"
            if landmarkShapeType_ == landmarkShapeType.LandmarkShapeType.Contour:
                writeLabelme.writeLabelmeJson(imgDir, imgPath, jsonPath,
                                              frontLandmarks2d, 'insightface', writeLabelme.LabelmeShapeType.HULL)
            else:
                writeLabelme.writeLabelmeJson(imgDir, imgPath, jsonPath,
                                            frontLandmarks2d, 'insightface')
        return frontLandmarks2d
    def figureIdrMask(self, imgPath,idrDir):
        outImgDir = os.path.join(idrDir, 'image')
        outMaskDir = os.path.join(idrDir, 'mask')
        if not os.path.exists(outImgDir):
            os.mkdir(outImgDir)
        if not os.path.exists(outMaskDir):
            os.mkdir(outMaskDir)
        imgOri = cv2.imread(imgPath)
        img, scale = cropImg(imgOri, self.detector._feat_stride_fpn[-1])
        det, kpss = self.detector.detect(img, 1)
        if len(det) == 0:
            return None
        landmark = self.landmarkFinder.get(
            img, {'bbox': det[0], 'kps': kpss[0]})
        frontLandmarks2d = landmark/scale
        imgDir, imgPath = os.path.split(imgPath)
        shortName,ext = os.path.splitext(imgPath)
        cv2.imwrite(os.path.join(outImgDir, shortName+'.png'), imgOri)
        mask = np.zeros(imgOri.shape)
        hull = cv2.convexHull(frontLandmarks2d)
        hull = np.round(hull).astype(np.int32)
        cv2.fillPoly(mask, [hull],   (255,255,255) )
        cv2.imwrite(os.path.join(outMaskDir, shortName+'.png'), mask)

if __name__ == '__main__':
    print(os.getcwd())

    faceParamPath = 'models/buffalo_l/det_10g.onnx'
    landmarkParamPath = 'models/buffalo_l/2d106det.onnx'
    landmarkFinder = InsightFaceFinder(faceParamPath, landmarkParamPath)

    imgOri = cv2.imread('data/c.jpg')
    imgCopy = cv2.imread('data/c.jpg')
    imgCopy = np.ones(imgOri.shape, np.uint8)*255
    img, scale = cropImg(imgOri, landmarkFinder.detector._feat_stride_fpn[-1])
    det, kpss = landmarkFinder.detector.detect(img, 1)
    borderFactor = -1
    if len(det) == 0:
        borderSizeHeight = int(imgOri.shape[0]*0.8)
        borderSizeWidth = int(imgOri.shape[1]*0.8)
        borderImgOri = cv2.copyMakeBorder(
            imgOri, 0, borderSizeHeight, 0, borderSizeWidth, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        borderFactor = borderImgOri.shape[1]/imgOri.shape[1]
        imgOri = cv2.resize(
            borderImgOri, (imgOri.shape[1], imgOri.shape[0]))
    img, scale = cropImg(imgOri, landmarkFinder.detector._feat_stride_fpn[-1])
    det, kpss = landmarkFinder.detector.detect(img, 1)
    if len(det) == 0:
        exit(0)  
    landmark = landmarkFinder.landmarkFinder.get(
        img, {'bbox': det[0], 'kps': kpss[0]})
    frontLandmarks2d = landmark/scale
    if borderFactor > 1:
        frontLandmarks2d = frontLandmarks2d*borderFactor
    
   
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.4
    textColor = (0, 0, 0)  # 蓝色文本
    thickness = 1
 
    color = (200, 160, 75)
    lmk = np.round(frontLandmarks2d).astype(np.int32)
    for i in landmarkFinder.landmarkFinder.allMarkIdx:
        cv2.circle(imgCopy, lmk[i], 1, color, 1, cv2.LINE_AA)
        text = str(i)
        cv2.putText(imgCopy, text, lmk[i], fontFace,
                    fontScale, textColor, thickness)
    cv2.imwrite('c.jpg', imgCopy)

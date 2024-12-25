import cv2
import numpy as np
import os
import insightface 
import numpy as np
import cv2
import onnx
import onnxruntime
import pickle
from insightface.data import get_image as ins_get_image
# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      :


# from __future__ import division

import glob
import os.path as osp

import numpy as np
import onnxruntime
from numpy.linalg import norm
 
class Landmark:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session
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

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])

    def get(self, img, face):
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0] / (max(w, h)*1.5)
        # print('param:', img.shape, bbox, center, self.input_size, _scale, rotate)
        aimg, M = face_align.transform(
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
        pred = face_align.trans_points(pred, IM)
        face[self.taskname] = pred
        if self.require_pose:
            P = transform.estimate_affine_matrix_3d23d(self.mean_lmk, pred)
            s, R, t = transform.P2sRt(P)
            rx, ry, rz = transform.matrix2angle(R)
            pose = np.array([rx, ry, rz], dtype=np.float32)
            face['pose'] = pose  # pitch, yaw, roll
        return pred

  
class FaceAnalysis:
    def __init__(self):
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        self.model_dir = 'C:/Users/Administrator/.insightface/models/buffalo_l'
        # onnx_file = 'C:/Users/Administrator/.insightface/models/buffalo_l/1k3d68.onnx'
        onnx_file = 'C:/Users/Administrator/.insightface/models/buffalo_l/2d106det.onnx'
        # onnx_file = 'C:/Users/Administrator/.insightface/models/buffalo_l/det_10g.onnx'
        # onnx_file = 'C:/Users/Administrator/.insightface/models/buffalo_l/genderage.onnx'
        # onnx_file = 'C:/Users/Administrator/.insightface/models/buffalo_l/w600k_r50.onnx'
        model = Landmark(onnx_file, None)
 

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname == 'detection':
                model.prepare(ctx_id, input_size=det_size,
                              det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname == 'detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return ret

    def draw_on(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(np.int)
                # print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg, '%s,%d' % (face.sex, face.age),
                            (box[0]-1, box[1]-4), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)

            # for key, value in face.items():
            #    if key.startswith('landmark_3d'):
            #        print(key, value.shape)
            #        print(value[0:10,:])
            #        lmk = np.round(value).astype(np.int)
            #        for l in range(lmk.shape[0]):
            #            color = (255, 0, 0)
            #            cv2.circle(dimg, (lmk[l][0], lmk[l][1]), 1, color,
            #                       2)
        return dimg


if __name__ == '__main__':
    print(os.getcwd())
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(544, 960))
    img = cv2.imread('data/00094.jpg')
    faces = app.get(img)
    # assert len(faces)==6
    tim = img.copy()
    color = (200, 160, 75)
    for face in faces:
        lmk = face.landmark_2d_106
        lmk = np.round(lmk).astype(np.int32)
        for i in range(lmk.shape[0]):
            p = tuple(lmk[i])
            cv2.circle(tim, p, 1, color, 1, cv2.LINE_AA)
    cv2.imwrite('00094.jpg', tim)

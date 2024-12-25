import cv2
import numpy as np
import os
import numpy as np
import cv2
import onnx
import onnxruntime
import datetime
import pickle
# from insightface.data import get_image as ins_get_image
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



class FacialDetect:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session 
        self.model = onnx.load(self.model_file)
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
        print(len(self.output_names))
        self.pixel_means = [103.52, 116.28, 123.675]
        self.pixel_stds = [57.375, 57.12, 58.395]
    def detect(self, img, threshold=0.5, scales=[1.0], do_flip=False):
        #print('in_detect', threshold, scales, do_flip, do_nms)
        proposals_list = []
        scores_list = []
        landmarks_list = []
        strides_list = []
        timea = datetime.datetime.now()
        flips = [0]
        if do_flip:
            flips = [0, 1]

        imgs = [img]
        if isinstance(img, list):
            imgs = img
        for img in imgs:
            for im_scale in scales:
                for flip in flips:
                    if im_scale != 1.0:
                        im = cv2.resize(img,
                                        None,
                                        None,
                                        fx=im_scale,
                                        fy=im_scale,
                                        interpolation=cv2.INTER_LINEAR)
                    else:
                        im = img.copy()
                    if flip:
                        im = im[:, ::-1, :]
                    im = im.astype(np.float32) 
                    for i in range(3):
                        im[:, :, 2 - i] = (
                            im[:, :, 2 - i]   -
                            self.pixel_means[2 - i]) / self.pixel_stds[2 - i]
                    blob = cv2.dnn.blobFromImage(im,1.,  [im.shape[1],im.shape[0]], swapRB=True)
                    pred = self.session.run(self.output_names, {self.input_name: blob})

                    #post_nms_topN = self._rpn_post_nms_top_n
                    #min_size_dict = self._rpn_min_size_fpn

                    sym_idx = 0

                    for _idx, s in enumerate(self._feat_stride_fpn):
                        #if len(scales)>1 and s==32 and im_scale==scales[-1]:
                        #  continue
                        _key = 'stride%s' % s
                        stride = int(s)
                        is_cascade = False
                        if self.cascade:
                            is_cascade = True
                        #if self.vote and stride==4 and len(scales)>2 and (im_scale==scales[0]):
                        #  continue
                        #print('getting', im_scale, stride, idx, len(net_out), data.shape, file=sys.stderr)
                        scores = net_out[sym_idx].asnumpy()
                        if self.debug:
                            timeb = datetime.datetime.now()
                            diff = timeb - timea
                            print('A uses', diff.total_seconds(), 'seconds')
                        #print(scores.shape)
                        #print('scores',stride, scores.shape, file=sys.stderr)
                        scores = scores[:, self._num_anchors['stride%s' %
                                                             s]:, :, :]

                        bbox_deltas = net_out[sym_idx + 1].asnumpy()

                        #if DEBUG:
                        #    print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
                        #    print 'scale: {}'.format(im_info[2])

                        #_height, _width = int(im_info[0] / stride), int(im_info[1] / stride)
                        height, width = bbox_deltas.shape[
                            2], bbox_deltas.shape[3]

                        A = self._num_anchors['stride%s' % s]
                        K = height * width
                        anchors_fpn = self._anchors_fpn['stride%s' % s]
                        anchors = anchors_plane(height, width, stride,
                                                anchors_fpn)
                        #print((height, width), (_height, _width), anchors.shape, bbox_deltas.shape, scores.shape, file=sys.stderr)
                        anchors = anchors.reshape((K * A, 4))
                        #print('num_anchors', self._num_anchors['stride%s'%s], file=sys.stderr)
                        #print('HW', (height, width), file=sys.stderr)
                        #print('anchors_fpn', anchors_fpn.shape, file=sys.stderr)
                        #print('anchors', anchors.shape, file=sys.stderr)
                        #print('bbox_deltas', bbox_deltas.shape, file=sys.stderr)
                        #print('scores', scores.shape, file=sys.stderr)

                        #scores = self._clip_pad(scores, (height, width))
                        scores = scores.transpose((0, 2, 3, 1)).reshape(
                            (-1, 1))

                        #print('pre', bbox_deltas.shape, height, width)
                        #bbox_deltas = self._clip_pad(bbox_deltas, (height, width))
                        #print('after', bbox_deltas.shape, height, width)
                        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
                        bbox_pred_len = bbox_deltas.shape[3] // A
                        #print(bbox_deltas.shape)
                        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
                        bbox_deltas[:,
                                    0::4] = bbox_deltas[:, 0::
                                                        4] * self.bbox_stds[0]
                        bbox_deltas[:,
                                    1::4] = bbox_deltas[:, 1::
                                                        4] * self.bbox_stds[1]
                        bbox_deltas[:,
                                    2::4] = bbox_deltas[:, 2::
                                                        4] * self.bbox_stds[2]
                        bbox_deltas[:,
                                    3::4] = bbox_deltas[:, 3::
                                                        4] * self.bbox_stds[3]
                        proposals = self.bbox_pred(anchors, bbox_deltas)

                        #print(anchors.shape, bbox_deltas.shape, A, K, file=sys.stderr)
                        if is_cascade:
                            cascade_sym_num = 0
                            cls_cascade = False
                            bbox_cascade = False
                            __idx = [3, 4]
                            if not self.use_landmarks:
                                __idx = [2, 3]
                            for diff_idx in __idx:
                                if sym_idx + diff_idx >= len(net_out):
                                    break
                                body = net_out[sym_idx + diff_idx].asnumpy()
                                if body.shape[1] // A == 2:  #cls branch
                                    if cls_cascade or bbox_cascade:
                                        break
                                    else:
                                        cascade_scores = body[:, self.
                                                              _num_anchors[
                                                                  'stride%s' %
                                                                  s]:, :, :]
                                        cascade_scores = cascade_scores.transpose(
                                            (0, 2, 3, 1)).reshape((-1, 1))
                                        #scores = (scores+cascade_scores)/2.0
                                        scores = cascade_scores  #TODO?
                                        cascade_sym_num += 1
                                        cls_cascade = True
                                        #print('find cascade cls at stride', stride)
                                elif body.shape[1] // A == 4:  #bbox branch
                                    cascade_deltas = body.transpose(
                                        (0, 2, 3, 1)).reshape(
                                            (-1, bbox_pred_len))
                                    cascade_deltas[:, 0::
                                                   4] = cascade_deltas[:, 0::
                                                                       4] * self.bbox_stds[
                                                                           0]
                                    cascade_deltas[:, 1::
                                                   4] = cascade_deltas[:, 1::
                                                                       4] * self.bbox_stds[
                                                                           1]
                                    cascade_deltas[:, 2::
                                                   4] = cascade_deltas[:, 2::
                                                                       4] * self.bbox_stds[
                                                                           2]
                                    cascade_deltas[:, 3::
                                                   4] = cascade_deltas[:, 3::
                                                                       4] * self.bbox_stds[
                                                                           3]
                                    proposals = self.bbox_pred(
                                        proposals, cascade_deltas)
                                    cascade_sym_num += 1
                                    bbox_cascade = True
                                    #print('find cascade bbox at stride', stride)

                        proposals = clip_boxes(proposals, im_info[:2])

                        #if self.vote:
                        #  if im_scale>1.0:
                        #    keep = self._filter_boxes2(proposals, 160*im_scale, -1)
                        #  else:
                        #    keep = self._filter_boxes2(proposals, -1, 100*im_scale)
                        #  if stride==4:
                        #    keep = self._filter_boxes2(proposals, 12*im_scale, -1)
                        #    proposals = proposals[keep, :]
                        #    scores = scores[keep]

                        #keep = self._filter_boxes(proposals, min_size_dict['stride%s'%s] * im_info[2])
                        #proposals = proposals[keep, :]
                        #scores = scores[keep]
                        #print('333', proposals.shape)
                        if stride == 4 and self.decay4 < 1.0:
                            scores *= self.decay4

                        scores_ravel = scores.ravel()
                        #print('__shapes', proposals.shape, scores_ravel.shape)
                        #print('max score', np.max(scores_ravel))
                        order = np.where(scores_ravel >= threshold)[0]
                        #_scores = scores_ravel[order]
                        #_order = _scores.argsort()[::-1]
                        #order = order[_order]
                        proposals = proposals[order, :]
                        scores = scores[order]
                        if flip:
                            oldx1 = proposals[:, 0].copy()
                            oldx2 = proposals[:, 2].copy()
                            proposals[:, 0] = im.shape[1] - oldx2 - 1
                            proposals[:, 2] = im.shape[1] - oldx1 - 1

                        proposals[:, 0:4] /= im_scale

                        proposals_list.append(proposals)
                        scores_list.append(scores)
                        if self.nms_threshold < 0.0:
                            _strides = np.empty(shape=(scores.shape),
                                                dtype=np.float32)
                            _strides.fill(stride)
                            strides_list.append(_strides)

                        if not self.vote and self.use_landmarks:
                            landmark_deltas = net_out[sym_idx + 2].asnumpy()
                            #landmark_deltas = self._clip_pad(landmark_deltas, (height, width))
                            landmark_pred_len = landmark_deltas.shape[1] // A
                            landmark_deltas = landmark_deltas.transpose(
                                (0, 2, 3, 1)).reshape(
                                    (-1, 5, landmark_pred_len // 5))
                            landmark_deltas *= self.landmark_std
                            #print(landmark_deltas.shape, landmark_deltas)
                            landmarks = self.landmark_pred(
                                anchors, landmark_deltas)
                            landmarks = landmarks[order, :]

                            if flip:
                                landmarks[:, :,
                                          0] = im.shape[1] - landmarks[:, :,
                                                                       0] - 1
                                #for a in range(5):
                                #  oldx1 = landmarks[:, a].copy()
                                #  landmarks[:,a] = im.shape[1] - oldx1 - 1
                                order = [1, 0, 2, 4, 3]
                                flandmarks = landmarks.copy()
                                for idx, a in enumerate(order):
                                    flandmarks[:, idx, :] = landmarks[:, a, :]
                                    #flandmarks[:, idx*2] = landmarks[:,a*2]
                                    #flandmarks[:, idx*2+1] = landmarks[:,a*2+1]
                                landmarks = flandmarks
                            landmarks[:, :, 0:2] /= im_scale
                            #landmarks /= im_scale
                            #landmarks = landmarks.reshape( (-1, landmark_pred_len) )
                            landmarks_list.append(landmarks)
                            #proposals = np.hstack((proposals, landmarks))
                        if self.use_landmarks:
                            sym_idx += 3
                        else:
                            sym_idx += 2
                        if is_cascade:
                            sym_idx += cascade_sym_num

        if self.debug:
            timeb = datetime.datetime.now()
            diff = timeb - timea
            print('B uses', diff.total_seconds(), 'seconds')
        proposals = np.vstack(proposals_list)
        landmarks = None
        if proposals.shape[0] == 0:
            if self.use_landmarks:
                landmarks = np.zeros((0, 5, 2))
            if self.nms_threshold < 0.0:
                return np.zeros((0, 6)), landmarks
            else:
                return np.zeros((0, 5)), landmarks
        scores = np.vstack(scores_list)
        #print('shapes', proposals.shape, scores.shape)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        #if config.TEST.SCORE_THRESH>0.0:
        #  _count = np.sum(scores_ravel>config.TEST.SCORE_THRESH)
        #  order = order[:_count]
        proposals = proposals[order, :]
        scores = scores[order]
        if self.nms_threshold < 0.0:
            strides = np.vstack(strides_list)
            strides = strides[order]
        if not self.vote and self.use_landmarks:
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)

        if self.nms_threshold > 0.0:
            pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32,
                                                                    copy=False)
            if not self.vote:
                keep = self.nms(pre_det)
                det = np.hstack((pre_det, proposals[:, 4:]))
                det = det[keep, :]
                if self.use_landmarks:
                    landmarks = landmarks[keep]
            else:
                det = np.hstack((pre_det, proposals[:, 4:]))
                det = self.bbox_vote(det)
        elif self.nms_threshold < 0.0:
            det = np.hstack(
                (proposals[:, 0:4], scores, strides)).astype(np.float32,
                                                             copy=False)
        else:
            det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32,
                                                                copy=False)

        if self.debug:
            timeb = datetime.datetime.now()
            diff = timeb - timea
            print('C uses', diff.total_seconds(), 'seconds')
        return det, landmarks

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
        # onnx_file = 'C:/Users/Administrator/.insightface/models/buffalo_l/1k3d68.onnx'
        self.landmarkParam = 'models/buffalo_l/2d106det.onnx'
        self.detectParam = 'models/buffalo_l/det_10g.onnx'
        # onnx_file = 'C:/Users/Administrator/.insightface/models/buffalo_l/det_10g.onnx'
        # onnx_file = 'C:/Users/Administrator/.insightface/models/buffalo_l/genderage.onnx'
        # onnx_file = 'C:/Users/Administrator/.insightface/models/buffalo_l/w600k_r50.onnx'
        detectModel = Landmark(self.detectParam)
        landmarkModel = Landmark(self.landmarkParam)
 

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
    img = cv2.imread('data/00050.jpg')
    detector = FacialDetect('models/buffalo_l/det_10g.onnx')
    detector.detect(img)

    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(544, 960))
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

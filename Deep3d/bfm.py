# conv1x1(last_dim, 80, bias=True), # id layer   组训练  共享形状参数
# conv1x1(last_dim, 64, bias=True), # exp layer   组训练  共享表情参数
# conv1x1(last_dim, 80, bias=True), # tex layer   组训练  共享纹理参数
# conv1x1(last_dim, 3, bias=True),  # angle layer 组训练 不敢去反传该参数R
# conv1x1(last_dim, 27, bias=True), # gamma layer 组训练 共享环节sh
# conv1x1(last_dim, 2, bias=True),  # tx, ty       组训练 不敢去反传该参数t
# conv1x1(last_dim, 1, bias=True)   # tz           组训练 不敢去反传该参数t

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import load_mats
import save
import h5py

class SH:
    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        self.c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) /
                  np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]


class ParametricFaceModel:
    def __init__(self,
                 bfm_folder='./BFM',
                 recenter=True,
                 init_lit=np.array([
                     0.8, 0, 0, 0, 0, 0, 0, 0, 0
                 ]),
                 cameraMatrix=np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
                 is_train=True,
                 default_name='BFM_model_front.mat'):

        if not os.path.isfile(os.path.join(bfm_folder, default_name)):
            load_mats.transferBFM09(bfm_folder)
        model = load_mats.loadmat(os.path.join(bfm_folder, default_name))
        # mean face shape. [3*N,1]
        self.mean_shape = model['meanshape'].astype(np.float32)
        # identity basis. [3*N,80]
        self.id_base = model['idBase'].astype(np.float32)
        # expression basis. [3*N,64]
        self.exp_base = model['exBase'].astype(np.float32)
        # mean face texture. [3*N,1] (0-255)
        self.mean_tex = model['meantex'].astype(np.float32)
        # texture basis. [3*N,80]
        self.tex_base = model['texBase'].astype(np.float32)
        # face indices for each vertex that lies in. starts from 0. [N,8]
        self.point_buf = model['point_buf'].astype(np.int64) - 1
        # vertex indices for each face. starts from 0. [F,3]
        self.face_buf = model['tri'].astype(np.int64) - 1
        self.face_tri = self.face_buf
        # vertex indices for 68 landmarks. starts from 0. [68,1]
        self.keypoints = np.load(os.path.join(
            bfm_folder, 'index_mp468_from_mesh35709.npy')).astype(np.int64)
        #unmatch_mask = self.keypoints < 0
        #self.keypoints= self.keypoints[self.keypoints >= 0]
        if is_train:
            # vertex indices for small face region to compute photometric error. starts from 0.
            self.front_mask = np.squeeze(
                model['frontmask2_idx']).astype(np.int64) - 1
            # vertex indices for each face from small face region. starts from 0. [f,3]
            self.front_face_buf = model['tri_mask2'].astype(np.int64) - 1
            # vertex indices for pre-defined skin region to compute reflectance loss
            self.skin_mask = np.squeeze(model['skinmask'])

        if recenter:
            mean_shape = self.mean_shape.reshape([-1, 3])
            mean_shape = mean_shape - \
                np.mean(mean_shape, axis=0, keepdims=True)
            self.mean_shape = mean_shape.reshape([-1, 1])

        self.persc_proj = cameraMatrix.transpose()  # 相机矩阵,并且进行了装置
        self.device = 'cpu'
        self.SH = SH()
        self.init_lit = init_lit.reshape(
            [1, 1, -1]).astype(np.float32)  # 光线的方向 吗,

    def transferMediapipe486(self, pts478):
        return 0

    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value).to(device))

    def getMediapipe486BfmBase(self,):
        mediapipe486Len = len(self.keypoints)  # 只有480  里没有6个点的对应
        id_486part = np.zeros([mediapipe486Len*3, 80])
        exp_486part = np.zeros([mediapipe486Len*3, 64])
        mean_486part = np.zeros([mediapipe486Len*3, 1])
        for i in range(mediapipe486Len):
            I = int(self.keypoints[i])
            if I<0:
                continue
            id_486part[3*i, ...] = self.id_base[3*I, ...]
            id_486part[3*i+1, ...] = self.id_base[3*I+1, ...]
            id_486part[3*i+2, ...] = self.id_base[3*I+2, ...]

            exp_486part[3*i, ...] = self.exp_base[3*I, ...]
            exp_486part[3*i+1, ...] = self.exp_base[3*I+1, ...]
            exp_486part[3*i+2, ...] = self.exp_base[3*I+2, ...]

            mean_486part[3*i, ...] = self.mean_shape[3*I, ...]
            mean_486part[3*i+1, ...] = self.mean_shape[3*I+1, ...]
            mean_486part[3*i+2, ...] = self.mean_shape[3*I+2, ...]
        return id_486part, exp_486part, mean_486part

    def compute_shape(self, id_coeff, exp_coeff):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3)

        Parameters:
            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs
        """
        batch_size = id_coeff.shape[0]
        id_part = torch.einsum('ij,aj->ai', self.id_base, id_coeff)
        exp_part = torch.einsum('ij,aj->ai', self.exp_base, exp_coeff)
        face_shape = id_part + exp_part + self.mean_shape.reshape([1, -1])
        return face_shape.reshape([batch_size, -1, 3])

    def compute_texture(self, tex_coeff, normalize=True):
        """
        Return:
            face_texture     -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.)

        Parameters:
            tex_coeff        -- torch.tensor, size (B, 80)
        """
        batch_size = tex_coeff.shape[0]
        face_texture = torch.einsum(
            'ij,aj->ai', self.tex_base, tex_coeff) + self.mean_tex
        if normalize:
            face_texture = face_texture / 255.
        return face_texture.reshape([batch_size, -1, 3])

    def compute_norm(self, face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """

        v1 = face_shape[:, self.face_buf[:, 0]]
        v2 = face_shape[:, self.face_buf[:, 1]]
        v3 = face_shape[:, self.face_buf[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat([face_norm, torch.zeros(
            face_norm.shape[0], 1, 3).to(self.device)], dim=1)

        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm

    def compute_color(self, face_texture, face_norm, gamma):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_texture     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_norm        -- torch.tensor, size (B, N, 3), rotated face normal
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        batch_size = gamma.shape[0]
        v_num = face_texture.shape[1]
        a, c = self.SH.a, self.SH.c
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + self.init_lit
        gamma = gamma.permute(0, 2, 1)
        Y = torch.cat([
            a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.device),
            -a[1] * c[1] * face_norm[..., 1:2],
            a[1] * c[1] * face_norm[..., 2:],
            -a[1] * c[1] * face_norm[..., :1],
            a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) *
            (3 * face_norm[..., 2:] ** 2 - 1),
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
            0.5 * a[2] * c[2] * (face_norm[..., :1] **
                                 2 - face_norm[..., 1:2] ** 2)
        ], dim=-1)
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        face_color = torch.cat([r, g, b], dim=-1) * face_texture
        return face_color

    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x),
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def to_camera(self, face_shape):
        # face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        """
        Return:
            face_proj        -- torch.tensor, size (B, N, 2), y direction is opposite to v direction

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        face_proj = face_shape @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]

        return face_proj

    def transform(self, face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + trans.unsqueeze(1)

    def get_landmarks(self, face_proj):
        """
        Return:
            face_lms         -- torch.tensor, size (B, 68, 2)

        Parameters:
            face_proj       -- torch.tensor, size (B, N, 2)
        """
        return face_proj[:, self.keypoints]

    def split_coeff(self, coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }

    def compute_for_render(self, coeffs):
        """
        Return:
            face_vertex     -- torch.tensor, size (B, N, 3), in camera coordinate
            face_color      -- torch.tensor, size (B, N, 3), in RGB order
            landmark        -- torch.tensor, size (B, 68, 2), y direction is opposite to v direction
        Parameters:
            coeffs          -- torch.tensor, size (B, 257)
        """
        coef_dict = self.split_coeff(coeffs)
        face_shape = self.compute_shape(coef_dict['id'], coef_dict['exp'])
        #save.saveFacePts(face_shape,'face.pts')
        rotation = self.compute_rotation(coef_dict['angle'])

        face_shape_transformed = self.transform(
            face_shape, rotation, coef_dict['trans'])
        face_vertex = self.to_camera(face_shape_transformed)

        face_proj = self.to_image(face_vertex)
        landmark = self.get_landmarks(face_proj)

        face_texture = self.compute_texture(coef_dict['tex'])
        #save.saveFacePts(face_shape,face_texture,'face.pts')
        face_norm = self.compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation
        face_color = self.compute_color(
            face_texture, face_norm_roted, coef_dict['gamma'])
        # save.saveColorFacePts(face_shape,face_texture,'face0.pts')
        # save.saveFacePts(landmark,'face.pts')
        return face_vertex, face_texture, face_color, landmark

class Bfm2019:
    def __init__(self,
                 bfm_folder='./BFM',
                 defaultFaceFile='model2019_bfm(47439p94464f).h5',
                 defaultHeadFile='model2019_fullHead(58203p116160f).h5',
                 defaultFaceLmIdxFile='index_mp468_from_model2019_47439p.npy'):
        faceH5Path = os.path.join(bfm_folder, defaultFaceFile)
        headH5Path = os.path.join(bfm_folder, defaultHeadFile)
        faceLmIdxFile = os.path.join(bfm_folder, defaultFaceLmIdxFile)
        if not os.path.isfile(faceH5Path):
            print("not exists : ",faceH5Path)
            return
        if not os.path.isfile(headH5Path):
            print("not exists : ",headH5Path)
            return         
        if not os.path.isfile(faceLmIdxFile):
            print("not exists : ", faceLmIdxFile)
            return            
        with h5py.File(faceH5Path, 'r') as h5_file:
            shape_points = h5_file['shape/representer/points'][:]
            shape_cells = h5_file['shape/representer/cells'][:]
            shape_mean = h5_file['shape/model/mean'][:]
            shape_pcaBasis = h5_file['shape/model/pcaBasis'][:]
            expression_points = h5_file['expression/representer/points'][:]
            expression_cells = h5_file['expression/representer/cells'][:]
            expression_mean = h5_file['expression/model/mean'][:]
            expression_pcaBasis = h5_file['expression/model/pcaBasis'][:]

        # mean face shape. [3*N,1]
        self.mean_shape = (shape_mean+expression_mean).astype(np.float32)
        # identity basis. [3*N,80]
        self.id_base = shape_pcaBasis.astype(np.float32)
        # expression basis. [3*N,64]
        self.exp_base = expression_pcaBasis.astype(np.float32) 
        self.face_tri = np.array(shape_cells.T, dtype=np.int64)
        # vertex indices for 68 landmarks. starts from 0. [68,1]
        self.keypoints = np.load(faceLmIdxFile).astype(np.int64)
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        for key, value in self.__dict__.items():
            if type(value).__module__ == np.__name__:
                setattr(self, key, torch.tensor(value).to(device))

    def getMediapipe486BfmBase(self,):
        mediapipe486Len = len(self.keypoints)  # 只有480  里没有6个点的对应
        id_486part = np.zeros([mediapipe486Len*3, 199])
        exp_486part = np.zeros([mediapipe486Len*3, 100])
        mean_486part = np.zeros([mediapipe486Len*3, 1])
        for i in range(mediapipe486Len):
            I = int(self.keypoints[i])
            if I < 0:
                continue
            id_486part[3*i, ...] = self.id_base[3*I, ...]
            id_486part[3*i+1, ...] = self.id_base[3*I+1, ...]
            id_486part[3*i+2, ...] = self.id_base[3*I+2, ...]

            exp_486part[3*i, ...] = self.exp_base[3*I, ...]
            exp_486part[3*i+1, ...] = self.exp_base[3*I+1, ...]
            exp_486part[3*i+2, ...] = self.exp_base[3*I+2, ...]

            mean_486part[3*i, ...] = self.mean_shape[3*I, ...]
            mean_486part[3*i+1, ...] = self.mean_shape[3*I+1, ...]
            mean_486part[3*i+2, ...] = self.mean_shape[3*I+2, ...]
        return id_486part, exp_486part, mean_486part

if __name__ == '__main__':

    facemodel2019 = Bfm2019('Deep3d/BFM')

    facemodel = ParametricFaceModel()
    shape_base = facemodel.id_base.reshape(-1,3,80)
    expression_base = facemodel.exp_base.reshape(-1,3,64)
    mean_base = facemodel.mean_shape.reshape(-1,3)
    shape_weight = np.zeros([80,1])
    expression_weight = np.zeros([64,1])
    vts = (shape_base@shape_weight + expression_base@expression_weight).reshape(-1,3)+mean_base
    vts=vts*100
    save.saveObj("bfm09.obj",vts,facemodel.face_tri)
    print()
import os
import torch
import numpy as np
import os
import utils
import open3d as o3d


class SpaceMap:
    def __init__(self, unit, regionStart=None, regionEnd=None):
        regionStart = np.array(regionStart)
        regionEnd = np.array(regionEnd)
        self.imgWidth = -1
        self.imgHeight = -1
        print(np.iinfo(np.uint64).max)
        if (regionStart == None).any():
            self.xSize = 2
            self.ySize = 2
            self.zSize = 2
            self.maxUint64 = np.iinfo(np.uint64).max
            resolution = np.uint64(np.power(self.maxUint64, 0.333))
            self.unit = self.xSize/resolution
            assert self.unit < unit, "parameter error"
            self.unit = unit
            self.resolutionX = np.uint64(self.xSize/self.unit)
            self.resolutionY = np.uint64(self.ySize/self.unit)
            self.resolutionZ = np.uint64(self.zSize/self.unit)
            self.xSize = self.unit*self.resolutionX
            self.ySize = self.unit*self.resolutionY
            self.zSize = self.unit*self.resolutionZ
            self.regionStartX = 0-self.xSize*0.5
            self.regionStartY = 0-self.ySize*0.5
            self.regionStartZ = 0-self.zSize*0.5
            self.regionEndX = self.xSize-self.regionStartX
            self.regionEndY = self.ySize-self.regionStartY
            self.regionEndZ = self.zSize-self.regionStartZ
        else:

            assert len(regionStart) == 3 and len(
                regionEnd) == 3, "parameter error"
            assert np.min(
                regionEnd-regionStart) > 0., "Data directory is empty"
            self.xSize = regionEnd[0]-regionStart[0]
            self.ySize = regionEnd[1]-regionStart[1]
            self.zSize = regionEnd[2]-regionStart[1]
            self.maxUint64 = np.iinfo(np.uint64).max
            resolution = np.uint64(
                np.power(self.maxUint64/self.zSize/self.zSize*self.xSize*self.ySize, 0.333))
            self.unit = self.xSize/resolution
            assert self.unit < unit, "parameter error"
            self.unit = unit
            self.resolutionX = np.uint64(self.xSize/self.unit)
            self.resolutionY = np.uint64(self.ySize/self.unit)
            self.resolutionZ = np.uint64(self.zSize/self.unit)
            self.xSize = self.unit*self.resolutionX
            self.ySize = self.unit*self.resolutionY
            self.zSize = self.unit*self.resolutionZ
            self.regionStartX = regionStart[0]
            self.regionStartY = regionStart[1]
            self.regionStartZ = regionStart[2]
            self.regionEndX = self.xSize+self.regionStartX
            self.regionEndY = self.ySize+self.regionStartY
            self.regionEndZ = self.zSize+self.regionStartZ
        self.gridFlag = np.ones(self.resolutionX*self.resolutionY*self.resolutionZ).astype(bool)
        self.gridCenter = np.mgrid[0:self.resolutionX,
                                   0:self.resolutionY, 0:self.resolutionZ].astype(np.float32).reshape(3, -1)+np.array([[0.5], [0.5], [0.5]]).astype(np.float32)
        self.gridCenter = self.gridCenter*self.unit + \
            np.array([[self.regionStartX], [
                     self.regionStartY], [self.regionStartZ]])
        self.gridCenter = np.vstack(
            [self.gridCenter, np.ones([1, self.gridCenter.shape[1]]).astype(np.float32)])
    def inputData(self, data_dir, cam_file=None):
        self.instance_dir = data_dir
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        image_dir = '{0}/image'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))
        mask_dir = '{0}/mask'.format(self.instance_dir)
        mask_paths = sorted(utils.glob_imgs(mask_dir))

        self.n_images = len(image_paths)

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        if cam_file is not None:
            self.cam_file = '{0}/{1}'.format(self.instance_dir, cam_file)

        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(
            np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(
            np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = utils.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(intrinsics.astype(np.float32))
            self.pose_all.append(pose.astype(np.float32))

        self.rgb_images = []
        for path in image_paths:
            rgb = utils.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(rgb.astype(np.float32))

        self.object_masks = []
        for path in mask_paths:
            object_mask = utils.load_mask(path)
            if self.imgHeight < 0:
                self.imgHeight = object_mask.shape[0]
            if self.imgWidth < 0:
                self.imgWidth = object_mask.shape[1]
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(object_mask.astype(bool))

        for picIdx in range(self.n_images):
            intrinsics = self.intrinsics_all[picIdx]
            pose = self.pose_all[picIdx]
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]
            sk = intrinsics[0, 1]
            fxfy = np.array([fx, fy]).reshape(2, -1)
            validPos = self.gridFlag==True
            xyz = pose.T@self.gridCenter[:, validPos]
            zPositive = xyz[2, :]>0.01
            xyz = xyz/xyz[2,:]

            xyz[0, :] = xyz[0, :]*fx+cx
            xyz[1, :] = xyz[1, :]*fy+cy
            xyz = np.round(xyz).astype(np.int32)
            xyminFlag = np.logical_and(xyz[0, :] > 0, xyz[1, :] > 0)
            xymaxFlag = np.logical_and(
                xyz[0, :] < self.imgWidth, xyz[1, :] < self.imgHeight)
            imgRectFlag = np.logical_and(zPositive, xyminFlag)
            imgRectFlag = np.logical_and(imgRectFlag, xymaxFlag)
            x = np.round(xyz[0, :][imgRectFlag])
            y = self.imgWidth*np.round(xyz[1, :][imgRectFlag])
            xy = x+y
            a = np.where(self.object_masks[picIdx][xy]==False)[0]
            viewedPos = np.where(imgRectFlag==True)[0]
            self.gridFlag[viewedPos][a]=False
            self.getCloud()
    def getCloud(self):
        a=np.where(self.gridFlag==True)
        x = a%self.resolutionX
        yz = a//self.resolutionX
        y = yz%self.resolutionY
        z = yz//self.resolutionY
        a=self.gridFlag.reshape([self.resolutionZ,self.resolutionY,self.resolutionX])
        print()

class SceneDataset:
    def __init__(self,
                 data_dir,
                 regionStart,
                 regionEnd,
                 cam_file=None
                 ):
        print(np.iinfo(np.uint64).max)
        self.instance_dir = data_dir
        self.imgWidth = -1
        self.imgHeight = -1
        self.regionStart = regionStart
        self.regionEnd = regionEnd
        self.octreeDepth = 8
        self.octreeOrigin = np.array([0, 0, 0])
        self.octreeSize = 1
        self.octree = o3d.geometry.Octree(
            max_depth=self.octreeDepth, origin=self.octreeOrigin, size=self.octreeSize)
        # self.octree.convert_from_point_cloud(pcd)
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        image_dir = '{0}/image'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))
        mask_dir = '{0}/mask'.format(self.instance_dir)
        mask_paths = sorted(utils.glob_imgs(mask_dir))

        self.n_images = len(image_paths)

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        if cam_file is not None:
            self.cam_file = '{0}/{1}'.format(self.instance_dir, cam_file)

        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(
            np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(
            np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = utils.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(intrinsics.astype(np.float32))
            self.pose_all.append(pose.astype(np.float32))

        self.rgb_images = []
        for path in image_paths:
            rgb = utils.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(rgb.astype(np.float32))

        self.object_masks = []
        for path in mask_paths:
            object_mask = utils.load_mask(path)
            if self.imgHeight < 0:
                self.imgHeight = object_mask.shape[0]
            if self.imgWidth < 0:
                self.imgWidth = object_mask.shape[1]
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(object_mask.astype(bool))

    def len(self):
        return self.n_images

    def getitem(self, idx):
        uv = np.mgrid[0:self.imgHeight, 0:self.imgWidth].astype(np.int32)
        uv = uv.reshape(2, -1).astype(np.float32)
        uv = uv[:, self.object_masks[idx]]

        intrinsics = self.intrinsics_all[idx]
        pose = self.pose_all[idx]
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        sk = intrinsics[0, 1]
        fxfy = np.array([fx, fy]).reshape(2, -1)

        xyz = np.ones([3, uv.shape[1]])
        xyz[:2, :] = (uv-intrinsics[:2, 2].reshape(2, -1)) / fxfy
        xyz = xyz/np.linalg.norm(xyz, axis=0)

        f_init = o3d.geometry.OctreeColorLeafNode.get_init_function()
        f_update = o3d.geometry.OctreeColorLeafNode.get_update_function([
                                                                        0.0, 0.0, 0.0])
        for depthIter in range(10000):
            if depthIter == 0:
                continue
            if 0.01*depthIter > 0.5*self.octreeSize:
                break
            xyz2 = xyz*(0.01*depthIter)
            xyz3 = pose[:3, :3]@xyz2+pose[:3, 3].reshape(3, 1)
            for i in range(uv.shape[1]):
                self.octree.insert_point(xyz3.T[i], f_init, f_update)

        o3d.visualization.draw_geometries([self.octree])
        return idx

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(
                self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def get_gt_pose(self, scaled=False):
        # Load gt pose without normalization to unit sphere
        camera_dict = np.load(self.cam_file)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(
            np.float32) for idx in range(self.n_images)]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(
            np.float32) for idx in range(self.n_images)]

        pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat
            if scaled:
                P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = utils.load_K_Rt_from_P(None, P)
            pose_all.append(torch.from_numpy(pose).float())

        return torch.cat([p.float().unsqueeze(0) for p in pose_all], 0)

    def get_pose_init(self):
        # get noisy initializations obtained with the linear method
        cam_file = '{0}/cameras_linear_init.npz'.format(self.instance_dir)
        camera_dict = np.load(cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(
            np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(
            np.float32) for idx in range(self.n_images)]

        init_pose = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = utils.load_K_Rt_from_P(None, P)
            init_pose.append(pose)
        init_pose = torch.cat([torch.Tensor(pose).float().unsqueeze(0)
                              for pose in init_pose], 0).cuda()
        init_quat = utils.rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat

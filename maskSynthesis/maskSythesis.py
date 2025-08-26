# -*- coding: UTF-8 -*-
import os
import numpy as np
import numpy as np
import open3d as o3d 
import data

if __name__=='__main__':
    # a = data.SpaceMap(0.005, regionStart=np.array([-1, -1, -1]), regionEnd=np.array([1, 1, 1]))
    # a.inputData('../idr/data/DTU/scan122')  

    cam_file = 'data/a/result/shapeMask/cam_file.npy'
    assert os.path.exists(cam_file), "cam_file is empty"
    cam_info = np.load(cam_file, allow_pickle=True).item()

    # intr = cam_info['resulta@00037.jpg@intr']
    # Rt = cam_info['resulta@00037.jpg@Rt']
    # pts3d = np.array([1.66263, -0.125554, 10.7359, 1,
    #                   1.23785, 0.158018, 10.0801, 1,
    #                   1.24995, 0.674161, 10.0609, 1,
    #                   1.55609, -0.566569, 10.5188, 1,
    #                   -1.83245, -0.469614, 8.64266, 1,
    #                   0.416358, -0.5012, 8.41182, 1,
    #                   -1.26225, 1.09625, 8.30688, 1,
    #                   -0.070143, 1.0553, 8.20302, 1,
    #                   -0.551219, 0.884708, 7.82925, 1,
    #                   -0.66592, 1.24098, 7.91844, 1,
    #                   -0.351713, -0.987281, 8.03881, 1,
    #                   -1.9065, -0.911543, 8.56162, 1,
    #                   -0.629218, 0.433139, 7.80185, 1,
    #                   -1.15032, -0.439842, 8.3314, 1,
    #                   -0.865296, 0.436891, 7.8632, 1,
    #                   -0.273799, -0.45967, 8.25304, 1,
    #                   -2.63195, -0.101635, 10.9875, 1,
    #                   ]).reshape(-1,4)
    # pts2d = Rt@pts3d.T
    # pts2d = pts2d/pts2d[2]
    # pts2d = pts2d[:3]
    # uv = intr@pts2d


    a = data.SpaceMap(0.1, cam_info['regionStart'], cam_info['regionEnd'])
    a.inputData2('data/a/result/shapeMask', cam_file)
    exit(-1)
# -*- coding: UTF-8 -*-
import os
import numpy as np
import numpy as np
import open3d as o3d 
import data

if __name__=='__main__':
    # a = data.SpaceMap(0.005, regionStart=np.array([-1, -1, -1]), regionEnd=np.array([1, 1, 1]))
    # a.inputData('../idr/data/DTU/scan122')  

    cam_file = 'data/shapeMask/cam_file.npy'
    assert os.path.exists(cam_file), "cam_file is empty"
    cam_info = np.load(cam_file, allow_pickle=True).item()
    a = data.SpaceMap(0.005, cam_info['regionStart'], cam_info['regionEnd'])
    a.inputData2('data/shapeMask', cam_file)
    exit(-1)
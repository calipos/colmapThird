# -*- coding: UTF-8 -*-
import numpy as np
import numpy as np
import open3d as o3d 
import data

if __name__=='__main__':
    a = data.SpaceMap(0.005,regionStart=np.array([-1, -1, -1]), regionEnd=np.array([1,1,1]))
    a.inputData('../idr/data/DTU/scan40')
    data = data.SceneDataset('../idr/data/DTU/scan40', regionStart=[-5, -5, -5], regionEnd = [5, 5, 5])
    for i in range(data.len()):
        data.getitem(i)
    pcd = o3d.io.read_point_cloud('D:/ucl360/imu/intensityPano.ply')
    pts = np.asarray(pcd.points)
    # octree = o3d.geometry.Octree(max_depth=16)    
    octree = o3d.geometry.Octree(max_depth=4)
    octree.convert_from_point_cloud(pcd)
    #o3d.visualization.draw_geometries([octree])

    Node,  NodeInfoo = octree.locate_leaf_node(pts[2])

    print()
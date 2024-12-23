import json
import numpy as np
import h5py
import scipy.io as sio
import load_mats 
from scipy.spatial import KDTree
def review_h5_file(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        def print_hierarchy(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"    Dataset: {name}, Shape: {obj.shape}, Data type: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")

        h5_file.visititems(print_hierarchy)
 
def print_first_line(file_path, name_group_or_dataset):
    with h5py.File(file_path, 'r') as h5_file:
        if name_group_or_dataset in h5_file:
            obj = h5_file[name_group_or_dataset]
            if isinstance(obj, h5py.Dataset):
                print(f"First line of dataset '{name_group_or_dataset}': {obj[0]}")
            elif isinstance(obj, h5py.Group):
                keys = list(obj.keys())
                if keys:
                    print(f"First item in group '{name_group_or_dataset}': {keys[0]}")
                else:
                    print(f"Group '{name_group_or_dataset}' is empty.")
            else:
                print("Invalid input, please provide a group or dataset name")
        else:
            print(f"'{name_group_or_dataset}' not found in the file.")         


def get_face_color_from_h5(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        shape_points = h5_file['color/representer/points'][:]
        shape_cells = h5_file['color/representer/cells'][:]
        color_mean = h5_file['color/model/mean'][:] 
        color_pcaBasis = h5_file['color/model/pcaBasis'][:] 
        shape_pcaBasis = h5_file['shape/model/pcaBasis'][:] 
    vertices = np.array(shape_points.T, dtype=np.float32)
    faces = np.array(shape_cells.T, dtype=np.int64)
    colors = np.array(color_mean.reshape(-1, 3), dtype=np.float32)
    return vertices, faces, colors
 

def get_face_metadata_landmarks60_from_h5(file_path):
    # mat_data = sio.loadmat('BFM/similarity_Lm3D_all.mat')
    with h5py.File(file_path, 'r') as h5_file:
        landmarkJson = h5_file['metadata/landmarks/json'][:][0]
        decoded_data = json.loads(landmarkJson)

def get_face_shape_from_h5(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        shape_points = h5_file['shape/representer/points'][:]
        shape_cells = h5_file['shape/representer/cells'][:]
        shape_mean = h5_file['shape/model/mean'][:]
        shape_pcaBasis = h5_file['shape/model/pcaBasis'][:]
    vertices = np.array(shape_points.T, dtype=np.float32)
    faces = np.array(shape_cells.T, dtype=np.int64)
    mean = np.array(shape_mean.reshape(-1, 3), dtype=np.float32)
    vertixCnt = mean.shape[0]
    pcaBasis = np.array(shape_pcaBasis.reshape(
        vertixCnt, 3, -1), dtype=np.float32)
    shapeBasisDim = pcaBasis.shape[2]
    print('pts num = ', vertixCnt)
    print('faces num = ', faces.shape[0])
    print('shapeBasisDim = ', shapeBasisDim)
    return vertices, faces, mean, pcaBasis, shapeBasisDim


def get_face_expression_from_h5(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        expression_points = h5_file['expression/representer/points'][:]
        expression_cells = h5_file['expression/representer/cells'][:]
        expression_mean = h5_file['expression/model/mean'][:]
        expression_pcaBasis = h5_file['expression/model/pcaBasis'][:]
    vertices = np.array(expression_points.T, dtype=np.float32)
    faces = np.array(expression_cells.T, dtype=np.int64)
    mean = np.array(expression_mean.reshape(-1, 3), dtype=np.float32)
    vertixCnt = mean.shape[0]
    pcaBasis = np.array(
        expression_pcaBasis.reshape(vertixCnt, 3, -1), dtype=np.float32)
    expressionBasisDim = pcaBasis.shape[2]
    print('expressionBasisDim = ', expressionBasisDim)
    return vertices, faces, mean, pcaBasis, expressionBasisDim

def saveObj(filepath,faces,verts):
    # verts=shapeMean+expressionMean+(shapeBias@shapeWeight).reshape(-1,3)+(expressionBias@expressionWeight).reshape(-1,3)
    # shapeBias@shapeWeight == np.dot(shapeBias, shapeWeight)
    thefile = open(filepath, 'w')
    for item in verts:
        thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2])) 
    for item in faces:
        thefile.write("f {0} {1} {2}\n".format(item[0]+1,item[1]+1,item[2]+1))  
    thefile.close()            
def saveRgbObj(filepath,faces,verts,rgb):
    # verts=shapeMean+expressionMean+(shapeBias@shapeWeight).reshape(-1,3)+(expressionBias@expressionWeight).reshape(-1,3)
    # shapeBias@shapeWeight == np.dot(shapeBias, shapeWeight)
    thefile = open(filepath, 'w')
    for i,item in enumerate( verts):
        thefile.write("v {0} {1} {2} {3} {4} {5}\n".format(item[0],item[1],item[2],rgb[i,0],rgb[i,1],rgb[i,2])) 
    for item in faces:
        thefile.write("f {0} {1} {2}\n".format(item[0]+1,item[1]+1,item[2]+1))  
    thefile.close()      
def figureBFM2019CorrespondTo2009():
    bfm2019file = 'BFM/model2019_bfm(47439p94464f).h5'
    bfm2009file = 'BFM/BFM_model_front.mat'
    index_mp468_from_bfm2009file = 'BFM/index_mp468_from_mesh35709.npy'
    scale = 101.047
    R=np.array([1.012078881264, 0.000035081626, -0.002887555165 ,
-0.000098854864, 1.011836051941, -0.022355282679,
0.002886074129, 0.022355468944, 1.011832118034],dtype=np.float32).reshape(3,3)
    t=np.array([ 0.386107414961,0.755384385586,-2.180266141891],dtype=np.float32).reshape(3,1)
    model = load_mats.loadmat(bfm2009file)
    bfm2009pts = model['meanshape'].astype(np.float32).reshape(-1,3)*100
    bfm2009pts = R@(bfm2009pts.T)+t
    bfm2009pts=bfm2009pts.T
    with h5py.File(bfm2019file, 'r') as h5_file:
        shape_mean = h5_file['shape/model/mean'][:]
        expression_mean = h5_file['expression/model/mean'][:]
    bfm2019pts = np.array((shape_mean+expression_mean).reshape(-1, 3), dtype=np.float32)
    # np.savetxt('bfm2009pts.pts',bfm2009pts)
    # np.savetxt('bfm2019pts.pts',bfm2019pts)

    index_mp468_from_bfm2009 = np.load(index_mp468_from_bfm2009file).astype(np.int64)
    index_mp468_from_bfm2009_valid = index_mp468_from_bfm2009[index_mp468_from_bfm2009>=0]
    lm2009 = bfm2009pts[index_mp468_from_bfm2009_valid,]
    kdtree=KDTree(bfm2019pts) #创建kdtree
    nearest_dist, nearest_idx = kdtree.query(lm2009,k=1)
    np.savetxt('bfm2019lm.pts',bfm2019pts[nearest_idx])
    np.savetxt('bfm2009lm.pts',lm2009)
    index_mp468_from_bfm2019=index_mp468_from_bfm2009
    index_mp468_from_bfm2019[index_mp468_from_bfm2009 >= 0] = nearest_idx
    np.save('BFM/index_mp468_from_model2019_47439p.npy', index_mp468_from_bfm2019)
    print()
if __name__ == '__main__': 

    figureBFM2019CorrespondTo2009()
    # review_h5_file('C:/Users/Administrator/Downloads/model2019_face12.h5')
    # review_h5_file('C:/Users/Administrator/Downloads/model2019_fullHead.h5')
    # print(vertices.shape, faces.shape, colors.shape)

    h5file = 'BFM/model2019_bfm(47439p94464f).h5'
    # h5file = 'BFM/model2019_fullHead(58203p116160f).h5'
    # h5file = 'BFM/model2019_face12(27657p55040f).h5'
    get_face_metadata_landmarks60_from_h5(h5file)
    vertices, faces, colors = get_face_color_from_h5(h5file)
    saveRgbObj('rgb.obj',faces,vertices,  colors)
    vertices_shape, faces_shape, mean_shape, pcaBasis_shape, shapeBasisDim = get_face_shape_from_h5(
        h5file)
    vertices_expression, faces_expression, mean_expression, pcaBasis_expression, expressionBasisDim = get_face_expression_from_h5(
        h5file)
    
    shapeWeight = np.zeros([shapeBasisDim,1])
    expressionWeight = np.zeros([expressionBasisDim, 1])
    verts=mean_shape+mean_expression+\
        (pcaBasis_shape@shapeWeight).reshape(-1,3)+\
        (pcaBasis_expression@expressionWeight).reshape(-1,3)
    
    saveObj('test2019head.obj',faces_shape,verts )
      
    print()
    # review_h5_file('C:/Users/Administrator/Downloads/model2019_bfm.h5')
    # print_first_line('C:/Users/Administrator/Downloads/model2019_bfm.h5','color/representer/colorspace')
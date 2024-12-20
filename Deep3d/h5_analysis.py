
import numpy as np
import h5py
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
    vertices = np.array(shape_points.T, dtype=np.float32)
    faces = np.array(shape_cells.T, dtype=np.int64)
    colors = np.array(color_mean.reshape(-1, 3), dtype=np.float32)
    return vertices, faces, colors


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

def saveObj(filepath,faces,shapeMean,shapeBias,shapeWeight,expressionMean,expressionBias,expressionWeight):
    verts=shapeMean+expressionMean+(shapeBias@shapeWeight).reshape(-1,3)+(expressionBias@expressionWeight).reshape(-1,3)
    # shapeBias@shapeWeight == np.dot(shapeBias, shapeWeight)
    thefile = open(filepath, 'w')
    for item in verts:
        thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2])) 
    for item in faces:
        thefile.write("f {0} {1} {2}\n".format(item[0]+1,item[1]+1,item[2]+1))  
    thefile.close()            

if __name__ == '__main__': 

    # review_h5_file('C:/Users/Administrator/Downloads/model2019_face12.h5')
    # review_h5_file('C:/Users/Administrator/Downloads/model2019_fullHead.h5')
    # vertices, faces, colors = get_face_color_from_h5('C:/Users/Administrator/Downloads/model2019_bfm.h5')
    # print(vertices.shape, faces.shape, colors.shape)

    h5file = 'C:/Users/Administrator/Downloads/model2019_bfm(47439p94464f).h5'
    # h5file = 'C:/Users/Administrator/Downloads/model2019_fullHead(58203p116160f).h5'
    # h5file = 'C:/Users/Administrator/Downloads/model2019_face12(27657p55040f).h5'
    vertices_shape, faces_shape, mean_shape, pcaBasis_shape, shapeBasisDim = get_face_shape_from_h5(
        h5file)
    vertices_expression, faces_expression, mean_expression, pcaBasis_expression, expressionBasisDim = get_face_expression_from_h5(
        h5file)
    
    shapeBias = np.zeros([shapeBasisDim,1])
    expressionBias = np.zeros([expressionBasisDim, 1])
    saveObj('test2019face.obj',faces_shape,mean_shape, pcaBasis_shape,shapeBias,mean_expression, pcaBasis_expression,expressionBias)
    print()
    # review_h5_file('C:/Users/Administrator/Downloads/model2019_bfm.h5')
    # print_first_line('C:/Users/Administrator/Downloads/model2019_bfm.h5','color/representer/colorspace')
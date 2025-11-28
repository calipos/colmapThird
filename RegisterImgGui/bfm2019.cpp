#include <iostream>
#include <filesystem>
#include <string>
#include "H5Cpp.h"
#include "log.h"
#include "opencv2/opencv.hpp"
using namespace H5;
/* bfm2019
catalog
catalog/MorphableModel
catalog/MorphableModel/modelPath
catalog/MorphableModel/modelType
catalog/MorphableModel.color
catalog/MorphableModel.color/modelPath
catalog/MorphableModel.color/modelType
catalog/MorphableModel.expression
catalog/MorphableModel.expression/modelPath
catalog/MorphableModel.expression/modelType
catalog/MorphableModel.shape
catalog/MorphableModel.shape/modelPath
catalog/MorphableModel.shape/modelType
color
color/model
color/model/mean
color/model/noiseVariance
color/model/pcaBasis
color/model/pcaVariance
color/modelinfo
color/modelinfo/build-time
color/modelinfo/scores
color/representer
color/representer/cells
color/representer/colorspace
color/representer/points
color/version
color/version/majorVersion
color/version/minorVersion
expression
expression/model
expression/model/mean
expression/model/noiseVariance
expression/model/pcaBasis
expression/model/pcaVariance
expression/modelinfo
expression/modelinfo/build-time
expression/modelinfo/scores
expression/representer
expression/representer/cells
expression/representer/points
expression/version
expression/version/majorVersion
expression/version/minorVersion
metadata
metadata/landmarks
metadata/landmarks/json
shape
shape/model
shape/model/mean
shape/model/noiseVariance
shape/model/pcaBasis
shape/model/pcaVariance
shape/modelinfo
shape/modelinfo/build-time
shape/modelinfo/scores
shape/representer
shape/representer/cells
shape/representer/points
shape/version
shape/version/majorVersion
shape/version/minorVersion
version
version/majorVersion
version/minorVersion
*/
int test_bfm(void)
{  
    
    std::string bfmFacePath = "../models/model2019_face12.h5";
    if (!std::filesystem::exists(bfmFacePath))
    {
        LOG_ERR_OUT << "need models/model2019_face12.h5";
        return -1;
    } 
    cv::Mat shape_mean,shape_points, shape_cells, shape_pcaBasis;
    cv::Mat expression_mean, expression_points,  expression_pcaBasis;
    cv::Mat color_mean, color_points,  color_pcaBasis;
    { 
        H5File file(bfmFacePath.c_str(), H5F_ACC_RDONLY);
        {
            DataSet dataset = file.openDataSet("shape/model/mean");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            shape_mean = cv::Mat(dims[0], dims[1], CV_32FC1);
            dataset.read(shape_mean.data, H5::PredType::NATIVE_FLOAT);
        }
        {
            DataSet dataset = file.openDataSet("shape/representer/points");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            shape_points = cv::Mat(dims[0], dims[1], CV_32FC1);
            dataset.read(shape_points.data, H5::PredType::NATIVE_FLOAT);
        }
        {
            DataSet dataset = file.openDataSet("shape/representer/cells");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            shape_cells = cv::Mat(dims[0], dims[1], CV_32SC1);
            dataset.read(shape_cells.data, H5::PredType::NATIVE_INT);
        }
        {
            DataSet dataset = file.openDataSet("shape/model/pcaBasis");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            shape_pcaBasis = cv::Mat(dims[0], dims[1], CV_32FC1);
            dataset.read(shape_pcaBasis.data, H5::PredType::NATIVE_FLOAT);
        }
        {
            DataSet dataset = file.openDataSet("expression/model/mean");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            expression_mean = cv::Mat(dims[0], dims[1], CV_32FC1);
            dataset.read(expression_mean.data, H5::PredType::NATIVE_FLOAT);
        }
        {
            DataSet dataset = file.openDataSet("expression/representer/points");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            expression_points = cv::Mat(dims[0], dims[1], CV_32FC1);
            dataset.read(expression_points.data, H5::PredType::NATIVE_FLOAT);
        }
        {
            DataSet dataset = file.openDataSet("expression/model/pcaBasis");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            expression_pcaBasis = cv::Mat(dims[0], dims[1], CV_32FC1);
            dataset.read(expression_pcaBasis.data, H5::PredType::NATIVE_FLOAT);
        }
        {
            DataSet dataset = file.openDataSet("color/model/mean");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            color_mean = cv::Mat(dims[0], dims[1], CV_32FC1);
            dataset.read(color_mean.data, H5::PredType::NATIVE_FLOAT);
        }
        {
            DataSet dataset = file.openDataSet("color/representer/points");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            color_points = cv::Mat(dims[0], dims[1], CV_32FC1);
            dataset.read(color_points.data, H5::PredType::NATIVE_FLOAT);
        }
        {
            DataSet dataset = file.openDataSet("color/model/pcaBasis");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            color_pcaBasis = cv::Mat(dims[0], dims[1], CV_32FC1);
            dataset.read(color_pcaBasis.data, H5::PredType::NATIVE_FLOAT);
        }
        file.close();
        shape_mean.reshape(1, shape_points.rows);
        expression_mean.reshape(1, expression_points.rows);
        color_mean.reshape(1, color_points.rows);
    }


    return 0; // successfully terminated
}

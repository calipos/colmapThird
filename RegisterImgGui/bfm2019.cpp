#include <fstream>
#include <iostream>
#include <random>
#include <filesystem>
#include <string>
#include "H5Cpp.h"
#include "log.h"
#include "opencv2/opencv.hpp"
#include "Eigen/Core"
#include "igl/writeOBJ.h"
#include "json/json.h"
using namespace H5;
namespace bfm
{
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
    std::list<cv::Vec2i> triangle(const cv::Vec2i& p0, const cv::Vec2i& p1, const cv::Vec2i& p2) {
        std::list<cv::Vec2i> ret;
        //triangle area = 0
        if (p0[1] == p1[1] && p0[1] == p2[1])
        {
            int xmin = (std::min)((std::min)(p0[0], p1[0]), p2[0]);
            int xmax = (std::max)((std::max)(p0[0], p1[0]), p2[0]);
            for (int i = xmin; i <= xmax; i++)
            {
                ret.emplace_back(i, p0[1]);
            }
            return ret;
        }
        //sort base on Y
        cv::Vec2i t0 = p0;
        cv::Vec2i t1 = p1;
        cv::Vec2i t2 = p2;
        if (t0[1] > t1[1]) std::swap(t0, t1);
        if (t0[1] > t2[1]) std::swap(t0, t2);
        if (t1[1] > t2[1]) std::swap(t1, t2);
        int total_height = t2[1] - t0[1];
        for (int i = 0; i < total_height; i++) {
            //separate
            bool second_half = i > t1[1] - t0[1] || t1[1] == t0[1];
            int segment_height = second_half ? t2[1] - t1[1] : t1[1] - t0[1];
            float alpha = (float)i / total_height;
            float beta = (float)(i - (second_half ? t1[1] - t0[1] : 0)) / segment_height;

            cv::Vec2i A = t0 + (t2 - t0) * alpha;
            cv::Vec2i B = second_half ? t1 + (t2 - t1) * beta : t0 + (t1 - t0) * beta;
            if (A[0] > B[0]) std::swap(A, B);
            for (int j = A[0]; j <= B[0]; j++) {
                ret.emplace_back(j, t0[1] + i);
            }
        }
        return ret;
    }
    void savePts(const std::filesystem::path&path  ,const cv::Mat&pts)
    {
        std::fstream fout(path, std::ios::out);
        for (int i = 0; i < pts.cols; i++)
        {
            fout << pts.ptr<float>(0)[i] << " " << pts.ptr<float>(1)[i] << " " << pts.ptr<float>(2)[i] << std::endl;
        }
        fout.close();
        return;
    }
    class Bfm2019
    {
    public:
        Bfm2019() {}
        Bfm2019(const std::filesystem::path& bfmFacePath)
        {
            H5File file(bfmFacePath.string().c_str(), H5F_ACC_RDONLY);
            {
                DataSet dataset = file.openDataSet("shape/representer/points");
                DataSpace dataspace = dataset.getSpace();
                H5S_class_t type = dataspace.getSimpleExtentType();
                int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
                hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
                dataspace.getSimpleExtentDims(dims, NULL);
                points = cv::Mat(dims[0], dims[1], CV_32FC1);
                dataset.read(points.data, H5::PredType::NATIVE_FLOAT);
            }
            {
                cv::Mat cells;
                DataSet dataset = file.openDataSet("shape/representer/cells");
                DataSpace dataspace = dataset.getSpace();
                H5S_class_t type = dataspace.getSimpleExtentType();
                int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
                hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
                dataspace.getSimpleExtentDims(dims, NULL);
                cells = cv::Mat(dims[0], dims[1], CV_32SC1);
                dataset.read(cells.data, H5::PredType::NATIVE_INT);
                F = Eigen::MatrixXi(cells.cols, cells.rows);
                for (int c = 0; c < cells.cols; c++)
                {
                    F(c, 0) = cells.ptr<int>(0)[c];
                    F(c, 1) = cells.ptr<int>(1)[c];
                    F(c, 2) = cells.ptr<int>(2)[c];
                }
            }


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
                DataSet dataset = file.openDataSet("shape/model/pcaVariance");
                DataSpace dataspace = dataset.getSpace();
                H5S_class_t type = dataspace.getSimpleExtentType();
                int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
                hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
                dataspace.getSimpleExtentDims(dims, NULL);
                shape_pcaStandardDeviation = cv::Mat(dims[0], dims[1], CV_32FC1);
                dataset.read(shape_pcaStandardDeviation.data, H5::PredType::NATIVE_FLOAT);
                for (int r = 0; r < shape_pcaStandardDeviation.rows; r++)
                {
                    for (int c = 0; c < shape_pcaStandardDeviation.cols; c++)
                    {
                        shape_pcaStandardDeviation.ptr<float>(r)[c] = sqrt(shape_pcaStandardDeviation.ptr<float>(r)[c]);
                    }
                }
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
                DataSet dataset = file.openDataSet("expression/model/pcaVariance");
                DataSpace dataspace = dataset.getSpace();
                H5S_class_t type = dataspace.getSimpleExtentType();
                int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
                hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
                dataspace.getSimpleExtentDims(dims, NULL);
                expression_pcaStandardDeviation = cv::Mat(dims[0], dims[1], CV_32FC1);
                dataset.read(expression_pcaStandardDeviation.data, H5::PredType::NATIVE_FLOAT);
                for (int r = 0; r < expression_pcaStandardDeviation.rows; r++)
                {
                    for (int c = 0; c < expression_pcaStandardDeviation.cols; c++)
                    {
                        expression_pcaStandardDeviation.ptr<float>(r)[c] = sqrt(expression_pcaStandardDeviation.ptr<float>(r)[c]);
                    }
                }
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
                DataSet dataset = file.openDataSet("color/model/pcaBasis");
                DataSpace dataspace = dataset.getSpace();
                H5S_class_t type = dataspace.getSimpleExtentType();
                int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
                hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
                dataspace.getSimpleExtentDims(dims, NULL);
                color_pcaBasis = cv::Mat(dims[0], dims[1], CV_32FC1);
                dataset.read(color_pcaBasis.data, H5::PredType::NATIVE_FLOAT);
            }
            {
                DataSet dataset = file.openDataSet("color/model/pcaVariance");
                DataSpace dataspace = dataset.getSpace();
                H5S_class_t type = dataspace.getSimpleExtentType();
                int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
                hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
                dataspace.getSimpleExtentDims(dims, NULL);
                color_pcaStandardDeviation = cv::Mat(dims[0], dims[1], CV_32FC1);
                dataset.read(color_pcaStandardDeviation.data, H5::PredType::NATIVE_FLOAT);
                for (int r = 0; r < color_pcaStandardDeviation.rows; r++)
                {
                    for (int c = 0; c < color_pcaStandardDeviation.cols; c++)
                    {
                        color_pcaStandardDeviation.ptr<float>(r)[c] = sqrt(color_pcaStandardDeviation.ptr<float>(r)[c]);
                    }
                }
            }
            {
                //metadata/landmarks/json
                H5::DataSet dataset = file.openDataSet("/metadata/landmarks/json");
                std::string json_string;
                dataset.read(json_string, dataset.getStrType());
                JSONCPP_STRING err;
                Json::Value newRoot;
                int rawJsonLength = json_string.capacity();
                Json::CharReaderBuilder newBuilder;
                const std::unique_ptr<Json::CharReader> newReader(newBuilder.newCharReader());
                bool parseRet = newReader->parse(json_string.c_str(), json_string.c_str() + rawJsonLength, &newRoot,
                    &err);
                if (!parseRet)
                { 
                    LOG_ERR_OUT << err;
                }
                auto newMemberNames = newRoot.getMemberNames();
 
                LOG_OUT<< json_string;
                //json parsed_json = json::parse(json_string);
                //for (const auto& item : parsed_json) {
                //    Landmark lm;
                //    lm.name = item["id"];
                //    lm.coordinates.x = item["coordinates"][0];
                //    lm.coordinates.y = item["coordinates"][1];
                //    lm.coordinates.z = item["coordinates"][2];
                //    landmarks.push_back(lm);
                //}
                LOG_OUT;
            }
            file.close();
            shapeDim = shape_pcaBasis.cols;
            expressionDim = expression_pcaBasis.cols;
            colorDim = color_pcaBasis.cols;
        }
        int shapeDim{ 0 };
        int expressionDim{ 0 };
        int colorDim{ 0 };
        cv::Mat points;
        Eigen::MatrixXi F;
        cv::Mat shape_mean, shape_pcaBasis, shape_pcaStandardDeviation;
        cv::Mat expression_mean, expression_pcaBasis, expression_pcaStandardDeviation;
        cv::Mat color_mean, color_pcaBasis, color_pcaStandardDeviation;
        void generateRandomFace(Eigen::MatrixXf& V, Eigen::MatrixXf&C)const
        {
            std::default_random_engine e; 
            std::uniform_real_distribution<float> u(-1, 1);
            e.seed(time(0));
            cv::Mat shapeParam(shapeDim, 1, CV_32FC1);
            cv::Mat expressionParam(expressionDim, 1, CV_32FC1);
            cv::Mat colorParam;
            for (int i = 0; i < shapeDim; i++) {
                shapeParam.ptr<float>(0)[i] =   u(e);
            }
            for (int i = 0; i < expressionDim; i++) {
                expressionParam.ptr<float>(0)[i] =   u(e);
            }

            colorParam = shapeParam.mul(color_pcaStandardDeviation);
            shapeParam = shapeParam.mul(shape_pcaStandardDeviation);
            expressionParam = expressionParam.mul(expression_pcaStandardDeviation);
            
            cv::Mat face = shape_mean + shape_pcaBasis * shapeParam +expression_mean + expression_pcaBasis * expressionParam;
            cv::Mat color = color_mean + color_pcaBasis * colorParam;
            int ptsCnt = face.rows / 3;
            V = Eigen::MatrixXf(ptsCnt, 3);
            C = Eigen::MatrixXf(ptsCnt, 3);
#pragma omp parallel for
            for (int i = 0; i < ptsCnt; i++)
            {
                int i3 = 3 * i;
                V(i, 0) = face.ptr<float>(i3 + 0)[0];
                V(i, 1) = face.ptr<float>(i3 + 1)[0];
                V(i, 2) = face.ptr<float>(i3 + 2)[0];
                C(i, 0) = color.ptr<float>(i3 + 0)[0];
                C(i, 1) = color.ptr<float>(i3 + 1)[0];
                C(i, 2) = color.ptr<float>(i3 + 2)[0];
            }
            return;
        }
        void capture(const Eigen::MatrixXf& V, const Eigen::MatrixXf& C)const
        {
            float minX = V.col(0).minCoeff();
            float minY = V.col(1).minCoeff();
            float minZ = V.col(2).minCoeff();
            float maxX = V.col(0).maxCoeff();
            float maxY = V.col(1).maxCoeff();
            float maxZ = V.col(2).maxCoeff();            
            float widthFloat = maxX - minX;
            float heightFloat = maxY - minY;
            int tarImgSize = 600;
            float scale = 1;
            if (widthFloat > heightFloat)
            {
                scale = tarImgSize / widthFloat;
            }
            else
            {
                scale = tarImgSize / heightFloat;
            }
            int offsetX = -minX * scale;
            int offsetY = -minY * scale;
            cv::Mat img = cv::Mat::zeros(static_cast<int>(heightFloat * scale) + 1, static_cast<int>(heightFloat * scale) + 1, CV_8UC3);
            cv::Mat depth = cv::Mat::ones(static_cast<int>(heightFloat * scale) + 1, static_cast<int>(heightFloat * scale) + 1, CV_32FC1)* minZ;
            std::vector<cv::Point2i> uv(V.rows());
            int ptsCnt = V.rows(); 
            int faceCnt = F.rows();
#pragma omp parallel for
            for (int i = 0; i < ptsCnt; i++)
            {
                uv[i].x = static_cast<int>(scale * V(i, 0)+ offsetX);
                uv[i].y = static_cast<int>(scale * V(i, 1)+ offsetY);
            }
            for (int f = 0; f < faceCnt; f++)
            {
                const int& a = F(f, 0);
                const int& b = F(f, 1);
                const int& c = F(f, 2);
                int minx = (std::min)((std::min)(uv[c].x, uv[b].x), uv[c].x);
                int miny = (std::min)((std::min)(uv[c].y, uv[b].y), uv[c].y);
                int maxx = (std::max)((std::max)(uv[c].x, uv[b].x), uv[c].x);
                int maxy = (std::max)((std::max)(uv[c].y, uv[b].y), uv[c].y);
                float depth = (std::max)((std::max)(V(a, 2), V(b, 2)), V(c, 2));
                for (int r = miny; r <= maxy; ++r)
                {
                    for (int c = minx; c <= maxx; ++c)
                    {

                    }
                }
            }
            return;
        }
        void saveObj(const std::filesystem::path&path, const Eigen::MatrixXf& V, const Eigen::MatrixXf& C)const
        {
            //igl::writeOBJ(path.string(),V,F);
            std::fstream fout(path, std::ios::out);
            int ptsCnt = V.rows();
            int faceCnt = F.rows();
            for (int i = 0; i < ptsCnt; i++)
            {
                fout << "v " << V(i,0) <<" " << V(i, 1) << " " << V(i, 2) << " "
                    << C(i, 0) << " " << C(i, 1) << " " << C(i, 2) << std::endl;
            }
            for (int i = 0; i < faceCnt; i++)
            {
                fout << "f " << F(i, 0)+1 << " " << F(i, 1)+1 << " " << F(i, 2)+1 << std::endl;
            }
            fout.close();
            return;
        }
 
    };
}
int test_bfm(void)
{  
    
    std::filesystem::path bfmFacePath = "../models/model2019_face12.h5";
    if (!std::filesystem::exists(bfmFacePath))
    {
        LOG_ERR_OUT << "need models/model2019_face12.h5";
        return -1;
    }
    bfm::Bfm2019 model(bfmFacePath);
    cv::Mat face;
    cv::Mat color;
    for (int i = 0; i < 20; i++)
    {
        Eigen::MatrixXf V;
        Eigen::MatrixXf C;
        model.generateRandomFace(V, C);
        //model.saveObj("../surf/rand"+std::to_string(i)+".obj", V, C);
        model.capture(V, C);
    }
    return 0; // successfully terminated
}

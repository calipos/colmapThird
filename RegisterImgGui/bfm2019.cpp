#include <map>
#include <fstream>
#include <iostream>
#include <random>
#include <filesystem>
#include <string>
#include "bfm.h"
#include "H5Cpp.h"
#include "log.h"
#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"
#include "Eigen/Core"
#include "igl/writeOBJ.h"
#include "json/json.h"
#include "meshDraw.h"
#include "face.h"
#include "labelme.h"
#include "misc.h"
#include "triangulation.h"
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

    Bfm2019::Bfm2019() {}
    Bfm2019::Bfm2019(const std::filesystem::path& bfmFacePath)
    {
        H5File file(bfmFacePath.string().c_str(), H5F_ACC_RDONLY);
        {
            DataSet dataset = file.openDataSet("shape/representer/points");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            std::vector<float>rawdata(dims[1] * dims[0]);
            dataset.read(&rawdata[0], H5::PredType::NATIVE_FLOAT);
            points = Eigen::MatrixX3f(dims[1], dims[0]);
            points = Eigen::Map<Eigen::MatrixX3f>(&rawdata[0], dims[1], dims[0]);
        }
        {
            DataSet dataset = file.openDataSet("shape/representer/cells");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            std::vector<int>rawdata(dims[1] * dims[0]);
            dataset.read(&rawdata[0], H5::PredType::NATIVE_INT);
            F = Eigen::MatrixX3i(dims[1], dims[0]);
            F = Eigen::Map<Eigen::MatrixX3i>(&rawdata[0], dims[1], dims[0]);

        }


        {
            DataSet dataset = file.openDataSet("shape/model/mean");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            std::vector<float>rawdata(dims[1] * dims[0]);
            dataset.read(&rawdata[0], H5::PredType::NATIVE_FLOAT);
            shape_mean = Eigen::VectorXf(dims[1] * dims[0]);
            shape_mean = Eigen::Map<Eigen::VectorXf>(&rawdata[0], dims[1] * dims[0]);
        }
        {
            DataSet dataset = file.openDataSet("shape/model/pcaBasis");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            std::vector<float>rawdata(dims[1] * dims[0]);
            dataset.read(&rawdata[0], H5::PredType::NATIVE_FLOAT);
            shape_pcaBasis = Eigen::MatrixXf(dims[0], dims[1]);
            shape_pcaBasis = Eigen::Map<Eigen::MatrixXf>(&rawdata[0], dims[1], dims[0]);
            shape_pcaBasis.transposeInPlace();
        }
        {
            DataSet dataset = file.openDataSet("shape/model/pcaVariance");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            std::vector<float>rawdata(dims[1] * dims[0]);
            dataset.read(&rawdata[0], H5::PredType::NATIVE_FLOAT);
            shape_pcaStandardDeviation = Eigen::VectorXf(dims[1] * dims[0]);
            shape_pcaStandardDeviation = Eigen::Map<Eigen::VectorXf>(&rawdata[0], dims[1] * dims[0]);
            shape_pcaStandardDeviation = shape_pcaStandardDeviation.cwiseSqrt();
        }




        {
            DataSet dataset = file.openDataSet("expression/model/mean");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            std::vector<float>rawdata(dims[1] * dims[0]);
            dataset.read(&rawdata[0], H5::PredType::NATIVE_FLOAT);
            expression_mean = Eigen::VectorXf(dims[1] * dims[0]);
            expression_mean = Eigen::Map<Eigen::VectorXf>(&rawdata[0], dims[1] * dims[0]);
        }
        {
            DataSet dataset = file.openDataSet("expression/model/pcaBasis");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            std::vector<float>rawdata(dims[1] * dims[0]);
            dataset.read(&rawdata[0], H5::PredType::NATIVE_FLOAT);
            expression_pcaBasis = Eigen::MatrixXf(dims[0], dims[1]);
            expression_pcaBasis = Eigen::Map<Eigen::MatrixXf>(&rawdata[0], dims[1], dims[0]);
            expression_pcaBasis.transposeInPlace();
        }
        {
            DataSet dataset = file.openDataSet("expression/model/pcaVariance");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            std::vector<float>rawdata(dims[1] * dims[0]);
            dataset.read(&rawdata[0], H5::PredType::NATIVE_FLOAT);
            expression_pcaStandardDeviation = Eigen::VectorXf(dims[1] * dims[0]);
            expression_pcaStandardDeviation = Eigen::Map<Eigen::VectorXf>(&rawdata[0], dims[1] * dims[0]);
            expression_pcaStandardDeviation = expression_pcaStandardDeviation.cwiseSqrt();
        }



        {
            DataSet dataset = file.openDataSet("color/model/mean");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            std::vector<float>rawdata(dims[1] * dims[0]);
            dataset.read(&rawdata[0], H5::PredType::NATIVE_FLOAT);
            color_mean = Eigen::VectorXf(dims[1] * dims[0]);
            color_mean = Eigen::Map<Eigen::VectorXf>(&rawdata[0], dims[1] * dims[0]);
        }
        {
            DataSet dataset = file.openDataSet("color/model/pcaBasis");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            std::vector<float>rawdata(dims[1] * dims[0]);
            dataset.read(&rawdata[0], H5::PredType::NATIVE_FLOAT);
            color_pcaBasis = Eigen::MatrixXf(dims[0], dims[1]);
            color_pcaBasis = Eigen::Map<Eigen::MatrixXf>(&rawdata[0], dims[1], dims[0]);
            color_pcaBasis.transposeInPlace();
        }
        {
            DataSet dataset = file.openDataSet("color/model/pcaVariance");
            DataSpace dataspace = dataset.getSpace();
            H5S_class_t type = dataspace.getSimpleExtentType();
            int rank = dataspace.getSimpleExtentNdims(); // 获取数据集的维度数量
            hsize_t dims[2]; dims[1] = rank == 2 ? dims[1] : 1;
            dataspace.getSimpleExtentDims(dims, NULL);
            std::vector<float>rawdata(dims[1] * dims[0]);
            dataset.read(&rawdata[0], H5::PredType::NATIVE_FLOAT);
            color_pcaStandardDeviation = Eigen::VectorXf(dims[1] * dims[0]);
            color_pcaStandardDeviation = Eigen::Map<Eigen::VectorXf>(&rawdata[0], dims[1] * dims[0]);
            color_pcaStandardDeviation = color_pcaStandardDeviation.cwiseSqrt();
        }


        {
            //metadata/landmarks/json
            landmarks.clear();
            landmarkIdx.clear();
            H5::DataSet dataset = file.openDataSet("/metadata/landmarks/json");
            H5::DataType datatype = dataset.getDataType();
            H5T_class_t typeClass = datatype.getClass();
            DataSpace dataspace = dataset.getSpace();
            int rank = dataspace.getSimpleExtentNdims();
            hsize_t dims[1];
            dataspace.getSimpleExtentDims(dims, NULL);
            int numStrings = dims[0];
            if (typeClass == H5T_STRING) {
                if (datatype.isVariableStr()) {
                    LOG_OUT << 1;
                }
                else {
                    // 读取固定长度字符串
                    size_t typeSize = datatype.getSize();
                    char* buffer = new char[numStrings * typeSize];
                    dataset.read(buffer, datatype);
                    JSONCPP_STRING err;
                    Json::Value newRoot;
                    Json::CharReaderBuilder newBuilder;
                    const std::unique_ptr<Json::CharReader> newReader(newBuilder.newCharReader());
                    if (!newReader->parse(buffer, buffer + typeSize, &newRoot,
                        &err)) {
                        LOG_ERR_OUT << "newReader->parse error";
                        delete[] buffer;
                        return;
                    }
                    else
                    {
                        delete[] buffer;
                        LOG_OUT << newRoot.size();
                        for (int i = 0; i < newRoot.size(); i++)
                        {
                            auto newMemberNames = newRoot[i].getMemberNames();
                            if (std::find(newMemberNames.begin(), newMemberNames.end(), "id") != newMemberNames.end())
                            {
                                std::string landmarkName = newRoot[i]["id"].asString();
                                landmarks[landmarkName] = cv::Point3f();
                                landmarks[landmarkName].x = newRoot[i]["coordinates"][0].asFloat();
                                landmarks[landmarkName].y = newRoot[i]["coordinates"][1].asFloat();
                                landmarks[landmarkName].z = newRoot[i]["coordinates"][2].asFloat();
                            }
                        }
                    }
                }
            }
            std::vector<std::string>landmarksNameVect(landmarks.size());
            cv::Mat landmarksPos(landmarks.size(), 3, CV_32FC1);
            int i = 0;
            for (const auto& d : landmarks)
            {
                landmarksNameVect[i] = d.first;
                landmarksPos.ptr<float>(i)[0] = d.second.x;
                landmarksPos.ptr<float>(i)[1] = d.second.y;
                landmarksPos.ptr<float>(i)[2] = d.second.z;
                i += 1;
            }
            std::map<std::string, std::pair<int, float>>nearesetMatch;
            cv::flann::KDTreeIndexParams indexParams(4);
            int k = 1;
            cv::Mat pointMat;
            {
                pointMat = cv::Mat(points.rows(), points.cols(), CV_32FC1);
                for (int r = 0; r < pointMat.rows; r++)
                {
                    pointMat.ptr<float>(r)[0] = points(r, 0);
                    pointMat.ptr<float>(r)[1] = points(r, 1);
                    pointMat.ptr<float>(r)[2] = points(r, 2);
                }
            }
            cv::flann::Index tree(pointMat, indexParams);//此处用target构建k-d树
            cv::Mat ldIdx(landmarks.size(), k, CV_32SC1);   //装载搜索到的对应点的索引（即neibours在target这个矩阵的行数）
            cv::Mat ldDists(landmarks.size(), k, CV_32F);         //搜索到的最近邻的距离
            tree.knnSearch(landmarksPos, ldIdx, ldDists, k, cv::flann::SearchParams(32));

            for (int i = 0; i < landmarksNameVect.size(); i++)
            {
                landmarkIdx[landmarksNameVect[i]] = ldIdx.ptr<int>(i)[0];
            }
            //std::fstream fout("../surf/1.txt", std::ios::out);
            //for (int i = 0; i < this->points.rows; i++)
            //{
            //    fout << this->points.ptr<float>(i)[0] << " "
            //        << this->points.ptr<float>(i)[1] << " "
            //        << this->points.ptr<float>(i)[2] << std::endl;
            //}
            //fout.close();
            //std::fstream fout2("../surf/2.txt", std::ios::out);
            //for (int i = 0; i < landmarksPos.rows; i++)
            //{
            //    fout2 << landmarksPos.ptr<float>(i)[0] << " "
            //        << landmarksPos.ptr<float>(i)[1] << " "
            //        << landmarksPos.ptr<float>(i)[2] << std::endl;
            //}
            //fout2.close();
            //LOG_OUT;
        } 
        file.close();
        shapeDim = shape_pcaBasis.cols();
        expressionDim = expression_pcaBasis.cols();
        colorDim = color_pcaBasis.cols();
    } 
    void Bfm2019::generateRandomFace(Eigen::MatrixX3f& V, Eigen::MatrixX3f& C)const
    {
        std::default_random_engine e;
        std::uniform_real_distribution<float> u(-1, 1);
        e.seed(time(0));
        Eigen::VectorXf shapeParam(shapeDim);
        Eigen::VectorXf colorParam(colorDim);
        Eigen::VectorXf expressionParam(expressionDim);
        for (int i = 0; i < shapeDim; i++) {
            shapeParam[i] = u(e);
        }
        for (int i = 0; i < expressionDim; i++) {
            expressionParam[i] = u(e);
        }
        for (int i = 0; i < colorDim; i++) {
            colorParam[i] = u(e);
        }

        shapeParam = shapeParam.cwiseProduct(shape_pcaStandardDeviation);
        expressionParam = expressionParam.cwiseProduct(expression_pcaStandardDeviation);
        colorParam = colorParam.cwiseProduct(color_pcaStandardDeviation);

        Eigen::VectorXf face = shape_mean + shape_pcaBasis * shapeParam + expression_mean + expression_pcaBasis * expressionParam;
        Eigen::VectorXf color = color_mean + color_pcaBasis * colorParam;
        int ptsCnt = face.rows() / 3;
        V = Eigen::MatrixX3f(ptsCnt, 3);
        C = Eigen::MatrixX3f(ptsCnt, 3);
#pragma omp parallel for
        for (int i = 0; i < ptsCnt; i++)
        {
            int i3 = 3 * i;
            V(i, 0) = face[i3 + 0];
            V(i, 1) = face[i3 + 1];
            V(i, 2) = face[i3 + 2];
            C(i, 0) = color[i3 + 0];
            C(i, 1) = color[i3 + 1];
            C(i, 2) = color[i3 + 2];
        }
        return;
    }
    void Bfm2019::capture(const Eigen::MatrixX3f& V, const Eigen::MatrixX3f& C)const
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
        cv::Mat depth = cv::Mat::ones(static_cast<int>(heightFloat * scale) + 1, static_cast<int>(heightFloat * scale) + 1, CV_32FC1) * minZ;
        std::vector<cv::Point2i> uv(V.rows());
        int ptsCnt = V.rows();
        int faceCnt = F.rows();
#pragma omp parallel for
        for (int i = 0; i < ptsCnt; i++)
        {
            uv[i].x = static_cast<int>(scale * V(i, 0) + offsetX);
            uv[i].y = static_cast<int>(scale * V(i, 1) + offsetY);
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
    void Bfm2019::saveObj(const std::filesystem::path& path, const Eigen::MatrixX3f& V, const Eigen::MatrixX3f& C)const
    {
        //igl::writeOBJ(path.string(),V,F);
        std::fstream fout(path, std::ios::out);
        int ptsCnt = V.rows();
        int faceCnt = F.rows();
        for (int i = 0; i < ptsCnt; i++)
        {
            fout << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << " "
                << C(i, 0) << " " << C(i, 1) << " " << C(i, 2) << std::endl;
        }
        for (int i = 0; i < faceCnt; i++)
        {
            fout << "f " << F(i, 0) + 1 << " " << F(i, 1) + 1 << " " << F(i, 2) + 1 << std::endl;
        }
        fout.close();
        return;
    }
    //tar = scale*(R*src+t)
    bool figureRTS(const std::vector<Eigen::Vector3f>& src, const std::vector<Eigen::Vector3f>& tar, Eigen::Matrix3f& R, Eigen::RowVector3f& t, float& scale, const bool& doRt, const bool& doScale)
    {
        scale = 1.;
        R = Eigen::Matrix3f::Identity();
        t = Eigen::RowVector3f(0, 0, 0);
        if (src.size() < 3 || src.size() != tar.size())
        {
            LOG_ERR_OUT << "src.size() < 3 || src.size() != tar.size()";
            return false;
        }
        Eigen::MatrixX3f srcMat(src.size(), 3);
        Eigen::MatrixX3f tarMat(tar.size(), 3);
        for (int i = 0; i < src.size(); i++)
        {
            srcMat(i, 0) = src[i][0];
            srcMat(i, 1) = src[i][1];
            srcMat(i, 2) = src[i][2];
            tarMat(i, 0) = tar[i][0];
            tarMat(i, 1) = tar[i][1];
            tarMat(i, 2) = tar[i][2];
        }
        Eigen::RowVector3f meanSrc = srcMat.colwise().mean();
        Eigen::RowVector3f meanTar = tarMat.colwise().mean();
        if (doScale)
        {
            auto src_scale = (srcMat.rowwise() - meanSrc).rowwise().norm().mean();
            auto tar_mean = (tarMat.rowwise() - meanTar).rowwise().norm().mean();
            scale = tar_mean / src_scale;
            float scaleInv = src_scale / tar_mean;
            tarMat *= scaleInv; 
        }
        if (doRt)
        {
            Eigen::RowVector3f srcCenter = srcMat.colwise().mean();
            Eigen::RowVector3f tarCenter = tarMat.colwise().mean(); 
            Eigen::MatrixX3f srcMat2 = srcMat.rowwise() - srcCenter;//A
            Eigen::MatrixX3f tarMat2 = tarMat.rowwise() - tarCenter;//B             
            Eigen::Matrix3f H = srcMat2.transpose() * tarMat2;
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3f U = svd.matrixU();
            Eigen::Matrix3f V = svd.matrixV();
            R = V * U.transpose(); 
            // 步骤6：处理反射情况（确保是纯旋转，行列式=1）
            if (R.determinant() < 0) {
                V.col(2) *= -1;  // 将V的最后一列取反
                R = V * U.transpose();
            }
            t = tarCenter - srcCenter * R.transpose();
        }
        return true;
    }

    bool figureSharedPoint(const std::vector<Eigen::Vector2f>&imgPts, 
        const std::vector<Eigen::Vector4f>& camerafxfycxcy, const std::vector<Eigen::Matrix3f>& Rs, const std::vector<Eigen::RowVector3f>& ts,Eigen::Vector3f&pt)
    {
        if (imgPts.size() < 2)
        {
            LOG_ERR_OUT << "imgPts.size()<2";
            return false;
        }
        if (imgPts.size() != camerafxfycxcy.size() || imgPts.size() != Rs.size() || imgPts.size() != ts.size())
        {
            LOG_ERR_OUT << "size not match";
            return false;
        }


        Eigen::MatrixXf A(2 * imgPts.size(), 4);

        {
            Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);

            int rank = svd.rank();
            int cols = A.cols();

            if (rank < cols) {
                // 存在非零解
                Eigen::VectorXf null_vector = svd.matrixV().col(cols - 1); // 取最后一个奇异向量

                std::cout << "null_vector: " << null_vector << std::endl;;
                // 归一化，方便查看
                null_vector.normalize();

                // 验证
                std::cout << "A * null_vector =  " << A * null_vector << std::endl;;
                std::cout << "范数: " << (A * null_vector).norm() << std::endl;;
                 
            }
            else {
                std::cout << "只有零解！\n"; 
            }
        }

        {
            Eigen::MatrixXd A(3, 4);
            A << 1, 2, 3, 4,
                2, 4, 6, 8,
                3, 6, 9, 12;

            std::cout << "矩阵 A:\n" << A << std::endl;

            // 方法1：使用SVD分解求零空间
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
            Eigen::MatrixXd V = svd.matrixV(); // 右奇异向量矩阵
            int rank = svd.rank(); // 矩阵A的秩

            // 零空间的基向量是 V 中后 (cols - rank) 列
            int cols = A.cols();
            Eigen::MatrixXd nullspace_basis = V.rightCols(cols - rank);

            std::cout << "\n零空间基向量（列向量形式）:\n"
                << nullspace_basis << std::endl;

            // 验证：A * nullspace_basis 应接近零矩阵
            std::cout << "\n验证 A * nullspace_basis:\n"
                << A * nullspace_basis << std::endl;
        }

        return true;
    }

}
int test_figureRTS()
{
    std::default_random_engine e;
    std::uniform_real_distribution<float> u(-1, 1);
    e.seed(time(0));

    float gt_scale = abs(u(e));
    Eigen::AngleAxisf r(u(e),Eigen::Vector3f(u(e), u(e), u(e)).normalized());
    Eigen::Matrix3f gt_R = r.matrix();
    Eigen::RowVector3f gt_t(u(e), u(e), u(e));
    LOG_OUT << "gt_R=" << gt_R;
    LOG_OUT << "gt_t=" << gt_t;
    LOG_OUT << "gt_scale=" << gt_scale;
    int test_cnt = 10;
    std::vector<Eigen::Vector3f>src(test_cnt);
    std::vector<Eigen::Vector3f>tar(test_cnt);
    for (int i = 0; i < test_cnt; i++)
    {
        src[i] = Eigen::Vector3f(u(e), u(e), u(e)); 
        tar[i] = (gt_R* src[i]+ gt_t.transpose())* gt_scale;
    }
    bool figureRt = true;
    bool figureS = true;
    float scale = 1;
    Eigen::Matrix3f R;
    Eigen::RowVector3f t;
    if (bfm::figureRTS(src,tar,R,t,scale, figureRt, figureS))
    {
        LOG_OUT << "R=" << R;
        LOG_OUT << "t=" << t;
        LOG_OUT << "scale=" << scale;
    }
    return 0;
}
int test_TriangulateMultiViewPoint()
{
    std::vector<Eigen::Vector2f> imgPts;  
    std::vector<Eigen::Vector4f> camerafxfycxcy;  
    std::vector<Eigen::Matrix3f> Rs; 
    std::vector<Eigen::RowVector3f> ts;
    //0000
    imgPts.emplace_back(548, 623);
    camerafxfycxcy.emplace_back(1202.4190359473357, 1202.4190359473357,360.0, 640.0);
    Rs.emplace_back(Eigen::Quaternionf(1.0, 0.0, 0.0, 0.0).matrix());
    ts.emplace_back(0.0, 0.0, 0.0);
    //0021
    imgPts.emplace_back(515, 595);
    camerafxfycxcy.emplace_back(1202.4190359473357, 1202.4190359473357, 360.0, 640.0);
    Rs.emplace_back(Eigen::Quaternionf(0.931089568209165, 0.017631975672877137, .36064097475036161, -0.051955911473586476).matrix());;
    ts.emplace_back(-7.1564130482621779, 0.6496792304965231, 2.7521496852791203);
    //0041
    imgPts.emplace_back(496, 604);
    camerafxfycxcy.emplace_back(1202.4190359473357, 1202.4190359473357, 360.0, 640.0);
    Rs.emplace_back(Eigen::Quaternionf(0.99158533075176203, 0.087306421322532773, 0.086367645847644531, -0.040948142625852543).matrix());
    ts.emplace_back(1.5932503066297834,
        1.7476895482295607,
        0.59391886160274243);
    Eigen::Vector3f  pt;
    bfm::figureSharedPoint(imgPts, camerafxfycxcy,Rs,ts,pt);
    return 0;
}
int test_bfm(void)
{  
 
    if (0)//test face marks
    {
        face::FaceDet faceDetIns;
        if (!faceDetIns.init())
        {
            return -1;
        };
        face::FaceMark FaceMarkIns;
        if (!FaceMarkIns.init())
        {
            return -1;
        };
        cv::Mat image = cv::imread("image.jpg");
        std::vector<cv::Rect> rects; 
        std::vector<std::vector<cv::Point2f>> faceLandmarks; 
        std::vector<float> scores;
        faceDetIns.detect(image, rects, faceLandmarks, scores);
        FaceMarkIns.extract(image, rects, faceLandmarks);
        {
            for (const auto&d: faceLandmarks)
            {
                for (const auto&d2:d)
                {
                    cv::circle(image,d2,3,cv::Scalar(255,255,255),-1);
                }
            }
        }
        LOG_OUT;
    }
    if (1)
    {
        return test_TriangulateMultiViewPoint();
        return test_figureRTS();
    }

    std::filesystem::path bfmFacePath = "../models/model2019_face12.h5";
    if (!std::filesystem::exists(bfmFacePath))
    {
        LOG_ERR_OUT << "need models/model2019_face12.h5";
        return -1;
    }
    bfm::Bfm2019 model(bfmFacePath); 
    cv::Mat face;
    cv::Mat color;
    for (int i = 0; i < 10; i++)
    {
        Eigen::MatrixX3f V;
        Eigen::MatrixX3f C;
        model.generateRandomFace(V, C);


        meshdraw::Mesh msh(V, model.F, C);
        Eigen::Matrix3f R;
        Eigen::RowVector3f t;
        R << 1, 0, 0, 0, -1, 0, 0, 0, -1;
        t << 0, 0, 300; 
        msh.rotate(R,t); 

        msh.figureFacesNomral();
        meshdraw::Camera cam = meshdraw::utils::generateDefaultCamera();
        cv::Mat rgbMat;
        cv::Mat vertexMap;
        cv::Mat mask;
        meshdraw::render(msh, cam, rgbMat, vertexMap, mask);
        meshdraw::utils::savePtsMat("a.txt", vertexMap, mask);

        model.saveObj("../surf/rand"+std::to_string(i)+".obj", msh.V, msh.C);
        return 0;
        //model.capture(V, C);
    }
    return 0; // successfully terminated
}

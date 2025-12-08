#include <unordered_map>
#include <filesystem>
#include <iostream>
#include <fstream>
#include "Eigen/Core"
#include "opencv2/opencv.hpp"
#include "igl/per_face_normals.h"
#include "igl/per_vertex_normals.h"
#include "meshDraw.h"
#include "log.h"
#include "igl/per_face_normals.h"
#include "igl/per_vertex_normals.h"
#include "igl/centroid.h"
#include "igl/barycenter.h"
namespace meshdraw
{

    template <typename T>
    bool isEmpty(const Eigen::MatrixBase<T>& mat) {
        if (mat.rows() == 0 || mat.cols() == 0) {
            return true;
        }
        return false;
    }
    bool readFromSimpleObj(const std::filesystem::path& path,
        Eigen::MatrixX3f& pts,
        Eigen::MatrixX3i& faces)
    {
        std::list<cv::Point3f>pts_;
        std::list<cv::Point3i>faces_;
        std::fstream fin(path, std::ios::in);
        std::string aline;
        std::string flag = "";
        while (std::getline(fin, aline))
        {
            if (aline.length() < 4)continue;
            if (aline[0] == '#' && aline[1] == ' ')continue;

            if (aline[0] == 'v' && aline[1] == ' ')
            {
                std::stringstream ss(aline);
                float x, y, z;
                ss >> flag >> x >> y >> z;
                pts_.emplace_back(x, y, z);
            }
            else if (aline[0] == 'f' && aline[1] == ' ')
            {
                std::stringstream ss(aline);
                int face_x, face_y, face_z;
                ss >> flag >> face_x >> face_y >> face_z;
                face_x -= 1; face_y -= 1; face_z -= 1;
                faces_.emplace_back(face_x, face_y, face_z);
            }
        }
        fin.close();
        if (pts_.size() == 0 || faces_.size() == 0)
        {
            LOG_ERR_OUT << "pts_.size=0 OR faces_.size=0: " << pts_.size() << "; " << faces_.size();
            return false;
        }

        pts = Eigen::MatrixX3f(pts_.size(), 3);
        faces = Eigen::MatrixX3i(faces_.size(), 3);
        int i = 0;
        for (const auto& d : pts_)
        {
            pts(i, 0) = d.x;
            pts(i, 1) = d.y;
            pts(i, 2) = d.z;
            i += 1;
        }
        i = 0;
        for (const auto& d : faces_)
        {
            faces(i, 0) = d.x;
            faces(i, 1) = d.y;
            faces(i, 2) = d.z;
            i += 1;
        }
        return true;
    }
    namespace utils
    {
        Eigen::Matrix3f generRotateMatrix(const Eigen::Vector3f& direct, const Eigen::Vector3f& upDirect)
        {
            Eigen::Vector3f right = upDirect.cross(direct);
            Eigen::Matrix3f ret;
            ret << right[0], upDirect[0], direct[0],
                right[1], upDirect[1], direct[1], 
                right[2], upDirect[2], direct[2];
            return ret;
        }
        bool saveFacePickedMesh(const std::filesystem::path& path, const Mesh& msh, const std::vector<bool>& faceValid)
        {
            if (isEmpty(msh.F) || isEmpty(msh.V))
            {
                LOG_ERR_OUT << "empty VF";
                return false;
            }
            if (msh.F.rows() != faceValid.size())
            {
                LOG_ERR_OUT << "msh.F.rows() != faceValid.size()";
                return false;
            }
            int newPtsIdx = 0;
            std::unordered_map<int, int>oldPtsIdxToNew;
            std::unordered_map<int, int>newPtsIdxToOld;
            std::list<Eigen::Vector3i> newFaces;
            for (int f = 0; f < faceValid.size(); f++)
            {
                if (faceValid[f])
                {
                    const int& a = msh.F(f, 0);
                    const int& b = msh.F(f, 1);
                    const int& c = msh.F(f, 2);
                    if (oldPtsIdxToNew.count(a) == 0)
                    {
                        oldPtsIdxToNew[a] = newPtsIdx;
                        newPtsIdxToOld[newPtsIdx] = a;
                        newPtsIdx += 1;
                    }
                    if (oldPtsIdxToNew.count(b) == 0)
                    {
                        oldPtsIdxToNew[b] = newPtsIdx;
                        newPtsIdxToOld[newPtsIdx] = b;
                        newPtsIdx += 1;
                    }
                    if (oldPtsIdxToNew.count(c) == 0)
                    {
                        oldPtsIdxToNew[c] = newPtsIdx;
                        newPtsIdxToOld[newPtsIdx] = b;
                        newPtsIdx += 1;
                    }
                    newFaces.emplace_back(oldPtsIdxToNew[a], oldPtsIdxToNew[b], oldPtsIdxToNew[c]);
                }
            }
            std::fstream fout(path, std::ios::out);
            for (int i =0;i< newPtsIdxToOld.size();i++)
            {
                fout << "v " << msh.V(newPtsIdxToOld[i], 0) << " " << msh.V(newPtsIdxToOld[i], 1) << " " << msh.V(newPtsIdxToOld[i], 2) << std::endl;
            }
            for (const auto&d: newFaces)
            {
                fout << "f " << d[0]+1 << " " << d[1] + 1<< " " << d[2] + 1 << std::endl;
            }
            fout.close();
            return true;
        }

        std::list<cv::Vec2i> triangle(const cv::Vec2i& p0, const cv::Vec2i& p1, const cv::Vec2i& p2) {
            std::list<cv::Vec2i> ret;
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
        std::list<std::pair<cv::Vec2i, Eigen::Vector3f>> triangle(const cv::Vec2i& p0, const cv::Vec2i& p1, const cv::Vec2i& p2,
            const Eigen::Vector3f& value0, const Eigen::Vector3f& value1, const Eigen::Vector3f& value2) { 
            std::list<std::pair<cv::Vec2i, Eigen::Vector3f>> ret;
            if (p0[1] == p1[1] && p0[1] == p2[1])
            {
                int xmin = 0;
                int xmax = 0;
                Eigen::Vector3f minValue, maxValue;
                if (p0[0] >= p1[0] && p0[0] >= p1[0])
                {
                    xmax = p0[0];
                    maxValue = value0;
                }
                if (p1[0] >= p0[0] && p1[0] >= p2[0])
                {
                    xmax = p1[0];
                    maxValue = value1;
                }
                if (p2[0] >= p1[0] && p2[0] >= p0[0])
                {
                    xmax = p2[0];
                    maxValue = value2;
                }
                if (p0[0] <= p1[0] && p0[0] <= p1[0])
                {
                    xmin = p0[0];
                    minValue = value0;
                }
                if (p1[0] <= p0[0] && p1[0] <= p2[0])
                {
                    xmin = p1[0];
                    minValue = value1;
                }
                if (p2[0] <= p1[0] && p2[0] <= p0[0])
                {
                    xmin = p2[0];
                    minValue = value2;
                }
                int cnt = xmax - xmin + 1;
                Eigen::Vector3f each = (maxValue - minValue) / cnt;
                for (int i = xmin; i <= xmax; i++)
                {
                    ret.emplace_back(std::make_pair(cv::Vec2i{ i, p0[1] }, minValue + (i - xmin) * each));
                }
                return ret;
            }

            Eigen::Matrix3f A;
            Eigen::Vector3f b(1,1,1);
            A << p0[0], p1[0], p2[0], 
                p0[1], p1[1], p2[1], 
                1,1,1;
            Eigen::Matrix3f  A_1 = A.inverse();

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
                    b[0] = j;
                    b[1] = t0[1] + i;
                    Eigen::Vector3f x = A_1*(b);
                    Eigen::Vector3f v = x[0] * value0 + x[1] * value1 + x[2] * value2;
                    ret.emplace_back(std::make_pair(cv::Vec2i{ j, t0[1] + i }, v));
                }
            }
            return ret;
        }
    }

    Mesh::Mesh() {}
    Mesh::Mesh(const Eigen::MatrixX3f V_, const Eigen::MatrixX3i& F_, const Eigen::MatrixX3f& C_)
    {
        V = V_;
        F = F_;
        C = C_;
    }
    bool Mesh::figurePtsNomral()
    {
        if (isEmpty(F) || isEmpty(V))
        {
            LOG_ERR_OUT << "empty VF";
            return false;
        }
        igl::per_vertex_normals(V,F,ptsNormal);
        return true;
    }
    bool Mesh::figureFacesNomral()
    {
        if (isEmpty(F) || isEmpty(V))
        {
            LOG_ERR_OUT << "empty VF";
            return false;
        }
        igl::per_face_normals(V, F, facesNormal);
        return true;
    }

    bool render(const Mesh& msh, const Camera& cam, const RenderType& renderTpye)
    {
        if (renderTpye == RenderType::vertexColor)
        {
            if (isEmpty(msh.F) || isEmpty(msh.V) || isEmpty(msh.C) || isEmpty(msh.facesNormal))
            {
                LOG_ERR_OUT << "empty VFC";
                return false;
            }
            if (isEmpty(cam.R))
            {
                LOG_ERR_OUT << "empty R";
                return false;
            }
            if (cam.cameraType== CmaeraType::Pinhole && isEmpty(cam.t))
            {
                LOG_ERR_OUT << "empty t";
                return false;
            }
            Eigen::Matrix3f R_T = cam.R.transpose();
            Eigen::MatrixX3i ptsInPic;
            std::vector<bool>renderFaces(msh.F.rows(), false);
            Eigen::MatrixX3i colorInt= (msh.C*255.f).cast<int>();
            if (cam.cameraType == CmaeraType::Pinhole)
            {
                Eigen::MatrixX3f ptsInCam = (msh.V * R_T).rowwise() + cam.t;
                Eigen::MatrixX3f ptsInPicFloat = (ptsInCam.array().colwise() / ptsInCam.col(2).array()).matrix().eval();       
                Eigen::Matrix3f intr_T = cam.intr.transpose();    
                ptsInPicFloat = ptsInPicFloat * intr_T;
                ptsInPic = ptsInPicFloat.cast<int>();
                Eigen::MatrixX3f barycenter;
                igl::barycenter(msh.V, msh.F, barycenter);
                Eigen::MatrixX3f viewFaceDir = (barycenter.rowwise() - cam.t);// .rowwise().norm();
                Eigen::VectorXf dots = viewFaceDir.cwiseProduct(msh.facesNormal).rowwise().sum();
                cv::Mat drawMat = cv::Mat::zeros(cam.height, cam.width, CV_8UC3);
                cv::Mat drawMatDist = cv::Mat::zeros(cam.height, cam.width, CV_32FC1);
                cv::Mat drawMask = cv::Mat::zeros(cam.height, cam.width, CV_8UC1);
                cv::Mat ptsMat = cv::Mat::zeros(cam.height, cam.width, CV_32FC3);
                for (int f = 0; f < dots.size(); f++)
                {
                    if (dots[f]<0)
                    {
                        renderFaces[f] = true;
                        const int& fa = msh.F(f, 0);
                        const int& fb = msh.F(f, 1);
                        const int& fc = msh.F(f, 2);
                        float distFromCam = (barycenter.row(f)-cam.t).norm(); 
                        std::list<std::pair<cv::Vec2i, Eigen::Vector3f>>trianglePixels = utils::triangle({ ptsInPic(fa,0),ptsInPic(fa,1) }, { ptsInPic(fb,0),ptsInPic(fb,1) }, { ptsInPic(fc,0),ptsInPic(fc,1) }, msh.V.row(fa), msh.V.row(fb), msh.V.row(fc));
                        for (const auto&d: trianglePixels)
                        {
                            const cv::Vec2i& pixel = d.first;
                            const Eigen::Vector3f& value = d.second;
                            const int& r = pixel[1];
                            const int& c = pixel[0];
                            if (c >= 0 && r >= 0 && c < cam.width && r < cam.height)
                            {
                                if (drawMask.ptr<uchar>(r)[c]==0)
                                {
                                    drawMask.ptr<uchar>(r)[c] = 1;
                                    drawMatDist.ptr<float>(r)[c] = distFromCam;
                                    drawMat.at<cv::Vec3b>(r, c)[0] = colorInt(fa, 2);
                                    drawMat.at<cv::Vec3b>(r, c)[1] = colorInt(fa, 1);
                                    drawMat.at<cv::Vec3b>(r, c)[2] = colorInt(fa, 0);
                                    ptsMat.at<cv::Vec3f>(r, c)[0] = value[0];
                                    ptsMat.at<cv::Vec3f>(r, c)[1] = value[1];
                                    ptsMat.at<cv::Vec3f>(r, c)[2] = value[2];
                                    
                                }
                                else if (drawMatDist.ptr<float>(r)[c]> distFromCam)
                                {
                                    drawMatDist.ptr<float>(r)[c] = distFromCam;
                                    drawMat.at<cv::Vec3b>(r, c)[0] = colorInt(fa, 2);
                                    drawMat.at<cv::Vec3b>(r, c)[1] = colorInt(fa, 1);
                                    drawMat.at<cv::Vec3b>(r, c)[2] = colorInt(fa, 0);
                                    ptsMat.at<cv::Vec3f>(r, c)[0] = value[0];
                                    ptsMat.at<cv::Vec3f>(r, c)[1] = value[1];
                                    ptsMat.at<cv::Vec3f>(r, c)[2] = value[2];
                                } 
                            }
                        }
                    }
                }
                utils::saveFacePickedMesh("a.obj",msh, renderFaces);        
            }
            else if(cam.cameraType == CmaeraType::Ortho)
            {
                ptsInPic = (msh.V * R_T).cast<int>();
            }
            else
            {
                LOG_ERR_OUT << "not supported.";
                return false;
            } 
        }
        else
        {
            LOG_ERR_OUT << "not supported.";
            return false;
        }
        return true;
    }
}


int test_draw()
{
	return 0;
}
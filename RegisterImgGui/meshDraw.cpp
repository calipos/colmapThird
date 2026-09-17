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
        meshdraw::Camera generateBfmDefaultCamera()
        {
            meshdraw::Camera cam;
            cam.R = meshdraw::utils::generRotateMatrix({ 0,0,-1 }, { 0,-1,0 });
            cam.t = Eigen::RowVector3f(0, 0, 300);
            cam.intr = Eigen::Matrix3f::Identity();
            cam.intr(0, 0) = 1200;
            cam.intr(1, 1) = 1200;
            cam.intr(0, 2) = 800;
            cam.intr(1, 2) = 600;
            cam.height = 1200;
            cam.width = 1600;
            return cam;
        }
        meshdraw::Camera generateDefaultCamera()
        {
            meshdraw::Camera cam;
            cam.R = Eigen::Matrix3f::Identity();
            cam.t = Eigen::RowVector3f(0, 0, 0);
            cam.intr = Eigen::Matrix3f::Identity();
            cam.intr(0, 0) = 1200;
            cam.intr(1, 1) = 1200;
            cam.intr(0, 2) = 800;
            cam.intr(1, 2) = 600;
            cam.height = 1200;
            cam.width = 1600;
            return cam;
        }
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
        bool savePtsMat(const std::filesystem::path& path, const cv::Mat& ptsMat, const cv::Mat& mask)
        {
            if (ptsMat.size()!=mask.size())
            {
                LOG_ERR_OUT << "ptsMat.size()!=mask.size()";
                return false;
            }
            std::fstream fout(path,std::ios::out);
            for (int r = 0; r < ptsMat.rows; r++)
            {
                for (int c = 0; c < ptsMat.cols; c++)
                {
                    if (mask.ptr<uchar>(r)[c]>0)
                    {
                        fout << ptsMat.at<cv::Vec3f>(r, c)[0] << " " << ptsMat.at<cv::Vec3f>(r, c)[1] << " " << ptsMat.at<cv::Vec3f>(r, c)[2] << std::endl;
                    }
                }
            }
            fout.close();
            return true;
        }
        bool savePts(const std::filesystem::path& path, const Eigen::MatrixX3f& ptsMat)
        {
            std::fstream fout(path, std::ios::out);
            for (int i = 0; i < ptsMat.rows(); i++)
            {
                fout << ptsMat(i, 0) << " " << ptsMat(i, 1) << " " << ptsMat(i, 2) << std::endl;
            }
            fout.close();
            return true;
        }
        std::list<std::pair<cv::Vec2i, Eigen::Vector3f>> line(const cv::Vec2i& p0, const cv::Vec2i& p1,
            const Eigen::Vector3f& value0, const Eigen::Vector3f& value1)
        {
            if (p0[0] == p1[0] && p0[1] == p1[1])
            {
                return std::list<std::pair<cv::Vec2i, Eigen::Vector3f>>{ {p0, value0}};
            }
            std::list<std::pair<cv::Vec2i, Eigen::Vector3f>> ret;
            cv::Vec2f d = p1 - p0;
            int cnt = cv::norm(d) + 1;
            d /= cv::norm(d);
            Eigen::Vector3f each = (value1 - value0) / cnt;
            for (int i = 0; i <= cnt; i++)
            {
                cv::Vec2i pixel;
                pixel[0] = p0[0] + i * d[0];
                pixel[1] = p0[1] + i * d[1];
                ret.emplace_back(std::make_pair(pixel, value0 + i * each));
            }
            return ret;
        }

        std::vector<cv::Vec2i>getLinePixel(const cv::Vec2i& p0, const cv::Vec2i& p1)
        {
            int diffx01 = p0[0] - p1[0];
            int diffy01 = p0[1] - p1[1];
            if (diffx01==0 && diffy01 ==0)
            {
                return std::vector<cv::Vec2i>{p0};
            }
            if (abs(diffx01)>=abs(diffy01))
            {
                float k = (float)diffy01 / diffx01;
                int minx = p0[0] < p1[0] ? p0[0] : p1[0];
                int miny = p0[0] < p1[0] ? p0[1] : p1[1];
                int maxx = p0[0] < p1[0] ? p1[0] : p0[0];
                std::vector<cv::Vec2i>ret;
                ret.reserve(abs(diffx01)+1);
                for (int x = minx; x <= maxx; x++)
                {
                    ret.emplace_back(x, k * (x - minx) + miny+0.5);
                }
                return ret;
            }
            else
            {
                float k = (float)diffx01 / diffy01;
                int minx = p0[1] < p1[1] ? p0[0] : p1[0];
                int miny = p0[1] < p1[1] ? p0[1] : p1[1];
                int maxy = p0[1] < p1[1] ? p1[1] : p0[1];
                std::vector<cv::Vec2i>ret;
                ret.reserve(abs(diffx01) + 1);
                for (int y = miny; y <= maxy; y++)
                {
                    ret.emplace_back(k * (y - miny) + minx+0.5,y);
                }
                return ret;
            }
        }
        std::list<cv::Vec2i> triangle(const cv::Vec2i& p0, const cv::Vec2i& p1, const cv::Vec2i& p2,std::list<Eigen::Vector3f>&weights) {
            std::list<cv::Vec2i> ret;
            cv::Vec2i d1 = p2 - p0;
            cv::Vec2i d2 = p1 - p0;
            if (d1[0] == 0 && d1[1] == 0 && d2[0] == 0 && d2[1] == 0)
            {
                ret.emplace_back(p0);
                weights.emplace_back(1,0,0);
                return ret;
            }
            int temp_a = d1[0] * d2[1];
            int temp_b = d1[1] * d2[0];
            int xmin = (std::min)((std::min)(p0[0], p1[0]), p2[0]);
            int xmax = (std::max)((std::max)(p0[0], p1[0]), p2[0]);
            int ymin = (std::min)((std::min)(p0[1], p1[1]), p2[1]);
            int ymax = (std::max)((std::max)(p0[1], p1[1]), p2[1]);
            if (temp_a == temp_b)
            {
                int widthDiff = xmax - xmin;
                int heightDiff = ymax - ymin;
                if (widthDiff>= heightDiff)
                {
                    float k = (float)heightDiff / widthDiff;
                    for (int i = 0; i <= widthDiff; i++)
                    {
                        ret.emplace_back(xmin + i, ymin + k * i);
                    }
                }
                else
                {
                    float k = (float)widthDiff / heightDiff;
                    for (int i = 0; i <= widthDiff; i++)
                    {
                        ret.emplace_back(xmin + k * i, ymin + i);
                    } 
                }
                
                for (const auto&d: ret)
                {
                    float dists[3] = {
                     abs(d[0] - p0[0]),
                     abs(d[0] - p1[0]),
                     abs(d[0] - p2[0]) };
                    if (heightDiff> widthDiff)
                    {
                        dists[0] = abs(d[1] - p0[1]);
                        dists[1] = abs(d[1] - p1[1]);
                        dists[2] = abs(d[1] - p2[1]);
                    }
                    int maxDistIdx = 0;
                    if (dists[0] >= dists[1] && dists[0] >= dists[2] )
                    {

                    }
                    else if (dists[1] >= dists[0] && dists[1] >= dists[2])
                    {
                        maxDistIdx = 1;
                    }
                    else
                    {
                        maxDistIdx = 2;
                    }
                    float distSum = dists[0] + dists[1] + dists[2] - dists[maxDistIdx];
                    dists[maxDistIdx] = distSum;
                    dists[0] = distSum - dists[0];
                    dists[1] = distSum - dists[1];
                    dists[2] = distSum - dists[2];
                    distSum = 1. / distSum;
                    dists[0] *= distSum;
                    dists[1] *= distSum;
                    dists[2] *= distSum;
                    weights.emplace_back(dists[0], dists[1], dists[2]);
                } 
                return ret;
            }
            Eigen::Matrix3f A;
            Eigen::Vector3f b(1, 1, 1);
            A << p0[0], p1[0], p2[0],
                p0[1], p1[1], p2[1],
                1, 1, 1;
            Eigen::Matrix3f  A_1 = A.inverse();

             
            std::unordered_map<int, int>y_maxx;
            std::unordered_map<int, int>y_minx;

            std::vector<cv::Vec2i> linep01 = getLinePixel(p0, p1);
            std::vector<cv::Vec2i> linep02 = getLinePixel(p0, p2);
            std::vector<cv::Vec2i> linep12 = getLinePixel(p2, p1);

            for (const auto&d: linep01)
            {
                const auto& y = d[1];
                if (y_maxx.count(y)==0)
                {
                    y_minx[y] = std::numeric_limits<int>::max();
                    y_maxx[y] = -std::numeric_limits<int>::max();
                }
                if (d[0] > y_maxx[y])y_maxx[y] = d[0];
                if (d[0] < y_minx[y])y_minx[y] = d[0];
            }
            for (const auto& d : linep02)
            {
                const auto& y = d[1];
                if (y_maxx.count(y) == 0)
                {
                    y_minx[y] = std::numeric_limits<int>::max();
                    y_maxx[y] = -std::numeric_limits<int>::max();
                }
                if (d[0] > y_maxx[y])y_maxx[y] = d[0];
                if (d[0] < y_minx[y])y_minx[y] = d[0];
            }
            for (const auto& d : linep12)
            {
                const auto& y = d[1];
                if (y_maxx.count(y) == 0)
                {
                    y_minx[y] = std::numeric_limits<int>::max();
                    y_maxx[y] = -std::numeric_limits<int>::max();
                }
                if (d[0] > y_maxx[y])y_maxx[y] = d[0];
                if (d[0] < y_minx[y])y_minx[y] = d[0];
            }
             
            for (const auto&d: y_maxx)
            { 
                const auto&y = d.first;
                for (int x = y_minx[y]; x <= d.second; x++)
                {
                    ret.emplace_back(x, y);
                }
            }

             
            for (const auto&d: ret) {
                    b[0] = d[0];
                    b[1] = d[1];
                    Eigen::Vector3f x = A_1 * (b);
                    weights.emplace_back(x);
                
            }
            return ret;
        }


        std::list<std::pair<cv::Vec2i, Eigen::Vector3f>> triangle(const cv::Vec2i& p0, const cv::Vec2i& p1, const cv::Vec2i& p2,
            const Eigen::Vector3f& value0, const Eigen::Vector3f& value1, const Eigen::Vector3f& value2) { 
            std::list<std::pair<cv::Vec2i, Eigen::Vector3f>> ret;
            {
                cv::Vec2i d1 = p2 - p0;
                cv::Vec2i d2 = p1 - p0;
                int a = d1[0] * d2[1];
                int b = d1[1] * d2[0];
                if (a == b || a + b == 0)
                {
                    int maxx = (std::max)((std::max)(p0[0], p1[0]), p2[0]);
                    int maxy = (std::max)((std::max)(p0[1], p1[1]), p2[1]);
                    int minx = (std::min)((std::min)(p0[0], p1[0]), p2[0]);
                    int miny = (std::min)((std::min)(p0[1], p1[1]), p2[1]);
                    cv::Vec2i pa(minx, miny);
                    cv::Vec2i pb(maxx, maxy);
                    Eigen::Vector3f va, vb;
                    if (p0[0] == minx && p0[1] == miny)
                    {
                        va = value0;
                        if (p1[0] == maxx && p1[1] == maxy)
                        {
                            vb = value1; 
                        }
                        else
                        {
                            vb = value2;
                        }
                    }
                    else if (p1[0] == minx && p1[1] == miny)
                    {
                        va = value1;
                        if (p0[0] == maxx && p0[1] == maxy)
                        {
                            vb = value0;
                        }
                        else
                        {
                            vb = value2;
                        }
                    }
                    else if (p2[0] == minx && p2[1] == miny)
                    {
                        va = value2;
                        if (p0[0] == maxx && p0[1] == maxy)
                        {
                            vb = value0;
                        }
                        else
                        {
                            vb = value1;
                        }
                    }
                    else if (p0[0] == maxx && p0[1] == miny)
                    {
                        std::swap(pa[0], pb[0]);
                        va = value0;
                        if (p1[0] == minx && p1[1] == maxy)
                        {
                            vb = value1;
                        }
                        else
                        {
                            vb = value2;
                        }
                    }
                    else if (p1[0] == maxx && p1[1] == miny)
                    {
                        std::swap(pa[0], pb[0]);
                        va = value1;
                        if (p0[0] == minx && p0[1] == maxy)
                        {
                            vb = value0;
                        }
                        else
                        {
                            vb = value2;
                        }
                    }
                    else
                    {
                        std::swap(pa[0], pb[0]);
                        va = value2;
                        if (p0[0] == minx && p0[1] == maxy)
                        {
                            vb = value0;
                        }
                        else
                        {
                            vb = value1;
                        }
                    }
                    return line(pa, pb, va, vb);
                }
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
    bool Mesh::rotate(const Eigen::Matrix3f& R, const Eigen::RowVector3f& t,const float&scale)
    {
        if (isEmpty(V))
        {
            LOG_ERR_OUT << "empty V";
            return false;
        }
        Eigen::Matrix3f R_inv = R.transpose();
        V =(V * R_inv* scale).rowwise() + t;
        return true;
    }
    bool render(const Mesh& msh, const Camera& cam, const Eigen::Matrix3f& R, const Eigen::RowVector3f& t, const float& scale, cv::Mat& rgbMat, cv::Mat& vertexMap, cv::Mat& mask, const RenderType& renderTpye)
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
            if (cam.cameraType == CmaeraType::Pinhole && isEmpty(cam.t))
            {
                LOG_ERR_OUT << "empty t";
                return false;
            }
            Eigen::Matrix3f R_T = cam.R.transpose();
            Eigen::Vector3f cameraLookDir(R_T(2, 0), R_T(2, 1), R_T(2, 2));
            Eigen::MatrixX3i ptsInPic;
            Eigen::MatrixX3i colorInt = (msh.C * 255.f).cast<int>();
            Eigen::Matrix3f R_inv = R.transpose();
            Eigen::MatrixX3f meshVRotated = (msh.V * R_inv * scale).rowwise() + t;
            Eigen::MatrixX3f meshFacesNormalInCam = msh.facesNormal * R;
            Eigen::VectorXf dots = msh.facesNormal * cameraLookDir;
            if (cam.cameraType == CmaeraType::Pinhole)
            {
                Eigen::MatrixX3f ptsInCam = (meshVRotated * R_T).rowwise() + cam.t;
                Eigen::MatrixX3f ptsInPicFloat = (ptsInCam.array().colwise() / ptsInCam.col(2).array()).matrix().eval();
                Eigen::Matrix3f intr_T = cam.intr.transpose();
                ptsInPicFloat = ptsInPicFloat * intr_T;
                ptsInPic = ptsInPicFloat.cast<int>();
                Eigen::MatrixX3f barycenter;
                igl::barycenter(meshVRotated, msh.F, barycenter);
                Eigen::MatrixX3f viewFaceDir = (barycenter.rowwise() - cam.t);// .rowwise().norm();
                Eigen::VectorXf distFromCams = viewFaceDir.rowwise().norm(); 
                rgbMat = cv::Mat::zeros(cam.height, cam.width, CV_8UC3);
                cv::Mat drawMatDist = cv::Mat::zeros(cam.height, cam.width, CV_32FC1);
                mask = cv::Mat::zeros(cam.height, cam.width, CV_8UC1);
                vertexMap = cv::Mat::zeros(cam.height, cam.width, CV_32FC3);
                std::vector<bool>ptsInCanvas(msh.V.rows(), true);
#pragma omp parallel for  schedule(dynamic)
                for (int i = 0; i < ptsInCanvas.size(); i++)
                {
                    if (ptsInPic(i, 0) < 0 || ptsInPic(i, 1) < 0 || ptsInPic(i, 0) >= cam.width || ptsInPic(i, 1) >= cam.height)
                    {
                        ptsInCanvas[i] = false;
                    }
                }
                for (int f = 0; f < dots.size(); f++)
                {
                    if (dots[f]> 0)
                    {
                        const int& fa = msh.F(f, 0);
                        const int& fb = msh.F(f, 1);
                        const int& fc = msh.F(f, 2);
                        if (!ptsInCanvas[fa] || !ptsInCanvas[fb] || !ptsInCanvas[fc])
                        {
                            continue;
                        }
                        float distFromCam = distFromCams[f];// (barycenter.row(f) - cam.t).norm();
                        std::list<Eigen::Vector3f>weights;
                        std::list<cv::Vec2i>trianglePixels = utils::triangle({ ptsInPic(fa,0),ptsInPic(fa,1) }, { ptsInPic(fb,0),ptsInPic(fb,1) }, { ptsInPic(fc,0),ptsInPic(fc,1) }, weights);


                        auto itA = trianglePixels.begin();
                        auto itB = weights.begin();
                        int cnt = trianglePixels.size();
                        for (int i = 0; i < cnt; ++i, ++itA, ++itB) {
                            const cv::Vec2i& pixel = *itA;
                            if (pixel[1] > 0 && pixel[1] <= cam.height && pixel[0] >= 0 && pixel[0] < cam.width)
                            {
                                const float& w0 = (*itB)[0];
                                const float& w1 = (*itB)[1];
                                const float& w2 = (*itB)[2];
                                const int& r = cam.height - pixel[1];
                                const int& c = pixel[0];
                                if (c >= 0 && r >= 0 && c < cam.width && r < cam.height)
                                {
                                    if (mask.ptr<uchar>(r)[c] == 0)
                                    {
                                        mask.ptr<uchar>(r)[c] = 1;
                                        drawMatDist.ptr<float>(r)[c] = distFromCam;
                                        rgbMat.at<cv::Vec3b>(r, c)[0] = w0 * colorInt(fa, 2) + w1 * colorInt(fb, 2) + w2 * colorInt(fc, 2);
                                        rgbMat.at<cv::Vec3b>(r, c)[1] = w0 * colorInt(fa, 1) + w1 * colorInt(fb, 1) + w2 * colorInt(fc, 1);
                                        rgbMat.at<cv::Vec3b>(r, c)[2] = w0 * colorInt(fa, 0) + w1 * colorInt(fb, 0) + w2 * colorInt(fc, 0);
                                        vertexMap.at<cv::Vec3f>(r, c)[0] = w0 * msh.V(fa, 0) + w1 * msh.V(fb, 0) + w2 * msh.V(fc, 0);
                                        vertexMap.at<cv::Vec3f>(r, c)[1] = w0 * msh.V(fa, 1) + w1 * msh.V(fb, 1) + w2 * msh.V(fc, 1);
                                        vertexMap.at<cv::Vec3f>(r, c)[2] = w0 * msh.V(fa, 2) + w1 * msh.V(fb, 2) + w2 * msh.V(fc, 2);
                                    }
                                    else if (drawMatDist.ptr<float>(r)[c] > distFromCam)
                                    {
                                        drawMatDist.ptr<float>(r)[c] = distFromCam;
                                        rgbMat.at<cv::Vec3b>(r, c)[0] = w0 * colorInt(fa, 2) + w1 * colorInt(fb, 2) + w2 * colorInt(fc, 2);
                                        rgbMat.at<cv::Vec3b>(r, c)[1] = w0 * colorInt(fa, 1) + w1 * colorInt(fb, 1) + w2 * colorInt(fc, 1);
                                        rgbMat.at<cv::Vec3b>(r, c)[2] = w0 * colorInt(fa, 0) + w1 * colorInt(fb, 0) + w2 * colorInt(fc, 0);
                                        vertexMap.at<cv::Vec3f>(r, c)[0] = w0 * msh.V(fa, 0) + w1 * msh.V(fb, 0) + w2 * msh.V(fc, 0);
                                        vertexMap.at<cv::Vec3f>(r, c)[1] = w0 * msh.V(fa, 1) + w1 * msh.V(fb, 1) + w2 * msh.V(fc, 1);
                                        vertexMap.at<cv::Vec3f>(r, c)[2] = w0 * msh.V(fa, 2) + w1 * msh.V(fb, 2) + w2 * msh.V(fc, 2);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else if (cam.cameraType == CmaeraType::Ortho)
            {
                Eigen::Matrix3f intr = cam.intr;
                int imgWidth = static_cast<int>(intr(0, 2) * 2);
                int imgHeight = static_cast<int>(intr(1, 2) * 2);
                Eigen::MatrixX3f ptsFloat = msh.V * R_T; 
                Eigen::RowVector3f colMin = ptsFloat.colwise().minCoeff();
                Eigen::RowVector3f colMax = ptsFloat.colwise().maxCoeff(); 
                float minx = colMin[0];
                float miny = colMin[1];
                float maxx = colMax[0];
                float maxy = colMax[1];
                float scaleHeight = imgHeight / (maxy - miny);
                float scaleWidth = imgWidth / (maxx - minx);
                float scale = (std::min)(scaleHeight, scaleWidth);
                ptsInPic = ((msh.V.rowwise() - colMin)* scale).cast<int>();
                rgbMat = cv::Mat::zeros(imgHeight, imgWidth, CV_8UC3);
                mask = cv::Mat::zeros(imgHeight, imgWidth, CV_8UC1);
                vertexMap = cv::Mat::zeros(imgHeight, imgWidth, CV_32FC3);
                cv::Mat depthMat = cv::Mat::ones(imgHeight, imgWidth, CV_32FC1)*std::numeric_limits<float>::max();
                for (int f = 0; f < msh.F.rows(); f++)
                {
                    if (dots[f] > -0.)
                    {
                        const int& fa = msh.F(f, 0);
                        const int& fb = msh.F(f, 1);
                        const int& fc = msh.F(f, 2);
                        std::list<Eigen::Vector3f>weights;
                        std::list<cv::Vec2i>trianglePixels = utils::triangle({ ptsInPic(fa,0),ptsInPic(fa,1) }, { ptsInPic(fb,0),ptsInPic(fb,1) }, { ptsInPic(fc,0),ptsInPic(fc,1) }, weights);
                        auto itA = trianglePixels.begin();
                        auto itB = weights.begin();
                        int cnt = trianglePixels.size();
                        for (int i = 0; i < cnt; ++i, ++itA, ++itB) {
                            const cv::Vec2i& xy = *itA;
                            const float& w0 = (*itB)[0];
                            const float& w1 = (*itB)[1];
                            const float& w2 = (*itB)[2];
                            if (xy[1] > 0 && xy[1] <= imgHeight && xy[0] >= 0 && xy[0] < imgWidth)
                            {
                                const int& r = imgHeight - xy[1];
                                const int& c = xy[0];
                                float depth = w0 * ptsFloat(fa, 2) + w1 * ptsFloat(fb, 2) + w2 * ptsFloat(fc, 2);
                                if (mask.ptr<uchar>(r)[c] == 0 || depth > depthMat.ptr<float>(r)[c])
                                {
                                    depthMat.ptr<float>(r)[c] = depth;
                                    rgbMat.at<cv::Vec3b>(r, c)[0] = w0 * colorInt(fa, 2) + w1 * colorInt(fb, 2) + w2 * colorInt(fc, 2);
                                    rgbMat.at<cv::Vec3b>(r, c)[1] = w0 * colorInt(fa, 1) + w1 * colorInt(fb, 1) + w2 * colorInt(fc, 1);
                                    rgbMat.at<cv::Vec3b>(r, c)[2] = w0 * colorInt(fa, 0) + w1 * colorInt(fb, 0) + w2 * colorInt(fc, 0);
                                    vertexMap.at<cv::Vec3f>(r, c)[0] = w0 * msh.V(fa, 0) + w1 * msh.V(fb, 0) + w2 * msh.V(fc, 0);
                                    vertexMap.at<cv::Vec3f>(r, c)[1] = w0 * msh.V(fa, 1) + w1 * msh.V(fb, 1) + w2 * msh.V(fc, 1);
                                    vertexMap.at<cv::Vec3f>(r, c)[2] = w0 * msh.V(fa, 2) + w1 * msh.V(fb, 2) + w2 * msh.V(fc, 2);
                                }
                                mask.ptr<uchar>(r)[c] = 1;
                            }
                        }                         
                    }
                }
                LOG_OUT;
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
    bool render(const Mesh& msh, const Camera& cam, const Eigen::Matrix3f& R, const Eigen::RowVector3f& t, const float& scale, cv::Mat& vertexMap, cv::Mat& mask)
    {
        if (isEmpty(msh.F) || isEmpty(msh.V) || isEmpty(msh.facesNormal))
        {
            LOG_ERR_OUT << "empty VF";
            return false;
        }
        if (isEmpty(cam.R))
        {
            LOG_ERR_OUT << "empty R";
            return false;
        }
        if (cam.cameraType == CmaeraType::Pinhole && isEmpty(cam.t))
        {
            LOG_ERR_OUT << "empty t";
            return false;
        }
        Eigen::Matrix3f R_T = cam.R.transpose();
        Eigen::MatrixX3i ptsInPic;
        Eigen::MatrixX3i colorInt = (msh.C * 255.f).cast<int>();
        Eigen::Matrix3f R_inv = R.transpose();
        Eigen::MatrixX3f meshVRotated = (msh.V * R_inv * scale).rowwise() + t;
        Eigen::MatrixX3f meshFacesNormalInCam = msh.facesNormal * R;
        if (cam.cameraType == CmaeraType::Pinhole)
        {
            Eigen::MatrixX3f ptsInCam = (meshVRotated * R_T).rowwise() + cam.t;
            Eigen::MatrixX3f ptsInPicFloat = (ptsInCam.array().colwise() / ptsInCam.col(2).array()).matrix().eval();
            Eigen::Matrix3f intr_T = cam.intr.transpose();
            ptsInPicFloat = ptsInPicFloat * intr_T;
            ptsInPic = ptsInPicFloat.cast<int>();
            Eigen::MatrixX3f barycenter;
            igl::barycenter(meshVRotated, msh.F, barycenter);
            Eigen::MatrixX3f viewFaceDir = (barycenter.rowwise() - cam.t);// .rowwise().norm();
            Eigen::VectorXf faceDists = viewFaceDir.rowwise().norm();
            Eigen::Index minIndex;
            faceDists.minCoeff(&minIndex);
            float nearestFaceDot = viewFaceDir.row(minIndex).dot(meshFacesNormalInCam.row(minIndex));
            if (nearestFaceDot > 0)
            {
                nearestFaceDot = 1;
            }
            else
            {
                nearestFaceDot = -1;
            }

            Eigen::VectorXf dots = viewFaceDir.cwiseProduct(meshFacesNormalInCam).rowwise().sum();
            cv::Mat drawMatDist = cv::Mat::zeros(cam.height, cam.width, CV_32FC1);
            mask = cv::Mat::zeros(cam.height, cam.width, CV_8UC1);
            vertexMap = cv::Mat::zeros(cam.height, cam.width, CV_32FC3);
            std::vector<bool>ptsInCanvas(msh.V.rows(), true);
#pragma omp parallel for  schedule(dynamic)
            for (int i = 0; i < ptsInCanvas.size(); i++)
            {
                if (ptsInPic(i, 0) < 0 || ptsInPic(i, 1) < 0 || ptsInPic(i, 0) >= cam.width || ptsInPic(i, 1) >= cam.height)
                {
                    ptsInCanvas[i] = false;
                }
            }
            for (int f = 0; f < dots.size(); f++)
            {
                if (dots[f] * nearestFaceDot > 0)
                {
                    const int& fa = msh.F(f, 0);
                    const int& fb = msh.F(f, 1);
                    const int& fc = msh.F(f, 2);
                    if (!ptsInCanvas[fa] || !ptsInCanvas[fb] || !ptsInCanvas[fc])
                    {
                        continue;
                    }
                    float distFromCam = faceDists[f];
                    std::list<std::pair<cv::Vec2i, Eigen::Vector3f>>trianglePixels = utils::triangle({ ptsInPic(fa,0),ptsInPic(fa,1) }, { ptsInPic(fb,0),ptsInPic(fb,1) }, { ptsInPic(fc,0),ptsInPic(fc,1) }, msh.V.row(fa), msh.V.row(fb), msh.V.row(fc));
                    for (const auto& d : trianglePixels)
                    {
                        const cv::Vec2i& pixel = d.first;
                        const Eigen::Vector3f& value = d.second;
                        const int& r = pixel[1];
                        const int& c = pixel[0];
                        if (c >= 0 && r >= 0 && c < cam.width && r < cam.height)
                        {
                            if (mask.ptr<uchar>(r)[c] == 0)
                            {
                                mask.ptr<uchar>(r)[c] = 1;
                                drawMatDist.ptr<float>(r)[c] = distFromCam;
                                vertexMap.at<cv::Vec3f>(r, c)[0] = value[0];
                                vertexMap.at<cv::Vec3f>(r, c)[1] = value[1];
                                vertexMap.at<cv::Vec3f>(r, c)[2] = value[2];

                            }
                            else if (drawMatDist.ptr<float>(r)[c] > distFromCam)
                            {
                                drawMatDist.ptr<float>(r)[c] = distFromCam;
                                vertexMap.at<cv::Vec3f>(r, c)[0] = value[0];
                                vertexMap.at<cv::Vec3f>(r, c)[1] = value[1];
                                vertexMap.at<cv::Vec3f>(r, c)[2] = value[2];
                            }
                        }
                    }
                }
            }
        }
        else if (cam.cameraType == CmaeraType::Ortho)
        {
            ptsInPic = (msh.V * R_T).cast<int>();
        }
        else
        {
            LOG_ERR_OUT << "not supported.";
            return false;
        }


        return true;
    }

    bool render(const Mesh& msh, const Camera& cam, cv::Mat& rgbMat, cv::Mat& vertexMap, cv::Mat& mask, const RenderType& renderTpye)
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
            if (cam.cameraType == CmaeraType::Pinhole && isEmpty(cam.t))
            {
                LOG_ERR_OUT << "empty t";
                return false;
            }
            Eigen::Matrix3f R_T = cam.R.transpose();
            Eigen::MatrixX3i ptsInPic;
            Eigen::MatrixX3i colorInt = (msh.C * 255.f).cast<int>();
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
                Eigen::VectorXf faceDists = viewFaceDir.rowwise().norm();
                Eigen::Index minIndex;
                faceDists.minCoeff(&minIndex);
                float nearestFaceDot = viewFaceDir.row(minIndex).dot(msh.facesNormal.row(minIndex));
                if (nearestFaceDot > 0)
                {
                    nearestFaceDot = 1;
                }
                else
                {
                    nearestFaceDot = -1;
                }
                
                Eigen::VectorXf dots = viewFaceDir.cwiseProduct(msh.facesNormal).rowwise().sum();

                rgbMat = cv::Mat::zeros(cam.height, cam.width, CV_8UC3);
                cv::Mat drawMatDist = cv::Mat::zeros(cam.height, cam.width, CV_32FC1);
                mask = cv::Mat::zeros(cam.height, cam.width, CV_8UC1);
                vertexMap = cv::Mat::zeros(cam.height, cam.width, CV_32FC3);
                std::vector<bool>ptsInCanvas(msh.V.rows(), true);
#pragma omp parallel for  schedule(dynamic)
                for (int i = 0; i < ptsInCanvas.size(); i++)
                {
                    if (ptsInPic(i, 0) < 0 || ptsInPic(i, 1) < 0 || ptsInPic(i, 0) >= cam.width || ptsInPic(i, 1) >= cam.height)
                    {
                        ptsInCanvas[i] = false;
                    }
                }



                for (int f = 0; f < dots.size(); f++)
                {
                    if (dots[f]* nearestFaceDot > 0)
                    {
                        const int& fa = msh.F(f, 0);
                        const int& fb = msh.F(f, 1);
                        const int& fc = msh.F(f, 2);
                        if (!ptsInCanvas[fa] || !ptsInCanvas[fb] || !ptsInCanvas[fc])
                        {
                            continue;
                        }
                        float distFromCam = (barycenter.row(f) - cam.t).norm();
                        std::list<std::pair<cv::Vec2i, Eigen::Vector3f>>trianglePixels = utils::triangle({ ptsInPic(fa,0),ptsInPic(fa,1) }, { ptsInPic(fb,0),ptsInPic(fb,1) }, { ptsInPic(fc,0),ptsInPic(fc,1) }, msh.V.row(fa), msh.V.row(fb), msh.V.row(fc));
                        for (const auto& d : trianglePixels)
                        {
                            const cv::Vec2i& pixel = d.first;
                            const Eigen::Vector3f& value = d.second;
                            const int& r = pixel[1];
                            const int& c = pixel[0];
                            if (c >= 0 && r >= 0 && c < cam.width && r < cam.height)
                            {
                                if (mask.ptr<uchar>(r)[c] == 0)
                                {
                                    mask.ptr<uchar>(r)[c] = 1;
                                    drawMatDist.ptr<float>(r)[c] = distFromCam;
                                    rgbMat.at<cv::Vec3b>(r, c)[0] = colorInt(fa, 2);
                                    rgbMat.at<cv::Vec3b>(r, c)[1] = colorInt(fa, 1);
                                    rgbMat.at<cv::Vec3b>(r, c)[2] = colorInt(fa, 0);
                                    vertexMap.at<cv::Vec3f>(r, c)[0] = value[0];
                                    vertexMap.at<cv::Vec3f>(r, c)[1] = value[1];
                                    vertexMap.at<cv::Vec3f>(r, c)[2] = value[2];

                                }
                                else if (drawMatDist.ptr<float>(r)[c] > distFromCam)
                                {
                                    drawMatDist.ptr<float>(r)[c] = distFromCam;
                                    rgbMat.at<cv::Vec3b>(r, c)[0] = colorInt(fa, 2);
                                    rgbMat.at<cv::Vec3b>(r, c)[1] = colorInt(fa, 1);
                                    rgbMat.at<cv::Vec3b>(r, c)[2] = colorInt(fa, 0);
                                    vertexMap.at<cv::Vec3f>(r, c)[0] = value[0];
                                    vertexMap.at<cv::Vec3f>(r, c)[1] = value[1];
                                    vertexMap.at<cv::Vec3f>(r, c)[2] = value[2];
                                }
                            }
                        }
                    }
                }
            }
            else if (cam.cameraType == CmaeraType::Ortho)
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
    bool render(const Mesh& msh, const Camera& cam,  cv::Mat& vertexMap, cv::Mat& mask)
    {
        
        if (isEmpty(msh.F) || isEmpty(msh.V) || isEmpty(msh.facesNormal))
        {
            LOG_ERR_OUT << "empty VF";
            return false;
        }
        if (isEmpty(cam.R))
        {
            LOG_ERR_OUT << "empty R";
            return false;
        }
        if (cam.cameraType == CmaeraType::Pinhole && isEmpty(cam.t))
        {
            LOG_ERR_OUT << "empty t";
            return false;
        }
        Eigen::Matrix3f R_T = cam.R.transpose();
        Eigen::MatrixX3i ptsInPic;
        Eigen::MatrixX3i colorInt = (msh.C * 255.f).cast<int>();
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
            Eigen::VectorXf faceDists = viewFaceDir.rowwise().norm();
            Eigen::Index minIndex;
            faceDists.minCoeff(&minIndex);
            float nearestFaceDot = viewFaceDir.row(minIndex).dot(msh.facesNormal.row(minIndex));
            if (nearestFaceDot>0)
            {
                nearestFaceDot = 1;
            }
            else
            {
                nearestFaceDot = -1;
            }

            Eigen::VectorXf dots = viewFaceDir.cwiseProduct(msh.facesNormal).rowwise().sum();
            cv::Mat drawMatDist = cv::Mat::zeros(cam.height, cam.width, CV_32FC1);
            mask = cv::Mat::zeros(cam.height, cam.width, CV_8UC1);
            vertexMap = cv::Mat::zeros(cam.height, cam.width, CV_32FC3);
            std::vector<bool>ptsInCanvas(msh.V.rows(), true);
#pragma omp parallel for  schedule(dynamic)
            for (int i = 0; i < ptsInCanvas.size(); i++)
            {
                if (ptsInPic(i, 0) < 0 || ptsInPic(i, 1) < 0 || ptsInPic(i, 0) >= cam.width || ptsInPic(i, 1) >= cam.height)
                {
                    ptsInCanvas[i] = false;
                }
            }
            for (int f = 0; f < dots.size(); f++)
            {
                if (dots[f]* nearestFaceDot > 0)
                {
                    const int& fa = msh.F(f, 0);
                    const int& fb = msh.F(f, 1);
                    const int& fc = msh.F(f, 2);
                    if (!ptsInCanvas[fa] || !ptsInCanvas[fb] || !ptsInCanvas[fc])
                    {
                        continue;
                    }
                    float distFromCam = (barycenter.row(f) - cam.t).norm();
                    std::list<std::pair<cv::Vec2i, Eigen::Vector3f>>trianglePixels = utils::triangle({ ptsInPic(fa,0),ptsInPic(fa,1) }, { ptsInPic(fb,0),ptsInPic(fb,1) }, { ptsInPic(fc,0),ptsInPic(fc,1) }, msh.V.row(fa), msh.V.row(fb), msh.V.row(fc));
                    for (const auto& d : trianglePixels)
                    {
                        const cv::Vec2i& pixel = d.first;
                        const Eigen::Vector3f& value = d.second;
                        const int& r = pixel[1];
                        const int& c = pixel[0];
                        if (c >= 0 && r >= 0 && c < cam.width && r < cam.height)
                        {
                            if (mask.ptr<uchar>(r)[c] == 0)
                            {
                                mask.ptr<uchar>(r)[c] = 1;
                                drawMatDist.ptr<float>(r)[c] = distFromCam;
                                vertexMap.at<cv::Vec3f>(r, c)[0] = value[0];
                                vertexMap.at<cv::Vec3f>(r, c)[1] = value[1];
                                vertexMap.at<cv::Vec3f>(r, c)[2] = value[2];

                            }
                            else if (drawMatDist.ptr<float>(r)[c] > distFromCam)
                            {
                                drawMatDist.ptr<float>(r)[c] = distFromCam;
                                vertexMap.at<cv::Vec3f>(r, c)[0] = value[0];
                                vertexMap.at<cv::Vec3f>(r, c)[1] = value[1];
                                vertexMap.at<cv::Vec3f>(r, c)[2] = value[2];
                            }
                        }
                    }
                }
            }
        }
        else if (cam.cameraType == CmaeraType::Ortho)
        {
            ptsInPic = (msh.V * R_T).cast<int>();
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
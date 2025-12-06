#include <filesystem>
#include <iostream>
#include <fstream>
#include "Eigen/Core"
#include "opencv2/opencv.hpp"
#include "igl/per_face_normals.h"
#include "igl/per_vertex_normals.h"
#include "meshDraw.h"
#include "log.h"
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


	int meshOrthoDraw(const Eigen::MatrixX3f& V, const Eigen::MatrixX3i&F, const Eigen::MatrixX3f&C, const int&anchorIdx,const int&tarImgSize,const float&additionalScale)
	{
        Eigen::MatrixX3f vertexNormal, faceNormal;
        igl::per_face_normals(V, F, faceNormal);
        igl::per_vertex_normals(V, F, vertexNormal);
        Eigen::Vector3f anchorNormal = vertexNormal.row(anchorIdx);
        std::vector<bool>faceValid(F.rows(),true); 
        Eigen::Vector3f bboxMin = V.row(anchorIdx);
        Eigen::Vector3f bboxMax = V.row(anchorIdx);
        for (int f = 0; f < F.rows(); f++)
        {  
            //Eigen::Vector3f faceNormal = faceNormal.row(anchorIdx);
            //if (faceNormal.dot(anchorNormal)<=0)
            //{
            //    faceValid[f] = false;;
            //}
            //else
            //{
            //    const int& fa = F(f, 0);
            //    const int& fb = F(f, 1);
            //    const int& fc = F(f, 2);
            //    if (V(fa, 0) < bboxMin[0])bboxMin[0] = V(fa, 0);
            //    if (V(fa, 1) < bboxMin[1])bboxMin[1] = V(fa, 1);
            //    if (V(fa, 2) < bboxMin[2])bboxMin[2] = V(fa, 2);
            //    if (V(fb, 0) < bboxMin[0])bboxMin[0] = V(fb, 0);
            //    if (V(fb, 1) < bboxMin[1])bboxMin[1] = V(fb, 1);
            //    if (V(fb, 2) < bboxMin[2])bboxMin[2] = V(fb, 2);
            //    if (V(fc, 0) < bboxMin[0])bboxMin[0] = V(fc, 0);
            //    if (V(fc, 1) < bboxMin[1])bboxMin[1] = V(fc, 1);
            //    if (V(fc, 2) < bboxMin[2])bboxMin[2] = V(fc, 2);
            //    if (V(fa, 0) > bboxMax[0])bboxMax[0] = V(fa, 0);
            //    if (V(fa, 1) > bboxMax[1])bboxMax[1] = V(fa, 1);
            //    if (V(fa, 2) > bboxMax[2])bboxMax[2] = V(fa, 2);
            //    if (V(fb, 0) > bboxMax[0])bboxMax[0] = V(fb, 0);
            //    if (V(fb, 1) > bboxMax[1])bboxMax[1] = V(fb, 1);
            //    if (V(fb, 2) > bboxMax[2])bboxMax[2] = V(fb, 2);
            //    if (V(fc, 0) > bboxMax[0])bboxMax[0] = V(fc, 0);
            //    if (V(fc, 1) > bboxMax[1])bboxMax[1] = V(fc, 1);
            //    if (V(fc, 2) > bboxMax[2])bboxMax[2] = V(fc, 2);
            //}
        }
        //float widthFloat = bboxMax[0] - bboxMin[0];
        //float heightFloat = bboxMax[1] - bboxMin[1]; 
        //float scale = 1;
        //if (heightFloat> widthFloat)
        //{
        //    scale = tarImgSize / heightFloat;
        //}
        //else
        //{
        //    scale = tarImgSize / widthFloat;
        //}
        //scale *= 1.2;
        //scale *= additionalScale;
        //int height = static_cast<int>(scale * heightFloat);
        //int width = static_cast<int>(scale * widthFloat);
        //Eigen::Vector3i centerInImg = (bboxMin * scale *0.5+ bboxMax * scale * 0.5).cast<int>();
        //Eigen::Vector3i shiftXY;
        //shiftXY[0] = width * 0.5 - centerInImg[0];
        //shiftXY[1] = height * 0.5 - centerInImg[1];
        //shiftXY[2] = 0;
        //Eigen::MatrixX3i VinImg= (V * scale).cast<int>().colwise() + shiftXY;
        //Eigen::MatrixX3i Vcolor = (255 * C).cast<int>();
        //cv::Mat img = cv::Mat::zeros(height, width, CV_8UC3);
        //for (int f = 0; f < F.rows(); f++)
        //{
        //    if (faceValid[f])
        //    { 
        //        const int& fa = F(f, 0);
        //        const int& fb = F(f, 1);
        //        const int& fc = F(f, 2);
        //        img.at<cv::Vec3b>(VinImg(fa, 1), VinImg(fa, 0))[0] = Vcolor(fa, 0);
        //        img.at<cv::Vec3b>(VinImg(fa, 1), VinImg(fa, 0))[1] = Vcolor(fa, 1);
        //        img.at<cv::Vec3b>(VinImg(fa, 1), VinImg(fa, 0))[2] = Vcolor(fa, 2);
        //        img.at<cv::Vec3b>(VinImg(fb, 1), VinImg(fb, 0))[0] = Vcolor(fb, 0);
        //        img.at<cv::Vec3b>(VinImg(fb, 1), VinImg(fb, 0))[1] = Vcolor(fb, 1);
        //        img.at<cv::Vec3b>(VinImg(fb, 1), VinImg(fb, 0))[2] = Vcolor(fb, 2);
        //        img.at<cv::Vec3b>(VinImg(fc, 1), VinImg(fc, 0))[0] = Vcolor(fc, 0);
        //        img.at<cv::Vec3b>(VinImg(fc, 1), VinImg(fc, 0))[1] = Vcolor(fc, 1);
        //        img.at<cv::Vec3b>(VinImg(fc, 1), VinImg(fc, 0))[2] = Vcolor(fc, 2);
        //    }
        //}
		return 0;
	}
}


int test_draw()
{
	return 0;
}
#include <unordered_map>
#include <numeric>
#include <vector>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <map>
#include "Eigen/Core"
#include <Eigen/Geometry>
#include <igl/barycenter.h>
#include "igl/boundary_loop.h"
#include "igl/read_triangle_mesh.h"
#include "opencv2/opencv.hpp"
#include "json/json.h"
#include "log.h"
#include "opencvTools.h"
#include "pips2.h"

namespace rayhit
{
#define IGL_RAY_TRI_EPSILON 0.000001
#define IGL_RAY_TRI_CROSS(dest,v1,v2) \
          dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
          dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
          dest[2]=v1[0]*v2[1]-v1[1]*v2[0];
#define IGL_RAY_TRI_DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
#define IGL_RAY_TRI_SUB(dest,v1,v2) \
          dest[0]=v1[0]-v2[0]; \
          dest[1]=v1[1]-v2[1]; \
          dest[2]=v1[2]-v2[2]; 

	template<class Dtype = float>
	int intersect_triangle1(const Dtype* orig, const Dtype* dir,
		const Dtype* vert0, const Dtype* vert1, const Dtype* vert2,
		Dtype* t, Dtype* u, Dtype* v)
	{
		Dtype edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
		Dtype det, inv_det;

		/* find vectors for two edges sharing vert0 */
		IGL_RAY_TRI_SUB(edge1, vert1, vert0);
		IGL_RAY_TRI_SUB(edge2, vert2, vert0);

		/* begin calculating determinant - also used to calculate U parameter */
		IGL_RAY_TRI_CROSS(pvec, dir, edge2);

		/* if determinant is near zero, ray lies in plane of triangle */
		det = IGL_RAY_TRI_DOT(edge1, pvec);

		if (det > IGL_RAY_TRI_EPSILON)
		{
			/* calculate distance from vert0 to ray origin */
			IGL_RAY_TRI_SUB(tvec, orig, vert0);

			/* calculate U parameter and test bounds */
			*u = IGL_RAY_TRI_DOT(tvec, pvec);
			if (*u < 0.0 || *u > det)
				return 0;

			/* prepare to test V parameter */
			IGL_RAY_TRI_CROSS(qvec, tvec, edge1);

			/* calculate V parameter and test bounds */
			*v = IGL_RAY_TRI_DOT(dir, qvec);
			if (*v < 0.0 || *u + *v > det)
				return 0;

		}
		else if (det < -IGL_RAY_TRI_EPSILON)
		{
			/* calculate distance from vert0 to ray origin */
			IGL_RAY_TRI_SUB(tvec, orig, vert0);

			/* calculate U parameter and test bounds */
			*u = IGL_RAY_TRI_DOT(tvec, pvec);
			/*      printf("*u=%f\n",(float)*u); */
			/*      printf("det=%f\n",det); */
			if (*u > 0.0 || *u < det)
				return 0;

			/* prepare to test V parameter */
			IGL_RAY_TRI_CROSS(qvec, tvec, edge1);

			/* calculate V parameter and test bounds */
			*v = IGL_RAY_TRI_DOT(dir, qvec);
			if (*v > 0.0 || *u + *v < det)
				return 0;
		}
		else return 0;  /* ray is parallel to the plane of the triangle */


		inv_det = 1.0 / det;

		/* calculate t, ray intersects triangle */
		*t = IGL_RAY_TRI_DOT(edge2, qvec) * inv_det;
		(*u) *= inv_det;
		(*v) *= inv_det;

		return 1;
	}


}
bool readRsultJson(const std::filesystem::path& jsonPath, Eigen::Matrix4f& Rt, Eigen::Matrix4f& K,int&height,int&width, std::filesystem::path& imgPath)
{ 
	std::string aline;
	{
		std::fstream fin(jsonPath, std::ios::in);
		if (!fin.is_open())
		{
			LOG_ERR_OUT << "connot open:" << jsonPath;
			return false;
		}
		std::stringstream ss;
		ss << fin.rdbuf();
		aline = ss.str();
	} 
	JSONCPP_STRING err;
	Json::Value newRoot;
	const auto rawJsonLength = static_cast<int>(aline.length());
	Json::CharReaderBuilder newBuilder;
	const std::unique_ptr<Json::CharReader> newReader(newBuilder.newCharReader());
	if (!newReader->parse(aline.c_str(), aline.c_str() + rawJsonLength, &newRoot,
		&err)) {
		return  false;
	} 

	if (!newRoot.isObject()
		|| !newRoot.isMember("Qt")
		|| !newRoot.isMember("cx")
		|| !newRoot.isMember("cy")
		|| !newRoot.isMember("fx")
		|| !newRoot.isMember("fy")
		|| !newRoot.isMember("height")
		|| !newRoot.isMember("width")
		|| !newRoot.isMember("imagePath"))
	{
		return false;
	}

	auto QtNode = newRoot["Qt"];
	auto cxNode = newRoot["cx"];
	auto cyNode = newRoot["cy"];
	auto fxNode = newRoot["fx"];
	auto fyNode = newRoot["fy"];
	auto heightNode = newRoot["height"];
	auto widthNode = newRoot["width"];
	auto imagePathNode = newRoot["imagePath"];
	Rt = Eigen::Matrix4f::Identity();
	K = Eigen::Matrix4f::Identity();
	if (QtNode.isNull() 
		|| cxNode.isNull()
		|| cyNode.isNull()
		|| fxNode.isNull()
		|| fyNode.isNull()
		|| heightNode.isNull()
		|| widthNode.isNull()
		|| imagePathNode.isNull())
	{
		return  false;
	}
	try
	{
		if (QtNode.isArray() && QtNode.size() == 7)
		{
			double w = QtNode[0].asDouble();
			double x = QtNode[1].asDouble();
			double y = QtNode[2].asDouble();
			double z = QtNode[3].asDouble();
			Eigen::Quaternionf q(w, x, y, z);
			Rt.block(0, 0, 3, 3) = q.matrix();
			Rt(0, 3) = QtNode[4].asDouble();
			Rt(1, 3) = QtNode[5].asDouble();
			Rt(2, 3) = QtNode[6].asDouble();
		}
		K(0, 0) = fxNode.asFloat();
		K(1, 1) = fyNode.asFloat();
		K(0, 2) = cxNode.asFloat();
		K(1, 2) = cyNode.asFloat();
		height = heightNode.asInt();
		width = widthNode.asInt();
		imgPath = imagePathNode.asString();
		return true;
	}
	catch (const std::exception&)
	{
		return false;
	}
	return true;
}
void savePts(const std::filesystem::path& path, const Eigen::MatrixX3f& pts, std::vector<bool>* enable = nullptr)
{
	std::fstream fout(path,std::ios::out);
	bool enabled = (enable != nullptr && enable->size() == pts.rows());
	for (int i = 0; i < pts.rows(); i++)
	{
		if (enabled)
		{
			if ((*enable)[i])
			{ 
				fout << pts(i, 0) << " " << pts(i, 1) << " " << pts(i, 2) << std::endl;
			}
		}
		else
		{
			fout << pts(i, 0) << " " << pts(i, 1) << " " << pts(i, 2) << std::endl;
		}
	}
}
int load_cameras(const std::filesystem::path& resultDir, 
	std::vector<std::filesystem::path>&imgPaths,
	std::vector<cv::Mat>& imgs,
	std::vector<cv::Mat>& masks,
	std::vector<Eigen::Matrix4f>& Rts,
	std::vector<Eigen::Matrix4f>& Ks)
{
	imgPaths.reserve(64);
	imgs.reserve(64);
	masks.reserve(64);
	Rts.reserve(64);
	Ks.reserve(64);
	imgPaths.clear();
	imgs.clear();
	masks.clear();
	Rts.clear();
	Ks.clear();
	for (auto const& dir_entry : std::filesystem::directory_iterator{ resultDir })
	{
		if (dir_entry.is_regular_file())
		{ 
			const auto& filePath = dir_entry.path();
			if (filePath.has_extension() && filePath.extension().compare(".json")==0)
			{
				Eigen::Matrix4f Rt;
				Eigen::Matrix4f K;
				int height;
				int width;
				std::filesystem::path imgPath;
				LOG_OUT << "try load : " << filePath;
				if (readRsultJson(filePath, Rt, K, height, width, imgPath))
				{
					std::string filename = imgPath.filename().stem().string();
					std::filesystem::path maskPath = imgPath.parent_path() / ("mask_" + filename + ".dat");
					if (std::filesystem::exists(imgPath)&& std::filesystem::exists(maskPath))
					{
						cv::Mat mask = tools::loadMask(maskPath.string());
						cv::Mat img = cv::imread(imgPath.string());
						if (!img.empty() && !mask.empty())
						{
							imgPaths.emplace_back(imgPath);
							imgs.emplace_back(img);
							masks.emplace_back(mask);
							Rts.emplace_back(Rt);
							Ks.emplace_back(K);
						}
					}
				};
			}
		} 
	}
	
	return 0;
}
void projectToImg(const Eigen::MatrixX3f& V_, std::vector<bool>& picked, const cv::Mat& mask, const Eigen::Matrix4f& Rt_, const Eigen::Matrix4f& K_, std::vector<cv::Point2f>& imgPts, std::vector<int>& imgPtsInV)
{
	size_t count = std::count(picked.begin(), picked.end(), true);
	Eigen::Matrix4Xf pickedV;
	pickedV.resize(4, count);
	std::vector<int>pickedIdxToTotalIdx;
	pickedIdxToTotalIdx.reserve(count);
	int idx = 0;
	for (int i = 0; i < picked.size(); i++)
	{
		if (picked[i])
		{
			pickedV(0, idx) = V_(i, 0);
			pickedV(1, idx) = V_(i, 1);
			pickedV(2, idx) = V_(i, 2);
			pickedV(3, idx) = 1;
			pickedIdxToTotalIdx.emplace_back(i);
			idx++;
		}
	} 
	Eigen::Matrix<float, 3, 4> Rt = Rt_.block(0, 0, 3, 4);
	Eigen::Matrix<float, 3, 3> K = K_.block(0, 0, 3, 3);
	Eigen::Matrix3Xf V = K * (Rt * pickedV);
	V.row(0).array() /= V.row(2).array();  
	V.row(1).array() /= V.row(2).array(); 
	imgPts.reserve(count);
	imgPts.clear();
	imgPtsInV.reserve(count);
	imgPtsInV.clear();
	for (int i = 0; i < count; i++)
	{ 
		int idxTotal = pickedIdxToTotalIdx[i];
		const float& x = V(0, i);
		const float& y = V(1, i);
		int xInt = static_cast<int>(x);
		int yInt = static_cast<int>(y);
		if (xInt<0 || yInt<0 || xInt >= mask.cols || yInt >= mask.rows)
		{
			picked[idxTotal] = false;
		}
		else
		{
			if (mask.ptr<uchar>(yInt)[xInt]>0)
			{
				imgPts.emplace_back(x,y);
				imgPtsInV.emplace_back(idxTotal);
			}
			else
			{
				picked[idxTotal] = false;
			}
		}
	}
	return;
}
int test_iter_d()
{
	pips2::Pips2 ins("../models/pips2_base_ncnn.param", "../models/pips2_base_ncnn.bin", "../models/pips2_deltaBlock_ncnn.param", "../models/pips2_deltaBlock_ncnn.bin");
	std::vector<std::filesystem::path>imgPaths;
	std::vector<cv::Mat>imgs;
	std::vector<cv::Mat>masks;
	std::vector<Eigen::Matrix4f>Rts;
	std::vector<Eigen::Matrix4f>Ks;
	int loadRet = load_cameras("../data/a/result", imgPaths, imgs, masks, Rts, Ks);
	Eigen::MatrixX3f V; 
	Eigen::MatrixX3i F;
	bool readRet = igl::read_triangle_mesh("../InsMesh/out.obj",V,F);
	std::vector<bool>isBndV(V.rows(),false);
	{
		std::vector<std::vector<int>> bnd_loops;
		igl::boundary_loop(F, bnd_loops); 
		for (size_t i = 0; i < bnd_loops.size(); ++i) 
		{
			for (const auto& idx : bnd_loops[i]) 
			{
				isBndV[idx] = true;
			} 
		}
	}

	if (!ins.inputImage(imgPaths))
	{
		LOG_ERR_OUT << "load img error";
		return -1;
	}
	for (int imgI = 0; imgI < imgs.size(); imgI++)
	{
		const Eigen::Matrix4f& Rt = Rts[imgI];
		const Eigen::Matrix4f& K = Ks[imgI];
		Eigen::RowVector3f ray_origin;
		ray_origin[0] = Rt(0, 3);
		ray_origin[1] = Rt(1, 3);
		ray_origin[2] = Rt(2, 3);

		Eigen::MatrixX3f Vdir = V.rowwise() - ray_origin;
		Eigen::VectorXf Vdist = Vdir.rowwise().norm();
		Vdir.array().colwise() /= Vdist.array();
		Eigen::MatrixX3f BC;
		igl::barycenter(V, F, BC);
		BC.rowwise() -= ray_origin;
		BC.rowwise().normalize();
		Eigen::MatrixXf cosMat = Vdir * BC.transpose();

		Eigen::RowVector3f ori(0, 0, 0);
		std::vector<bool>frontalVert(cosMat.rows(),true);
		for (int vI = 0; vI < cosMat.rows(); vI++)
		{
			if (isBndV[vI])
			{
				continue;
			}
			for (int fI = 0; fI < cosMat.cols(); fI++)
			{
				if (cosMat(vI,fI)>0.99)
				{
					int ai = F(fI, 0);
					int bi = F(fI, 1);
					int ci = F(fI, 2);
					float d[3] = { Vdir(vI,0),Vdir(vI,1),Vdir(vI,2) };
					float pa[3] = { V(ai,0),V(ai,1),V(ai,2) };
					float pb[3] = { V(bi,0),V(bi,1),V(bi,2) };
					float pc[3] = { V(ci,0),V(ci,1),V(ci,2) };
					float t, u, v;
					int hit = rayhit::intersect_triangle1<float>(&ori[0], &d[0], pa, pb, pc, &t, &u, &v);
					if (hit>0 && t>0 && t+1e-5< Vdist[vI])
					{
						frontalVert[vI] = false;
						break;
					}
				}
			}
		}
		std::vector<cv::Point2f> imgPts;
		std::vector<int> imgPtsInV;
		projectToImg(V, frontalVert, masks[imgI], Rts[imgI], Ks[imgI], imgPts, imgPtsInV);

		{ 

			std::vector<std::vector<cv::Point2f>>trajs;
			bool trackRet = ins.trackLimit(imgPts, trajs, 2, 12);
			 
		}
	}



	return 0;
}
#include <numeric>
#include <fstream>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <functional>
#include <sstream>
#include <list>
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "json/json.h"
#include "opencv2/opencv.hpp"
#include "opencvTools.h"
#include "igl/per_vertex_normals.h"
#include "igl/per_face_normals.h"
#include "log.h"
namespace surf
{
	bool transform(const Eigen::MatrixXf& dataIn, std::vector<char>&dataOut)
	{
		int r = dataIn.rows();
		int c = dataIn.cols();
		dataOut.resize(2 * sizeof(int) + r * c * sizeof(float));
		*(int*)&dataOut[0] = r;
		*(int*)&dataOut[sizeof(int)] = c;
		int size = r * c;
		float* data_ = (float*)&dataOut[2 * sizeof(int)];
		for (int i = 0; i < size; i++)
		{
			data_[i] = dataIn(i / c, i % c);
		}
		return true;
	}
	bool transform(const Eigen::MatrixXi& dataIn, std::vector<char>&dataOut)
	{
		int r = dataIn.rows();
		int c = dataIn.cols();
		dataOut.resize(2 * sizeof(int) + r * c * sizeof(int));
		*(int*)&dataOut[0] = r;
		*(int*)&dataOut[sizeof(int)] = c;
		int size = r * c;
		int* data_ = (int*)&dataOut[2 * sizeof(int)];
		for (int i = 0; i < size; i++)
		{
			data_[i] = dataIn(i / c, i % c);
		}
		return true;
	}
	int transform(const char* dataIn, Eigen::MatrixXf& dataOut)
	{
		const int& r = *(int*)&dataIn[0];
		const int& c = *(int*)&dataIn[sizeof(int)];
		dataOut.resize(r, c);
		const float* data_ = (float*)&dataIn[2 * sizeof(int)];
		int size = r * c;
		for (int i = 0; i < size; i++)
		{
			dataOut(i / c, i % c) = data_[i];
		}
		return 2 * sizeof(int)+ size * sizeof(float);
	}
	int transform(const char* dataIn, Eigen::MatrixXi& dataOut)
	{
		const int& r = *(int*)&dataIn[0];
		const int& c = *(int*)&dataIn[sizeof(int)];
		dataOut.resize(r, c);
		const int* data_ = (int*)&dataIn[2 * sizeof(int)];
		int size = r * c;
		for (int i = 0; i < size; i++)
		{
			dataOut(i / c, i % c) = data_[i];
		}
		return 2 * sizeof(int) + size * sizeof(float);
	}
	bool transform(const std::string&line,std::vector<char>&data)
	{
		data.resize(sizeof(int)+ line.length());
		int& length = *(int*)&data[0];
		length = line.length();
		for (int i = 0; i < length; i++)
		{
			data[sizeof(int)+i]= line[i];
		}
		return true;
	}
	int transform(const char* data,std::string& line)
	{
		const int& strLength = *(int*)&data[0];
		line.resize(strLength);
		for (int i = 0; i < strLength; i++)
		{
			 line[i]= data[sizeof(int) + i];
		}
		return strLength+sizeof(int);
	}
	bool readObj(const std::filesystem::path&objPath, Eigen::MatrixXf& vertex, Eigen::MatrixXi& faces, Eigen::MatrixXf& vertex_normal, Eigen::MatrixXf& face_normal)
	{
		if (!std::filesystem::exists(objPath))
		{
			LOG_ERR_OUT << "not found : " << objPath;
			return false;
		}
		std::fstream fin(objPath, std::ios::in);
		std::string aline;
		std::list<Eigen::Vector3f>v;
		std::list<Eigen::Vector3i>f;
		while (std::getline(fin,aline))
		{
			if (aline.length()>2 && aline[0] == 'v'&& aline[1] == ' ')
			{
				std::stringstream ss(aline);
				char c;
				float x, y, z;
				ss >> c >> x >> y >> z;
				v.emplace_back(x,y,z);
			}
			if (aline.length() > 2 && aline[0] == 'f' && aline[1] == ' ')
			{
				std::stringstream ss(aline);
				char c;
				int x, y, z;
				ss >> c >> x >> y >> z;
				f.emplace_back(x-1, y - 1, z - 1);
			}
		}
		vertex = Eigen::MatrixXf( v.size(),3);
		faces = Eigen::MatrixXi( f.size(),3);
		int i = 0;
		for (const auto&d:v)
		{
			vertex(i, 0) = d.x();
			vertex(i, 1) = d.y();
			vertex(i, 2) = d.z();
			i += 1;
		}
		i = 0;
		for (const auto& d : f)
		{
			faces(i, 0) = d.x();
			faces(i, 1) = d.y();
			faces(i, 2) = d.z();
			i += 1;
		}
		igl::per_vertex_normals(vertex, faces, vertex_normal);
		igl::per_face_normals(vertex, faces, face_normal);
//		if (cameraT != nullptr)
//		{
//			std::vector<float>distFromT(f.size());
//#pragma omp parallel for
//			for (int i = 0; i < f.size(); i++)
//			{
//				const int& v = faces(i, 0);//choose the first vertex idx
//				float a = vertex(v, 0) - cameraT->x();
//				float b = vertex(v, 1) - cameraT->y();
//				float c = vertex(v, 2) - cameraT->z();
//				distFromT[i] = a * a + b * b + c * c;
//			}
//			int neareatFaceIdx = std::min_element(distFromT.begin(), distFromT.end()) - distFromT.begin();
//			int neareatVertexIdx = faces(neareatFaceIdx, 0);
//			Eigen::Vector3f vertexNormal(vertex_normal(neareatVertexIdx, 0), vertex_normal(neareatVertexIdx, 1), vertex_normal(neareatVertexIdx, 2));
//			Eigen::Vector3f cameraTDir(vertex(neareatVertexIdx, 0) - cameraT->x(), vertex(neareatVertexIdx, 1) - cameraT->y(), vertex(neareatVertexIdx, 2) - cameraT->z());
//			Eigen::Vector3f faceNormal(face_normal(neareatFaceIdx, 0), face_normal(neareatFaceIdx, 1), face_normal(neareatFaceIdx, 2));
//			cameraTDir.normalize();
//			if (vertexNormal.dot(cameraTDir) < 0)
//			{
//				LOG_OUT << "swtich normals.";
//				vertex_normal *= -1;
//			}
//			if (faceNormal.dot(cameraTDir) < 0)
//			{
//				LOG_OUT << "swtich normals.";
//				face_normal *= -1;
//			}
//		}
		vertex = Eigen::Matrix4Xf(4,v.size()); 
		i = 0;
		for (const auto& d : v)
		{
			vertex(0,i) = d.x();
			vertex(1,i) = d.y();
			vertex(2,i) = d.z();
			vertex(3,i) = 1;
			i += 1;
		}
		faces.transposeInPlace();
		vertex_normal.transposeInPlace();
		face_normal.transposeInPlace();
		return true;
	}
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
#define SH_DEGREE (4)
	void shEncoder(const float& x, const float& y, const float& z, std::vector<float>&outputs)
	{
		float xy = x * y, xz = x * z, yz = y * z, x2 = x * x, y2 = y * y, z2 = z * z, xyz = xy * z;
		float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
		float x6 = x4 * x2, y6 = y4 * y2, z6 = z4 * z2;
		outputs.resize(SH_DEGREE* SH_DEGREE);
		outputs[0] = 0.28209479177387814f;                          // 1/(2*sqrt(pi))
		if (SH_DEGREE <= 1) { return; }
		outputs[1] = -0.48860251190291987f * y;                               // -sqrt(3)*y/(2*sqrt(pi))
		outputs[2] = 0.48860251190291987f * z;                                // sqrt(3)*z/(2*sqrt(pi))
		outputs[3] = -0.48860251190291987f * x;                               // -sqrt(3)*x/(2*sqrt(pi))
		if (SH_DEGREE <= 2) { return; }
		outputs[4] = 1.0925484305920792f * xy;                                // sqrt(15)*xy/(2*sqrt(pi))
		outputs[5] = -1.0925484305920792f * yz;                               // -sqrt(15)*yz/(2*sqrt(pi))
		outputs[6] = 0.94617469575755997f * z2 - 0.31539156525251999f;                         // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
		outputs[7] = -1.0925484305920792f * xz;                               // -sqrt(15)*xz/(2*sqrt(pi))
		outputs[8] = 0.54627421529603959f * x2 - 0.54627421529603959f * y2;                              // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
		if (SH_DEGREE <= 3) { return; }
		outputs[9] = 0.59004358992664352f * y * (-3.0f * x2 + y2);                         // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
		outputs[10] = 2.8906114426405538f * xy * z;                             // sqrt(105)*xy*z/(2*sqrt(pi))
		outputs[11] = 0.45704579946446572f * y * (1.0f - 5.0f * z2);                                // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
		outputs[12] = 0.3731763325901154f * z * (5.0f * z2 - 3.0f);                         // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
		outputs[13] = 0.45704579946446572f * x * (1.0f - 5.0f * z2);                                // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
		outputs[14] = 1.4453057213202769f * z * (x2 - y2);                              // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
		outputs[15] = 0.59004358992664352f * x * (-x2 + 3.0f * y2);                                // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
		if (SH_DEGREE <= 4) { return; }
		outputs[16] = 2.5033429417967046f * xy * (x2 - y2);                             // 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
		outputs[17] = 1.7701307697799304f * yz * (-3.0f * x2 + y2);                                // 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
		outputs[18] = 0.94617469575756008f * xy * (7.0f * z2 - 1.0f);                               // 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
		outputs[19] = 0.66904654355728921f * yz * (3.0f - 7.0f * z2);                               // 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
		outputs[20] = -3.1735664074561294f * z2 + 3.7024941420321507f * z4 + 0.31735664074561293f;                                // 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
		outputs[21] = 0.66904654355728921f * xz * (3.0f - 7.0f * z2);                               // 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
		outputs[22] = 0.47308734787878004f * (x2 - y2) * (7.0f * z2 - 1.0f);                                // 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
		outputs[23] = 1.7701307697799304f * xz * (-x2 + 3.0f * y2);                                // 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
		outputs[24] = -3.7550144126950569f * x2 * y2 + 0.62583573544917614f * x4 + 0.62583573544917614f * y4;                         // 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
		if (SH_DEGREE <= 5) { return; }
		outputs[25] = 0.65638205684017015f * y * (10.0f * x2 * y2 - 5.0f * x4 - y4);                            // 3*sqrt(154)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		outputs[26] = 8.3026492595241645f * xy * z * (x2 - y2);                           // 3*sqrt(385)*xy*z*(x2 - y2)/(4*sqrt(pi))
		outputs[27] = -0.48923829943525038f * y * (3.0f * x2 - y2) * (9.0f * z2 - 1.0f);                         // -sqrt(770)*y*(3*x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
		outputs[28] = 4.7935367849733241f * xy * z * (3.0f * z2 - 1.0f);                              // sqrt(1155)*xy*z*(3*z2 - 1)/(4*sqrt(pi))
		outputs[29] = 0.45294665119569694f * y * (14.0f * z2 - 21.0f * z4 - 1.0f);                             // sqrt(165)*y*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
		outputs[30] = 0.1169503224534236f * z * (-70.0f * z2 + 63.0f * z4 + 15.0f);                            // sqrt(11)*z*(-70*z2 + 63*z4 + 15)/(16*sqrt(pi))
		outputs[31] = 0.45294665119569694f * x * (14.0f * z2 - 21.0f * z4 - 1.0f);                             // sqrt(165)*x*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
		outputs[32] = 2.3967683924866621f * z * (x2 - y2) * (3.0f * z2 - 1.0f);                               // sqrt(1155)*z*(x2 - y2)*(3*z2 - 1)/(8*sqrt(pi))
		outputs[33] = -0.48923829943525038f * x * (x2 - 3.0f * y2) * (9.0f * z2 - 1.0f);                         // -sqrt(770)*x*(x2 - 3*y2)*(9*z2 - 1)/(32*sqrt(pi))
		outputs[34] = 2.0756623148810411f * z * (-6.0f * x2 * y2 + x4 + y4);                         // 3*sqrt(385)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
		outputs[35] = 0.65638205684017015f * x * (10.0f * x2 * y2 - x4 - 5.0f * y4);                            // 3*sqrt(154)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
		if (SH_DEGREE <= 6) { return; }
		outputs[36] = 1.3663682103838286f * xy * (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4);                               // sqrt(6006)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
		outputs[37] = 2.3666191622317521f * yz * (10.0f * x2 * y2 - 5.0f * x4 - y4);                            // 3*sqrt(2002)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		outputs[38] = 2.0182596029148963f * xy * (x2 - y2) * (11.0f * z2 - 1.0f);                             // 3*sqrt(91)*xy*(x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
		outputs[39] = -0.92120525951492349f * yz * (3.0f * x2 - y2) * (11.0f * z2 - 3.0f);                               // -sqrt(2730)*yz*(3*x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
		outputs[40] = 0.92120525951492349f * xy * (-18.0f * z2 + 33.0f * z4 + 1.0f);                           // sqrt(2730)*xy*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
		outputs[41] = 0.58262136251873131f * yz * (30.0f * z2 - 33.0f * z4 - 5.0f);                            // sqrt(273)*yz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
		outputs[42] = 6.6747662381009842f * z2 - 20.024298714302954f * z4 + 14.684485723822165f * z6 - 0.31784601133814211f;                         // sqrt(13)*(105*z2 - 315*z4 + 231*z6 - 5)/(32*sqrt(pi))
		outputs[43] = 0.58262136251873131f * xz * (30.0f * z2 - 33.0f * z4 - 5.0f);                            // sqrt(273)*xz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
		outputs[44] = 0.46060262975746175f * (x2 - y2) * (11.0f * z2 * (3.0f * z2 - 1.0f) - 7.0f * z2 + 1.0f);                               // sqrt(2730)*(x2 - y2)*(11*z2*(3*z2 - 1) - 7*z2 + 1)/(64*sqrt(pi))
		outputs[45] = -0.92120525951492349f * xz * (x2 - 3.0f * y2) * (11.0f * z2 - 3.0f);                               // -sqrt(2730)*xz*(x2 - 3*y2)*(11*z2 - 3)/(32*sqrt(pi))
		outputs[46] = 0.50456490072872406f * (11.0f * z2 - 1.0f) * (-6.0f * x2 * y2 + x4 + y4);                          // 3*sqrt(91)*(11*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
		outputs[47] = 2.3666191622317521f * xz * (10.0f * x2 * y2 - x4 - 5.0f * y4);                            // 3*sqrt(2002)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
		outputs[48] = 10.247761577878714f * x2 * y4 - 10.247761577878714f * x4 * y2 + 0.6831841051919143f * x6 - 0.6831841051919143f * y6;                         // sqrt(6006)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
		if (SH_DEGREE <= 7) { return; }
		outputs[49] = 0.70716273252459627f * y * (-21.0f * x2 * y4 + 35.0f * x4 * y2 - 7.0f * x6 + y6);                              // 3*sqrt(715)*y*(-21*x2*y4 + 35*x4*y2 - 7*x6 + y6)/(64*sqrt(pi))
		outputs[50] = 5.2919213236038001f * xy * z * (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4);                             // 3*sqrt(10010)*xy*z*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
		outputs[51] = -0.51891557872026028f * y * (13.0f * z2 - 1.0f) * (-10.0f * x2 * y2 + 5.0f * x4 + y4);                          // -3*sqrt(385)*y*(13*z2 - 1)*(-10*x2*y2 + 5*x4 + y4)/(64*sqrt(pi))
		outputs[52] = 4.1513246297620823f * xy * z * (x2 - y2) * (13.0f * z2 - 3.0f);                           // 3*sqrt(385)*xy*z*(x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
		outputs[53] = -0.15645893386229404f * y * (3.0f * x2 - y2) * (13.0f * z2 * (11.0f * z2 - 3.0f) - 27.0f * z2 + 3.0f);                              // -3*sqrt(35)*y*(3*x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
		outputs[54] = 0.44253269244498261f * xy * z * (-110.0f * z2 + 143.0f * z4 + 15.0f);                              // 3*sqrt(70)*xy*z*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
		outputs[55] = 0.090331607582517306f * y * (-135.0f * z2 + 495.0f * z4 - 429.0f * z6 + 5.0f);                              // sqrt(105)*y*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
		outputs[56] = 0.068284276912004949f * z * (315.0f * z2 - 693.0f * z4 + 429.0f * z6 - 35.0f);                              // sqrt(15)*z*(315*z2 - 693*z4 + 429*z6 - 35)/(32*sqrt(pi))
		outputs[57] = 0.090331607582517306f * x * (-135.0f * z2 + 495.0f * z4 - 429.0f * z6 + 5.0f);                              // sqrt(105)*x*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
		outputs[58] = 0.07375544874083044f * z * (x2 - y2) * (143.0f * z2 * (3.0f * z2 - 1.0f) - 187.0f * z2 + 45.0f);                         // sqrt(70)*z*(x2 - y2)*(143*z2*(3*z2 - 1) - 187*z2 + 45)/(64*sqrt(pi))
		outputs[59] = -0.15645893386229404f * x * (x2 - 3.0f * y2) * (13.0f * z2 * (11.0f * z2 - 3.0f) - 27.0f * z2 + 3.0f);                              // -3*sqrt(35)*x*(x2 - 3*y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
		outputs[60] = 1.0378311574405206f * z * (13.0f * z2 - 3.0f) * (-6.0f * x2 * y2 + x4 + y4);                         // 3*sqrt(385)*z*(13*z2 - 3)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
		outputs[61] = -0.51891557872026028f * x * (13.0f * z2 - 1.0f) * (-10.0f * x2 * y2 + x4 + 5.0f * y4);                          // -3*sqrt(385)*x*(13*z2 - 1)*(-10*x2*y2 + x4 + 5*y4)/(64*sqrt(pi))
		outputs[62] = 2.6459606618019f * z * (15.0f * x2 * y4 - 15.0f * x4 * y2 + x6 - y6);                               // 3*sqrt(10010)*z*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
		outputs[63] = 0.70716273252459627f * x * (-35.0f * x2 * y4 + 21.0f * x4 * y2 - x6 + 7.0f * y6);                              // 3*sqrt(715)*x*(-35*x2*y4 + 21*x4*y2 - x6 + 7*y6)/(64*sqrt(pi))
	
	};
	struct GridConfig
	{
		float amplitudeHalf;
		float gridUnit;
		int gridLevelCnt;		
		std::vector<char>serialization()const
		{
			std::vector<char> ret(sizeof(float) * 2 + sizeof(int));
			*(float*)&ret[0] = amplitudeHalf;
			*(float*)&ret[sizeof(float)] = gridUnit;
			*(int*)&ret[sizeof(float)*2] = gridLevelCnt;
			return ret;
		}
		int deserialization(const char*data)
		{
			amplitudeHalf = *(float*)&data[0];
			gridUnit = *(float*)&data[sizeof(float)];
			gridLevelCnt = *(int*)&data[sizeof(float) * 2];
			return sizeof(float) * 2 + sizeof(int);
		}
	};
	struct Camera
	{
		Camera() {}
		Camera(const double& fx_, const double& fy_, const double& cx_, const double& cy_, const int& height_, const int& width_) 
		{
			fx = fx_;
			fy = fy_;
			cx = cx_;
			cy = cy_;
			height = height_;
			width = width_;
			intr = Eigen::Matrix4f::Identity();
			intr(0, 0) = fx;
			intr(1, 1) = fy;
			intr(0, 2) = cx;
			intr(1, 2) = cy;
			if (imgDirs.empty())
			{
				float fxInv = 1. / fx;
				float fyInv = 1. / fy;
				imgDirs = cv::Mat::zeros(height, width, CV_32FC3);
				for (int r = 0; r < height; r++)
				{
					for (int c = 0; c < width; c++)
					{
						Eigen::Vector3f dir((c + 0.5 - cx) * fxInv, (r + 0.5 - cy) * fyInv, 1);
						dir.normalize();
						imgDirs.at<cv::Vec3f>(r, c)[0] = dir.x();
						imgDirs.at<cv::Vec3f>(r, c)[1] = dir.y();
						imgDirs.at<cv::Vec3f>(r, c)[2] = dir.z();
					}
				}
			}
		}
		std::vector<char>serialization()const
		{
			std::vector<char>ret(6*sizeof(float));
			float* data = (float*)&ret[0];
			data[0] = fx;
			data[1] = fy;
			data[2] = cx;
			data[3] = cy;
			data[4] = height;
			data[5] = width;
			return ret;
		}
		int deserialization(const char*data)
		{
			const float* floatData = (float*)&data[0];
			fx = floatData[0];
			fy = floatData[1];
			cx = floatData[2];
			cy = floatData[3] ;
			height = floatData[4];
			width = floatData[5] ;
			intr = Eigen::Matrix4f::Identity();
			intr(0, 0) = fx;
			intr(1, 1) = fy;
			intr(0, 2) = cx;
			intr(1, 2) = cy;
			if (imgDirs.empty())
			{
				float fxInv = 1. / fx;
				float fyInv = 1. / fy;
				imgDirs = cv::Mat::zeros(height, width, CV_32FC3);
				for (int r = 0; r < height; r++)
				{
					for (int c = 0; c < width; c++)
					{
						Eigen::Vector3f dir((c + 0.5 - cx) * fxInv, (r + 0.5 - cy) * fyInv, 1);
						dir.normalize();
						imgDirs.at<cv::Vec3f>(r, c)[0] = dir.x();
						imgDirs.at<cv::Vec3f>(r, c)[1] = dir.y();
						imgDirs.at<cv::Vec3f>(r, c)[2] = dir.z();
					}
				}
			}
			return 6 * sizeof(float);
		}
		static std::vector<char>serialization(const std::unordered_map<int, Camera>& cameras)
		{
			std::vector<char>ret(sizeof(int));;
			*(int*)&ret[0] = cameras.size();
			for (const auto&d : cameras)
			{
				std::vector<char>ret0(sizeof(int));
				*(int*)&ret0[0] = d.first;
				std::vector<char>ret1 = d.second.serialization();
				ret.insert(ret.end(), ret0.begin(), ret0.end());
				ret.insert(ret.end(), ret1.begin(), ret1.end());
			}
			return ret;
		}
		static int deserialization(const char* data, std::unordered_map<int, Camera>& dataOut)
		{
			dataOut.clear();
			int elemCnt = *(int*)data;
			int pos = sizeof(int);
			for (int i = 0; i < elemCnt; i++)
			{
				int cameraId = *(int*)&data[pos]; pos += sizeof(int);
				dataOut[cameraId] = Camera();
				int deserialLength = dataOut[cameraId].deserialization(&data[pos]); pos += deserialLength;
			}
			return pos;
		}
		static int figureCameraHash(const Camera& c)
		{
			std::stringstream ss;
			ss << static_cast<int>(c.fx * 10000)
				<< static_cast<int>(c.fy * 10000)
				<< static_cast<int>(c.cx * 10000)
				<< static_cast<int>(c.cy * 10000)
				<< static_cast<int>(c.height)
				<< static_cast<int>(c.height);
			std::hash<std::string> hash_fn;
			return hash_fn(ss.str());
		}
		cv::Mat getImgDirs()const
		{
			return imgDirs;
		}
		double fx, fy, cx, cy;
		int height, width;
		Eigen::Matrix4f intr;
		cv::Mat imgDirs;
	};
	struct trainTerm1
	{
		float r, g, b;
		std::vector<float>worldDirSh;
		int potentialGridCnt;
		int featLevelCnt;
		std::vector<std::uint32_t>featsId;
	};
	struct View
	{
		int cameraId;
		Eigen::Matrix4f Rt;
		std::filesystem::path imgPath;
		std::filesystem::path maskPath;
		//std::unordered_map<std::uint32_t, std::unordered_set<std::uint32_t>>pixelGridBelong;
		std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>pixelGridBelong;
		std::vector<char>serialization()const
		{
			std::vector<char>sndata(sizeof(int));
			*(int*)&sndata[0]= cameraId;
			std::vector<char>Rtdata(16*sizeof(float));
			float* Rtdata_ = (float*)&Rtdata[0];
			for (int i = 0; i < 16; i++)
			{
				Rtdata_[i] = Rt(i / 4, i % 4);
			}
			std::vector<char>imgPathData, maskPathData;
			transform(imgPath.string(), imgPathData);
			transform(maskPath.string(), maskPathData);
			std::list<std::uint32_t>pixelGridBelongData;//total:[key:size:element]
			pixelGridBelongData.emplace_back(pixelGridBelong.size());
			for (const auto&d: pixelGridBelong)
			{
				pixelGridBelongData.emplace_back(d.first);
				pixelGridBelongData.emplace_back(d.second.size());
				for (const auto& d2 : d.second)
				{
					pixelGridBelongData.emplace_back(d2);
				}
			}
			std::vector<char>pixelGridBelongData_(pixelGridBelongData.size()*sizeof(std::uint32_t));
			std::uint32_t* data = (std::uint32_t*)&pixelGridBelongData_[0];
			for (const auto&d: pixelGridBelongData)
			{
				*data = d;
				data++;
			}
			sndata.insert(sndata.end(), Rtdata.begin(), Rtdata.end());
			sndata.insert(sndata.end(), imgPathData.begin(), imgPathData.end());
			sndata.insert(sndata.end(), maskPathData.begin(), maskPathData.end());
			sndata.insert(sndata.end(), pixelGridBelongData_.begin(), pixelGridBelongData_.end());
			return sndata;
		}
		int deserialization(const char*data)
		{
			int pos = 0;
			cameraId = *(int*)data;
			pos += sizeof(int);
			const float* Rtdata_ = (float*)&data[pos];
			for (int i = 0; i < 16; i++)
			{
				Rt(i / 4, i % 4) = Rtdata_[i];
				pos += sizeof(float);
			}
			std::string line;
			pos += transform(&data[pos], line);
			imgPath = std::filesystem::path(line);
			pos += transform(&data[pos], line);
			maskPath = std::filesystem::path(line);
			const std::uint32_t* intData = (const std::uint32_t*)&data[pos];
			std::uint32_t i = 0;
			std::uint32_t mapSize = intData[i]; i++;
			pixelGridBelong.clear();
			for (int pixel = 0; pixel < mapSize; pixel++)
			{
				std::uint32_t key_ = intData[i]; i++;
				std::uint32_t memberSize_ = intData[i]; i++;
				pixelGridBelong[key_].resize(memberSize_);
				for (int j = 0; j < memberSize_; j++)
				{
					pixelGridBelong[key_][j] = intData[i]; i++;
				}
			}
			return pos+sizeof(std::uint32_t)*i;
		}
		static std::vector<char>serialization(const std::unordered_map<int, View>& views)
		{
			std::vector<char>ret(sizeof(int));;
			*(int*)&ret[0] = views.size();
			for (const auto&d: views)
			{
				std::vector<char>ret0(sizeof(int));
				*(int*)&ret0[0] = d.first;
				std::vector<char>ret1 = d.second.serialization();
				ret.insert(ret.end(), ret0.begin(), ret0.end());
				ret.insert(ret.end(), ret1.begin(), ret1.end());
			}
			return ret;
		}
		static int deserialization(const char*data, std::unordered_map<int, View>&dataOut)
		{
			dataOut.clear();
			int elemCnt = *(int*)data;
			int pos = sizeof(int);
			for (int i = 0; i < elemCnt; i++)
			{
				int viewId = *(int*)&data[pos]; pos += sizeof(int);
				dataOut[viewId] = View();
				int deserialLength = dataOut[viewId].deserialization(&data[pos]); pos += deserialLength;				
			}
			return 0;
		}
		cv::Mat getImgWorldDir(const Camera&c)
		{
			if (this->imgWorldDir.empty())
			{
				const cv::Mat& imgDir = c.getImgDirs();
				const Eigen::Matrix4f Rtinv = this->Rt.inverse();
				imgWorldDir = cv::Mat::zeros(imgDir.size(), CV_32FC3);
				Eigen::Matrix4Xf A(4, imgWorldDir.rows* imgWorldDir.cols);
				for (int i = 0; i < A.cols(); i++)
				{
					int r = i / imgWorldDir.cols;
					int c = i % imgWorldDir.cols;
					const cv::Vec3f& dir = imgDir.at<cv::Vec3f>(r, c);
					A(0, i) = dir[0];
					A(1, i) = dir[1];
					A(2, i) = dir[2];
					A(3, i) = 1.;
				}
				Eigen::Matrix4Xf B= Rtinv *A;
				for (int i = 0; i < A.cols(); i++)
				{
					int r = i / imgWorldDir.cols;
					int c = i % imgWorldDir.cols;
					imgWorldDir.at<cv::Vec3f>(r, c)[0] = B(0, i);
					imgWorldDir.at<cv::Vec3f>(r, c)[1] = B(1, i);
					imgWorldDir.at<cv::Vec3f>(r, c)[2] = B(2, i);
				}
			}
			return this->imgWorldDir;
		}
		cv::Mat getImgWorldDir()const
		{
			return this->imgWorldDir;
		}
		cv::Mat imgWorldDir;
	};
	struct SurfData
	{
#define XyzToGridXYZ(x,y,z,xStart,yStart,zStart,gridUnitInv,gridX,gridY,gridZ) \
		int gridX = (x-xStart)*gridUnitInv;  \
		int gridY = (y-yStart)*gridUnitInv;  \
		int gridZ = (z-zStart)*gridUnitInv;  
#define GridXyzToPosEncode(gridX,gridY,gridZ,resolutionX,resolutionXY,posEncode) \
		std::uint32_t posEncode = gridX+gridY*resolutionX+gridZ*resolutionXY;
#define PosEncodeToGridXyz(posEncode,resolutionX,resolutionXY,gridX,gridY,gridZ) \
		std::uint32_t gridZ = posEncode / resolutionXY;							\
		std::uint32_t gridY = posEncode % resolutionXY;							\
		std::uint32_t gridX = gridY % resolutionX;								\
		gridY = gridY / resolutionX;
#define ImgXyToPosEncode(x,y,imgWidth)  (imgWidth*y+x)
#define PosEncodeToImgXy(posEncode,x,y,imgWidth)  \
		int y = posEncode / imgWidth;				\
		int x = posEncode & imgWidth;

		SurfData() {}
		SurfData(const std::filesystem::path&dataDir_, const std::filesystem::path& objPath_, const GridConfig& gridConfig_)
		{
			dataDir = dataDir_;
			objPath = objPath_;
			gridConfig = gridConfig_;
			this->gridUnitInv = 1. / gridConfig.gridUnit;
			this->vertex.resize(0, 0);
			{
				bool loadObj = surf::readObj(objPath, this->vertex, this->faces, this->vertex_normal, this->face_normal);
				if (!loadObj)
				{
					LOG_ERR_OUT << "read obj fail : " << objPath;
				}
				float objMinX = this->vertex.row(0).minCoeff();
				float objMinY = this->vertex.row(1).minCoeff();
				float objMinZ = this->vertex.row(2).minCoeff();
				float objMaxX = this->vertex.row(0).maxCoeff();
				float objMaxY = this->vertex.row(1).maxCoeff();
				float objMaxZ = this->vertex.row(2).maxCoeff();
				this->targetMinBorderX = objMinX - 0.06 * (objMaxX - objMinX);
				this->targetMinBorderY = objMinY - 0.06 * (objMaxY - objMinY);
				this->targetMinBorderZ = objMinZ - 0.06 * (objMaxZ - objMinZ);
				this->targetMaxBorderX = objMaxX + 0.06 * (objMaxX - objMinX);
				this->targetMaxBorderY = objMaxY + 0.06 * (objMaxY - objMinY);
				this->targetMaxBorderZ = objMaxZ + 0.06 * (objMaxZ - objMinZ);

				this->resolutionX = (this->targetMaxBorderX - this->targetMinBorderX) / gridConfig.gridUnit + 1;
				this->resolutionY = (this->targetMaxBorderY - this->targetMinBorderY) / gridConfig.gridUnit + 1;
				this->resolutionZ = (this->targetMaxBorderZ - this->targetMinBorderZ) / gridConfig.gridUnit + 1;
				if (!isValidResolutions(this->resolutionX, this->resolutionY, this->resolutionZ))
				{
					LOG_ERR_OUT << "grid unit too small!!!";
					exit(-1);
				}
				this->resolutionXY = this->resolutionX * this->resolutionY;

			}
			for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{ dataDir_ })
			{
				const auto& thisFilename = dir_entry.path();
				if (thisFilename.has_extension())
				{
					const auto& ext = thisFilename.extension().string();
					if (ext.compare(".json") == 0)
					{
						LOG_OUT << thisFilename;
						std::stringstream ss;
						std::string aline;
						std::fstream fin(thisFilename, std::ios::in);
						while (std::getline(fin, aline))
						{
							ss << aline;
						}
						fin.close();
						aline = ss.str();
						JSONCPP_STRING err;
						Json::Value newRoot;
						const auto rawJsonLength = static_cast<int>(aline.length());
						Json::CharReaderBuilder newBuilder;
						const std::unique_ptr<Json::CharReader> newReader(newBuilder.newCharReader());
						if (!newReader->parse(aline.c_str(), aline.c_str() + rawJsonLength, &newRoot,
							&err)) {
							LOG_ERR_OUT << "parse json fail.";
							return;
						}
						try
						{


							auto newMemberNames = newRoot.getMemberNames();
							if (std::find(newMemberNames.begin(), newMemberNames.end(), "imagePath") == newMemberNames.end())
							{
								continue;
							}
							if (std::find(newMemberNames.begin(), newMemberNames.end(), "fx") == newMemberNames.end())
							{
								continue;
							}
							if (std::find(newMemberNames.begin(), newMemberNames.end(), "fy") == newMemberNames.end())
							{
								continue;
							}
							if (std::find(newMemberNames.begin(), newMemberNames.end(), "cx") == newMemberNames.end())
							{
								continue;
							}
							if (std::find(newMemberNames.begin(), newMemberNames.end(), "cy") == newMemberNames.end())
							{
								continue;
							}
							if (std::find(newMemberNames.begin(), newMemberNames.end(), "height") == newMemberNames.end())
							{
								continue;
							}
							if (std::find(newMemberNames.begin(), newMemberNames.end(), "width") == newMemberNames.end())
							{
								continue;
							}
							if (std::find(newMemberNames.begin(), newMemberNames.end(), "Qt") == newMemberNames.end())
							{
								continue;
							}
						}
						catch (...)
						{
							continue;
						}
						std::filesystem::path imgPath;
						double fx = 0;
						double fy = 0;
						double cx = 0;
						double cy = 0;
						int height = 0;
						int width = 0;
						Eigen::Matrix4f Rt = Eigen::Matrix4f::Identity();
						try
						{
							imgPath = std::filesystem::path(newRoot["imagePath"].asString());
							imgPath = std::filesystem::canonical(imgPath);
							fx = newRoot["fx"].asDouble();
							fy = newRoot["fy"].asDouble();
							cx = newRoot["cx"].asDouble();
							cy = newRoot["cy"].asDouble();
							height = newRoot["height"].asInt();
							width = newRoot["width"].asInt();
							if (7 != newRoot["Qt"].size())
							{
								throw std::exception();
							}
							auto QtArray = newRoot["Qt"];
							float w = QtArray[0].asFloat();
							float x = QtArray[1].asFloat();
							float y = QtArray[2].asFloat();
							float z = QtArray[3].asFloat();
							Eigen::Quaternionf q(w, x, y, z);
							Rt.block(0, 0, 3, 3) = q.matrix();
							Rt(0, 3) = QtArray[4].asFloat();
							Rt(1, 3) = QtArray[5].asFloat();
							Rt(2, 3) = QtArray[6].asFloat();
						}
						catch (...)
						{
							LOG_ERR_OUT << "parse json fail : " << thisFilename;
							continue;
						}
						Camera thisCamera(fx, fy, cx, cy, height, width);
						int thisCameraSn = Camera::figureCameraHash(thisCamera);
						if (this->cameras.count(thisCameraSn) == 0)
						{
							this->cameras[thisCameraSn] = thisCamera;
						}
						if (!std::filesystem::exists(imgPath))
						{
							LOG_ERR_OUT << "not found : " << imgPath;
							continue;
						}
						auto shortName = imgPath.filename().stem().string();
						std::filesystem::path maskPath = imgPath.parent_path() / ("mask_" + shortName + ".dat");
						if (!std::filesystem::exists(maskPath))
						{
							LOG_ERR_OUT << "not found : " << maskPath;
							continue;
						}
						cv::Mat img = cv::imread(imgPath.string());
						cv::Mat mask = tools::loadMask(maskPath.string());
						if (img.rows != mask.rows || img.cols != mask.cols)
						{
							LOG_ERR_OUT << "img.rows != mask.rows || img.cols != mask.cols";
							continue;
						}
						View thisView = { thisCameraSn, Rt ,imgPath ,maskPath };
						int viewIdx = this->views.size();
						this->views[viewIdx]=thisView;
						cv::Mat rayDistance = cv::Mat::ones(mask.size(), CV_32FC1) * -1;
						const Eigen::Matrix3f R = Rt.block(0, 0, 3, 3);
						const Eigen::Matrix4f KRt = thisCamera.intr * Rt;
						std::vector<std::int8_t> visualFace(this->faces.cols(), 0);
						Eigen::Vector3f cameraT(Rt(0, 3), Rt(1, 3), Rt(2, 3));
						{
							std::vector<float>distFromT(this->faces.cols());
#pragma omp parallel for
							for (int i = 0; i < this->faces.cols(); i++)
							{
								const int& v = faces(0, i);//choose the first vertex idx
								const float a = Rt(0, 3) - this->vertex(0, v);
								const float b = Rt(1, 3) - this->vertex(1, v);
								const float c = Rt(2, 3) - this->vertex(2, v);
								float dirSign = a * this->face_normal(0, i) + b * this->face_normal(1, i) + c * this->face_normal(2, i);
								if (dirSign > 0)
								{
									visualFace[i] = 1;
								}
								else if (dirSign < 0)
								{
									visualFace[i] = -1;
								}
								distFromT[i] = a * a + b * b + c * c;
							}
							int neareatFaceIdx = std::min_element(distFromT.begin(), distFromT.end()) - distFromT.begin();
							std::int8_t nearestFaceNormalSigned = visualFace[neareatFaceIdx];
#pragma omp parallel for
							for (int i = 0; i < visualFace.size(); i++)
							{
								visualFace[i] *= nearestFaceNormalSigned;
							}
						}
						std::vector<cv::Vec2i>vertexInImg(this->vertex.cols());
						std::vector<float>vertexDist(this->vertex.cols());
						{
							Eigen::Matrix4Xf vertexInView = Rt * this->vertex;
							std::vector<bool> vertexInViewInCanvas(this->vertex.cols(), false);
							float fxInv = 1. / this->cameras[thisCameraSn].fx;
							float fyInv = 1. / this->cameras[thisCameraSn].fy;
#pragma omp parallel for
							for (int i = 0; i < this->vertex.cols(); i++)
							{
								vertexDist[i] = sqrt(vertexInView(0, i) * vertexInView(0, i) + vertexInView(1, i) * vertexInView(1, i) + vertexInView(2, i) * vertexInView(2, i));
								vertexInImg[i][0] = vertexInView(0, i) / vertexInView(2, i) * cameras[thisCameraSn].fx + cameras[thisCameraSn].cx + 0.5;
								vertexInImg[i][1] = vertexInView(1, i) / vertexInView(2, i) * cameras[thisCameraSn].fy + cameras[thisCameraSn].cy + 0.5;
								if (vertexInImg[i][0] >= 0 && vertexInImg[i][1] >= 0 && vertexInImg[i][0] < width && vertexInImg[i][1] < height)
								{
									vertexInViewInCanvas[i] = true;
								}
							}
							for (int i = 0; i < this->faces.cols(); i++)
							{
								if (visualFace[i] > 0)
								{
									const auto& a = this->faces(0, i);
									const auto& b = this->faces(1, i);
									const auto& c = this->faces(2, i);
									if (vertexInViewInCanvas[a] && vertexInViewInCanvas[b] && vertexInViewInCanvas[c])
									{
										const cv::Vec2i& uv0 = vertexInImg[a];
										const cv::Vec2i& uv1 = vertexInImg[b];
										const cv::Vec2i& uv2 = vertexInImg[c];
										std::list<cv::Vec2i> trianglePixel = triangle(uv0, uv1, uv2);
										for (const auto& d : trianglePixel)
										{
											float& dist = rayDistance.ptr<float>(d[1])[d[0]];
											if (dist<0 || dist>vertexDist[a])
											{
												dist = vertexDist[a];
											}
										}
									}
								}
							}
						}
						if (false)
						{//show rayDistant
							Eigen::MatrixXf pts = rayDistantMapToPts(rayDistance, this->cameras[thisCameraSn], Rt);
							tools::saveColMajorPts3d("../surf/a.ply", pts);
						}
						this->registerRayDistanceMap(rayDistance, this->cameras[thisCameraSn], this->views[viewIdx]);
					}
				}
			}
			std::vector<char>d = this->serialization();
			std::fstream fout("../surf/d.dat",std::ios::out|std::ios::binary);
			int dCnt = d.size();
			fout.write((char*)&dCnt, sizeof(int));
			fout.write(&d[0], dCnt* sizeof(char));
			fout.close();
		}
		static bool isValidResolutions(const int& resolutionX, const int& resolutionY, const int& resolutionZ)
		{
			if (resolutionX <= 0 || resolutionY <= 0 || resolutionZ <= 0)
			{
				return false;
			}
			int xy = resolutionX * resolutionY;
			int xz = resolutionX * resolutionZ;
			int zy = resolutionZ * resolutionY;
			if ((xy / resolutionX != resolutionY) || (xz / resolutionX != resolutionZ) || (zy / resolutionY != resolutionZ))
			{
				return false;
			}
			int xyz = xy * resolutionZ;
			if (xyz/ xy != resolutionZ)
			{
				return false;
			}
			return true;
		}
		bool registerRayDistanceMap(const cv::Mat& rayDistantMap, Camera& camera, View& view)
		{
			std::unordered_map<std::uint32_t, std::unordered_set<std::uint32_t>>pixelGridBelong_map;
			cv::Mat imgDirs = camera.getImgDirs();
			const Eigen::Matrix4f Rtinv = view.Rt.inverse();
			const Eigen::Matrix3f Rinv = Rtinv.block(0, 0, 3, 3);
			const Eigen::Vector3f tinv(Rtinv(0, 3), Rtinv(1, 3), Rtinv(2, 3));
			float rayStepUnit = this->gridConfig.gridUnit * 0.5;
			int amplitudeCnt = 2 * this->gridConfig.amplitudeHalf / gridConfig.gridUnit;
			{				
				for (int r = 0; r < camera.height; r++)
				{
					for (int c = 0; c < camera.width; c++)
					{
						const float& rayDist = rayDistantMap.ptr<float>(r)[c];
						if (1e-3 < rayDist)
						{
							Eigen::Vector3f pixelDir = Eigen::Vector3f(imgDirs.at<cv::Vec3f>(r, c)[0], imgDirs.at<cv::Vec3f>(r, c)[1], imgDirs.at<cv::Vec3f>(r, c)[2]);
							std::uint32_t imgPixelPos = ImgXyToPosEncode(c, r, camera.width);
							for (int i = 0; ; i++)
							{
								{
									Eigen::Vector3f p = (rayDist - i * rayStepUnit) * pixelDir;
									if (p[0] < this->targetMinBorderX || p[1] < this->targetMinBorderY || p[2] < this->targetMinBorderZ
										|| p[0] >= this->targetMaxBorderX || p[1] >= this->targetMaxBorderY || p[2] >= this->targetMaxBorderZ)
									{
										continue;
									}
									Eigen::Vector3f p1 = Rinv * p + tinv;
									XyzToGridXYZ(p1[0], p1[0], p1[2], this->targetMinBorderX, this->targetMinBorderY, this->targetMinBorderZ, this->gridUnitInv, gridX, gridY, gridZ);
									GridXyzToPosEncode(gridX, gridY, gridZ, this->resolutionX, this->resolutionXY, posEncode);
									pixelGridBelong_map[imgPixelPos].emplace(posEncode);
								}
								if (pixelGridBelong_map[imgPixelPos].size()>= amplitudeCnt)
								{
									break;
								}
								{
									Eigen::Vector3f p = (rayDist + i * rayStepUnit) * pixelDir;
									if (p[0] < this->targetMinBorderX || p[1] < this->targetMinBorderY || p[2] < this->targetMinBorderZ
										|| p[0] >= this->targetMaxBorderX || p[1] >= this->targetMaxBorderY || p[2] >= this->targetMaxBorderZ)
									{
										continue;
									}
									Eigen::Vector3f p1 = Rinv * p + tinv;
									XyzToGridXYZ(p1[0], p1[0], p1[2], this->targetMinBorderX, this->targetMinBorderY, this->targetMinBorderZ, this->gridUnitInv, gridX, gridY, gridZ);
									GridXyzToPosEncode(gridX, gridY, gridZ, this->resolutionX, this->resolutionXY, posEncode);
									pixelGridBelong_map[imgPixelPos].emplace(posEncode);
								}
								if (pixelGridBelong_map[imgPixelPos].size() >= amplitudeCnt)
								{
									break;
								}
							}
						}
					}
				}
				for (const auto&d: pixelGridBelong_map)
				{
					int elemSize = d.second.size();
					view.pixelGridBelong[d.first].clear();
					view.pixelGridBelong[d.first].reserve(elemSize);
					for (const auto&d2:d.second)
					{
						view.pixelGridBelong[d.first].emplace_back(d2);
					}
				}
			}

			return true;
		}
		static Eigen::MatrixXf rayDistantMapToPts(const cv::Mat& rayDistantMap, const Camera& camera, const Eigen::Matrix4f& viewRt)
		{
			Eigen::MatrixXf ret;
			cv::Mat imgDirs = camera.getImgDirs();
			{
				std::list<Eigen::Vector3f>listRet;
				for (int r = 0; r < camera.height; r++)
				{
					for (int c = 0; c < camera.width; c++)
					{
						const float& rayDist = rayDistantMap.ptr<float>(r)[c];
						if (1e-3 < rayDist)
						{
							listRet.emplace_back(rayDist * Eigen::Vector3f(imgDirs.at<cv::Vec3f>(r, c)[0], imgDirs.at<cv::Vec3f>(r, c)[1], imgDirs.at<cv::Vec3f>(r, c)[2]));
						}
					}
				}
				ret.resize(4, listRet.size());
				int i = 0;
				for (const auto&d: listRet)
				{
					ret(0, i) = d.x();
					ret(1, i) = d.y();
					ret(2, i) = d.z();
					ret(3, i) = 1;
					i += 1;
				}
			}
			const Eigen::Matrix4f Rtinv = viewRt.inverse();
			return Rtinv* ret;
		}
		std::vector<char>serialization()const
		{
			std::vector<char>ret;
			std::vector<char>imgPathData, maskPathData, vertexData, facesData, vertex_normalData, face_normalData;
			transform(this->dataDir.string(), imgPathData);
			transform(this->objPath.string(), maskPathData);
			transform(this->vertex, vertexData);
			transform(this->faces, facesData);
			transform(this->vertex_normal, vertex_normalData);
			transform(this->face_normal, face_normalData);
			ret.insert(ret.end(), imgPathData.begin(), imgPathData.end());
			ret.insert(ret.end(), maskPathData.begin(), maskPathData.end());
			ret.insert(ret.end(), vertexData.begin(), vertexData.end());
			ret.insert(ret.end(), facesData.begin(), facesData.end());
			ret.insert(ret.end(), vertex_normalData.begin(), vertex_normalData.end());
			ret.insert(ret.end(), face_normalData.begin(), face_normalData.end());
			{
				std::vector<char>floatDat(7*sizeof(float));
				*(float*)&floatDat[0 * sizeof(float)]= targetMinBorderX;
				*(float*)&floatDat[1 * sizeof(float)] = targetMinBorderY;
				*(float*)&floatDat[2 * sizeof(float)] = targetMinBorderZ;
				*(float*)&floatDat[3 * sizeof(float)] = targetMaxBorderX;
				*(float*)&floatDat[4 * sizeof(float)] = targetMaxBorderY;
				*(float*)&floatDat[5 * sizeof(float)]= targetMaxBorderZ;
				*(float*)&floatDat[6 * sizeof(float)] = gridUnitInv;
				std::vector<char>intDat(4 * sizeof(float));
				*(int*)&intDat[0 * sizeof(int)] = resolutionX;
				*(int*)&intDat[1 * sizeof(int)] = resolutionY;
				*(int*)&intDat[2 * sizeof(int)] = resolutionZ;
				*(int*)&intDat[3 * sizeof(int)] = resolutionXY;
				ret.insert(ret.end(), floatDat.begin(), floatDat.end());
				ret.insert(ret.end(), intDat.begin(), intDat.end());
			}
			std::vector<char>configDat = gridConfig.serialization();
			ret.insert(ret.end(), configDat.begin(), configDat.end());
			std::vector<char>camerasData = Camera::serialization(this->cameras);
			ret.insert(ret.end(), camerasData.begin(), camerasData.end());
			std::vector<char>viewData = View::serialization(this->views);
			ret.insert(ret.end(), viewData.begin(), viewData.end());
			return ret;
		}
		bool reload(const std::filesystem::path&path)
		{
			std::fstream fin(path,std::ios::in|std::ios::binary);
			int totalSize = 0;
			fin.read((char*)&totalSize,sizeof(int));
			std::vector<char>dat(totalSize);
			fin.read((char*)&dat[0], totalSize* sizeof(char));
			int pos = 0;
			std::string line;
			pos += transform(&dat[pos], line); this->dataDir = std::filesystem::path(line);
			pos += transform(&dat[pos], line); this->objPath = std::filesystem::path(line);
			pos += transform(&dat[pos], this->vertex);
			pos += transform(&dat[pos], this->faces);
			pos += transform(&dat[pos], this->vertex_normal);
			pos += transform(&dat[pos], this->face_normal);
			{
				targetMinBorderX = *(float*)&dat[pos]; pos += sizeof(float);
				targetMinBorderY = *(float*)&dat[pos]; pos += sizeof(float);
				targetMinBorderZ = *(float*)&dat[pos]; pos += sizeof(float);
				targetMaxBorderX = *(float*)&dat[pos]; pos += sizeof(float);
				targetMaxBorderY = *(float*)&dat[pos]; pos += sizeof(float);
				targetMaxBorderZ = *(float*)&dat[pos]; pos += sizeof(float);
				gridUnitInv = *(float*)&dat[pos]; pos += sizeof(float);
				resolutionX = *(int*)&dat[pos]; pos += sizeof(int);
				resolutionY = *(int*)&dat[pos]; pos += sizeof(int);
				resolutionZ = *(int*)&dat[pos]; pos += sizeof(int);
				resolutionXY = *(int*)&dat[pos]; pos += sizeof(int);
			}
			pos += gridConfig.deserialization(&dat[pos]);
			pos += Camera::deserialization(&dat[pos], this->cameras);
			pos += View::deserialization(&dat[pos], this->views);
			for (auto&v:this->views)
			{
				const auto& cameraId = v.second.cameraId;
				auto& thisCamera = this->cameras.at(cameraId);
				v.second.getImgWorldDir(thisCamera);
			}
			return true;
		}
		std::list<trainTerm1> getTrainDat()const
		{
			std::list<trainTerm1>ret;
			/// <summary>
			///    RGB  |  xyz0 xyz1 2 3 ...  pixelDir
			///         (map: posIdx->mapKey)     xyz[posEncode...]  ->  feat0gridID[0,1,2,3...]    ->  feat0gridID
			///									                     ->  feat1gridID[0,2,4,6...]    ->  feat1gridID
			///									                     ->  feat2gridID[0,4,8,12...]   ->  feat2gridID
			///									                     ->  feat3gridID[0,8,16,24...]  ->  feat3gridID
			std::vector<std::unordered_map<std::uint32_t, std::uint32_t>>featsId(this->gridConfig.gridLevelCnt);
			for (const auto&v:views)
			{
				const auto& viewId = v.first;
				const auto& cameraId = v.second.cameraId;
				const auto& thisCamera = this->cameras.at(cameraId);
				const auto& Rt = v.second.Rt;
				const auto& imgWorldDir = v.second.getImgWorldDir();
				cv::Mat img = cv::imread(v.second.imgPath.string());
				if (img.empty())
				{
					LOG_ERR_OUT << "not found : " << v.second.imgPath;
					return std::list<trainTerm1>();;
				}
				for (const auto&pixel: v.second.pixelGridBelong)
				{
					trainTerm1 dat;
					const auto& pixelId = pixel.first;
					PosEncodeToImgXy(pixelId, imgX, imgY, thisCamera.width);
					const float& wordDirX = imgWorldDir.at<cv::Vec3f>(imgY, imgX)[0];
					const float& wordDirY = imgWorldDir.at<cv::Vec3f>(imgY, imgX)[1];
					const float& wordDirZ = imgWorldDir.at<cv::Vec3f>(imgY, imgX)[2];
					std::vector<float>dirEncode;
					shEncoder(wordDirX, wordDirY, wordDirZ, dat.worldDirSh);
					dat.b = img.at<cv::Vec3b>(imgY, imgX)[0] * 0.0078125 - 1;
					dat.g = img.at<cv::Vec3b>(imgY, imgX)[1] * 0.0078125 - 1;
					dat.r = img.at<cv::Vec3b>(imgY, imgX)[2] * 0.0078125 - 1;
					dat.featLevelCnt = this->gridConfig.gridLevelCnt;
					dat.potentialGridCnt = pixel.second.size();
					std::vector<std::uint32_t>&thisPixelFeatId= dat.featsId;
					thisPixelFeatId.reserve(this->gridConfig.gridLevelCnt * pixel.second.size());
					for (const auto&rayPtPosEncode : pixel.second)
					{
						PosEncodeToGridXyz(rayPtPosEncode, this->resolutionX, this->resolutionXY, gridX, gridY, gridZ);
						if (featsId[0].count(rayPtPosEncode)==0)
						{
							std::uint32_t newId = featsId[0].size();
							featsId[0][rayPtPosEncode] = newId;
						}
						thisPixelFeatId.emplace_back(featsId[0][rayPtPosEncode]);
						for (int gridLevel = 1; gridLevel < this->gridConfig.gridLevelCnt; gridLevel++)
						{
							std::uint32_t gridX_temp = gridX>>gridLevel;
							std::uint32_t gridY_temp = gridY>>gridLevel;
							std::uint32_t gridZ_temp = gridZ>>gridLevel;
							GridXyzToPosEncode(gridX_temp, gridY_temp, gridZ_temp, this->resolutionX, this->resolutionXY, posEncode_temp);
							if (featsId[gridLevel].count(posEncode_temp) == 0)
							{
								std::uint32_t newId = featsId[gridLevel].size();
								featsId[gridLevel][posEncode_temp] = newId;
							}
							thisPixelFeatId.emplace_back(featsId[gridLevel][posEncode_temp]);
						}
					}
					ret.emplace_back(dat);
				}
			}
			return ret;
		}
		std::filesystem::path dataDir;
		std::filesystem::path objPath;
		Eigen::MatrixXf vertex;
		Eigen::MatrixXi faces;
		Eigen::MatrixXf vertex_normal;
		Eigen::MatrixXf face_normal;
		float targetMinBorderX;
		float targetMinBorderY;
		float targetMinBorderZ;
		float targetMaxBorderX;
		float targetMaxBorderY;
		float targetMaxBorderZ;
		float gridUnitInv;
		int resolutionX;
		int resolutionY;
		int resolutionZ;
		int resolutionXY;
		std::unordered_map<int, Camera>cameras;
		std::unordered_map<int, View>views;
		GridConfig gridConfig;
		
	};
	bool saveTrainData(const std::filesystem::path&path,const std::list<surf::trainTerm1>&data)
	{
		std::fstream fout(path, std::ios::out | std::ios::binary);
		int dataType = 1;
		fout.write((char*)&dataType, sizeof(dataType));
		int itemCnt = data.size();
		fout.write((char*)&itemCnt, sizeof(itemCnt));
		fout.write((char*)&data.begin()->featLevelCnt, sizeof(int));
		fout.write((char*)&data.begin()->potentialGridCnt, sizeof(int));
		int shSize = SH_DEGREE * SH_DEGREE;
		fout.write((char*)&shSize, sizeof(int));
		for (const auto&d:data)
		{
			fout.write((char*)&d.r, sizeof(float));
			fout.write((char*)&d.g, sizeof(float));
			fout.write((char*)&d.b, sizeof(float));
			fout.write((char*)&d.worldDirSh[0], d.worldDirSh.size()*sizeof(float));
			fout.write((char*)&d.featsId[0], d.featsId.size() * sizeof(std::uint32_t));
		}
		fout.close();
		return true;
	}
} 
int test_surf()
{
	float totalAmplitude = 0.3;//measured from obj data manually
	float gridUnit = totalAmplitude / 12;// the Amplitude, i wang to separet it into 16 picecs
	surf::GridConfig gridConfig = { totalAmplitude /2,gridUnit ,4};
	surf::SurfData asd("../data/a/result", "../data/a/result/dense.obj", gridConfig);

	surf::SurfData asd2;
	asd2.reload("../surf/d.dat");
	std::list<surf::trainTerm1>trainDat = asd2.getTrainDat();
	surf::saveTrainData("../surf/trainTerm1.dat", trainDat);
	return 0;
}
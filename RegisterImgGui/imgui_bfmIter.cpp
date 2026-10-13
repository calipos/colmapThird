#include <unordered_set>
#include <numeric>
#include <vector>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <map>
#include "log.h"
#include "Eigen/Core"
#include "opencv2/opencv.hpp"
#include "imgui.h" 
#include "browser.h"
#include "imgui_tools.h"
#include "imgui_bfmIter.h"
#include "labelme.h"
#include "bfm.h"
#include "meshDraw.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
static ProgressThread progress;
 
static BfmIter* BfmIterManger = nullptr;
namespace draw
{
	std::string getfullShortName(const std::filesystem::path& path_, const std::filesystem::path& baseDir_)
	{
		std::string shortName = path_.filename().stem().string();
		std::filesystem::path path = std::filesystem::canonical(path_);
		std::filesystem::path baseDir = std::filesystem::canonical(baseDir_);
		std::list<std::string>parentNames;
		std::filesystem::path parent = std::filesystem::canonical(path.parent_path());
		while (parent.compare(baseDir) != 0)
		{
			parentNames.emplace_back(parent.filename().stem().string());
			path = path.parent_path();
			parent = std::filesystem::canonical(path.parent_path());
		}
		std::string fullShortName;
		for (const auto& d : parentNames)
		{
			fullShortName = "_" + fullShortName;
			fullShortName = d + fullShortName;
		}
		fullShortName = fullShortName + shortName;
		return fullShortName;
	}
	struct BSplineGuider
	{
		BSplineGuider()
		{
		}
		std::vector< std::map<std::string, Eigen::RowVector3f>>bfmMeshControlPts;
		int lastControlIdx{ -1 };
		std::map<std::string, ImU32> colors;
		std::vector<std::string>tagsListName;
		std::map<std::string, std::vector<std::string>> hasTagFlags;
		static int tagPickIdx;
		bool pickedChanged;
		static cv::Mat shiftSnap(const cv::Mat& img, const ImVec2& shift)
		{
			cv::Mat rotation_matix = cv::Mat::eye(2,3,CV_32FC1);
			rotation_matix.ptr<float>(0)[2] = shift.x;
			rotation_matix.ptr<float>(1)[2] = shift.y;
			cv::Mat rotated_image;
			warpAffine(img, rotated_image, rotation_matix, img.size()); 
			return rotated_image;
		}
		ImVec2 drawPos;
	}; 
	int BSplineGuider::tagPickIdx = -1;
	class Draw
	{
	private:
		int currentImgHeight;
		int currentImgWidth;
		static Draw* instance;
		static ImVec2 zoom_start;
		static ImVec2 zoom_end;
		static GLuint image_texture;
		Draw() {};
		Draw(const Draw&) = delete;
		Draw& operator=(const Draw&) = delete;
	public:
		ImVec2 mouseDownStartPos; 
		BSplineGuider guideData;
		static ImVec2 draw_pos;
		static ImVec2 canvas;//(w,h)
		static ImVec2 canvasInv;//(1/w,1/h)
		static float resizeFactor;
		static int canvasMaxSide;
		bool hasImageContext{ false };
		static ImVec2 imgPt2GuiPt(const ImVec2& imgPt, const int& imgHeight, const int& imgWidth, const ImVec2& canvas_size, const ImVec2& zoom_start, const ImVec2& zoom_end, const ImVec2& canvas_location)
		{
			float x_inRatio = (imgPt.x / imgWidth - zoom_start.x) / (zoom_end.x - zoom_start.x);
			float y_inRatio = (imgPt.y / imgHeight - zoom_start.y) / (zoom_end.y - zoom_start.y);
			float x = x_inRatio * canvas_size.x + canvas_location.x + ImGui::GetWindowPos().x;
			float y = y_inRatio * canvas_size.y + canvas_location.y + ImGui::GetWindowPos().y;
			return ImVec2(x, y);
		}
		static Draw* getDrawIns(const ImVec2& draw_pos_, const int& canvasMaxSide_, const std::vector<std::string>& picShortNameForlist, const std::vector<std::filesystem::path>& picPaths)
		{
			if (Draw::instance == nullptr)
			{
				Draw::instance = new Draw();
				instance->guideData.bfmMeshControlPts.resize(picShortNameForlist.size()); 
				instance->mouseDownStartPos.x = -1;
			}
			else
			{
				return Draw::instance;
			}
			canvasMaxSide = canvasMaxSide_;
			draw_pos = draw_pos_;
			canvas = ImVec2(0, 0);
			zoom_start = ImVec2(0, 0);
			zoom_end = ImVec2(1, 1);
			return instance;
		}
		bool feedImg(const cv::Mat& img = cv::Mat())
		{
			if (img.empty())
			{
				hasImageContext = false;
			}
			else
			{
				hasImageContext = true;
				LoadTextureFromMat(img, &image_texture, &currentImgWidth, &currentImgHeight);
				float a = 1.f * currentImgWidth / Draw::canvasMaxSide;
				float b = 1.f * currentImgHeight / Draw::canvasMaxSide;
				if (a < 1.f && b < 1.f)
				{
					Draw::canvas = ImVec2(currentImgWidth, currentImgHeight);
				}
				else if (a > b)
				{
					Draw::canvas = ImVec2(currentImgWidth / a, currentImgHeight / a);
				}
				else
				{
					Draw::canvas = ImVec2(currentImgWidth / b, currentImgHeight / b);
				}
				Draw::canvasInv.x = 1. / Draw::canvas.x;
				Draw::canvasInv.y = 1. / Draw::canvas.y;
			}
			return true;
		}
		bool draw(const int& imgPickIdx)
		{
			if (hasImageContext)
			{
				ImGui::SetCursorPos(draw_pos);
				ImGui::Image((ImTextureID)(intptr_t)image_texture, Draw::canvas, Draw::zoom_start, Draw::zoom_end, ImVec4(1, 1, 1, 1), ImVec4(.5, .5, .5, .5));
				
			}
			return true;
		}
		ImVec2 control(const ImVec2&canvasPos, ImVec2& frameOffset, ImVec2& offset)
		{
			ImVec2 mousePosInImage;
			mousePosInImage.x = (ImGui::GetIO().MousePos.x - ImGui::GetWindowPos().x - canvasPos.x);
			mousePosInImage.y = (ImGui::GetIO().MousePos.y - ImGui::GetWindowPos().y - canvasPos.y);
			if (mousePosInImage.x < 0 || mousePosInImage.y < 0 || mousePosInImage.x >= canvas.x || mousePosInImage.y >canvas.y)
			{
				if (mouseDownStartPos.x != -1)
				{
					frameOffset.x = (ImGui::GetIO().MousePos.x - mouseDownStartPos.x);
					frameOffset.y = (ImGui::GetIO().MousePos.y - mouseDownStartPos.y);
					offset.x += frameOffset.x;
					offset.y += frameOffset.y;
				}
				frameOffset.x = 0;
				frameOffset.y = 0;
				mouseDownStartPos = ImVec2(-1, -1);
				return ImVec2(-1, -1);;
			}
			if (ImGui::GetIO().MouseDown[0])
			{
				if (mouseDownStartPos.x == -1)
				{
					mouseDownStartPos = ImGui::GetIO().MousePos;
				}
				else
				{
					frameOffset.x = (ImGui::GetIO().MousePos.x - mouseDownStartPos.x);
					frameOffset.y = (ImGui::GetIO().MousePos.y - mouseDownStartPos.y);
				}
			} 
			if (ImGui::GetIO().MouseReleased[0])
			{
				if (mouseDownStartPos.x == ImGui::GetIO().MousePos.x)
				{
					frameOffset.x = 0;
					frameOffset.y = 0;
					mouseDownStartPos = ImVec2(-1, -1);
					float x_inRatio = mousePosInImage.x * Draw::canvasInv.x * (Draw::zoom_end.x - Draw::zoom_start.x) + Draw::zoom_start.x;
					float y_inRatio = mousePosInImage.y * Draw::canvasInv.y * (Draw::zoom_end.y - Draw::zoom_start.y) + Draw::zoom_start.y;
					float x_inPic = x_inRatio * currentImgWidth;
					float y_inPic = y_inRatio * currentImgHeight;
					return ImVec2(x_inPic, y_inPic);
				}
				else
				{
					frameOffset.x = (ImGui::GetIO().MousePos.x - mouseDownStartPos.x);
					frameOffset.y = (ImGui::GetIO().MousePos.y - mouseDownStartPos.y);
					offset.x += frameOffset.x;
					offset.y += frameOffset.y;
					frameOffset.x = 0;
					frameOffset.y = 0;
					mouseDownStartPos = ImVec2(-1, -1);
				}
			}

			
			float wheel = ImGui::GetIO().MouseWheel;
			if (abs(wheel) > 0.01)
			{
				if (mousePosInImage.x >= 0 && mousePosInImage.y >= 0 && mousePosInImage.x < canvas.x && mousePosInImage.y < canvas.y)
				{
					float x_inRatio = mousePosInImage.x * Draw::canvasInv.x * (Draw::zoom_end.x - Draw::zoom_start.x) + Draw::zoom_start.x;
					float y_inRatio = mousePosInImage.y * Draw::canvasInv.y * (Draw::zoom_end.y - Draw::zoom_start.y) + Draw::zoom_start.y;
					float x_inPic = x_inRatio * currentImgWidth;
					float y_inPic = y_inRatio * currentImgHeight;

					{
						//zoom out: resizeFactor=1 (0,0)->(1,1);
						//zoom in:  resizeFactor=4 (0.25,0.25)->(.75,.75);
						resizeFactor += (0.02 * wheel);//    
						if (resizeFactor > 0.375)
						{
							resizeFactor = 0.375;
						}
						if (resizeFactor < 0)
						{
							resizeFactor = 0;
						}
						float regionRadius = 0.5 - resizeFactor;
						float regionDiameter = 2 * resizeFactor;
						Draw::zoom_start.x = x_inRatio - regionRadius;
						Draw::zoom_start.y = y_inRatio - regionRadius;
						Draw::zoom_end.x = x_inRatio + regionRadius;
						Draw::zoom_end.y = y_inRatio + regionRadius;
						if (Draw::zoom_start.x < 0)
						{
							Draw::zoom_start.x = 0;
							Draw::zoom_end.x = (1 - regionDiameter);
						}
						if (Draw::zoom_start.y < 0)
						{
							Draw::zoom_start.y = 0;
							Draw::zoom_end.y = (1 - regionDiameter);
						}
						if (Draw::zoom_end.x > 1)
						{
							Draw::zoom_start.x = regionDiameter;
							Draw::zoom_end.x = 1;
						}
						if (Draw::zoom_end.y > 1)
						{
							Draw::zoom_start.y = regionDiameter;
							Draw::zoom_end.y = 1;
						}

					}
				}
			}
			return ImVec2(-1, -1);
		}
	};
	Draw* Draw::instance = nullptr;
	ImVec2 Draw::draw_pos = ImVec2();
	ImVec2 Draw::canvas = ImVec2();
	ImVec2 Draw::canvasInv = ImVec2();
	ImVec2 Draw::zoom_start = ImVec2();
	ImVec2 Draw::zoom_end = ImVec2();
	GLuint Draw::image_texture = 0;
	int Draw::canvasMaxSide = 0;
	float Draw::resizeFactor = 1;
}
Eigen::Matrix<float, 3, 4> solveAffine(const Eigen::Matrix3f& A,
	const Eigen::Matrix3f& B)
{

	Eigen::Vector3f u = A.row(1).transpose() - A.row(0).transpose();
	Eigen::Vector3f v = A.row(2).transpose() - A.row(0).transpose();
	Eigen::Vector3f w = u.cross(v);

	Eigen::Vector3f u2 = B.row(1).transpose() - B.row(0).transpose();
	Eigen::Vector3f v2 = B.row(2).transpose() - B.row(0).transpose();
	Eigen::Vector3f w2 = u2.cross(v2);

	// X = [u, v, w]，Y = [u2, v2, w2]
	Eigen::Matrix3f X;
	X.col(0) = u;
	X.col(1) = v;
	X.col(2) = w;

	Eigen::Matrix3f Y;
	Y.col(0) = u2;
	Y.col(1) = v2;
	Y.col(2) = w2;

	if (std::abs(X.determinant()) < 1e-12)
		throw std::runtime_error("source points are collinear");

	Eigen::Matrix<float, 3, 4> Mt;
	// M = Y * X^{-1}
	// Eigen::Matrix3f M = Y * X.inverse();
	Mt.block(0, 0, 3, 3) = Y * X.inverse();

	// t = B0 - M * A0
	// Eigen::Matrix3f t = B[0] - M * A[0];
	Mt.col(3) =
		B.row(0).transpose() - Mt.block(0, 0, 3, 3) * A.row(0).transpose();

	return Mt;
}
bool initRts(const Eigen::MatrixX3f& bfmV)
{
	return true;
}
bool readInitBfmRts(const  std::filesystem::path& path, const Eigen::MatrixX3f& bfmMshV,
	Eigen::Matrix3f&bfm_R, Eigen::RowVector3f&bfm_t, float&bfm_scale)
{
	std::list<Eigen::Vector3f>bfmPts, objPts;
	std::ifstream file(path);
	if (!file.is_open()) {
		LOG_WARN_OUT << "not found : " << path;
		return false;
	}
	else
	{
		std::string aline;
		while (std::getline(file, aline))
		{
			if (aline.length() > 0)
			{
				if (aline[0] == '#')
				{
					continue;
				}
				else
				{
					for (auto&d: aline)
					{
						if (d==',')
						{
							d = ' ';
						}
					}
					std::uint32_t bfmIdx;
					float a, b, c;
					std::istringstream iss(aline);
					if (iss >> bfmIdx >> a >> b >> c) {
						bfmPts.emplace_back(bfmMshV.row(bfmIdx));
						objPts.emplace_back(a, b, c);
					}
					else {
						LOG_WARN_OUT << "parse failed : " << path;
						return false;
					}
				}
			}
		}
	}
	if (bfmPts.size()<3 || objPts.size()!= bfmPts.size())
	{
		LOG_WARN_OUT << "bfmPts.size()<3";
		return false;
	}
	Eigen::Matrix3Xf ptsTgt(3,bfmPts.size());
	Eigen::Matrix3Xf ptsSrc(3,bfmPts.size());
	int ii = 0;
	for (auto it1 = bfmPts.begin(), it2 = objPts.begin(); it1 != bfmPts.end(); ++it1, ++it2,++ii) 
	{
		ptsTgt(0,ii) = it2->x();
		ptsTgt(1,ii) = it2->y();
		ptsTgt(2,ii) = it2->z();
		ptsSrc(0,ii) = it1->x();
		ptsSrc(1,ii) = it1->y();
		ptsSrc(2,ii) = it1->z();
	}
	Eigen::Matrix4f T = Eigen::umeyama(ptsSrc, ptsTgt, true);
	Eigen::Matrix3f sR = T.block<3, 3>(0, 0);
	bfm_scale = std::cbrt(sR.determinant());
	bfm_R = sR / bfm_scale;
	bfm_t = T.block<3, 1>(0, 3);	 
	return true;
}
int figureDlibLandMarks(const  std::filesystem::path& mvsResultDir)
{
	for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{ mvsResultDir })
	{
		const auto& thisFilename = dir_entry.path();
		if (thisFilename.has_extension())
		{
			const auto& ext = thisFilename.extension().string();
			if (ext.compare(".json") == 0)
			{
				std::string stem = thisFilename.filename().stem().string();
				auto maskPath = thisFilename.parent_path() / ("mask_" + stem + ".dat");
				if (std::filesystem::exists(maskPath))
				{
					Eigen::Matrix4d cameraMatrix, Rt;
					bool readRet = labelme::readCmaeraFromRegisterJson(thisFilename, cameraMatrix, Rt);
					if (!readRet)
					{
						LOG_WARN_OUT << "read fail : " << thisFilename;
						continue;
					}
					std::string imgPath;
					if (!labelme::readJsonStringElement(thisFilename, "imagePath", imgPath))
					{
						LOG_WARN_OUT << "not found : " << imgPath;
						continue;
					};
					if (!std::filesystem::exists(imgPath))
					{
						LOG_WARN_OUT << "not found : " << imgPath;
					}

					cv::Mat image = cv::imread(imgPath);
					if (!image.empty())
					{
						
					}
				}
			}
		}
	}
	return 0;
}
bool BfmIter::readObj(const std::filesystem::path& objPath, Eigen::MatrixX3f& vertex, Eigen::MatrixX3i& faces)
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
	while (std::getline(fin, aline))
	{
		if (aline.length() > 2 && aline[0] == 'v' && aline[1] == ' ')
		{
			std::stringstream ss(aline);
			char c;
			float x, y, z;
			ss >> c >> x >> y >> z;
			v.emplace_back(x, y, z);
		}
		if (aline.length() > 2 && aline[0] == 'f' && aline[1] == ' ')
		{
			std::stringstream ss(aline);
			char c;
			int x, y, z;
			ss >> c >> x >> y >> z;
			f.emplace_back(x - 1, y - 1, z - 1);
		}
	}
	vertex = Eigen::MatrixX3f(v.size(), 3);
	faces = Eigen::MatrixX3i(f.size(), 3);
	int i = 0;
	for (const auto& d : v)
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
	return true;
}
BfmIter::BfmIter(const  std::filesystem::path& mvsResultDir, const  std::filesystem::path& modelDirPath)
{
	this->initialSuccess = false;
	progress.procRunning.fetch_add(1);
	progress.denominator.store(1);
	progress.numerator.store(1);
	std::filesystem::path bfmFacePath = modelDirPath / "model2019_face12.h5";
	if (!std::filesystem::exists(bfmFacePath))
	{
		LOG_ERR_OUT << "need models/model2019_face12.h5";
		return;
	}
	bfmIns = new bfm::Bfm2019(bfmFacePath);
	bfmIns->generateRandomFace(bfmMsh.V, bfmMsh.C);
	//meshdraw::utils::savePts("0.txt", bfmMsh.V);
	this->bfmMsh.F = bfmIns->F;
	if (meshdraw::isEmpty(this->bfmMsh.facesNormal))
	{
		this->bfmMsh.figureFacesNomral();
	}	 
	{
		if (!readInitBfmRts(mvsResultDir / "match3.txt", bfmMsh.V, bfm_R, bfm_t, bfm_scale))
		{ 
			progress.procRunning.store(0);
			progress.denominator.store(-1);
			progress.numerator.store(-1);
		} 
		//Eigen::MatrixX3f meshVRotated = (bfmMsh.V * bfm_R.transpose() * bfm_scale).rowwise() + bfm_t;
		//meshdraw::utils::savePts(mvsResultDir / "1.txt", meshVRotated);
	} 


	//meshdraw::Camera cam;
	//cam.cameraType = meshdraw::CmaeraType::Pinhole;
	//cam.intr << 1536, 0, 360, 0, 1536, 640, 0, 0, 1;
	//cam.R  << 1, 0, 0, 0, 1, 0, 0, 0, 1;
	//cam.t  <<0,0,0;
	//cam.height = 384;
	//cam.width = 384;
	//cv::Mat render3d;
	//cv::Mat render3dPts;
	//cv::Mat mask;
	//meshdraw::render(this->bfmMsh, cam, bfm_R, bfm_t, bfm_scale, render3d, render3dPts, mask);


	imgPaths.clear();
	imgNameForlist.clear();
	imgDirPath_ = mvsResultDir;

	for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{ mvsResultDir })
	{
		const auto& thisFilename = dir_entry.path();
		if (thisFilename.has_extension())
		{
			const auto& ext = thisFilename.extension().string();
			if (ext.compare(".json") == 0)
			{
				std::string stem = thisFilename.filename().stem().string();
				auto maskPath = thisFilename.parent_path() / ("mask_" + stem + ".dat");
				//if (std::filesystem::exists(maskPath))
				{
					Eigen::Matrix4d cameraMatrix, Rt;
					bool readRet = labelme::readCmaeraFromRegisterJson(thisFilename, cameraMatrix, Rt);
					if (!readRet)
					{
						LOG_WARN_OUT << "read fail : " << thisFilename;
						continue;
					}
					std::string imgPath;
					if (!labelme::readJsonStringElement(thisFilename, "imagePath", imgPath))
					{
						LOG_WARN_OUT << "not found : " << imgPath;
						continue;
					};
					if (!std::filesystem::exists(imgPath))
					{
						LOG_WARN_OUT << "not found : " << imgPath;
					}
				 
					cv::Mat image = cv::imread(imgPath);
					if (!image.empty())
					{
						meshdraw::Camera cam;
						cam.cameraType = meshdraw::CmaeraType::Pinhole;
						cam.intr = cameraMatrix.cast<float>().block(0, 0, 3, 3);
						cam.R = Rt.cast<float>().block(0, 0, 3, 3);
						cam.t = Eigen::RowVector3f(Rt(0, 3), Rt(1, 3), Rt(2, 3));
						cam.height = image.rows;
						cam.width = image.cols;
						imgCameras.emplace_back(cam);
						imgNameForlist.emplace_back("   "+stem);
						imgPaths.emplace_back(imgPath);
						imgs.emplace_back(image);
						cv::Mat render3d;
						cv::Mat render3dPts;
						cv::Mat mask;						
						if (meshdraw::isEmpty(this->bfmMsh.facesNormal))
						{
							this->bfmMsh.figureFacesNomral();
						}
						meshdraw::render(this->bfmMsh, cam, bfm_R, bfm_t, bfm_scale,render3d, render3dPts, mask);
						bfmRenders.emplace_back(render3d);
						bfmRenderPts.emplace_back(render3dPts);
						bfmRenderMasks.emplace_back(mask); 
						shifts.emplace_back(ImVec2(0, 0));
						
						progress.denominator.fetch_add(1);
						progress.numerator.fetch_add(1);
					}
				}
			}
		}
	}
	if (imgNameForlist.size() < 1)
	{
		progress.procRunning.store(0);
		progress.denominator.store(-1);
		progress.numerator.store(-1);
		LOG_ERR_OUT << "imgNameForlist.size()<1";
		return;
	}
	this->initialSuccess = true;
	progress.procRunning.store(0);
	progress.denominator.store(-1);
	progress.numerator.store(-1);
	 
}
BfmIter::~BfmIter() {};
bool BfmIter::iter(const std::vector<cv::Point3f>& src, const std::vector<cv::Point3f>& tar, const IterType& type)
{
    return false;
}
bool BfmIter::updataRts(const Eigen::Matrix3f& R, const  Eigen::RowVector3f& t, const  float& scale)
{
	//V = (V * R' * scale).rowwise() + t;
	//(Y=s1XR1'+t1)
	//X = (Y-t1)R1/s1
	//Z = s2/s1( Y-t1 )R1R2' + t2 = s2/s1 Y R1R2' - s2/s1*t1*R1R2' +t2

	bfm_scale = scale;
	bfm_R = R;
	bfm_t = t;
	//bfm_scale = scale / bfm_scale;
	//bfm_R = R * bfm_R.transpose();
	//bfm_t = t - bfm_scale * bfm_t * bfm_R.transpose();
	//bfmMsh.rotate(bfm_R, bfm_t, bfm_scale);

	{
		//Eigen::Matrix3f R_inv = R.transpose();
		//Eigen::MatrixX3f  V2 = (BfmIterManger->bfmMsh.V * R_inv * scale).rowwise() + t;
		//meshdraw::utils::savePts("1.txt", V2);
	}

	

	//bfm_scale = scale* bfm_scale;
	//bfm_R = R* bfm_R.transpose();
	//bfm_t = t - scale / bfm_scale* bfm_t * bfm_R * R.transpose();

	
	progress.denominator.fetch_add(this->bfmRenders.size());
	for (int i = 0; i < this->bfmRenders.size(); i++)
	{
		progress.numerator.fetch_add(1);
		cv::Mat&render3d = this->bfmRenders[i];
		cv::Mat&render3dPts = this->bfmRenderPts[i];
		cv::Mat&mask = this->bfmRenderMasks[i];
		//BfmIterManger->bfmMsh.figureFacesNomral();		
		meshdraw::render(BfmIterManger->bfmMsh, imgCameras[i], bfm_R, bfm_t, bfm_scale, render3d, render3dPts, mask);
	} 	 
	progress.procRunning.store(0);
	progress.denominator.store(-1);
	progress.numerator.store(-1);
	return true;
}
bool BfmIter::figureSharedPoint(const std::vector<Eigen::Vector2f>& imgPts, const std::vector<meshdraw::Camera>& cams, Eigen::Vector3f& pt)
{
	if (imgPts.size() < 2)
	{
		LOG_ERR_OUT << "imgPts.size()<2";
		return false;
	}
	if (imgPts.size() != cams.size())
	{
		LOG_ERR_OUT << "size not match";
		return false;
	}


	Eigen::MatrixXf A(2 * imgPts.size(), 4);
	for (int i = 0; i < imgPts.size(); i++)
	{
		Eigen::Matrix4f intr = Eigen::Matrix4f::Identity();
		intr.block(0, 0, 3, 3) = cams[i].intr;
		Eigen::Matrix4f Rt = Eigen::Matrix4f::Identity();
		Rt.block(0, 0, 3, 3) = cams[i].R;
		Rt(0, 3) = cams[i].t[0];
		Rt(1, 3) = cams[i].t[1];
		Rt(2, 3) = cams[i].t[2];
		Eigen::Matrix4f P = intr * Rt;
		A.row(2 * i) = imgPts[i][0] * P.row(2) - P.row(0);
		A.row(2 * i + 1) = imgPts[i][0] * P.row(2) - P.row(1);
	}
	// 使用SVD求解最小二乘意义下的解
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);
	Eigen::Vector4f point_homogeneous = svd.matrixV().col(3);

	// 将齐次坐标转换为三维欧氏坐标
	pt = point_homogeneous.head<3>() / point_homogeneous(3);
 


	return true;
}

 
static bool showImgDirBrowser = false;
static std::filesystem::path imgDirPath;
static std::filesystem::path modelDirPath;
static browser::Browser* imgDirPicker = nullptr;
static browser::Browser* modelDirPicker = nullptr;

GLuint BfmIter::image_texture = 0;
int BfmIter::viewWindowHeight = 720;
int BfmIter::viewWindowWidth = 960;
int BfmIter::imgPickIdx = -1;
static float mixFactor = 0.5;
static cv::Mat showImgMix;
static cv::Mat ImgShift;
bool bfmIterFrame(bool* show_bfmIter_window)
{ 
	ImGui::SetNextWindowSize(ImVec2(1280, 960));//ImVec2(x, y)
	ImGui::Begin("bfmIter", show_bfmIter_window, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);
	//ImGui::Begin("bfmIter", show_bfmIter_window);
	ImGui::Text("bfmIter");

	int isProcRunning = progress.procRunning.load();
	if (progress.proc != nullptr && isProcRunning > 0)
	{
		progress.procRunning.fetch_add(1);
		progress.msg = "--------------------";
		//int timeProgress= align::getCaptureTimestamp() % 40;
		progress.msg[ProgressThread::getCaptureTimestamp() % 1000 / 50] = '@';
		int num = progress.numerator.load();
		int den = progress.denominator.load();
		if (progress.numerator.load() >= 0)
		{
			progress.msg += (std::to_string(num) + "/" + std::to_string(den));
		}
	}
	if (progress.proc != nullptr && isProcRunning == 0)
	{
		if (progress.proc != nullptr)
		{
			if (progress.proc->joinable())
			{
				progress.msg = "";
				progress.proc->join();
				progress.procRunning.store(false);
			}
			progress.proc = nullptr;
		}
	}
	ImVec2 deployButtom(-1, -1);
	while (isProcRunning == 0)
	{
		if (BfmIterManger == nullptr)
		{
#define PRE_FILL 1
#if PRE_FILL==0
			if (ImGui::Button("pick image dir"))
			{
				imgDirPath = "";
				if (imgDirPicker == nullptr)
				{
					imgDirPicker = new browser::Browser(browser::BrowserPick::PICK_DIR);
				}
			}
			if (imgDirPicker != nullptr && imgDirPath.string().length() == 0)
			{
				if (imgDirPicker->pick(imgDirPath))
				{
					if (!imgDirPath.empty())
					{
						LOG_OUT << imgDirPath;
					}
					delete imgDirPicker;
					imgDirPicker = nullptr;
				}
			}
			ImGui::SameLine();
			ImGui::Text(imgDirPath.string().c_str());
			if (ImGui::Button("pick model dir"))
			{
				modelDirPath = "";
				if (modelDirPicker == nullptr)
				{
					modelDirPicker = new browser::Browser(browser::BrowserPick::PICK_DIR);
				}
			}
			if (modelDirPicker != nullptr && modelDirPath.string().length() == 0)
			{
				if (modelDirPicker->pick(modelDirPath))
				{
					if (!modelDirPath.empty())
					{
						LOG_OUT << modelDirPath;
					}
					delete modelDirPicker;
					modelDirPicker = nullptr;
				}
			}
			ImGui::SameLine();
			ImGui::Text(modelDirPath.string().c_str());
#else
			//imgDirPath = "../data/a/result";
			imgDirPath = "../data/a/2";
			modelDirPath = "../models";
#endif // PRE_FILL==0
			if (imgDirPath.string().length() > 0 && modelDirPath.string().length() > 0)
			{
				progress.numerator.store(-1);
				progress.denominator.store(-1);
				progress.procRunning.fetch_add(1);
				progress.proc = new std::thread(
					[&]() {
						BfmIterManger = new BfmIter(imgDirPath, modelDirPath);
					}
				);
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}
		}
		if (BfmIterManger != nullptr && (BfmIterManger->imgNameForlist.size() < 1 || BfmIterManger->initialSuccess==false))
		{
			delete BfmIterManger;
			BfmIterManger = nullptr;
			imgDirPath = "";
			modelDirPath = "";
		}
		if (BfmIterManger != nullptr && BfmIterManger->imgNameForlist.size() > 0)
		{
			bool pickedChanged = false;
			bool mixFactorChanged = ImGui::SliderFloat(u8"混合度", &mixFactor, 0.f, 1.f);

			auto imgListLocation = ImGui::GetCursorPos();
			ImVec2 listPicSize(100, 500);
			ImVec2 canvas_location = imgListLocation;
			canvas_location.x += listPicSize.x;
			draw::Draw* labelControlPtr = draw::Draw::getDrawIns(canvas_location, 720, BfmIterManger->imgNameForlist, BfmIterManger->imgPaths);
			labelControlPtr->guideData.drawPos = ImVec2(canvas_location.x+ draw::Draw::canvas.x+5, canvas_location.y);
			listComponent("picPick", listPicSize, BfmIterManger->imgNameForlist, BfmIter::imgPickIdx, pickedChanged);
			

			//ImGui::Text("Mouse pos: (%d, %d)", BfmIterManger::imgPickIdx, pickedChanged);

			if (BfmIter::imgPickIdx >= 0 && (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) || ImGui::IsKeyPressed(ImGuiKey_UpArrow)))
			{
				pickedChanged = true;
				BfmIterManger->imgNameForlist[BfmIter::imgPickIdx][0] = ' ';
				BfmIter::imgPickIdx--;
				if (BfmIter::imgPickIdx < 0)
				{
					BfmIter::imgPickIdx = 0;
					pickedChanged = false;
				}
				BfmIterManger->imgNameForlist[BfmIter::imgPickIdx][0] = '-';
				//continue;
			}
			if (BfmIter::imgPickIdx >= 0 && (ImGui::IsKeyPressed(ImGuiKey_RightArrow) || ImGui::IsKeyPressed(ImGuiKey_DownArrow)))
			{
				pickedChanged = true;
				BfmIterManger->imgNameForlist[BfmIter::imgPickIdx][0] = ' ';
				BfmIter::imgPickIdx++;
				if (BfmIter::imgPickIdx >= BfmIterManger->imgPaths.size())
				{
					BfmIter::imgPickIdx = BfmIterManger->imgPaths.size() - 1;
					pickedChanged = false;
				}
				BfmIterManger->imgNameForlist[BfmIter::imgPickIdx][0] = '-';
				//continue;
			} 
			ImVec2 maybeClik(-1, -1);
			ImVec2 frameOffset(0, 0);
			if (BfmIter::imgPickIdx >= 0)
			{
				maybeClik = labelControlPtr->control(labelControlPtr->draw_pos, frameOffset, BfmIterManger->shifts[BfmIter::imgPickIdx]);
			}
			
			if (BfmIter::imgPickIdx >= 0 && (pickedChanged || mixFactorChanged || (frameOffset.x != 0 || frameOffset.y != 0) || maybeClik.x>=0))
			{
				if (pickedChanged)
				{
					ImgShift.release();
				}
				const cv::Mat& img = BfmIterManger->imgs[BfmIter::imgPickIdx];
				const meshdraw::Camera&cam= BfmIterManger->imgCameras[BfmIter::imgPickIdx];
				cv::Mat& bfmRender3d = BfmIterManger->bfmRenders[BfmIter::imgPickIdx];
				cv::Mat& bfmRender3dPts = BfmIterManger->bfmRenderPts[BfmIter::imgPickIdx];
				cv::Mat& bfmMask = BfmIterManger->bfmRenderMasks[BfmIter::imgPickIdx];
				if (bfmRender3d.empty())
				{
					if (meshdraw::isEmpty(BfmIterManger->bfmMsh.facesNormal))
					{
						BfmIterManger->bfmMsh.figureFacesNomral();
					}
					meshdraw::render(BfmIterManger->bfmMsh, cam, BfmIterManger->bfm_R, BfmIterManger->bfm_t, BfmIterManger->bfm_scale, bfmRender3d, bfmRender3dPts, bfmMask);
				}
				if (ImgShift.empty())
				{
					bfmRender3d.copyTo(ImgShift);
				}
				{ 
					ImVec2 offset(0, 0);
					offset.x = BfmIterManger->shifts[BfmIter::imgPickIdx].x + frameOffset.x;
					offset.y = BfmIterManger->shifts[BfmIter::imgPickIdx].y + frameOffset.y;					 
					ImgShift = draw::BSplineGuider::shiftSnap(bfmRender3d, offset);
				}
				cv::addWeighted(img, mixFactor, ImgShift, 1 - mixFactor, 0, showImgMix);
				labelControlPtr->feedImg(showImgMix); 

				 
				if ( maybeClik.x >= 0)
				{
					int maybeXint = static_cast<int>(maybeClik.x + 0.5);
					int maybeYint = static_cast<int>(maybeClik.y + 0.5);
					int maybeXinBfmmap = maybeXint - BfmIterManger->shifts[BfmIter::imgPickIdx].x ;
					int maybeYinBfmmap = maybeYint - BfmIterManger->shifts[BfmIter::imgPickIdx].y ;
					LOG_OUT << maybeXinBfmmap << "," << maybeYinBfmmap<<" "<< maybeXint << "," << maybeYint;//render pos; picture pos
				}
			}
			if (BfmIter::imgPickIdx >= 0)
			{
				labelControlPtr->draw(BfmIter::imgPickIdx);
			}
			if (labelControlPtr->guideData.tagsListName.size() > 0)
			{
				labelControlPtr->guideData.pickedChanged = false;
				auto pushPos = ImGui::GetCursorPos();
				ImGui::SetCursorPos(labelControlPtr->guideData.drawPos);
				listComponent("tarPick", listPicSize, labelControlPtr->guideData.tagsListName, labelControlPtr->guideData.tagPickIdx, labelControlPtr->guideData.pickedChanged);
				ImGui::SetCursorPos(pushPos);
			}
		}
		break;
	}
	if (deployButtom.x > 0)
	{
		ImGui::SetCursorScreenPos(deployButtom);
	}
	if (progress.msg.length() > 0)
	{
		ImGui::Text(progress.msg.c_str());
	}
	if (ImGui::Button("Close Me") && *show_bfmIter_window) *show_bfmIter_window = false;
	ImGui::End();
	return true;
}
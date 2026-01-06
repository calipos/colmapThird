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
	struct ControlLogic
	{
		ControlLogic()
		{
		}
		std::vector< std::map<std::string, ImVec2>>borderMeshControlPtsInImg;
		std::vector< std::map<std::string, cv::Vec3f>>borderMeshControlPts;
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
		bool figureRts(const std::vector<meshdraw::Camera>& imgCameras, Eigen::Matrix3f& R, Eigen::RowVector3f& t, float& scale)const
		{
			std::vector<cv::Vec3f>objPtsFromBorderMesh;
			std::vector<Eigen::RowVector3f>objPtsFromBfm;
			for (int i = 0; i < borderMeshControlPts.size(); i++)
			{
				for (const auto& d : borderMeshControlPts[i])
				{
					objPtsFromBorderMesh.emplace_back(d.second);
					objPtsFromBfm.emplace_back(bfmMeshControlPts[i].at(d.first));
				}
			}
			if (objPtsFromBfm.size() < 4)
			{
				LOG_WARN_OUT << "insufficient 2d pts";
				return false;
			}
			else
			{
				scale = 1.;
				R = Eigen::Matrix3f::Identity();
				t = Eigen::RowVector3f(0, 0, 0);
				Eigen::MatrixX3f srcMat(objPtsFromBfm.size(), 3);
				Eigen::MatrixX3f tarMat(objPtsFromBorderMesh.size(), 3);
				for (int i = 0; i < objPtsFromBfm.size(); i++)
				{
					srcMat(i, 0) = objPtsFromBfm[i][0];
					srcMat(i, 1) = objPtsFromBfm[i][1];
					srcMat(i, 2) = objPtsFromBfm[i][2];
					tarMat(i, 0) = objPtsFromBorderMesh[i][0];
					tarMat(i, 1) = objPtsFromBorderMesh[i][1];
					tarMat(i, 2) = objPtsFromBorderMesh[i][2];

				}
				LOG_OUT << srcMat;
				LOG_OUT << tarMat;
				Eigen::RowVector3f meanSrc = srcMat.colwise().mean();
				Eigen::RowVector3f meanTar = tarMat.colwise().mean();
				{
					auto src_scale = (srcMat.rowwise() - meanSrc).rowwise().norm().mean();
					auto tar_mean = (tarMat.rowwise() - meanTar).rowwise().norm().mean();
					scale = tar_mean / src_scale;
					srcMat *= scale;
				}
				{
					Eigen::MatrixX3f srcMat2 = srcMat.rowwise() - meanSrc * scale;//A
					Eigen::MatrixX3f tarMat2 = tarMat.rowwise() - meanTar;//B             
					Eigen::Matrix3f H = srcMat2.transpose() * tarMat2;
					Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
					Eigen::Matrix3f U = svd.matrixU();
					Eigen::Matrix3f V = svd.matrixV();
					R = V * U.transpose();
					if (R.determinant() < 0) {
						V.col(2) *= -1;
						R = V * U.transpose();
					}
					t = meanTar - (meanSrc * scale) * R.transpose();
				}
				//LOG_OUT<< (srcMat* R.transpose()* scale).rowwise()+t;
				return true;
			}
			return true;
		}
		void updataTagsListName(const int&imgPickedIdx)
		{
			tagsListName.clear();
			if (imgPickedIdx>=0 && imgPickedIdx< bfmMeshControlPts.size())
			{
				tagsListName.reserve(this->borderMeshControlPts[imgPickedIdx].size());
				for (const auto& d : this->borderMeshControlPts[imgPickedIdx])
				{
					tagsListName.emplace_back("   " + d.first);
				}
			}
		}
		/*
		bool save(const std::vector<std::filesystem::path>& picPath)const
		{
			if (picShortNameForlist.size() != picPath.size())
			{
				return false;
			}
			for (int i = 0; i < picPath.size(); i++)
			{
				const auto& imgPath = picPath[i];
				std::filesystem::path parentDir = imgPath.parent_path();
				std::string shortName = imgPath.filename().stem().string();
				std::map<std::string, Eigen::Vector2d>picLabel;
				for (const auto& d : borderMeshControlPtsInImg[i])
				{
					picLabel[d.first] = Eigen::Vector2d(d.second.x, d.second.y);
				}
				labelme::writeLabelMeLinestripJson(imgPath, picLabel);
			}
			return true;
		}
		bool loadOnlyOnce(const std::vector<std::filesystem::path>& picPath)
		{
			std::unordered_set<std::string>labelNames;
			for (int i = 0; i < picPath.size(); i++)
			{
				const auto& imgPath = picPath[i];
				std::string shortName = imgPath.filename().stem().string();
				std::filesystem::path parentDir = imgPath.parent_path();
				std::string tempShort = picShortNameForlist[i].substr(picShortNameForlist[i].length() - shortName.length(), shortName.length());
				if (tempShort.compare(shortName) != 0)
				{
					LOG_ERR_OUT << "name err! jump.";
					continue;
				}
				auto jsonPath = parentDir / (shortName + ".json");
				if (std::filesystem::exists(jsonPath))
				{
					std::map<std::string, Eigen::Vector2d> imgPts;
					Eigen::Vector2i imgSizeWH;
					labelme::readPtsFromLabelMeJson(jsonPath, imgPts, imgSizeWH);
					borderMeshControlPtsInImg[i].clear();
					for (const auto& d : imgPts)
					{
						borderMeshControlPtsInImg[i][d.first] = ImVec2(d.second[0], d.second[1]);
						labelNames.insert(d.first);
					}
				}
			}
			tagsListName.reserve(labelNames.size());
			tagsName.reserve(labelNames.size());
			for (const auto& d : labelNames)
			{
				tagsListName.emplace_back(" " + d);
				tagsName.emplace_back(d);
				colors[d] = getImguiColor();
			}
			updata();
			return true;
		}
		*/
		ImVec2 drawPos;
		static bool tryFind3d3dPair;
		static bool figureRtsFlag; 
		static bool figureParamFlag;
	}; 
	int ControlLogic::tagPickIdx = -1;
	bool ControlLogic::tryFind3d3dPair = true;
	bool ControlLogic::figureRtsFlag = true;
	bool ControlLogic::figureParamFlag = false;
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
		ControlLogic ptsData;
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
				instance->ptsData.borderMeshControlPtsInImg.resize(picShortNameForlist.size());
				instance->ptsData.borderMeshControlPts.resize(picShortNameForlist.size());
				instance->ptsData.bfmMeshControlPts.resize(picShortNameForlist.size()); 
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
				
				if (ptsData.tagPickIdx >= 0 && ptsData.tagPickIdx<ptsData.tagsListName.size())
				{
					std::string picktarName = ptsData.tagsListName[ptsData.tagPickIdx];
			 
					const std::map<std::string, ImVec2>& tags = ptsData.borderMeshControlPtsInImg[imgPickIdx];
					if (tags.size() > 0)
					{
						for (const auto& d : tags)
						{
							ImVec2 guiPt = imgPt2GuiPt(d.second, currentImgHeight, currentImgWidth, canvas, zoom_start, zoom_end, draw_pos);
							if (d.first.compare(picktarName)==0)
							{
								ImGui::GetForegroundDrawList()->AddCircleFilled(guiPt, 4.0f, 0xC80688FB);
							}
							else
							{
								ImGui::GetForegroundDrawList()->AddCircleFilled(guiPt, 4.0f, ptsData.colors.at(d.first));
							}
						}
					}
				} 
			}
			return true;
		}
		ImVec2 control(const ImVec2&canvasPos, ImVec2& temporaryOffset, ImVec2& offset)
		{
			ImVec2 mousePosInImage;
			mousePosInImage.x = (ImGui::GetIO().MousePos.x - ImGui::GetWindowPos().x - canvasPos.x);
			mousePosInImage.y = (ImGui::GetIO().MousePos.y - ImGui::GetWindowPos().y - canvasPos.y);
			if (mousePosInImage.x < 0 || mousePosInImage.y < 0 || mousePosInImage.x >= canvas.x || mousePosInImage.y >canvas.y)
			{
				return ImVec2(-1, -1);;
			}
			float wheel = ImGui::GetIO().MouseWheel;
			if (ImGui::GetIO().MouseClicked[2])
			{
				float x_inRatio = mousePosInImage.x * Draw::canvasInv.x * (Draw::zoom_end.x - Draw::zoom_start.x) + Draw::zoom_start.x;
				float y_inRatio = mousePosInImage.y * Draw::canvasInv.y * (Draw::zoom_end.y - Draw::zoom_start.y) + Draw::zoom_start.y;
				float x_inPic = x_inRatio * currentImgWidth;
				float y_inPic = y_inRatio * currentImgHeight;
				return ImVec2(x_inPic, y_inPic);
			}
			else if (ImGui::GetIO().MouseDown[0])
			{
				mouseDownStartPos = ImGui::GetIO().MouseClickedPos[0]; 
 
				temporaryOffset.x = (ImGui::GetIO().MousePos.x - mouseDownStartPos.x);
				temporaryOffset.y = (ImGui::GetIO().MousePos.y - mouseDownStartPos.y);
			}
			else if (ImGui::GetIO().MouseReleased[0])
			{
				offset.x += (ImGui::GetIO().MousePos.x - mouseDownStartPos.x);
				offset.y += (ImGui::GetIO().MousePos.y - mouseDownStartPos.y);
				temporaryOffset.x = 0;
				temporaryOffset.y = 0;
			} 
			else if (abs(wheel) > 0.01)
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
	this->denseObjPath = mvsResultDir / "dense.obj";
	if (!std::filesystem::exists(this->denseObjPath))
	{
		LOG_ERR_OUT << "need denseObjPath";
		return;
	}
	else
	{
		readObj(this->denseObjPath, borderMsh.V, borderMsh.F);
	}
	std::filesystem::path bfmFacePath = modelDirPath / "model2019_face12.h5";
	if (!std::filesystem::exists(bfmFacePath))
	{
		LOG_ERR_OUT << "need models/model2019_face12.h5";
		return;
	}
	bfmIns = new bfm::Bfm2019(bfmFacePath);
	bfmIns->generateRandomFace(bfmMsh.V, bfmMsh.C);
	meshdraw::utils::savePts("0.txt", bfmMsh.V);
	this->bfmMsh.F = bfmIns->F;
    bfm_R << 1, 0, 0, 0, -1, 0, 0, 0, -1;
    bfm_t << 0, 0, 300;  
	bfm_scale = 1.f;
	this->bfmMsh.rotate(bfm_R, bfm_t, bfm_scale);
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
						meshdraw::render(this->bfmMsh, cam, render3d, render3dPts, mask);
						bfmRenders.emplace_back(render3d);
						bfmRenderPts.emplace_back(render3dPts);
						bfmRenderMasks.emplace_back(mask);
						cv::Mat borderMshRender3dPts;
						cv::Mat borderMshMask;
						if (meshdraw::isEmpty(this->borderMsh.facesNormal))
						{
							this->borderMsh.figureFacesNomral();
						}
						meshdraw::render(this->borderMsh, cam, borderMshRender3dPts, borderMshMask);
						this->borderMshRenderPts.emplace_back(borderMshRender3dPts);
						this->borderMshMasks.emplace_back(borderMshMask);
						shifts.emplace_back(ImVec2(0, 0));
						if (this->borderMshMasks.size()>=5)
						{
							break;
						}
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


	bfm_scale = scale / bfm_scale;
	bfm_R = R * bfm_R.transpose();
	bfm_t = t - bfm_scale * bfm_t * bfm_R.transpose();
	bfmMsh.rotate(bfm_R, bfm_t, bfm_scale);


	meshdraw::utils::savePts("1.txt", bfmMsh.V);

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
		BfmIterManger->bfmMsh.figureFacesNomral();		
		meshdraw::render(BfmIterManger->bfmMsh, imgCameras[i], render3d, render3dPts, mask);
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
			labelControlPtr->ptsData.drawPos = ImVec2(canvas_location.x+ draw::Draw::canvas.x+5, canvas_location.y);
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
			ImVec2 temporaryOffset(0, 0);
			if (BfmIter::imgPickIdx >= 0)
			{
				maybeClik = labelControlPtr->control(labelControlPtr->draw_pos, temporaryOffset, BfmIterManger->shifts[BfmIter::imgPickIdx]);
			}
			if (BfmIter::imgPickIdx >= 0 && (pickedChanged || mixFactorChanged || (temporaryOffset.x != 0 || temporaryOffset.y != 0) || maybeClik.x>=0))
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
				cv::Mat& borderMeshRrender3dPts = BfmIterManger->borderMshRenderPts[BfmIter::imgPickIdx];
				cv::Mat& borderMeshMask = BfmIterManger->borderMshMasks[BfmIter::imgPickIdx];
				if (bfmRender3d.empty())
				{
					if (meshdraw::isEmpty(BfmIterManger->bfmMsh.facesNormal))
					{
						BfmIterManger->bfmMsh.figureFacesNomral();
					}
					meshdraw::render(BfmIterManger->bfmMsh, cam, bfmRender3d, bfmRender3dPts, bfmMask);
				}
				if (borderMeshRrender3dPts.empty())
				{
					if (meshdraw::isEmpty(BfmIterManger->bfmMsh.facesNormal))
					{
						BfmIterManger->bfmMsh.figureFacesNomral();
					}
					meshdraw::render(BfmIterManger->borderMsh, cam, borderMeshRrender3dPts, borderMeshMask);
				}
				if (ImgShift.empty())
				{
					bfmRender3d.copyTo(ImgShift);
				}
				//if (temporaryOffset.x != 0 || temporaryOffset.y != 0)
				{ 
					ImVec2 offset(0, 0);
					offset.x = BfmIterManger->shifts[BfmIter::imgPickIdx].x + temporaryOffset.x;
					offset.y = BfmIterManger->shifts[BfmIter::imgPickIdx].y + temporaryOffset.y;					 
					ImgShift = draw::ControlLogic::shiftSnap(bfmRender3d, offset);
				}
				cv::addWeighted(img, mixFactor, ImgShift, 1 - mixFactor, 0, showImgMix);
				labelControlPtr->feedImg(showImgMix); 

				 
				if ( maybeClik.x >= 0)
				{
					int maybeXint = static_cast<int>(maybeClik.x + 0.5);
					int maybeYint = static_cast<int>(maybeClik.y + 0.5);
					int maybeXinBfmmap = maybeXint - BfmIterManger->shifts[BfmIter::imgPickIdx].x ;
					int maybeYinBfmmap = maybeYint - BfmIterManger->shifts[BfmIter::imgPickIdx].y ;
					if (draw::ControlLogic::tryFind3d3dPair)
					{
						if (bfmMask.ptr<uchar>(maybeYinBfmmap)[maybeXinBfmmap] != 0 && borderMeshMask.ptr<uchar>(maybeYint)[maybeXint] != 0)
						{ 
							labelControlPtr->ptsData.lastControlIdx += 1;
							const std::string& thisTarName = std::to_string(labelControlPtr->ptsData.lastControlIdx);
							labelControlPtr->ptsData.borderMeshControlPtsInImg[BfmIter::imgPickIdx][thisTarName] = maybeClik;

							const cv::Vec3f&bfmPt = bfmRender3dPts.at<cv::Vec3f>(maybeYinBfmmap, maybeXinBfmmap);
							Eigen::RowVector3f pickedBfmPt(bfmPt[0], bfmPt[1], bfmPt[2]);
							pickedBfmPt = (pickedBfmPt- BfmIterManger->bfm_t)*BfmIterManger->bfm_R* (1./BfmIterManger->bfm_scale);

							labelControlPtr->ptsData.bfmMeshControlPts[BfmIter::imgPickIdx][thisTarName] = pickedBfmPt;
							labelControlPtr->ptsData.borderMeshControlPts[BfmIter::imgPickIdx][thisTarName] = borderMeshRrender3dPts.at<cv::Vec3f>(maybeYint, maybeXint);
							labelControlPtr->ptsData.updataTagsListName(BfmIter::imgPickIdx);
							labelControlPtr->ptsData.colors[thisTarName] = getImguiColor();
							draw::ControlLogic::tagPickIdx = std::find(labelControlPtr->ptsData.tagsListName.begin(), labelControlPtr->ptsData.tagsListName.begin(), thisTarName) - labelControlPtr->ptsData.tagsListName.begin();
						}
					}
					else
					{
						 
					} 		
				}
			}
			if (BfmIter::imgPickIdx >= 0)
			{
				labelControlPtr->draw(BfmIter::imgPickIdx);
			}
			if (labelControlPtr->ptsData.tagsListName.size() > 0)
			{
				labelControlPtr->ptsData.pickedChanged = false;
				auto pushPos = ImGui::GetCursorPos();
				ImGui::SetCursorPos(labelControlPtr->ptsData.drawPos);
				listComponent("tarPick", listPicSize, labelControlPtr->ptsData.tagsListName, labelControlPtr->ptsData.tagPickIdx, labelControlPtr->ptsData.pickedChanged);
				ImGui::SetCursorPos(pushPos);
			}
			if (ImGui::Checkbox("try3D3D", &draw::ControlLogic::tryFind3d3dPair))
			{

			} 
			bool trigerRts = ImGui::Checkbox("figureRtS", &draw::ControlLogic::figureRtsFlag); ImGui::SameLine();
			bool trigerParam = ImGui::Checkbox("figureParamFlag", &draw::ControlLogic::figureParamFlag); ImGui::SameLine();
			if (trigerRts || trigerParam)
			{
				if (draw::ControlLogic::figureRtsFlag || draw::ControlLogic::figureParamFlag)
				{

				}
				else
				{
					draw::ControlLogic::figureRtsFlag = true;
					draw::ControlLogic::figureParamFlag = true;
				}
			}
			if (ImGui::Button("figure"))
			{ 
				progress.numerator.store(-1);
				progress.denominator.store(-1);
				progress.procRunning.fetch_add(1);
				progress.proc = new std::thread(
					[&]() {
						float scale = 1;
						Eigen::Matrix3f R;
						Eigen::RowVector3f t;
						if (labelControlPtr->ptsData.figureRts(BfmIterManger->imgCameras, R, t, scale))
						{
							LOG_OUT << scale;
							LOG_OUT << R;
							LOG_OUT << t;
							BfmIterManger->updataRts(R, t, scale);
						}
					}
				);
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
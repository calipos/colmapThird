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
			memset(tarStr,0, tagLengthMax);
		}
		std::vector< std::map<std::string, ImVec2>>controlPts2dAndTag;
		std::map<std::string, cv::Vec3f>controlPts3dAndTag;
		//ImVec2 tempPt{ -1,-1 };  // for first track ,the temp point wil be showed on canvas

		std::map<std::string, ImU32> colors;
		std::vector<std::string>tagsListName;
		std::vector<std::string>tagsName;
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
		 
		bool figureRts(const std::vector<meshdraw::Camera>&imgCameras, Eigen::Matrix3f& R, Eigen::RowVector3f& t, float& scale)const
		{
			std::map<std::string, std::vector<std::pair<int, ImVec2>>> pts2ds;
			for (int i = 0; i < controlPts2dAndTag.size(); i++)
			{
				for (const auto&d: controlPts2dAndTag[i])
				{
					const std::string& ptName = d.first;
					const ImVec2& pt = d.second;
					pts2ds[ptName].emplace_back(i,pt);
				}
			}
			std::vector<std::string>targetPtsNames;
			targetPtsNames.reserve(pts2ds.size());
			for (const auto&d: pts2ds)
			{
				if (d.second.size()>1 && controlPts3dAndTag.count(d.first)>0)
				{
					targetPtsNames.emplace_back(d.first);
				}
			}
			if (targetPtsNames.size()<4)
			{
				LOG_WARN_OUT << "insufficient 2d pts";
				return false;
			}
			else
			{
				std::vector<Eigen::Vector3f>objPtsFromImgs;
				std::vector<Eigen::Vector3f>objPtsFromBfm;
				objPtsFromImgs.reserve(targetPtsNames.size());
				objPtsFromBfm.reserve(targetPtsNames.size());
				for (const auto&ptName: targetPtsNames)
				{
					std::vector<Eigen::Vector2f>imgPts;
					std::vector<meshdraw::Camera>cams;
					imgPts.reserve(pts2ds[ptName].size());
					cams.reserve(pts2ds[ptName].size());
					for (int i = 0; i < pts2ds[ptName].size(); i++)
					{
						const ImVec2& imgPt = pts2ds[ptName][i].second;
						imgPts.emplace_back(imgPt.x, imgPt.y);
						cams.emplace_back(imgCameras[pts2ds[ptName][i].first]);
					}
					Eigen::Vector3f pt;
					BfmIter::figureSharedPoint(imgPts, cams, pt);
					objPtsFromImgs.emplace_back(pt);
					const cv::Vec3f&pt3d = controlPts3dAndTag.at(ptName);
					objPtsFromBfm.emplace_back(pt3d[0], pt3d[1], pt3d[2]);
				}

				scale = 1.;
				R = Eigen::Matrix3f::Identity();
				t = Eigen::RowVector3f(0, 0, 0); 
				Eigen::MatrixX3f srcMat(objPtsFromBfm.size(), 3);
				Eigen::MatrixX3f tarMat(objPtsFromImgs.size(), 3);
				for (int i = 0; i < objPtsFromBfm.size(); i++)
				{
					srcMat(i, 0) = objPtsFromBfm[i][0];
					srcMat(i, 1) = objPtsFromBfm[i][1];
					srcMat(i, 2) = objPtsFromBfm[i][2];
					tarMat(i, 0) = objPtsFromImgs[i][0];
					tarMat(i, 1) = objPtsFromImgs[i][1];
					tarMat(i, 2) = objPtsFromImgs[i][2];
				}
				Eigen::RowVector3f meanSrc = srcMat.colwise().mean();
				Eigen::RowVector3f meanTar = tarMat.colwise().mean();
				{
					auto src_scale = (srcMat.rowwise() - meanSrc).rowwise().norm().mean();
					auto tar_mean = (tarMat.rowwise() - meanTar).rowwise().norm().mean();
					scale = tar_mean / src_scale;
					srcMat *= scale; 
				} 
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
		}
		void updataFalgs()
		{
			if (draw::ControlLogic::tagPickIdx < 0)
			{
				for (auto&d: BfmIterManger->imgNameForlist)
				{
					d[1] = ' ';
				}				
			}
			else
			{
				const std::string& pickedTagName = this->tagsName[draw::ControlLogic::tagPickIdx];
				for (int i = 0; i < this->controlPts2dAndTag.size(); i++)
				{
					if (controlPts2dAndTag[i].count(pickedTagName)!=0)
					{
						BfmIterManger->imgNameForlist[i][1] = '.';
					}
					else
					{
						BfmIterManger->imgNameForlist[i][1] = ' ';
					}
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
				for (const auto& d : controlPts2dAndTag[i])
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
					controlPts2dAndTag[i].clear();
					for (const auto& d : imgPts)
					{
						controlPts2dAndTag[i][d.first] = ImVec2(d.second[0], d.second[1]);
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
		static const int tagLengthMax{ 24 };
		char tarStr[tagLengthMax];
		static bool tryFind2d3dPair;
		static bool figureRtsFlag; 
		static bool figureParamFlag;
	}; 
	int ControlLogic::tagPickIdx = -1;
	bool ControlLogic::tryFind2d3dPair = true;
	bool ControlLogic::figureRtsFlag = true;
	bool ControlLogic::figureParamFlag = true;
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

		static const int tagLengthMax{ 24 };
		static char tarStr[tagLengthMax];
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
				instance->ptsData.controlPts2dAndTag.resize(picShortNameForlist.size());
				instance->ptsData.controlPts3dAndTag.clear();
				//instance->ptsData.loadOnlyOnce(picPaths);
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
				std::string pickedTag;
				if (ptsData.tagPickIdx >= 0)
				{
					pickedTag = ptsData.tagsName[ptsData.tagPickIdx];
					//if (alignPt)
					{
						const std::map<std::string, ImVec2>& tags = ptsData.controlPts2dAndTag[imgPickIdx];
						if (tags.count(pickedTag) > 0)
						{
							const auto p = tags.at(pickedTag);
							float x_ratio = p.x / currentImgWidth;
							float y_ratio = p.y / currentImgHeight;
							float ratioCnterX = (Draw::zoom_end.x + Draw::zoom_start.x) * .5;
							float ratioCnterY = (Draw::zoom_end.y + Draw::zoom_start.y) * .5;
							float shiftX = x_ratio - ratioCnterX;
							float shiftY = y_ratio - ratioCnterY;
							Draw::zoom_start.x += shiftX;
							Draw::zoom_end.x += shiftX;
							Draw::zoom_start.y += shiftY;
							Draw::zoom_end.y += shiftY;
							if (Draw::zoom_start.x < 0)
							{
								Draw::zoom_end.x -= Draw::zoom_start.x;
								Draw::zoom_start.x = 0;
							}
							if (Draw::zoom_start.y < 0)
							{
								Draw::zoom_end.y -= Draw::zoom_start.y;
								Draw::zoom_start.y = 0;
							}
							if (Draw::zoom_end.x > 1)
							{
								Draw::zoom_start.x += (1 - Draw::zoom_end.x);
								Draw::zoom_end.x = 1;
							}
							if (Draw::zoom_end.y > 1)
							{
								Draw::zoom_start.y += (1 - Draw::zoom_end.y);
								Draw::zoom_end.y = 1;
							}
						}
					}
				}
				ImGui::Image((ImTextureID)(intptr_t)image_texture, Draw::canvas, Draw::zoom_start, Draw::zoom_end, ImVec4(1, 1, 1, 1), ImVec4(.5, .5, .5, .5));
				std::string picktarName = "";
				if (ptsData.tagPickIdx >= 0)
				{
					picktarName = ptsData.tagsName[ptsData.tagPickIdx];
				}
				{
					const std::map<std::string, ImVec2>& tags = ptsData.controlPts2dAndTag[imgPickIdx];
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
				mouseDownStartPos.x = -900000;
				return ImVec2(-1, -1);;
			}
			float wheel = ImGui::GetIO().MouseWheel;
			if (ImGui::GetIO().MouseClicked[2])
			{
				mouseDownStartPos.x = -900000;
				float x_inRatio = mousePosInImage.x * Draw::canvasInv.x * (Draw::zoom_end.x - Draw::zoom_start.x) + Draw::zoom_start.x;
				float y_inRatio = mousePosInImage.y * Draw::canvasInv.y * (Draw::zoom_end.y - Draw::zoom_start.y) + Draw::zoom_start.y;
				float x_inPic = x_inRatio * currentImgWidth;
				float y_inPic = y_inRatio * currentImgHeight;
				LOG_OUT << x_inPic << " " << y_inPic;
				return ImVec2(x_inPic, y_inPic);
			}
			else if (ImGui::GetIO().MouseDown[0] && mouseDownStartPos.x < -800000)
			{
				mouseDownStartPos = mousePosInImage;
			}
			else if (ImGui::GetIO().MouseReleased[0])
			{
				offset.x += (mousePosInImage.x - mouseDownStartPos.x);
				offset.y += (mousePosInImage.y - mouseDownStartPos.y);
				temporaryOffset.x = 0;
				temporaryOffset.y = 0;
				mouseDownStartPos.x = -900000;
			}
			else if (ImGui::GetIO().MouseDown[0])
			{ 
				temporaryOffset.x = (mousePosInImage.x- mouseDownStartPos.x );
				temporaryOffset.y = (mousePosInImage.y - mouseDownStartPos.y); 
			}
			else if (abs(wheel) > 0.01)
			{
				mouseDownStartPos.x = -900000;
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
	char Draw::tarStr[tagLengthMax] = "\0";
}
 
BfmIter::BfmIter(const  std::filesystem::path& mvsResultDir, const  std::filesystem::path& modelDirPath)
{
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
	bfmIns->generateRandomFace(msh.V, msh.C);
	this->msh.F = bfmIns->F;
    bfm_R << 1, 0, 0, 0, -1, 0, 0, 0, -1;
    bfm_t << 0, 0, 300;  
	bfm_scale = 1.f;
	this->msh.rotate(bfm_R, bfm_t, bfm_scale);
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
						if (meshdraw::isEmpty(this->msh.facesNormal))
						{
							this->msh.figureFacesNomral();
						}
						meshdraw::render(this->msh, cam, render3d, render3dPts, mask);
						renders.emplace_back(render3d);
						renderPts.emplace_back(render3dPts);
						renderMasks.emplace_back(mask);
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
	msh.rotate(R, t, scale);
	bfm_scale *= scale;
	bfm_R = R*bfm_R;
	bfm_t = t + bfm_t * R.transpose() * scale;
	for (int i = 0; i < this->renders.size(); i++)
	{
		cv::Mat&render3d = this->renders[i];
		cv::Mat&render3dPts = this->renderPts[i];
		cv::Mat&mask = this->renderMasks[i];
		if (meshdraw::isEmpty(BfmIterManger->msh.facesNormal))
		{
			BfmIterManger->msh.figureFacesNomral();
		}
		meshdraw::render(BfmIterManger->msh, imgCameras[i], render3d, render3dPts, mask);
	}
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
		if (BfmIterManger != nullptr && BfmIterManger->imgNameForlist.size() < 1)
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
			if (BfmIter::imgPickIdx >= 0 && (pickedChanged || mixFactorChanged || BfmIterManger->shifts[BfmIter::imgPickIdx].x != 0 || maybeClik.x>=0))
			{
				if (pickedChanged)
				{
					ImgShift.release();
				}
				const cv::Mat& img = BfmIterManger->imgs[BfmIter::imgPickIdx];
				const meshdraw::Camera&cam= BfmIterManger->imgCameras[BfmIter::imgPickIdx];
				cv::Mat& render3d = BfmIterManger->renders[BfmIter::imgPickIdx];
				cv::Mat& render3dPts = BfmIterManger->renderPts[BfmIter::imgPickIdx];
				cv::Mat& mask = BfmIterManger->renderMasks[BfmIter::imgPickIdx];
				if (render3d.empty())
				{ 
					if (meshdraw::isEmpty(BfmIterManger->msh.facesNormal))
					{
						BfmIterManger->msh.figureFacesNomral();
					}
					meshdraw::render(BfmIterManger->msh, cam, render3d, render3dPts, mask);
				}  
				if (ImgShift.empty())
				{
					render3d.copyTo(ImgShift);
				}
				if (temporaryOffset.x != 0 || temporaryOffset.y != 0)
				{ 
					ImVec2 offset(0, 0);
					offset.x = BfmIterManger->shifts[BfmIter::imgPickIdx].x + temporaryOffset.x;
					offset.y = BfmIterManger->shifts[BfmIter::imgPickIdx].y + temporaryOffset.y;					 
					ImgShift = draw::ControlLogic::shiftSnap(render3d, offset); 
				}
				cv::addWeighted(img, mixFactor, ImgShift, 1 - mixFactor, 0, showImgMix);
				labelControlPtr->feedImg(showImgMix); 

				 
				if (labelControlPtr->ptsData.tarStr[0] != '\0' && maybeClik.x >= 0)
				{
					std::string thisTarName(labelControlPtr->ptsData.tarStr);

					int maybeXint = static_cast<int>(maybeClik.x + 0.5);
					int maybeYint = static_cast<int>(maybeClik.y + 0.5);
					if (draw::ControlLogic::tryFind2d3dPair)
					{
						if (mask.ptr<uchar>(maybeYint)[maybeXint] != 0)
						{
							labelControlPtr->ptsData.controlPts2dAndTag[BfmIter::imgPickIdx][thisTarName] = maybeClik;
							labelControlPtr->ptsData.controlPts3dAndTag[thisTarName] = render3dPts.at<cv::Vec3f>(maybeYint, maybeXint);
						}
					}
					else
					{
						labelControlPtr->ptsData.controlPts2dAndTag[BfmIter::imgPickIdx][thisTarName] = maybeClik;
					}


					auto fidExisted = std::find(labelControlPtr->ptsData.tagsName.begin(), labelControlPtr->ptsData.tagsName.end(), thisTarName);
					if (labelControlPtr->ptsData.tagsName.end() == fidExisted)
					{ 
						draw::ControlLogic::tagPickIdx = labelControlPtr->ptsData.tagsName.size();
						labelControlPtr->ptsData.tagsName.emplace_back(thisTarName);
						labelControlPtr->ptsData.tagsListName.emplace_back("  " + thisTarName);
						labelControlPtr->ptsData.colors[thisTarName] = (getImguiColor());
						labelControlPtr->ptsData.updataFalgs(); 
						listComponentReChoose(labelControlPtr->ptsData.tagsListName, labelControlPtr->ptsData.tagPickIdx);
						
					}
					else
					{ 
						draw::ControlLogic::tagPickIdx = fidExisted - labelControlPtr->ptsData.tagsName.begin();
					}



					
				}
				else if (labelControlPtr->ptsData.tarStr[0] == '\0' && maybeClik.x >= 0)
				{
					draw::ControlLogic::tagPickIdx = -1;
					labelControlPtr->ptsData.updataFalgs();
					listComponentReChoose(labelControlPtr->ptsData.tagsListName, labelControlPtr->ptsData.tagPickIdx);
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
				if (labelControlPtr->ptsData.pickedChanged)
				{
					labelControlPtr->ptsData.updataFalgs();
				}
				ImGui::SetCursorPos(pushPos);
			}
			if (ImGui::Checkbox("try2D3D", &draw::ControlLogic::tryFind2d3dPair))
			{

			}
			ImGui::SameLine();
			ImGui::InputTextMultiline("<-tag", labelControlPtr->ptsData.tarStr, draw::ControlLogic::tagLengthMax, ImVec2(200, 20), ImGuiTreeNodeFlags_None + ImGuiInputTextFlags_CharsNoBlank);

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
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
enum class IterType
{};
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
		std::vector< std::map<std::string, ImVec2>>controlPtsAndTag;
		//ImVec2 tempPt{ -1,-1 };  // for first track ,the temp point wil be showed on canvas

		std::map<std::string, ImU32> colors;
		std::vector<std::string>tagsListName;
		std::vector<std::string>tagsName;
		std::map<std::string, std::vector<std::string>> hasTagFlags;
		static int tagPickIdx;
		bool pickedChanged;
		static void shiftSnap(cv::Mat& img, const ImVec2& shift)
		{
			cv::Mat rotation_matix = cv::Mat::eye(2,3,CV_32FC1);
			rotation_matix.ptr<float>(0)[2] = shift.x;
			rotation_matix.ptr<float>(1)[2] = shift.y;
			cv::Mat rotated_image;
			warpAffine(img, rotated_image, rotation_matix, img.size());
			img = rotated_image;
			return;
		}
		/*
		void updata()
		{
			std::string maybePickedTag = (tagPickIdx >= 0) ? tagsName[tagPickIdx] : "";
			std::string maybePickedTag2 = (tagPickIdx >= 0) ? tagsListName[tagPickIdx] : "";
			std::unordered_set<std::string>newTagsName;
			for (const auto& d : controlPtsAndTag)
			{
				for (const auto& d2 : d)
				{
					newTagsName.insert(d2.first);
				}
			}
			std::vector<std::string>::iterator iter;
			for (iter = tagsName.begin(); iter != tagsName.end(); iter++)
			{
				std::string tagStr = *iter;
				if (newTagsName.find(tagStr) == newTagsName.end())
				{
					tagsListName.erase(tagsListName.begin() + (iter - tagsName.begin()));
					iter = tagsName.erase(iter);
				}
			}
			if (maybePickedTag.length() > 0)
			{
				if (newTagsName.find(maybePickedTag) == newTagsName.end())
				{
					tagPickIdx = -1;
				}
			}
			hasTagFlags.clear();
			for (const auto& d : colors)
			{
				auto& tagAndShortName = hasTagFlags[d.first];
				tagAndShortName.insert(tagAndShortName.end(), picShortNameForlist.begin(), picShortNameForlist.end());
				for (int j = 0; j < picShortNameForlist.size(); j++)
				{
					if (controlPtsAndTag[j].count(d.first) > 0)
					{
						hasTagFlags[d.first][j][1] = '*';
					}
				}
			}
		}
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
				for (const auto& d : controlPtsAndTag[i])
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
					controlPtsAndTag[i].clear();
					for (const auto& d : imgPts)
					{
						controlPtsAndTag[i][d.first] = ImVec2(d.second[0], d.second[1]);
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
	}; 
	int ControlLogic::tagPickIdx = -1;
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
				instance->ptsData.controlPtsAndTag.resize(picShortNameForlist.size());
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
						const std::map<std::string, ImVec2>& tags = ptsData.controlPtsAndTag[imgPickIdx];
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
					const std::map<std::string, ImVec2>& tags = ptsData.controlPtsAndTag[imgPickIdx];
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
		ImVec2 control(const ImVec2&canvasPos, ImVec2&offset)
		{
			ImVec2 mousePosInImage;
			mousePosInImage.x = (ImGui::GetIO().MousePos.x - ImGui::GetWindowPos().x - canvasPos.x);
			mousePosInImage.y = (ImGui::GetIO().MousePos.y - ImGui::GetWindowPos().y - canvasPos.y);
			float wheel = ImGui::GetIO().MouseWheel;
			if (ImGui::GetIO().MouseClicked[0])
			{
				mouseDownStartPos.x = -900000;
				if (mousePosInImage.x >= 0 && mousePosInImage.y >= 0 && mousePosInImage.x < canvas.x && mousePosInImage.y < canvas.y)
				{
					float x_inRatio = mousePosInImage.x * Draw::canvasInv.x * (Draw::zoom_end.x - Draw::zoom_start.x) + Draw::zoom_start.x;
					float y_inRatio = mousePosInImage.y * Draw::canvasInv.y * (Draw::zoom_end.y - Draw::zoom_start.y) + Draw::zoom_start.y;
					float x_inPic = x_inRatio * currentImgWidth;
					float y_inPic = y_inRatio * currentImgHeight;
					return ImVec2(x_inPic, y_inPic);
				}
			}
			else if (ImGui::GetIO().MouseDown[0] && mouseDownStartPos.x < -800000)
			{
				mouseDownStartPos = mousePosInImage;
			}
			else if (ImGui::GetIO().MouseReleased[0])
			{
				mouseDownStartPos.x = -900000;
			}
			else if (ImGui::GetIO().MouseDown[0])
			{ 
				offset.x += (mousePosInImage.x- mouseDownStartPos.x );
				offset.y += (mousePosInImage.y - mouseDownStartPos.y);
				//LOG_OUT << mousePosInImage.x - mouseDownStartPos.x << ", " << mousePosInImage.y - mouseDownStartPos.y;
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
class BfmIter
{
public:
	bfm::Bfm2019* bfmIns{ nullptr };
	BfmIter() = default;
    BfmIter(const  std::filesystem::path& mvsResultDir, const  std::filesystem::path& modelDirPath)
    {
		std::filesystem::path bfmFacePath = modelDirPath / "model2019_face12.h5";
		if (!std::filesystem::exists(bfmFacePath))
		{
			LOG_ERR_OUT << "need models/model2019_face12.h5";
			return;
		}
		bfmIns = new bfm::Bfm2019(bfmFacePath);
		bfmIns->generateRandomFace(msh.V, msh.C);
		msh.F = bfmIns->F;
        bfm_R << 1, 0, 0, 0, -1, 0, 0, 0, -1;
        bfm_t << 0, 0, 300;  
		msh.rotate(bfm_R, bfm_t);
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
							imgNameForlist.emplace_back(stem); 
							imgPaths.emplace_back(imgPath);
							imgs.emplace_back(image);
							renders.emplace_back(cv::Mat());
							renderPts.emplace_back(cv::Mat()); 
							shifts.emplace_back(ImVec2(0, 0));
						}
						
					}
				}
			}
		}
		if (imgNameForlist.size() < 1)
		{
			progress.procRunning.store(0);
			LOG_ERR_OUT << "imgNameForlist.size()<1";
			return;
		}
		progress.procRunning.store(0);
    }
    ~BfmIter() {};
    bool iter(const std::vector<cv::Point3f>& src, const std::vector<cv::Point3f>& tar, const IterType& type)
    {
        return false;
    }
    Eigen::Matrix3f bfm_R;
    Eigen::RowVector3f bfm_t;
	std::vector<cv::Mat>imgs;
	std::vector<cv::Mat>renders;
	std::vector<cv::Mat>renderPts; 
	std::vector<ImVec2>shifts;;
	std::vector<std::filesystem::path>imgPaths;
	std::vector<std::string>imgNameForlist;
	std::filesystem::path imgDirPath_;
	std::filesystem::path modelDirPath_;
	static GLuint image_texture;
	static int viewWindowHeight;
	static int viewWindowWidth; 
	static int imgPickIdx;
	meshdraw::Mesh msh;
	std::vector<meshdraw::Camera> imgCameras;
};
static bool showImgDirBrowser = false;
static std::filesystem::path imgDirPath;
static std::filesystem::path modelDirPath;
static browser::Browser* imgDirPicker = nullptr;
static browser::Browser* modelDirPicker = nullptr;
static BfmIter* BfmIterManger = nullptr;
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


		if (BfmIterManger == nullptr && imgDirPath.string().length() > 0 && modelDirPath.string().length() > 0)
		{
			progress.numerator.store(-1);
			progress.procRunning.fetch_add(1);
			progress.proc = new std::thread(
				[&]() {
					BfmIterManger = new BfmIter(imgDirPath, modelDirPath);
				}
			);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
			bool mixFactorChanged = ImGui::SliderFloat(u8"»ìºÏ¶È", &mixFactor, 0.f, 1.f);

			auto imgListLocation = ImGui::GetCursorPos();
			ImVec2 listPicSize(100, 500);
			ImVec2 canvas_location = imgListLocation;
			canvas_location.x += listPicSize.x;
			draw::Draw* labelControlPtr = draw::Draw::getDrawIns(canvas_location, 720, BfmIterManger->imgNameForlist, BfmIterManger->imgPaths);
			labelControlPtr->ptsData.drawPos = ImVec2(canvas_location.x+ draw::Draw::canvasMaxSide, canvas_location.y);
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
			if (BfmIter::imgPickIdx >= 0)
			{
				ImVec2& shift = BfmIterManger->shifts[BfmIter::imgPickIdx];
				maybeClik = labelControlPtr->control(labelControlPtr->draw_pos, shift);
			}
			if (BfmIter::imgPickIdx >= 0 && (pickedChanged || mixFactorChanged || BfmIterManger->shifts[BfmIter::imgPickIdx].x != 0))
			{
				if (pickedChanged)
				{
					ImgShift.release();
				}
				const cv::Mat& img = BfmIterManger->imgs[BfmIter::imgPickIdx];
				const meshdraw::Camera&cam= BfmIterManger->imgCameras[BfmIter::imgPickIdx];
				cv::Mat& render3d = BfmIterManger->renders[BfmIter::imgPickIdx];
				cv::Mat& render3dPts = BfmIterManger->renderPts[BfmIter::imgPickIdx];
				if (render3d.empty())
				{ 
					if (meshdraw::isEmpty(BfmIterManger->msh.facesNormal))
					{
						BfmIterManger->msh.figureFacesNomral();
					}
					cv::Mat mask; 
					meshdraw::render(BfmIterManger->msh, cam, render3d, render3dPts, mask);
				}  
				if (ImgShift.empty())
				{
					render3d.copyTo(ImgShift);
				}
				if (BfmIterManger->shifts[BfmIter::imgPickIdx].x!=0)
				{
					render3d.copyTo(ImgShift);
					draw::ControlLogic::shiftSnap(ImgShift, BfmIterManger->shifts[BfmIter::imgPickIdx]);
				} 
				cv::addWeighted(img, mixFactor, ImgShift, 1 - mixFactor, 0, showImgMix);
				labelControlPtr->feedImg(showImgMix); 


				if (labelControlPtr->ptsData.tarStr[0] != '\0')
				{
					std::string thisTarName(labelControlPtr->ptsData.tarStr);

					if (labelControlPtr->ptsData.tagsName.end() == std::find(labelControlPtr->ptsData.tagsName.begin(), labelControlPtr->ptsData.tagsName.end(), thisTarName))
					{
						labelControlPtr->ptsData.tagsName.emplace_back(thisTarName);
						labelControlPtr->ptsData.tagsListName.emplace_back("  " + thisTarName);
						labelControlPtr->ptsData.colors[thisTarName] = (getImguiColor());
					}
					labelControlPtr->ptsData.controlPtsAndTag[BfmIter::imgPickIdx][thisTarName] = maybeClik;
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
			
			ImGui::InputTextMultiline("<-tag", labelControlPtr->ptsData.tarStr, draw::ControlLogic::tagLengthMax, ImVec2(200, 20), ImGuiTreeNodeFlags_None + ImGuiInputTextFlags_CharsNoBlank);

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
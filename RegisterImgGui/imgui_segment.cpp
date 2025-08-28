#include <numeric>
#include <vector>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <map>
#include "imgui.h" 
#include "browser.h"
#include "segmentFrame.h"
#include "imgui_tools.h"
#include "log.h"
#include "sam2.h"
static bool showImgDirBrowser = false;
static std::filesystem::path imgDirPath;
static std::filesystem::path modelDirPath;
static browser::Browser* imgDirPicker = nullptr;
static browser::Browser* modelDirPicker = nullptr;
static ProgressThread progress;

class SegmentMgr
{
public:
	SegmentMgr() = default;
	SegmentMgr(const  std::filesystem::path& dirPath, const  std::filesystem::path& modelDirPath)
	{

		std::filesystem::path encoderParamPath = modelDirPath / "ncnnEncoder.param";
		std::filesystem::path encoderBinPath = modelDirPath / "ncnnEncoder.bin";
		std::filesystem::path dencoderOnnxPath = modelDirPath / "opencv_decoder.onnx";
		if (!std::filesystem::exists(encoderParamPath))
		{
			progress.procRunning.store(0);
			LOG_ERR_OUT << "not found : " << encoderParamPath;
			return;
		}
		if (!std::filesystem::exists(encoderBinPath))
		{
			progress.procRunning.store(0);
			LOG_ERR_OUT << "not found : " << encoderBinPath;
			return;
		}
		if (!std::filesystem::exists(dencoderOnnxPath))
		{
			progress.procRunning.store(0);
			LOG_ERR_OUT << "not found : " << dencoderOnnxPath;
			return;
		}


		imgPaths.clear();
		imgName.clear();
		imgNameForList.clear();
		imgDat.clear();
		imgDirPath_ = dirPath;
		for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{ imgDirPath_ })
		{
			const auto& thisFilename = dir_entry.path();
			if (thisFilename.has_extension())
			{
				const auto& ext = thisFilename.extension().string();
				if (ext.compare(".bmp") == 0 || ext.compare(".jpg") == 0 || ext.compare(".jpeg") == 0)
				{
					cv::Mat img = cv::imread(thisFilename.string());
					if (!img.empty())
					{
						imgPaths.emplace_back(thisFilename);
						std::string stem = thisFilename.filename().stem().string();
						imgName.emplace_back(stem);
						imgNameForList.emplace_back("   "+stem);
						imgDat.emplace_back(img);
					}
					
				}
			}
		}
		if (imgName.size()<1)
		{
			progress.procRunning.store(0);
			LOG_ERR_OUT << "imgName.size()<1";
			return;
		}
		progress.denominator.store(imgName.size());
		progress.numerator.store(0);
		progress.procRunning.fetch_add(1);
		sam2Ins = new sam2::Sam2(encoderParamPath, encoderBinPath, dencoderOnnxPath);
		for (size_t i = 0; i < imgName.size(); i++)
		{
			const auto& parentDir = std::filesystem::canonical(imgPaths[i].parent_path());
			std::string shortName = imgPaths[i].filename().stem().string();
			const auto& segData = parentDir / (shortName + ".samDat");
			bool reload = false;
			if (std::filesystem::exists(segData))
			{
				reload = sam2Ins->deserializationFeat(segData);
			}
			if(!reload)
			{
				sam2Ins->inputImage(imgPaths[i]);
				sam2Ins->serializationFeat(segData);
			}
			progress.numerator.fetch_add(1);			
		}
		progress.procRunning.store(0);
	}
	~SegmentMgr()
	{}
	sam2::Sam2* sam2Ins{ nullptr };
	std::vector<std::filesystem::path>imgPaths;
	std::vector<std::string>imgName;
	std::vector<std::string>imgNameForList;
	std::vector<cv::Mat> imgDat;
	std::filesystem::path imgDirPath_;
	std::filesystem::path modelDirPath_;
	static GLuint image_texture;
	static int viewWindowHeight;
	static int viewWindowWidth;
	static int imgPickIdx;
	static bool showMask;
};
static SegmentMgr* segmentMgr = nullptr;
GLuint SegmentMgr::image_texture = 0;
int SegmentMgr::viewWindowHeight = 720;
int SegmentMgr::viewWindowWidth = 960;
int SegmentMgr::imgPickIdx = -1;
bool SegmentMgr::showMask = true;


struct SegmentControl
{
	std::vector<ImVec2>tempPt;
	std::vector<ImVec2>tempNegPt;
	std::vector<std::vector<std::pair<int, cv::Point2i>>>hints;
	std::vector<cv::Mat>masks;
	static bool load(const std::filesystem::path& picPath, std::vector<std::pair<int, cv::Point2i>>& hints, cv::Mat& mask)
	{
		if (!std::filesystem::exists(picPath))
		{
			LOG_ERR_OUT << "not found : " << picPath;
			return false;
		}
		cv::Mat img = cv::imread(picPath.string());
		if (img.empty())
		{
			LOG_ERR_OUT << "img.empty";
			return false;
		}
		auto parentDir = picPath.parent_path();
		auto shortName = picPath.filename().stem();
		auto jsonPath = parentDir / (shortName.string() + ".json");
		auto binPath = parentDir / (shortName.string() + ".segBin");
		auto showPath = parentDir / ("mask_" + shortName.string() + ".jpg");
		if (std::filesystem::exists(jsonPath) && std::filesystem::exists(binPath))
		{

		}
		else
		{
			mask = cv::Mat(img.size(),CV_8UC3);
			mask.setTo(cv::Scalar(0,0,0));
		}
		return true;
	}
	static bool save(const std::filesystem::path& picPath)
	{
		return true;
	}
};
class SegmentGui
{
private:
	int currentImgHeight;
	int currentImgWidth;
	static SegmentGui* instance;
	static ImVec2 zoom_start;
	static ImVec2 zoom_end;
	static GLuint image_texture;
	static int canvasMaxSide;
	SegmentGui() {};
	SegmentGui(const SegmentGui&) = delete;
	SegmentGui& operator=(const SegmentGui&) = delete;
public:
	static const int segmentNameStrLengthMax{ 24 };
	static char segmentNameStr[segmentNameStrLengthMax];
	SegmentControl ptsData;
	static ImVec2 draw_pos;
	static ImVec2 canvas;//(w,h)
	static ImVec2 canvasInv;//(1/w,1/h)
	static float resizeFactor;
	bool hasImageContext{ false };
	bool showLabeled{ true };
	static ImVec2 imgPt2GuiPt(const ImVec2& imgPt, const int& imgHeight, const int& imgWidth, const ImVec2& canvas_size, const ImVec2& zoom_start, const ImVec2& zoom_end, const ImVec2& canvas_location)
	{
		float x_inRatio = (imgPt.x / imgWidth - zoom_start.x) / (zoom_end.x - zoom_start.x);
		float y_inRatio = (imgPt.y / imgHeight - zoom_start.y) / (zoom_end.y - zoom_start.y);
		float x = x_inRatio * canvas_size.x + canvas_location.x + ImGui::GetWindowPos().x;
		float y = y_inRatio * canvas_size.y + canvas_location.y + ImGui::GetWindowPos().y;
		return ImVec2(x, y);
	}
	static SegmentGui* getSegmentGui(const ImVec2& draw_pos_, const int& canvasMaxSide_, const std::vector<std::string>& picShortName, const std::vector<std::filesystem::path>& picPaths)
	{
		if (SegmentGui::instance == nullptr)
		{
			SegmentGui::instance = new SegmentGui();
			instance->ptsData.hints.resize(picPaths.size());
			instance->ptsData.masks.resize(picPaths.size());
			for (int i = 0; i < picPaths.size(); i++)
			{
				SegmentControl::load(picPaths[i], instance->ptsData.hints[i], instance->ptsData.masks[i]);
			}
			//instance->ptsData.picShortName.insert(instance->ptsData.picShortName.end(), picShortName.begin(), picShortName.end());
			//instance->ptsData.controlPtsAndTag.resize(picShortName.size());
			//instance->ptsData.loadOnlyOnce(picPaths);
		}
		else
		{
			return SegmentGui::instance;
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
			float a = 1.f * currentImgWidth / SegmentGui::canvasMaxSide;
			float b = 1.f * currentImgHeight / SegmentGui::canvasMaxSide;
			if (a < 1.f && b < 1.f)
			{
				SegmentGui::canvas = ImVec2(currentImgWidth, currentImgHeight);
			}
			else if (a > b)
			{
				SegmentGui::canvas = ImVec2(currentImgWidth / a, currentImgHeight / a);
			}
			else
			{
				SegmentGui::canvas = ImVec2(currentImgWidth / b, currentImgHeight / b);
			}
			SegmentGui::canvasInv.x = 1. / SegmentGui::canvas.x;
			SegmentGui::canvasInv.y = 1. / SegmentGui::canvas.y;
		}
		return true;
	}
	bool draw(const int& imgPickIdx)
	{
		if (hasImageContext)
		{
			ImGui::SetCursorPos(draw_pos);
			ImGui::Image((ImTextureID)(intptr_t)image_texture, SegmentGui::canvas, SegmentGui::zoom_start, SegmentGui::zoom_end, ImVec4(1, 1, 1, 1), ImVec4(.5, .5, .5, .5));
			for (int i = 0; i < ptsData.tempPt.size(); i++)
			{
				ImVec2 guiPt = imgPt2GuiPt(ptsData.tempPt[i], currentImgHeight, currentImgWidth, canvas, zoom_start, zoom_end, draw_pos);
				ImGui::GetForegroundDrawList()->AddCircleFilled(guiPt, 4.0f, 0xC80688FB);
				ImGui::GetForegroundDrawList()->AddCircle(guiPt, 4.0f, 0xC8000000);
			}
			for (int i = 0; i < ptsData.tempNegPt.size(); i++)
			{
				ImVec2 guiPt = imgPt2GuiPt(ptsData.tempNegPt[i], currentImgHeight, currentImgWidth, canvas, zoom_start, zoom_end, draw_pos);
				ImGui::GetForegroundDrawList()->AddCircleFilled(guiPt, 4.0f, 0xFF54FF36);
				ImGui::GetForegroundDrawList()->AddCircle(guiPt, 4.0f, 0xC8000000);
			}
		}
		return true;
	}
	ImVec2 control(const ImVec2& mousePosInImage, const float wheel, const bool& mouseLeftDown, const bool& mouseRightDown)
	{
		if (abs(wheel) > 0.01)
		{
			if (mousePosInImage.x >= 0 && mousePosInImage.y >= 0 && mousePosInImage.x < canvas.x && mousePosInImage.y < canvas.y)
			{
				float x_inRatio = mousePosInImage.x * SegmentGui::canvasInv.x * (SegmentGui::zoom_end.x - SegmentGui::zoom_start.x) + SegmentGui::zoom_start.x;
				float y_inRatio = mousePosInImage.y * SegmentGui::canvasInv.y * (SegmentGui::zoom_end.y - SegmentGui::zoom_start.y) + SegmentGui::zoom_start.y;
				float x_inPic = x_inRatio * currentImgWidth;
				float y_inPic = y_inRatio * currentImgHeight;
				{
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
					SegmentGui::zoom_start.x = x_inRatio - regionRadius;
					SegmentGui::zoom_start.y = y_inRatio - regionRadius;
					SegmentGui::zoom_end.x = x_inRatio + regionRadius;
					SegmentGui::zoom_end.y = y_inRatio + regionRadius;
					if (SegmentGui::zoom_start.x < 0)
					{
						SegmentGui::zoom_start.x = 0;
						SegmentGui::zoom_end.x = (1 - regionDiameter);
					}
					if (SegmentGui::zoom_start.y < 0)
					{
						SegmentGui::zoom_start.y = 0;
						SegmentGui::zoom_end.y = (1 - regionDiameter);
					}
					if (SegmentGui::zoom_end.x > 1)
					{
						SegmentGui::zoom_start.x = regionDiameter;
						SegmentGui::zoom_end.x = 1;
					}
					if (SegmentGui::zoom_end.y > 1)
					{
						SegmentGui::zoom_start.y = regionDiameter;
						SegmentGui::zoom_end.y = 1;
					}
				}
			}
		}
		if (mouseLeftDown|| mouseRightDown)
		{
			if (mousePosInImage.x >= 0 && mousePosInImage.y >= 0 && mousePosInImage.x < canvas.x && mousePosInImage.y < canvas.y)
			{
				float x_inRatio = mousePosInImage.x * SegmentGui::canvasInv.x * (SegmentGui::zoom_end.x - SegmentGui::zoom_start.x) + SegmentGui::zoom_start.x;
				float y_inRatio = mousePosInImage.y * SegmentGui::canvasInv.y * (SegmentGui::zoom_end.y - SegmentGui::zoom_start.y) + SegmentGui::zoom_start.y;
				float x_inPic = x_inRatio * currentImgWidth;
				float y_inPic = y_inRatio * currentImgHeight;
				return ImVec2(x_inPic, y_inPic);
			}
		}
		return ImVec2(-1, -1);
	}
};
SegmentGui* SegmentGui::instance = nullptr;
ImVec2 SegmentGui::draw_pos = ImVec2();
ImVec2 SegmentGui::canvas = ImVec2();
ImVec2 SegmentGui::canvasInv = ImVec2();
ImVec2 SegmentGui::zoom_start = ImVec2();
ImVec2 SegmentGui::zoom_end = ImVec2();
GLuint SegmentGui::image_texture = 0;
int SegmentGui::canvasMaxSide = 0;
float SegmentGui::resizeFactor = 1;
char SegmentGui::segmentNameStr[segmentNameStrLengthMax] = "\0";


bool segmentFrame(bool* show_regist_window)
{
	ImGui::SetNextWindowSize(ImVec2(1280, 960));//ImVec2(x, y)
	ImGui::Begin("segment", show_regist_window, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);
	ImGui::Text("segment"); 

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
		if (imgDirPicker != nullptr && imgDirPath.string().length()==0)
		{
			if (imgDirPicker->pick(imgDirPath))
			{
				if (!imgDirPath.empty())
				{
					LOG_OUT << imgDirPath;
				}
				delete modelDirPicker;
				modelDirPicker = nullptr;
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
		if (segmentMgr == nullptr && imgDirPath.string().length() > 0 && modelDirPath.string().length() > 0 )
		{
			progress.procRunning.fetch_add(1);
			progress.proc = new std::thread(
				[&]() {
					segmentMgr = new SegmentMgr(imgDirPath, modelDirPath);
				}
			);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		if (segmentMgr != nullptr  && segmentMgr->imgName.size()<1)
		{
			delete segmentMgr;
			segmentMgr = nullptr;
			imgDirPath = "";
			modelDirPath = "";
		}
		ImGui::ColorButton("drag me", ImGui::ColorConvertU32ToFloat4(0xC80688FB), ImGuiColorEditFlags_NoPicker| ImGuiColorEditFlags_NoOptions| ImGuiColorEditFlags_NoSmallPreview| ImGuiColorEditFlags_NoTooltip| ImGuiColorEditFlags_NoLabel| ImGuiColorEditFlags_InputRGB);
		if (segmentMgr != nullptr && segmentMgr->imgName.size() > 0)
		{
			auto imgListLocation = ImGui::GetCursorPos();
			bool pickedChanged = false;
			ImVec2 listPicSize(100, 500);
			listComponent("picPick", listPicSize, segmentMgr->imgName, SegmentMgr::imgPickIdx, pickedChanged);
			ImVec2 canvas_location = imgListLocation;
			canvas_location.x += listPicSize.x;
			canvas_location.x += 10;
			SegmentGui* segmentControlPtr = SegmentGui::getSegmentGui(canvas_location, 720, segmentMgr->imgName, segmentMgr->imgPaths);
			if (SegmentMgr::imgPickIdx>=0 && (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) || ImGui::IsKeyPressed(ImGuiKey_UpArrow)))
			{
				pickedChanged = true;
				segmentMgr->imgName[SegmentMgr::imgPickIdx][0] = ' ';
				SegmentMgr::imgPickIdx--;
				if (SegmentMgr::imgPickIdx < 0)
				{
					SegmentMgr::imgPickIdx = 0;
					pickedChanged = false;
				}
				segmentMgr->imgName[SegmentMgr::imgPickIdx][0] = '-';
				//continue;
			}
			if (SegmentMgr::imgPickIdx >= 0 && (ImGui::IsKeyPressed(ImGuiKey_RightArrow) || ImGui::IsKeyPressed(ImGuiKey_DownArrow)))
			{
				pickedChanged = true;
				segmentMgr->imgName[SegmentMgr::imgPickIdx][0] = ' ';
				SegmentMgr::imgPickIdx++;
				if (SegmentMgr::imgPickIdx >= segmentMgr->imgPaths.size())
				{
					SegmentMgr::imgPickIdx = segmentMgr->imgPaths.size() - 1;
					pickedChanged = false;
				}
				segmentMgr->imgName[SegmentMgr::imgPickIdx][0] = '-';
				//continue;
			}
			if (SegmentMgr::imgPickIdx >= 0 && pickedChanged)
			{
				segmentControlPtr->ptsData.tempPt.clear();
				segmentControlPtr->ptsData.tempNegPt.clear();
				cv::Mat asd = cv::imread(segmentMgr->imgPaths[SegmentMgr::imgPickIdx].string());
				segmentControlPtr->feedImg(asd);
				bool& mouseLeftDown = ImGui::GetIO().MouseReleased[0];
				bool& mouseRightDown = ImGui::GetIO().MouseReleased[1];
				mouseRightDown = false;
			}
			segmentControlPtr->draw(SegmentMgr::imgPickIdx);
			if (segmentControlPtr->hasImageContext)
			{
				ImVec2 tagListlocation = SegmentGui::draw_pos;
				tagListlocation.x += segmentControlPtr->canvas.x;
				tagListlocation.x += ImGui::GetWindowPos().x;
				tagListlocation.y += ImGui::GetWindowPos().y;
				ImGui::SetCursorScreenPos(tagListlocation);
			}
			if (segmentControlPtr->hasImageContext)
			{
				ImVec2 mousePosInImage;
				mousePosInImage.x = (ImGui::GetIO().MousePos.x - ImGui::GetWindowPos().x - segmentControlPtr->draw_pos.x);
				mousePosInImage.y = (ImGui::GetIO().MousePos.y - ImGui::GetWindowPos().y - segmentControlPtr->draw_pos.y);
				float wheel = ImGui::GetIO().MouseWheel;
				//ImGui::Text("Mouse pos: (%g, %g)", mousePosInImage.x, mousePosInImage.y);
				//ImGui::Text("Mouse wheel: %.1f", ImGui::GetIO().MouseWheel);
				bool mouseLeftDown = ImGui::GetIO().MouseReleased[0];
				bool mouseRightDown = ImGui::GetIO().MouseReleased[1];
				ImVec2 maybeClik = segmentControlPtr->control(mousePosInImage, wheel, mouseLeftDown, mouseRightDown);
				if (maybeClik.x > 0)
				{
					if (mouseLeftDown)
					{
						segmentControlPtr->ptsData.tempPt.emplace_back(maybeClik);
					}
					else
					{
						segmentControlPtr->ptsData.tempNegPt.emplace_back(maybeClik);
					}
				}
			}
			deployButtom = ImVec2(imgListLocation.x + ImGui::GetWindowPos().x, std::max(listPicSize.y, segmentControlPtr->canvas.y) + imgListLocation.y + ImGui::GetWindowPos().y);
			if (SegmentMgr::imgPickIdx >= 0)
			{
				ImGui::SetCursorScreenPos(deployButtom);
				if (segmentControlPtr->ptsData.tempPt.size() <= 0)
				{
					ImGui::BeginDisabled();
				}
				if (ImGui::Button("seg"))
				{
					LOG_OUT << "seg at " << SegmentMgr::imgPickIdx;

					progress.numerator.store(-1);
					progress.procRunning.fetch_add(1);
					progress.proc = new std::thread(
						[&]() {


							segmentControlPtr->ptsData.hints[SegmentMgr::imgPickIdx].clear();
							for (const auto&d: segmentControlPtr->ptsData.tempPt)
							{
								segmentControlPtr->ptsData.hints[SegmentMgr::imgPickIdx].emplace_back(std::make_pair(1, cv::Point2i(d.x, d.y)));
							}
							for (const auto& d : segmentControlPtr->ptsData.tempNegPt)
							{
								segmentControlPtr->ptsData.hints[SegmentMgr::imgPickIdx].emplace_back(std::make_pair(0, cv::Point2i(d.x, d.y)));
							}
							segmentMgr->sam2Ins->inputHint(segmentControlPtr->ptsData.hints[SegmentMgr::imgPickIdx], segmentControlPtr->ptsData.masks[SegmentMgr::imgPickIdx]);
							segmentControlPtr->ptsData.tempPt.clear();
							segmentControlPtr->ptsData.tempNegPt.clear();
							progress.procRunning.store(0);
						}
					);
					std::this_thread::sleep_for(std::chrono::milliseconds(100));
				}
				if (segmentControlPtr->ptsData.tempPt.size() <= 0)
				{
					ImGui::EndDisabled();
				}
				ImGui::SameLine();
				if (ImGui::Checkbox("showMask", &SegmentMgr::showMask))
				{ 
					cv::Mat asd = cv::imread(segmentMgr->imgPaths[SegmentMgr::imgPickIdx].string());
					const cv::Mat& mask = segmentControlPtr->ptsData.masks[SegmentMgr::imgPickIdx];
					const float gamma = 0;
					const float mixFacter = 0.5;
					cv::Mat mixImg;
					cv::addWeighted(asd, mixFacter, mask, 1 - mixFacter, gamma, mixImg);
					segmentControlPtr->feedImg(mixImg);
				}
				ImGui::SameLine();
				if (ImGui::Button("~Tag") || ImGui::IsKeyPressed(ImGuiKey_Delete))
				{
					segmentControlPtr->ptsData.tempPt.clear();
					segmentControlPtr->ptsData.tempNegPt.clear();
				}
				ImGui::SameLine();
				if (ImGui::Button("save"))
				{
				}
				deployButtom.y += 45;
			}
		}
		break;
	}
	if (progress.msg.length() > 0)
	{
		ImGui::Text(progress.msg.c_str());
	}
	if (ImGui::Button("Close Me") && *show_regist_window) *show_regist_window = false;
	ImGui::End();
	return true;
}
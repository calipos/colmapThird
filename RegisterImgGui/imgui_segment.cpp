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
						std::string stem = "   " + thisFilename.filename().stem().string();
						imgName.emplace_back(stem);
						imgDat.emplace_back();
					}
					
				}
			}
		}
		if (imgName.size()<3)
		{
			progress.procRunning.store(0);
			LOG_ERR_OUT << "imgName.size()<3";
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
			const auto& segData = parentDir / (shortName + ".segDat");
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
	std::vector<cv::Mat> imgDat;
	std::filesystem::path imgDirPath_;
	std::filesystem::path modelDirPath_;
	static GLuint image_texture;
	static int viewWindowHeight;
	static int viewWindowWidth;
	int imgPickIdx { 0};
};
static SegmentMgr* segmentMgr = nullptr;
GLuint SegmentMgr::image_texture = 0;
int SegmentMgr::viewWindowHeight = 720;
int SegmentMgr::viewWindowWidth = 960;


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
				progress.proc->join();
				progress.procRunning.store(false);
			}
			progress.proc = nullptr;
		}
	}
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
		if (segmentMgr != nullptr  && segmentMgr->imgName.size()<3)
		{
			delete segmentMgr;
			segmentMgr = nullptr;
			imgDirPath = "";
			modelDirPath = "";
		}
		//if (segmentMgr != nullptr && segmentMgr->imgName.size() > 0)
		//{
		//	{
		//		lidarPics[imgPickIdx].copyTo(showInternsity);
		//		float gamma = 0;
		//		shiftSnap(showInternsity, thisSnapShift);
		//		cv::addWeighted(showImg, mixFactor, showInternsity, 1 - mixFactor, gamma, showImgMix);
		//		bool ret = LoadTextureFromMat(showImgMix, &image2_texture, &image2_width, &image2_height);
		//		//canvas = ImVec2(image2_width / 2, image2_height / 2);
		//		viewWindowHeight = static_cast<int>(1. * image2_height * viewWindowWidth / image2_width);
		//		canvas = ImVec2(viewWindowWidth, viewWindowHeight);
		//	}
		//	ImVec2 canvasPos = ImGui::GetCursorPos();
		//	ImGui::Image((ImTextureID)(intptr_t)image2_texture, canvas, zoom_start, zoom_end, ImVec4(1, 1, 1, 1), ImVec4(.5, .5, .5, .5));
		//}
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
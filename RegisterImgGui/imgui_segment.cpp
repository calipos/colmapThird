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
static bool showImgDirBrowser = false;
static std::filesystem::path dirPath;
static browser::Browser* imgDirPicker = nullptr;
static browser::Browser* modelDirPicker = nullptr;
static ProgressThread progress;
class SegmentMgr
{
public:
	SegmentMgr() = default;
	SegmentMgr(const  std::filesystem::path& dirPath)
	{
		imgPaths.clear();
		imgName.clear();
		dirPath_ = dirPath;
		for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{ dirPath_ })
		{
			const auto& thisFilename = dir_entry.path();
			if (thisFilename.has_extension())
			{
				const auto& ext = thisFilename.extension().string();
				//const auto& parentDir = std::filesystem::canonical(dir_entry.path().parent_path());
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
	}
	~SegmentMgr()
	{}
	std::vector<std::filesystem::path>imgPaths;
	std::vector<std::string>imgName;
	std::vector<cv::Mat> imgDat;
	std::filesystem::path dirPath_;
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
			if (imgDirPicker == nullptr)
			{
				dirPath = "";
				imgDirPicker = new browser::Browser(browser::BrowserPick::PICK_DIR);
			}
		}
		if (imgDirPicker != nullptr)
		{
			if (imgDirPicker->pick(dirPath))
			{
				progress.proc = new std::thread(
					[&]() {
						if (!dirPath.empty())
						{
							segmentMgr = new SegmentMgr(dirPath);
							LOG_OUT << dirPath;
						}
						delete imgDirPicker;
						imgDirPicker = nullptr;
					}
				);
			}
		}
		if (ImGui::Button("pick model dir"))
		{
			if (modelDirPicker == nullptr)
			{
				dirPath = "";
				modelDirPicker = new browser::Browser(browser::BrowserPick::PICK_DIR);
			}
		}
		if (modelDirPicker != nullptr)
		{
			if (modelDirPicker->pick(dirPath))
			{
				if (!dirPath.empty())
				{
					LOG_OUT << dirPath;
				}
				delete modelDirPicker;
				modelDirPicker = nullptr;
			}
		}
		if (segmentMgr != nullptr && segmentMgr->imgName.size() > 0)
		{
			{
				lidarPics[imgPickIdx].copyTo(showInternsity);
				float gamma = 0;
				shiftSnap(showInternsity, thisSnapShift);
				cv::addWeighted(showImg, mixFactor, showInternsity, 1 - mixFactor, gamma, showImgMix);
				bool ret = LoadTextureFromMat(showImgMix, &image2_texture, &image2_width, &image2_height);
				//canvas = ImVec2(image2_width / 2, image2_height / 2);
				viewWindowHeight = static_cast<int>(1. * image2_height * viewWindowWidth / image2_width);
				canvas = ImVec2(viewWindowWidth, viewWindowHeight);
			}
			ImVec2 canvasPos = ImGui::GetCursorPos();
			ImGui::Image((ImTextureID)(intptr_t)image2_texture, canvas, zoom_start, zoom_end, ImVec4(1, 1, 1, 1), ImVec4(.5, .5, .5, .5));
		}
		break;
	}

	if (ImGui::Button("Close Me") && *show_regist_window) *show_regist_window = false;
	ImGui::End();
	return true;
}
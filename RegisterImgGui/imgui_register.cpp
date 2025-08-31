#include <numeric>
#include <vector>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <map>
#include "imgui.h" 
#include "omp.h"
#include "log.h"
#include "browser.h"
#include "labelme.h"
#include "registerFrame.h"
#include "sam2.h"
#include "imgui_tools.h"
static bool showImgDirBrowser = false;
static std::filesystem::path imgDirPath;
static browser::Browser* imgDirPicker = nullptr;

  
class ImgDir
{
public:
	ImgDir() {};
	ImgDir(const std::filesystem::path&dir) 
	{
		imgDir_ = dir;
		updata();
	};
	void updata()
	{
		ImgPts.clear();
		if (!std::filesystem::exists(imgDir_))
		{
			return;
		}
		for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{ imgDir_ })
		{
			const auto& thisFilename = dir_entry.path();
			if (thisFilename.has_extension())
			{
				const auto& shortName = thisFilename.filename().stem().string();
				const auto& ext = thisFilename.extension().string();
				if (ext.compare(".json") == 0)
				{
					std::map<std::string, Eigen::Vector2d>& imgPts = ImgPts[shortName];
					Eigen::Vector2i imgSizeWH;
					labelme::readPtsFromLabelMeJson(thisFilename, imgPts, imgSizeWH);

				}
			}
		}
		return;
	}
	std::map<std::string, std::map<std::string, Eigen::Vector2d>>ImgPts;
	~ImgDir() {};
	std::vector<std::string>getTotal()
	{
		std::vector<std::string> ret;
		ret.reserve(ImgPts.size());
		for (const auto&d: ImgPts)
		{
			ret.emplace_back(d.first);
		}
		return ret;
	}
private:
	std::filesystem::path imgDir_;
};

static ProgressThread progress;
void endThread(ProgressThread&progress)
{
	progress.denominator.store(-1);
	progress.numerator.store(-1);
	progress.procRunning.store(0);
	return;
}
bool registFrame(bool* show_regist_window)
{
	ImGui::SetNextWindowSize(ImVec2(1280, 960));//ImVec2(x, y)
	ImGui::Begin("register", show_regist_window, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);
	//ImGui::Begin(u8"¶ÔÆë", show_far_align_window);
	ImGui::Text("register");
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
		if (std::filesystem::exists(imgDirPath))
		{
			progress.numerator.store(-1);
			progress.procRunning.fetch_add(1);
			progress.proc = new std::thread(
				[&]() {
					std::filesystem::path imgDirPath_ = imgDirPath;
					int registRet = register_incremental(imgDirPath_.string());
					endThread(progress);
				}
			);
			std::this_thread::sleep_for(std::chrono::milliseconds(100)); 
			imgDirPath = "";
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
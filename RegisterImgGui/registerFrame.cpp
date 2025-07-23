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
static bool showImgDirBrowser = false;
static std::filesystem::path imgPath;
static browser::Browser* filePicker = nullptr;

  
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
		for (auto const& dir_entry : std::filesystem::directory_iterator{ imgDir_ })
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


bool registFrame(bool* show_regist_window)
{
	ImGui::SetNextWindowSize(ImVec2(1280, 960));//ImVec2(x, y)
	ImGui::Begin("register", show_regist_window, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);
	//ImGui::Begin(u8"¶ÔÆë", show_far_align_window);
	ImGui::Text("register");

	if (ImGui::Button("pick image dir"))
	{
		if (filePicker ==nullptr)
		{
			imgPath = "";
			filePicker = new browser::Browser(browser::BrowserPick::PICK_DIR);
		}
	}
	if (filePicker != nullptr)
	{
		if (filePicker->pick(imgPath))
		{
			if (!imgPath.empty())
			{
				new ImgDir(imgPath);
			}
			delete filePicker;
			filePicker = nullptr;
		}
	}


	if (ImGui::Button("Close Me") && *show_regist_window) *show_regist_window = false;
	ImGui::End();
	return true;
}
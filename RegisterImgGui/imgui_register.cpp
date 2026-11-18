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
#include "colString.h"
#include "registerFrame.h"
#include "sam2.h"
#include "imgui_tools.h"
#include <Eigen/Geometry>
static bool showImgDirBrowser = false;
static std::filesystem::path imgDirPath;
static std::filesystem::path initFilePath;
static browser::Browser* imgDirPicker = nullptr;
static browser::Browser* initFilePicker = nullptr;

std::vector<std::string>listPicName(const std::filesystem::path& imgDir_)
{ 
	std::vector<std::string> ret;
	ret.reserve(256);
	for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{ imgDir_ })
	{
		const auto& thisFilename = dir_entry.path();
		auto parentName = thisFilename.parent_path().filename().string();
		if (parentName.compare("result") == 0)
		{
			continue;
		}
		if (thisFilename.has_extension())
		{
			const auto& shortName = thisFilename.filename().stem().string(); 
			if (thisFilename.has_extension())
			{
				const auto& shortName = thisFilename.filename().stem().string();
				const auto& ext = thisFilename.extension().string();
				if (ext.compare(".json") == 0)
				{
					std::map<std::string, Eigen::Vector2d> cornerInfo;
					Eigen::Vector2i imgSizeWH;
					std::string imgPath;
					bool readret = labelme::readPtsFromLabelMeJson(thisFilename, cornerInfo, imgSizeWH, &imgPath);
					if (readret && std::filesystem::exists(thisFilename.parent_path() / imgPath))
					{
						ret.emplace_back(shortName);
					}
				}
			}
		}
	}
	return ret;
}
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
void clearScreen() {
#ifdef _WIN32
	std::system("cls");
#else
	std::system("clear");
#endif
}
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
	static bool* imgpicks = nullptr;
	static std::vector<std::string>picNames;
	static std::vector<std::string>initView;
	static std::vector<std::string>discardView;
	const int QT_DATA_LENGTH_MAX = 512;
	static char qt1Data[QT_DATA_LENGTH_MAX] = "0\0";
	static char qt2Data[QT_DATA_LENGTH_MAX] = "0\0";
	static float qtRatio = 0.5;
	while (isProcRunning == 0)
	{
		if (ImGui::Button("pick image dir"))
		{
			picNames.clear();
			if (imgpicks != nullptr)
			{
				delete[]imgpicks;
				imgpicks = nullptr;
			}
			imgDirPath = "";
			if (imgDirPicker == nullptr)
			{
				imgDirPicker = new browser::Browser(browser::BrowserPick::PICK_DIR);
			}
		}
		if (ImGui::Button("pick initFile"))
		{  
			initFilePath = "";
			if (initFilePicker == nullptr)
			{
				initFilePicker = new browser::Browser(browser::BrowserPick::PICK_FILE);
			}
		}
		{
			ImGui::InputTextMultiline("qt1", qt1Data, QT_DATA_LENGTH_MAX, ImVec2(180, ImGui::GetTextLineHeight() * 7), ImGuiInputTextFlags_AllowTabInput| ImGuiInputTextFlags_EnterReturnsTrue| ImGuiInputTextFlags_CharsNoBlank);
			ImGui::SameLine();
			ImGui::InputTextMultiline("qt2", qt2Data, QT_DATA_LENGTH_MAX, ImVec2(180, ImGui::GetTextLineHeight() * 7), ImGuiInputTextFlags_AllowTabInput | ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CharsNoBlank);
			ImGui::PushItemWidth(300);
			ImGui::SliderFloat("qtRatio", &qtRatio, 0.01f, 0.99f, "%.1f", ImGuiSliderFlags_Logarithmic);
			ImGui::PopItemWidth();
			ImGui::SameLine();
			if (ImGui::Button("R"))
			{
				for (int i = 0; i < QT_DATA_LENGTH_MAX; i++)
				{
					if ((qt1Data[i] < '0' || qt1Data[i] > '9') && qt1Data[i] != '.' && qt1Data[i] != '\r' && qt1Data[i] != '\n')
					{
						qt1Data[i] = ' ';
					}
					if ((qt2Data[i] < '0' || qt2Data[i] > '9') && qt2Data[i] != '.' && qt2Data[i] != '\r' && qt2Data[i] != '\n')
					{
						qt2Data[i] = ' ';
					}
					qt1Data[QT_DATA_LENGTH_MAX - 1] = '\0';
					qt2Data[QT_DATA_LENGTH_MAX - 1] = '\0';
				}
				std::vector<std::string>segs1 = splitString(std::string(qt1Data), " \n\r", true);
				std::vector<std::string>segs2 = splitString(std::string(qt2Data), " \n\r", true);
				if (segs1.size()>=7 && segs2.size() >= 7)
				{
					Eigen::Quaterniond q1, q2;
					Eigen::Vector3d t1, t2;
					try
					{
						q1.w() = std::stod(segs1[0]);
						q1.x() = std::stod(segs1[1]);
						q1.y() = std::stod(segs1[2]);
						q1.z() = std::stod(segs1[3]);
						t1.x() = std::stod(segs1[4]);
						t1.y() = std::stod(segs1[5]);
						t1.z() = std::stod(segs1[6]);
						q2.w() = std::stod(segs2[0]);
						q2.x() = std::stod(segs2[1]);
						q2.y() = std::stod(segs2[2]);
						q2.z() = std::stod(segs2[3]);
						t2.x() = std::stod(segs2[4]);
						t2.y() = std::stod(segs2[5]);
						t2.z() = std::stod(segs2[6]);
						LOG_OUT << q1;
						LOG_OUT << q2;
						LOG_OUT << qtRatio;
						auto interpQ = q1.slerp(qtRatio, q2);
						auto interpT = t1 * (1 - qtRatio) + t2 * qtRatio;
						LOG_OUT << interpQ.w() << "," << interpQ.x() << "," << interpQ.y() << "," << interpQ.z() << ","
							<< interpT.x() << "," << interpT.y() << "," << interpT.z();
					}
					catch (...)
					{

					}
				}
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
		if (initFilePicker != nullptr && initFilePath.string().length() == 0)
		{
			if (initFilePicker->pick(initFilePath))
			{
				LOG_OUT << initFilePath;
				delete initFilePicker;
				initFilePicker = nullptr;
			}
		}
		if (std::filesystem::exists(imgDirPath) && picNames.size() == 0 && imgpicks == nullptr)
		{
			picNames = listPicName(imgDirPath);
			for (auto&d: picNames)
			{
#define NAME_TAIL_SPACE_CNT 4
				d += std::string(NAME_TAIL_SPACE_CNT, ' ');
			}
			imgpicks = new bool[picNames.size()];
			for (size_t i = 0; i < picNames.size(); i++)
			{
				imgpicks[i] = false;
			}
		}
		if (std::filesystem::exists(imgDirPath)  )
		{
			if (picNames.size()>3)
			{
				for (int i = 0; i < picNames.size(); i++)
				{
					if (i%9==0)
					{
					}
					else
					{
						ImGui::SameLine();
					}
					bool picNamesButton = ImGui::Button(picNames[i].c_str());
					if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
						if (picNames[i].back() == 'x')
						{
							picNames[i].back() = ' ';
						}
						else
						{
							picNames[i].back() = 'x';
						}
					}
					if (picNamesButton)
					{
						if (picNames[i].back() == 'O')
						{
							picNames[i].back() = ' ';
						}
						else
						{
							initView.clear();
							initView.reserve(4);
							for (size_t j = 0; j < picNames.size(); j++)
							{
								if (picNames[j].back() == 'O')
								{
									initView.emplace_back(picNames[j]);
								}
							}
							if (initView.size()<3)
							{
								picNames[i].back() = 'O';
							}
						}						
					} 
				}
				ImGui::NewLine();
			}
			static bool optimCameraIntr = false;
			static bool saveResult = false;
			ImGui::Checkbox("optimCameraIntr", &optimCameraIntr);
			ImGui::Checkbox("saveResult", &saveResult);
			if (ImGui::Button("figure!"))
			{
				initView.clear();
				discardView.clear();
				initView.reserve(4);
				discardView.reserve(64);
				for (size_t j = 0; j < picNames.size(); j++)
				{
					if (picNames[j].back() == 'O')
					{
						int len = picNames[j].length();
						initView.emplace_back(picNames[j].substr(0, len - NAME_TAIL_SPACE_CNT));
					}if (picNames[j].back() == 'x')
					{
						int len = picNames[j].length();
						discardView.emplace_back(picNames[j].substr(0, len - NAME_TAIL_SPACE_CNT));
					}
				}
				if (initView.size() == 3)
				{
					progress.numerator.store(-1);
					progress.procRunning.fetch_add(1);
					progress.proc = new std::thread(
						[&]() {
							//clearScreen();
							std::filesystem::path imgDirPath_ = imgDirPath;
							int registRet = register_incremental_base_hint(imgDirPath_.string(), initView, discardView, optimCameraIntr, saveResult);
							endThread(progress);
						}
					);
				}
				else
				{
					LOG_WARN_OUT << "pick initialed 3 views!";
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
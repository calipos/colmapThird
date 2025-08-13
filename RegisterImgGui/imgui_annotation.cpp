#include <numeric>
#include <vector>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <map>
#include "imgui.h" 
#include "browser.h"
#include "imgui_tools.h"
#include "log.h"
#include "pips2.h"
#include "imgui_annotation.h"
static bool showImgDirBrowser = false;
static std::filesystem::path imgDirPath;
static std::filesystem::path modelDirPath;
static browser::Browser* imgDirPicker = nullptr;
static browser::Browser* modelDirPicker = nullptr;
static ProgressThread progress;
namespace label
{
	class ImageLabel
	{
	private:
		int currentImgHeight;
		int currentImgWidth;
		static ImageLabel* instance;
		static ImVec2 draw_pos;
		static ImVec2 canvas;//(w,h)
		static ImVec2 zoom_start;
		static ImVec2 zoom_end;
		static GLuint image_texture;
		ImageLabel() {};
		ImageLabel(const ImageLabel&) = delete;
		ImageLabel& operator=(const ImageLabel&) = delete;
	public:
		bool hasImageContext{ false };
		static ImageLabel* getImageLabel(const ImVec2& draw_pos_, const int& canvasHeight_, const int& canvasWidth_)
		{
			if (ImageLabel::instance == nullptr)
			{
				ImageLabel::instance = new ImageLabel();
			}
			else
			{
				return ImageLabel::instance;
			}
			draw_pos = draw_pos_;
			canvas = ImVec2(canvasWidth_, canvasHeight_);
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
			}
			return true;
		}
		bool draw()
		{
			if (hasImageContext)
			{
				ImGui::SetCursorPos(draw_pos);
				ImGui::Image((ImTextureID)(intptr_t)image_texture, ImageLabel::canvas, ImageLabel::zoom_start, ImageLabel::zoom_end, ImVec4(1, 1, 1, 1), ImVec4(.5, .5, .5, .5));
			}
			return true;
		}
	};
	ImageLabel* ImageLabel::instance = nullptr;
	ImVec2 ImageLabel::draw_pos = ImVec2();
	ImVec2 ImageLabel::canvas = ImVec2();
	ImVec2 ImageLabel::zoom_start = ImVec2();
	ImVec2 ImageLabel::zoom_end = ImVec2();
	GLuint ImageLabel::image_texture = 0;
}
class AnnotationManger
{
public:
	AnnotationManger() = default;
	AnnotationManger(const  std::filesystem::path& dirPath, const  std::filesystem::path& modelDirPath)
	{

		std::filesystem::path encoderParamPath = modelDirPath / "pips2_base_ncnn.param";
		std::filesystem::path encoderBinPath = modelDirPath / "pips2_base_ncnn.bin";
		std::filesystem::path deltaParamPath = modelDirPath / "pips2_deltaBlock_ncnn.param";
		std::filesystem::path deltaBinPath = modelDirPath / "pips2_deltaBlock_ncnn.bin";
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
		if (!std::filesystem::exists(deltaParamPath))
		{
			progress.procRunning.store(0);
			LOG_ERR_OUT << "not found : " << deltaParamPath;
			return;
		}
		if (!std::filesystem::exists(deltaBinPath))
		{
			progress.procRunning.store(0);
			LOG_ERR_OUT << "not found : " << deltaBinPath;
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
		if (imgName.size() < 3)
		{
			progress.procRunning.store(0);
			LOG_ERR_OUT << "imgName.size()<3";
			return;
		}
		progress.denominator.store(imgName.size());
		progress.numerator.store(-1);
		progress.procRunning.fetch_add(1);
		pips2Ins = new pips2::Pips2(encoderParamPath, encoderBinPath, deltaParamPath, deltaBinPath);
		pips2Ins->inputImage(imgPaths); 
		progress.procRunning.store(0);
	}
	~AnnotationManger()
	{}
	pips2::Pips2* pips2Ins{ nullptr };
	std::vector<std::filesystem::path>imgPaths;
	std::vector<std::string>imgName;
	std::vector<cv::Mat> imgDat;
	std::filesystem::path imgDirPath_;
	std::filesystem::path modelDirPath_;
	static GLuint image_texture;
	static int viewWindowHeight;
	static int viewWindowWidth;
	int imgPickIdx{ 0 };
};
static AnnotationManger* annotationManger = nullptr;
GLuint AnnotationManger::image_texture = 0;
int AnnotationManger::viewWindowHeight = 720;
int AnnotationManger::viewWindowWidth = 960;
bool annotationFrame(bool* show_regist_window)
{
	ImGui::SetNextWindowSize(ImVec2(1280, 960));//ImVec2(x, y)
	ImGui::Begin("annotation", show_regist_window, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);
	ImGui::Text("annotation");

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


		if (annotationManger == nullptr && imgDirPath.string().length() > 0 && modelDirPath.string().length() > 0)
		{
			progress.numerator.store(-1);
			progress.procRunning.fetch_add(1);
			progress.proc = new std::thread(
				[&]() {
					annotationManger = new AnnotationManger(imgDirPath, modelDirPath);
				}
			);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		if (annotationManger != nullptr && annotationManger->imgName.size() < 3)
		{
			delete annotationManger;
			annotationManger = nullptr;
			imgDirPath = "";
			modelDirPath = "";
		}
		{
			auto currentPos = ImGui::GetCursorPos();
			label::ImageLabel* labelControlPtr = label::ImageLabel::getImageLabel(currentPos,720,960);
			if (!labelControlPtr->hasImageContext)
			{
				cv::Mat asd = cv::imread("D:/repo/colmapThird/a.bmp");
				labelControlPtr->feedImg(asd);
			}
			labelControlPtr->draw();
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
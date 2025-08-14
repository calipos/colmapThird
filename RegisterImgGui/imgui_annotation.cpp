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
	struct ControlLogic
	{
		std::vector<std::string>picShortName;
		std::vector<std::pair<cv::Point2f, std::string>>controlPtsAndTag;
		ImVec2 tempPt{-1,-1};
	};
	class ImageLabel
	{
	private:
		int currentImgHeight;
		int currentImgWidth;
		static ImageLabel* instance;
		static ImVec2 zoom_start;
		static ImVec2 zoom_end;
		static GLuint image_texture;
		static int canvasMaxSide;
		ImageLabel() {};
		ImageLabel(const ImageLabel&) = delete;
		ImageLabel& operator=(const ImageLabel&) = delete;
	public:
		ControlLogic ptsData;
		static ImVec2 draw_pos;
		static ImVec2 canvas;//(w,h)
		static ImVec2 canvasInv;//(1/w,1/h)
		static float resizeFactor;
		bool hasImageContext{ false };
		static ImVec2 imgPt2GuiPt(const ImVec2& imgPt, const int& imgHeight, const int& imgWidth, const ImVec2& canvas_size, const ImVec2& zoom_start, const ImVec2& zoom_end, const ImVec2& canvas_location)
		{
			float x_inRatio = (imgPt.x / imgWidth - zoom_start.x) / (zoom_end.x - zoom_start.x);
			float y_inRatio = (imgPt.y / imgHeight - zoom_start.y) / (zoom_end.y - zoom_start.y);
			float x = x_inRatio * canvas_size.x + canvas_location.x + ImGui::GetWindowPos().x;
			float y = y_inRatio * canvas_size.y + canvas_location.y + ImGui::GetWindowPos().y;
			return ImVec2(x, y);
		}
		static ImageLabel* getImageLabel(const ImVec2& draw_pos_, const int& canvasMaxSide_)
		{
			if (ImageLabel::instance == nullptr)
			{
				ImageLabel::instance = new ImageLabel();
			}
			else
			{
				return ImageLabel::instance;
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
				float a = 1.f * currentImgWidth / ImageLabel::canvasMaxSide;
				float b = 1.f * currentImgHeight / ImageLabel::canvasMaxSide;
				if (a < 1.f && b < 1.f)
				{
					ImageLabel::canvas = ImVec2(currentImgWidth, currentImgHeight);
				}
				else if(a>b)
				{
					ImageLabel::canvas = ImVec2(currentImgWidth / a, currentImgHeight / a);
				}
				else
				{
					ImageLabel::canvas = ImVec2(currentImgWidth / b, currentImgHeight / b);
				}
				ImageLabel::canvasInv.x = 1. / ImageLabel::canvas.x;
				ImageLabel::canvasInv.y = 1. / ImageLabel::canvas.y;
			}
			return true;
		}
		bool draw()
		{
			if (hasImageContext)
			{
				ImGui::SetCursorPos(draw_pos);
				ImGui::Image((ImTextureID)(intptr_t)image_texture, ImageLabel::canvas, ImageLabel::zoom_start, ImageLabel::zoom_end, ImVec4(1, 1, 1, 1), ImVec4(.5, .5, .5, .5));
				if (ptsData.tempPt.x>0)
				{
					ImVec2 guiPt = imgPt2GuiPt(ptsData.tempPt, currentImgHeight, currentImgWidth, canvas, zoom_start, zoom_end, draw_pos);
					ImGui::GetForegroundDrawList()->AddCircleFilled(guiPt, 4.0f, 0xC80688FB);
				}
			}
			return true;
		}
		ImVec2 control(const ImVec2&mousePosInImage,const float wheel, const bool&mouseLeftDown)
		{
			if (abs(wheel) > 0.01)
			{
				if (mousePosInImage.x >= 0 && mousePosInImage.y >= 0 && mousePosInImage.x < canvas.x && mousePosInImage.y < canvas.y)
				{
					float x_inRatio = mousePosInImage.x * ImageLabel::canvasInv.x * (ImageLabel::zoom_end.x - ImageLabel::zoom_start.x) + ImageLabel::zoom_start.x;
					float y_inRatio = mousePosInImage.y * ImageLabel::canvasInv.y * (ImageLabel::zoom_end.y - ImageLabel::zoom_start.y) + ImageLabel::zoom_start.y;
					float x_inPic = x_inRatio * currentImgWidth;
					float y_inPic = y_inRatio * currentImgHeight;

					{
						//zoom out: resizeFactor=1 (0,0)->(1,1);
						//zoom in:  resizeFactor=4 (0.25,0.25)->(.75,.75);
						resizeFactor += (0.05 * wheel);//    
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
						ImageLabel::zoom_start.x = x_inRatio - regionRadius;
						ImageLabel::zoom_start.y = y_inRatio - regionRadius;
						ImageLabel::zoom_end.x = x_inRatio + regionRadius;
						ImageLabel::zoom_end.y = y_inRatio + regionRadius;
						if (ImageLabel::zoom_start.x < 0)
						{
							ImageLabel::zoom_start.x = 0;
							ImageLabel::zoom_end.x = (1 - regionDiameter);
						}
						if (ImageLabel::zoom_start.y < 0)
						{
							ImageLabel::zoom_start.y = 0;
							ImageLabel::zoom_end.y = (1 - regionDiameter);
						}
						if (ImageLabel::zoom_end.x > 1)
						{
							ImageLabel::zoom_start.x = regionDiameter;
							ImageLabel::zoom_end.x = 1;
						}
						if (ImageLabel::zoom_end.y > 1)
						{
							ImageLabel::zoom_start.y = regionDiameter;
							ImageLabel::zoom_end.y = 1;
						}

					}
				}
			}
			if (mouseLeftDown)
			{
				if (mousePosInImage.x >= 0 && mousePosInImage.y >= 0 && mousePosInImage.x < canvas.x && mousePosInImage.y < canvas.y)
				{
					float x_inRatio = mousePosInImage.x * ImageLabel::canvasInv.x * (ImageLabel::zoom_end.x - ImageLabel::zoom_start.x) + ImageLabel::zoom_start.x;
					float y_inRatio = mousePosInImage.y * ImageLabel::canvasInv.y * (ImageLabel::zoom_end.y - ImageLabel::zoom_start.y) + ImageLabel::zoom_start.y;
					float x_inPic = x_inRatio * currentImgWidth;
					float y_inPic = y_inRatio * currentImgHeight;
					return ImVec2(x_inPic, y_inPic);
				}
			}
			return ImVec2(-1,-1);
		}
	};
	ImageLabel* ImageLabel::instance = nullptr;
	ImVec2 ImageLabel::draw_pos = ImVec2();
	ImVec2 ImageLabel::canvas = ImVec2();
	ImVec2 ImageLabel::canvasInv = ImVec2();
	ImVec2 ImageLabel::zoom_start = ImVec2();
	ImVec2 ImageLabel::zoom_end = ImVec2();
	GLuint ImageLabel::image_texture = 0; 
	int ImageLabel::canvasMaxSide = 0;
	float ImageLabel::resizeFactor = 1;
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
			label::ImageLabel* labelControlPtr = label::ImageLabel::getImageLabel(currentPos,720);
			if (!labelControlPtr->hasImageContext)
			{
				cv::Mat asd = cv::imread("../a.bmp");
				labelControlPtr->feedImg(asd);
				bool&mouseLeftDown = ImGui::GetIO().MouseReleased[0];
				mouseLeftDown = false;
			}
			labelControlPtr->draw();
			if (labelControlPtr->hasImageContext)
			{
				ImVec2 mousePosInImage;
				mousePosInImage.x = (ImGui::GetIO().MousePos.x - ImGui::GetWindowPos().x - labelControlPtr->draw_pos.x);
				mousePosInImage.y = (ImGui::GetIO().MousePos.y - ImGui::GetWindowPos().y - labelControlPtr->draw_pos.y);
				float wheel = ImGui::GetIO().MouseWheel;
				//ImGui::Text("Mouse pos: (%g, %g)", mousePosInImage.x, mousePosInImage.y);
				//ImGui::Text("Mouse wheel: %.1f", ImGui::GetIO().MouseWheel);
				bool mouseLeftDown = ImGui::GetIO().MouseReleased[0];
				ImVec2 maybeClik = labelControlPtr->control(mousePosInImage, wheel, mouseLeftDown);
				if (maybeClik.x>0)
				{
					labelControlPtr->ptsData.tempPt = maybeClik;
				}
				

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
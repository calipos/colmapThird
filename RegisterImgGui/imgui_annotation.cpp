#include <unordered_set>
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
		std::vector< std::map<std::string, ImVec2>>controlPtsAndTag;
		ImVec2 tempPt{ -1,-1 };  // for first track ,the temp point wil be showed on canvas
		ImVec2 tempPt2{ -1,-1 }; // for re-track only,  the temp2 point  = the re-new controlPtsAndTag

		std::map<std::string, ImU32> colors;
		std::vector<std::string>tagsListName;
		std::vector<std::string>tagsName;
		std::map<std::string, std::vector<std::string>> hasTagFlags;
		static int tagPickIdx;
		bool pickedChanged;
		void updata()
		{
			std::string maybePickedTag = (tagPickIdx >= 0) ? tagsName[tagPickIdx] : "";
			std::string maybePickedTag2 = (tagPickIdx >= 0) ? tagsListName[tagPickIdx] : "";
			std::unordered_set<std::string>newTagsName;
			for (const auto&d: controlPtsAndTag)
			{
				for (const auto&d2:d)
				{
					newTagsName.insert(d2.first);
				}
			}
			std::vector<std::string>::iterator iter;
			for (iter = tagsName.begin(); iter != tagsName.end(); iter++)
			{
				std::string tagStr = *iter;
				if (newTagsName.find(tagStr)== newTagsName.end())
				{
					tagsListName.erase(tagsListName.begin() + (iter - tagsName.begin()));
					iter = tagsName.erase(iter);
				}
			}
			if (maybePickedTag.length()>0)
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
				tagAndShortName.insert(tagAndShortName.end(), picShortName.begin(), picShortName.end());
				for (int j = 0; j < picShortName.size(); j++)
				{
					if (controlPtsAndTag[j].count(d.first) > 0)
					{
						hasTagFlags[d.first][j][0] = '*';
					}
				}
			}
		}
		bool save(const std::vector<std::filesystem::path>& picPath)const
		{
			if (true)
			{

			}
		}
	};
	int ControlLogic::tagPickIdx = -1;
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
		static const int tagLengthMax{24};
		static char tarStr[tagLengthMax];
		ControlLogic ptsData;
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
		static ImageLabel* getImageLabel(const ImVec2& draw_pos_, const int& canvasMaxSide_,const std::vector<std::string>& picShortName)
		{
			if (ImageLabel::instance == nullptr)
			{
				ImageLabel::instance = new ImageLabel();
				instance->ptsData.picShortName.insert(instance->ptsData.picShortName.end(), picShortName.begin(), picShortName.end());
				instance->ptsData.controlPtsAndTag.resize(picShortName.size());
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
		bool draw(const int& imgPickIdx, const bool& alignPt = false)
		{
			if (hasImageContext)
			{
				ImGui::SetCursorPos(draw_pos);
				std::string pickedTag;
				if (!showLabeled && ptsData.tagPickIdx >= 0)
				{
					pickedTag = ptsData.tagsName[ptsData.tagPickIdx];
					if (alignPt)
					{
						const std::map<std::string, ImVec2>& tags = ptsData.controlPtsAndTag[imgPickIdx];
						if (tags.count(pickedTag) > 0)
						{
							const auto p = tags.at(pickedTag);
							float x_ratio = p.x / currentImgWidth;
							float y_ratio = p.y / currentImgHeight;
							float ratioCnterX = (ImageLabel::zoom_end.x + ImageLabel::zoom_start.x) * .5;
							float ratioCnterY = (ImageLabel::zoom_end.y + ImageLabel::zoom_start.y) * .5;
							float shiftX = x_ratio - ratioCnterX;
							float shiftY = y_ratio - ratioCnterY;
							ImageLabel::zoom_start.x += shiftX;
							ImageLabel::zoom_end.x += shiftX;
							ImageLabel::zoom_start.y += shiftY;
							ImageLabel::zoom_end.y += shiftY;
							if (ImageLabel::zoom_start.x < 0)
							{
								ImageLabel::zoom_end.x -= ImageLabel::zoom_start.x;
								ImageLabel::zoom_start.x = 0;
							}
							if (ImageLabel::zoom_start.y < 0)
							{
								ImageLabel::zoom_end.y -= ImageLabel::zoom_start.y;
								ImageLabel::zoom_start.y = 0;
							}
							if (ImageLabel::zoom_end.x > 1)
							{
								ImageLabel::zoom_start.x += (1 - ImageLabel::zoom_end.x);
								ImageLabel::zoom_end.x = 1;
							}
							if (ImageLabel::zoom_end.y > 1)
							{
								ImageLabel::zoom_start.y += (1 - ImageLabel::zoom_end.y);
								ImageLabel::zoom_end.y = 1;
							}
						}
					}
				}
				ImGui::Image((ImTextureID)(intptr_t)image_texture, ImageLabel::canvas, ImageLabel::zoom_start, ImageLabel::zoom_end, ImVec4(1, 1, 1, 1), ImVec4(.5, .5, .5, .5));
				if (ptsData.tempPt.x>0)
				{
					ImVec2 guiPt = imgPt2GuiPt(ptsData.tempPt, currentImgHeight, currentImgWidth, canvas, zoom_start, zoom_end, draw_pos);
					ImGui::GetForegroundDrawList()->AddCircleFilled(guiPt, 4.0f, 0xC80688FB);
				}
				if (showLabeled)
				{
					const std::map<std::string, ImVec2>& tags = ptsData.controlPtsAndTag[imgPickIdx];
					if (tags.size() > 0)
					{
						for (const auto& d : tags)
						{
							ImVec2 guiPt = imgPt2GuiPt(d.second, currentImgHeight, currentImgWidth, canvas, zoom_start, zoom_end, draw_pos);
							ImGui::GetForegroundDrawList()->AddCircleFilled(guiPt, 4.0f, ptsData.colors.at(d.first));
						}
					}
				}

				if (!showLabeled && ptsData.tagPickIdx>=0)
				{
					const std::map<std::string, ImVec2>& tags = ptsData.controlPtsAndTag[imgPickIdx];
					if (tags.size() > 0)
					{
						if (tags.count(pickedTag)>0)
						{
							ImVec2 guiPt = imgPt2GuiPt(tags.at(pickedTag), currentImgHeight, currentImgWidth, canvas, zoom_start, zoom_end, draw_pos);
							ImGui::GetForegroundDrawList()->AddCircleFilled(guiPt, 4.0f, ptsData.colors.at(pickedTag));
						}
					}
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
	char ImageLabel::tarStr[tagLengthMax] = "\0";
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
	std::vector<std::filesystem::path>tempTrackImgPaths;//for other thread do track,keep the temporary variable
	std::vector<cv::Point2f> tempHint;//for other thread do track,keep the temporary variable
	std::vector<std::filesystem::path>imgPaths;
	std::vector<std::string>imgName;
	std::filesystem::path imgDirPath_;
	std::filesystem::path modelDirPath_;
	static GLuint image_texture;
	static int viewWindowHeight;
	static int viewWindowWidth;
	static int imgPickIdx;
	static std::vector<std::vector<cv::Point2f>>trajs;
	static bool alignTrack;
};
static AnnotationManger* annotationManger = nullptr;
GLuint AnnotationManger::image_texture = 0;
int AnnotationManger::viewWindowHeight = 720;
int AnnotationManger::viewWindowWidth = 960;
int AnnotationManger::imgPickIdx = -1;
std::vector<std::vector<cv::Point2f>>AnnotationManger::trajs;
bool AnnotationManger::alignTrack =false;
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
	ImVec2 deployButtom(-1,-1);
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
		if(annotationManger != nullptr && annotationManger->imgName.size()>0)
		{
			auto imgListLocation = ImGui::GetCursorPos();
			bool pickedChanged = false;
			ImVec2 listPicSize(100,500);
			listComponent("picPick", listPicSize,annotationManger->imgName, AnnotationManger::imgPickIdx, pickedChanged);
			//ImGui::Text("Mouse pos: (%d, %d)", AnnotationManger::imgPickIdx, pickedChanged);
			ImVec2 canvas_location = imgListLocation;
			canvas_location.x += listPicSize.x;
			label::ImageLabel* labelControlPtr = label::ImageLabel::getImageLabel(canvas_location,720, annotationManger->imgName);
			if (ImGui::IsKeyPressed(ImGuiKey_Z))
			{
				pickedChanged = true;
				annotationManger->imgName[AnnotationManger::imgPickIdx][0] = ' ';
				AnnotationManger::imgPickIdx--;
				if (AnnotationManger::imgPickIdx < 0)
				{
					AnnotationManger::imgPickIdx = 0;
					pickedChanged = false;
				}
				annotationManger->imgName[AnnotationManger::imgPickIdx][0] = '-';
				//continue;
			}
			if (ImGui::IsKeyPressed(ImGuiKey_X))
			{
				pickedChanged = true;
				annotationManger->imgName[AnnotationManger::imgPickIdx][0] = ' ';
				AnnotationManger::imgPickIdx++;
				if (AnnotationManger::imgPickIdx >= annotationManger->imgPaths.size())
				{
					AnnotationManger::imgPickIdx = annotationManger->imgPaths.size() - 1;
					pickedChanged = false;
				}
				annotationManger->imgName[AnnotationManger::imgPickIdx][0] = '-';
				//continue;
			}
			if (AnnotationManger::imgPickIdx >=0 && pickedChanged)
			{
				annotationManger->tempTrackImgPaths.clear();
				cv::Mat asd = cv::imread(annotationManger->imgPaths[AnnotationManger::imgPickIdx].string());
				labelControlPtr->feedImg(asd);
				bool&mouseLeftDown = ImGui::GetIO().MouseReleased[0];
				mouseLeftDown = false;
			}
			labelControlPtr->draw(AnnotationManger::imgPickIdx, AnnotationManger::alignTrack);
			if (labelControlPtr->hasImageContext)
			{
				labelControlPtr->ptsData.pickedChanged = false;
				ImVec2 tagListlocation = label::ImageLabel::draw_pos;
				tagListlocation.x += labelControlPtr->canvas.x;
				tagListlocation.x += ImGui::GetWindowPos().x;
				tagListlocation.y += ImGui::GetWindowPos().y;
				ImGui::SetCursorScreenPos(tagListlocation);
				listComponent("tag", listPicSize, labelControlPtr->ptsData.tagsListName, labelControlPtr->ptsData.tagPickIdx, labelControlPtr->ptsData.pickedChanged);
			}
			if (labelControlPtr->ptsData.pickedChanged && labelControlPtr->ptsData.tagPickIdx>=0)
			{
				labelControlPtr->ptsData.tempPt.x = -1;
				labelControlPtr->ptsData.tempPt.y = -1;
				labelControlPtr->ptsData.tempPt2.x = -1;
				labelControlPtr->ptsData.tempPt2.y = -1;
				annotationManger->imgName.clear();
				const auto& newPicName = labelControlPtr->ptsData.hasTagFlags[labelControlPtr->ptsData.tagsName[labelControlPtr->ptsData.tagPickIdx]];
				annotationManger->imgName.insert(annotationManger->imgName.end(), newPicName.begin(), newPicName.end());
			}
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
				if (labelControlPtr->showLabeled == true)
				{
					if (maybeClik.x > 0 && label::ImageLabel::tarStr[0] == '\0')
					{
						labelControlPtr->ptsData.tempPt = maybeClik;
					}
				}
				else
				{
					if (maybeClik.x > 0 && labelControlPtr->ptsData.tagPickIdx >= 0)
					{
						const std::string& pickedTag = labelControlPtr->ptsData.tagsName[labelControlPtr->ptsData.tagPickIdx];
						std::map<std::string, ImVec2>& tags = labelControlPtr->ptsData.controlPtsAndTag[AnnotationManger::imgPickIdx];
						tags[pickedTag] = maybeClik;
						labelControlPtr->ptsData.tempPt2 = maybeClik; // may re-track base the fixed point.
						labelControlPtr->ptsData.hasTagFlags[pickedTag][AnnotationManger::imgPickIdx][1] = '*';

						annotationManger->imgName.clear();
						const auto& newPicName = labelControlPtr->ptsData.hasTagFlags[pickedTag];
						annotationManger->imgName.insert(annotationManger->imgName.end(), newPicName.begin(), newPicName.end());
					}
				}				
			}
			deployButtom = ImVec2(imgListLocation.x + ImGui::GetWindowPos().x, std::max(listPicSize.y, labelControlPtr->canvas.y) + imgListLocation.y+ ImGui::GetWindowPos().y);
			if (AnnotationManger::imgPickIdx >= 0)
			{
				ImGui::SetCursorScreenPos(deployButtom);
				ImGui::InputTextMultiline("<-tag", label::ImageLabel::tarStr, label::ImageLabel::tagLengthMax, ImVec2(200, 20), ImGuiInputTextFlags_CharsHexadecimal+ ImGuiInputTextFlags_CharsNoBlank);
				std::string tagStr = std::string(label::ImageLabel::tarStr);
				if ((labelControlPtr->ptsData.tempPt.x < 0&& labelControlPtr->ptsData.tempPt2.x < 0) || tagStr.length() == 0)
				{
					ImGui::BeginDisabled();
				}
				if (ImGui::Button("track"))
				{
					LOG_OUT << "tag = " << tagStr;
					LOG_OUT << "track from " << AnnotationManger::imgPickIdx;
					annotationManger->tempTrackImgPaths.clear();
					annotationManger->tempTrackImgPaths.insert(annotationManger->tempTrackImgPaths.end(), annotationManger->imgPaths.begin() + AnnotationManger::imgPickIdx, annotationManger->imgPaths.end());
					annotationManger->tempHint = std::vector<cv::Point2f>{ {0, 0} };
					if (labelControlPtr->ptsData.tempPt.x>=0)
					{
						annotationManger->tempHint[0].x = labelControlPtr->ptsData.tempPt.x;
						annotationManger->tempHint[0].y = labelControlPtr->ptsData.tempPt.y;
					}
					else if (true)
					{
						annotationManger->tempHint[0].x = labelControlPtr->ptsData.tempPt2.x;
						annotationManger->tempHint[0].y = labelControlPtr->ptsData.tempPt2.y;
					}
					progress.numerator.store(-1);
					progress.procRunning.fetch_add(1);
					progress.proc = new std::thread(
						[&]() {
							annotationManger->pips2Ins->inputImage(annotationManger->tempTrackImgPaths);
							annotationManger->pips2Ins->trackLimit(annotationManger->tempHint, annotationManger->trajs, 16, 6);
							progress.procRunning.store(0);
						}
					);
					std::this_thread::sleep_for(std::chrono::milliseconds(100));
				}
				if ((labelControlPtr->ptsData.tempPt.x < 0 && labelControlPtr->ptsData.tempPt2.x < 0) || tagStr.length() == 0)
				{
					ImGui::EndDisabled();
				}
				ImGui::SameLine();
				if (ImGui::Button("clearTag"))
				{
					labelControlPtr->ptsData.tempPt.x = -1;
					labelControlPtr->ptsData.tempPt.y = -1;
					labelControlPtr->ptsData.tempPt2.x = -1;
					labelControlPtr->ptsData.tempPt2.y = -1;
					label::ImageLabel::tarStr[0] = '\0';
				}
				ImGui::SameLine();
				if (ImGui::Checkbox("showAllTag", &labelControlPtr->showLabeled))
				{
					labelControlPtr->ptsData.tagPickIdx = -1;
					for (auto&d: labelControlPtr->ptsData.tagsListName)
					{
						d[0] = ' ';
					}
					for (auto&d: annotationManger->imgName)
					{
						d[0] = ' ';
					}
				}
				if (!labelControlPtr->showLabeled && labelControlPtr->ptsData.tagPickIdx<0)
				{
					labelControlPtr->showLabeled = true;
				}
				ImGui::SameLine();
				if (ImGui::Checkbox("alignTrack", &AnnotationManger::alignTrack))
				{

				}
				ImGui::SameLine();
				if (labelControlPtr->ptsData.tagPickIdx<0 || AnnotationManger::imgPickIdx<0)
				{
					ImGui::BeginDisabled();
				}
				if (ImGui::Button("~Tag"))
				{
					auto&thisPicTags = labelControlPtr->ptsData.controlPtsAndTag[AnnotationManger::imgPickIdx];
					const auto&tagName = labelControlPtr->ptsData.tagsName[labelControlPtr->ptsData.tagPickIdx];
					auto iter = thisPicTags.find(tagName);					
					if (iter != thisPicTags.end())
					{
						thisPicTags.erase(iter);
						labelControlPtr->ptsData.updata();
					}
				}
				ImGui::SameLine();
				if (ImGui::Button("~TagBehindAll"))
				{
					const auto& tagName = labelControlPtr->ptsData.tagsName[labelControlPtr->ptsData.tagPickIdx];
					for (int ip = AnnotationManger::imgPickIdx; ip < annotationManger->imgPaths.size(); ip++)
					{
						auto& thisPicTags = labelControlPtr->ptsData.controlPtsAndTag[ip];
						if (thisPicTags.size() > 0)
						{
							auto iter = thisPicTags.find(tagName);
							if (iter != thisPicTags.end())
							{
								thisPicTags.erase(iter);
							}
						}
					}
					labelControlPtr->ptsData.updata();
				}
				if (labelControlPtr->ptsData.tagPickIdx < 0 || AnnotationManger::imgPickIdx < 0)
				{
					ImGui::EndDisabled();
				}
				ImGui::SameLine();
				if (ImGui::Button("save"))
				{

				}
				deployButtom.y += 45;
			}
			if (AnnotationManger::trajs.size()>0)
			{
				std::string tagStr = std::string(label::ImageLabel::tarStr);
				for (int i = AnnotationManger::imgPickIdx; i < labelControlPtr->ptsData.picShortName.size(); i++)
				{
					const cv::Point2f pt = AnnotationManger::trajs[i - AnnotationManger::imgPickIdx][0];
					labelControlPtr->ptsData.controlPtsAndTag[i][tagStr].x = pt.x;
					labelControlPtr->ptsData.controlPtsAndTag[i][tagStr].y = pt.y;
				}
				if (labelControlPtr->ptsData.colors.count(tagStr)==0)
				{
					labelControlPtr->ptsData.colors[tagStr] = getImguiColor();
				}
				labelControlPtr->showLabeled = true;
				labelControlPtr->ptsData.tagPickIdx = -1;
				AnnotationManger::trajs.clear();
				labelControlPtr->ptsData.tempPt.x = -1;
				labelControlPtr->ptsData.tempPt.y = -1;
				labelControlPtr->ptsData.tempPt2.x = -1;
				labelControlPtr->ptsData.tempPt2.y = -1;
				label::ImageLabel::tarStr[0] = '\0';
				labelControlPtr->ptsData.tagsListName.clear();
				labelControlPtr->ptsData.tagsListName.reserve(labelControlPtr->ptsData.colors.size());
				labelControlPtr->ptsData.tagsName.clear();
				labelControlPtr->ptsData.tagsName.reserve(labelControlPtr->ptsData.colors.size());
				labelControlPtr->ptsData.hasTagFlags.clear();
				for (const auto&d: labelControlPtr->ptsData.colors)
				{
					labelControlPtr->ptsData.tagsListName.emplace_back("  "+d.first);
					labelControlPtr->ptsData.tagsName.emplace_back(d.first);
					auto&tagAndShortName = labelControlPtr->ptsData.hasTagFlags[d.first];
					tagAndShortName.insert(tagAndShortName.end(), labelControlPtr->ptsData.picShortName.begin(), labelControlPtr->ptsData.picShortName.end());
					for (int j = 0; j < labelControlPtr->ptsData.picShortName.size(); j++)
					{
						if (labelControlPtr->ptsData.controlPtsAndTag[j].count(d.first)>0)
						{
							labelControlPtr->ptsData.hasTagFlags[d.first][j][1]='*';
						}
					}
				} 
			}
			if (annotationManger->imgPaths.size()>0 && labelControlPtr->ptsData.tagPickIdx>=0)
			{
				labelControlPtr->showLabeled = false;
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
	if (ImGui::Button("Close Me") && *show_regist_window) *show_regist_window = false;
	ImGui::End();
	return true;
}
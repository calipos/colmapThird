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
enum class IterType
{};
class BfmIter
{
public:
	BfmIter() = default;
    BfmIter(const  std::filesystem::path& dirPath, const  std::filesystem::path& modelDirPath)
    {
        R << 1, 0, 0, 0, -1, 0, 0, 0, -1;
        t << 0, 0, 300; 

		imgPaths.clear();
		imgNameForlist.clear();
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
						std::string fullShortName = label::getfullShortName(thisFilename, imgDirPath_);
						std::string stem = "   " + fullShortName;
						imgNameForlist.emplace_back(stem);
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
    Eigen::Matrix3f R;
    Eigen::RowVector3f t;
	std::vector<std::filesystem::path>imgPaths;
	std::vector<std::string>imgNameForlist;
	std::filesystem::path imgDirPath_;
	std::filesystem::path modelDirPath_;
	static GLuint image_texture;
	static int viewWindowHeight;
	static int viewWindowWidth; 

};
static ProgressThread progress;
static bool showImgDirBrowser = false;
static std::filesystem::path imgDirPath;
static std::filesystem::path modelDirPath;
static browser::Browser* imgDirPicker = nullptr;
static browser::Browser* modelDirPicker = nullptr;
static BfmIter* annotationManger = nullptr;
GLuint BfmIter::image_texture = 0;
int BfmIter::viewWindowHeight = 720;
int BfmIter::viewWindowWidth = 960;
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
		if (annotationManger != nullptr && annotationManger->imgNameForlist.size() < 1)
		{
			delete annotationManger;
			annotationManger = nullptr;
			imgDirPath = "";
			modelDirPath = "";
		}
		if (annotationManger != nullptr && annotationManger->imgNameForlist.size() > 0)
		{
			auto imgListLocation = ImGui::GetCursorPos();
			bool pickedChanged = false;
			ImVec2 listPicSize(100, 500);
			listComponent("picPick", listPicSize, annotationManger->imgNameForlist, AnnotationManger::imgPickIdx, pickedChanged);
			//ImGui::Text("Mouse pos: (%d, %d)", AnnotationManger::imgPickIdx, pickedChanged);
			ImVec2 canvas_location = imgListLocation;
			canvas_location.x += listPicSize.x;
			label::ImageLabel* labelControlPtr = label::ImageLabel::getImageLabel(canvas_location, 720, annotationManger->imgNameForlist, annotationManger->imgPaths);
			if (AnnotationManger::imgPickIdx >= 0 && (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) || ImGui::IsKeyPressed(ImGuiKey_UpArrow)))
			{
				pickedChanged = true;
				annotationManger->imgNameForlist[AnnotationManger::imgPickIdx][0] = ' ';
				AnnotationManger::imgPickIdx--;
				if (AnnotationManger::imgPickIdx < 0)
				{
					AnnotationManger::imgPickIdx = 0;
					pickedChanged = false;
				}
				annotationManger->imgNameForlist[AnnotationManger::imgPickIdx][0] = '-';
				//continue;
			}
			if (AnnotationManger::imgPickIdx >= 0 && (ImGui::IsKeyPressed(ImGuiKey_RightArrow) || ImGui::IsKeyPressed(ImGuiKey_DownArrow)))
			{
				pickedChanged = true;
				annotationManger->imgNameForlist[AnnotationManger::imgPickIdx][0] = ' ';
				AnnotationManger::imgPickIdx++;
				if (AnnotationManger::imgPickIdx >= annotationManger->imgPaths.size())
				{
					AnnotationManger::imgPickIdx = annotationManger->imgPaths.size() - 1;
					pickedChanged = false;
				}
				annotationManger->imgNameForlist[AnnotationManger::imgPickIdx][0] = '-';
				//continue;
			}
			if (AnnotationManger::imgPickIdx >= 0 && pickedChanged)
			{
				annotationManger->tempTrackImgPaths.clear();
				cv::Mat asd = cv::imread(annotationManger->imgPaths[AnnotationManger::imgPickIdx].string());
				labelControlPtr->feedImg(asd);
				bool& mouseLeftDown = ImGui::GetIO().MouseReleased[0];
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
				ImVec2 listTagSize = listPicSize;
				listTagSize.x += 100;
				listComponent("tag", listTagSize, labelControlPtr->ptsData.tagsListName, labelControlPtr->ptsData.tagPickIdx, labelControlPtr->ptsData.pickedChanged);
			}
			if (labelControlPtr->ptsData.pickedChanged && labelControlPtr->ptsData.tagPickIdx >= 0)
			{
				labelControlPtr->ptsData.tempPt.x = -1;
				labelControlPtr->ptsData.tempPt.y = -1;
				labelControlPtr->ptsData.tempPt2.x = -1;
				labelControlPtr->ptsData.tempPt2.y = -1;
				annotationManger->imgNameForlist.clear();
				const auto& newPicNameForList = labelControlPtr->ptsData.hasTagFlags[labelControlPtr->ptsData.tagsName[labelControlPtr->ptsData.tagPickIdx]];
				annotationManger->imgNameForlist.insert(annotationManger->imgNameForlist.end(), newPicNameForList.begin(), newPicNameForList.end());
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

						annotationManger->imgNameForlist.clear();
						const auto& newPicNameForList = labelControlPtr->ptsData.hasTagFlags[pickedTag];
						annotationManger->imgNameForlist.insert(annotationManger->imgNameForlist.end(), newPicNameForList.begin(), newPicNameForList.end());
					}
				}
			}
			deployButtom = ImVec2(imgListLocation.x + ImGui::GetWindowPos().x, std::max(listPicSize.y, labelControlPtr->canvas.y) + imgListLocation.y + ImGui::GetWindowPos().y);
			if (AnnotationManger::imgPickIdx >= 0)
			{
				ImGui::SetCursorScreenPos(deployButtom);
				ImGui::InputTextMultiline("<-tag", label::ImageLabel::tarStr, label::ImageLabel::tagLengthMax, ImVec2(200, 20), ImGuiTreeNodeFlags_None + ImGuiInputTextFlags_CharsNoBlank);
				std::string tagStr = std::string(label::ImageLabel::tarStr);
				if ((labelControlPtr->ptsData.tempPt.x < 0 && labelControlPtr->ptsData.tempPt2.x < 0) || tagStr.length() == 0)
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
					if (labelControlPtr->ptsData.tempPt.x >= 0)
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
					for (auto& d : labelControlPtr->ptsData.tagsListName)
					{
						d[0] = ' ';
					}
					for (auto& d : annotationManger->imgNameForlist)
					{
						d[0] = ' ';
					}
				}
				if (!labelControlPtr->showLabeled && labelControlPtr->ptsData.tagPickIdx < 0)
				{
					labelControlPtr->showLabeled = true;
				}
				ImGui::SameLine();
				if (ImGui::Checkbox("alignTrack", &AnnotationManger::alignTrack))
				{

				}
				ImGui::SameLine();
				if (labelControlPtr->ptsData.tagPickIdx < 0 || AnnotationManger::imgPickIdx < 0)
				{
					ImGui::BeginDisabled();
				}
				if (ImGui::Button("~Tag") || ImGui::IsKeyPressed(ImGuiKey_Delete))
				{
					auto& thisPicTags = labelControlPtr->ptsData.controlPtsAndTag[AnnotationManger::imgPickIdx];
					const auto& tagName = labelControlPtr->ptsData.tagsName[labelControlPtr->ptsData.tagPickIdx];
					auto iter = thisPicTags.find(tagName);
					if (iter != thisPicTags.end())
					{
						thisPicTags.erase(iter);
						labelControlPtr->ptsData.updata();
						labelControlPtr->ptsData.hasTagFlags[tagName][AnnotationManger::imgPickIdx][1] = ' ';

						annotationManger->imgNameForlist.clear();
						const auto& newPicNameForList = labelControlPtr->ptsData.hasTagFlags[tagName];
						annotationManger->imgNameForlist.insert(annotationManger->imgNameForlist.end(), newPicNameForList.begin(), newPicNameForList.end());
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
								labelControlPtr->ptsData.hasTagFlags[tagName][ip][1] = ' ';
							}
						}
					}
					annotationManger->imgNameForlist.clear();
					const auto& newPicNameForList = labelControlPtr->ptsData.hasTagFlags[tagName];
					annotationManger->imgNameForlist.insert(annotationManger->imgNameForlist.end(), newPicNameForList.begin(), newPicNameForList.end());
					labelControlPtr->ptsData.updata();
				}
				if (labelControlPtr->ptsData.tagPickIdx < 0 || AnnotationManger::imgPickIdx < 0)
				{
					ImGui::EndDisabled();
				}
				ImGui::SameLine();
				if (ImGui::Button("save"))
				{
					labelControlPtr->ptsData.save(annotationManger->imgPaths);
				}
				deployButtom.y += 45;
			}
			if (AnnotationManger::trajs.size() > 0)
			{
				std::string tagStr = std::string(label::ImageLabel::tarStr);
				for (int i = AnnotationManger::imgPickIdx; i < labelControlPtr->ptsData.picShortNameForlist.size(); i++)
				{
					const cv::Point2f pt = AnnotationManger::trajs[i - AnnotationManger::imgPickIdx][0];
					labelControlPtr->ptsData.controlPtsAndTag[i][tagStr].x = pt.x;
					labelControlPtr->ptsData.controlPtsAndTag[i][tagStr].y = pt.y;
				}
				if (labelControlPtr->ptsData.colors.count(tagStr) == 0)
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
				for (const auto& d : labelControlPtr->ptsData.colors)
				{
					labelControlPtr->ptsData.tagsListName.emplace_back("  " + d.first);
					labelControlPtr->ptsData.tagsName.emplace_back(d.first);
					auto& tagAndShortName = labelControlPtr->ptsData.hasTagFlags[d.first];
					tagAndShortName.insert(tagAndShortName.end(), labelControlPtr->ptsData.picShortNameForlist.begin(), labelControlPtr->ptsData.picShortNameForlist.end());
					for (int j = 0; j < labelControlPtr->ptsData.picShortNameForlist.size(); j++)
					{
						if (labelControlPtr->ptsData.controlPtsAndTag[j].count(d.first) > 0)
						{
							labelControlPtr->ptsData.hasTagFlags[d.first][j][1] = '*';
						}
					}
				}
			}
			if (annotationManger->imgPaths.size() > 0 && labelControlPtr->ptsData.tagPickIdx >= 0)
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
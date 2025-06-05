
#include <filesystem>
#include "log.h"
#include "browser.h" 
#include "imgui.h"
#include "imfilebrowser.h"
namespace browser
{
	 
	Browser::Browser()
	{
	}
	Browser::Browser(const BrowserPick& p)
	{
		if (p == BrowserPick::PICK_DIR)
		{
			dialogPtr = new ImGui::FileBrowser(ImGuiFileBrowserFlags_::ImGuiFileBrowserFlags_SelectDirectory);
		}
		else if (p == BrowserPick::PICK_FILE)
		{
			dialogPtr = new ImGui::FileBrowser(ImGuiFileBrowserFlags_::ImGuiFileBrowserFlags_EnterNewFilename);
		}
		else
		{
			CHECK(false);
		}
	}
	bool Browser::pick(std::filesystem::path&pickRet)
	{
		pickRet = "";
		ImGui::FileBrowser& dialog = *(ImGui::FileBrowser*)dialogPtr;
		dialog.Open();
		dialog.Display();
		if (dialog.HasSelected())
		{
			pickRet = dialog.GetSelected();
			pickRet = std::filesystem::canonical(pickRet);    
			dialog.ClearSelected();
			return true;
		}		
		if (dialog.triggerCancel)
		{
			return true;
		}
		return false;
	}
	Browser::~Browser()
	{
		if (dialogPtr !=nullptr)
		{
			delete dialogPtr;
			dialogPtr = nullptr;
		}
	}
}




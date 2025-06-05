#ifndef _BROWSER_H_
#define _BROWSER_H_
#include <filesystem>
namespace browser
{
	 enum class BrowserPick
	{
		 PICK_DIR = 1,
		 PICK_FILE = 2,
	};
	class Browser
	{
	public:
		Browser();
		Browser(const BrowserPick&p);
		~Browser();
		bool pick(std::filesystem::path& pickRet);
	private:
		void* dialogPtr{nullptr};
	};
 
}

#endif // _BROWSER_H_

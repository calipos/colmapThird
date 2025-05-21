#include "utils.h"



namespace utils
{

	std::vector<std::string> splitString(const std::string& src, const std::string& symbols, bool repeat)
	{
		std::vector<std::string> result;
		int startIdx = 0;
		for (int i = 0; i < src.length(); i++)
		{
			bool isMatch = false;
			for (int j = 0; j < symbols.length(); j++)
			{
				if (src[i] == symbols[j])
				{
					isMatch = true;
					break;
				}
				if (!repeat)
				{
					break;
				}
			}
			if (isMatch)
			{
				std::string sub = src.substr(startIdx, i - startIdx);
				startIdx = i + 1;
				if (sub.length() > 0)
				{
					result.push_back(sub);
				}
			}
			if (i + 1 == src.length())
			{
				std::string sub = src.substr(startIdx, src.length() - startIdx);
				startIdx = i + 1;
				if (sub.length() > 0)
				{
					result.push_back(sub);
				}
			}
		}
		return result;
	}
}
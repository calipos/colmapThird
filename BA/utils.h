#ifndef _UTILS_H_
#define _UTILS_H_
#include <string>
#include <iostream>
#include <cstdint>
#include <vector>
#include <sstream>
namespace utils
{
	template<class Dtype>
	bool stringToNumber(const std::string& s, Dtype& out)
	{
		if (s.length() == 0)
		{
			return false;
		}
		std::stringstream ss;
		ss << s;
		out = 0;
		ss >> out;
		{
			std::string same = std::to_string(out);
			if (typeid(Dtype) == typeid(std::int32_t)
				|| typeid(Dtype) == typeid(std::uint32_t)
				|| typeid(Dtype) == typeid(std::uint64_t)
				|| typeid(Dtype) == typeid(std::int64_t)
				|| typeid(Dtype) == typeid(std::int16_t)
				|| typeid(Dtype) == typeid(std::uint16_t)
				|| typeid(Dtype) == typeid(std::int8_t)
				|| typeid(Dtype) == typeid(std::uint8_t))
			{
				if (0 != same.compare(s))
				{
					return false;
				}
			}
			if (typeid(Dtype) == typeid(float)
				|| typeid(Dtype) == typeid(double))
			{
				std::stringstream ss2;
				ss2 << same;
				Dtype exp = 0;
				ss2 >> exp;
				if (abs(exp - out) > 1e-4)
				{
					return false;
				}
				return true;
			}
		}
		return true;
	}
	std::vector<std::string> splitString(const std::string& src, const std::string& symbols, bool repeat = true);
}

#endif // !_UTILS_H_

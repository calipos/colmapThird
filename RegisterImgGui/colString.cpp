#include "colString.h"
#include <algorithm>
#include <cstdarg>
#include <fstream>
#include <sstream>



void StringAppendV(std::string* dst, const char* format, va_list ap) {
    // First try with a small fixed size buffer.
    static const int kFixedBufferSize = 1024;
    char fixed_buffer[kFixedBufferSize];

    // It is possible for methods that use a va_list to invalidate
    // the data in it upon use.  The fix is to make a copy
    // of the structure before using it and use that copy instead.
    va_list backup_ap;
    va_copy(backup_ap, ap);
    int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
    va_end(backup_ap);

    if (result < kFixedBufferSize) {
        if (result >= 0) {
            // Normal case - everything fits.
            dst->append(fixed_buffer, result);
            return;
        }

#ifdef _MSC_VER
        // Error or MSVC running out of space.  MSVC 8.0 and higher
        // can be asked about space needed with the special idiom below:
        va_copy(backup_ap, ap);
        result = vsnprintf(nullptr, 0, format, backup_ap);
        va_end(backup_ap);
#endif

        if (result < 0) {
            // Just an error.
            return;
        }
    }

    // Increase the buffer size to the size requested by vsnprintf,
    // plus one for the closing \0.
    const int variable_buffer_size = result + 1;
    std::unique_ptr<char[]> variable_buffer(new char[variable_buffer_size]);

    // Restore the va_list before we use it again.
    va_copy(backup_ap, ap);
    result =
        vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
    va_end(backup_ap);

    if (result >= 0 && result < variable_buffer_size) {
        dst->append(variable_buffer.get(), result);
    }
}

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


std::string StringPrintf(const char* format, ...) {
    va_list ap;
    va_start(ap, format);
    std::string result;
    StringAppendV(&result, format, ap);
    va_end(ap);
    return result;
}
bool IsNotWhiteSpace(const int character) {
    return character != ' ' && character != '\n' && character != '\r' &&
        character != '\t';
}
std::string StringReplace(const std::string& str,
    const std::string& old_str,
    const std::string& new_str) {
    if (old_str.empty()) {
        return str;
    }
    size_t position = 0;
    std::string mod_str = str;
    while ((position = mod_str.find(old_str, position)) != std::string::npos) {
        mod_str.replace(position, old_str.size(), new_str);
        position += new_str.size();
    }
    return mod_str;
}

std::string StringGetAfter(const std::string& str, const std::string& key) {
    if (key.empty()) {
        return str;
    }
    std::size_t found = str.rfind(key);
    if (found != std::string::npos) {
        return str.substr(found + key.length(),
            str.length() - (found + key.length()));
    }
    return "";
}

std::vector<std::string> StringSplit(const std::string& str,
    const std::string& delim) {
    return splitString(str, delim,true);
}

bool StringStartsWith(const std::string& str, const std::string& prefix) {
    return !prefix.empty() && prefix.size() <= str.size() &&
        str.substr(0, prefix.size()) == prefix;
}

void StringLeftTrim(std::string* str) {
    str->erase(str->begin(),
        std::find_if(str->begin(), str->end(), IsNotWhiteSpace));
}

void StringRightTrim(std::string* str) {
    str->erase(std::find_if(str->rbegin(), str->rend(), IsNotWhiteSpace).base(),
        str->end());
}

void StringTrim(std::string* str) {
    StringLeftTrim(str);
    StringRightTrim(str);
}

void StringToLower(std::string* str) {
    std::transform(str->begin(), str->end(), str->begin(), ::tolower);
}

void StringToUpper(std::string* str) {
    std::transform(str->begin(), str->end(), str->begin(), ::toupper);
}

bool StringContains(const std::string& str, const std::string& sub_str) {
    return str.find(sub_str) != std::string::npos;
}
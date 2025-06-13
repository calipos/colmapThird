#include "misc.h"
#include "log.h"

#include <cstdarg>
#include <sstream>
#include "colString.h"

void PrintHeading1(const std::string& heading) {
    std::ostringstream log;
    log << "\n" << std::string(78, '=') << "\n";
    log << heading << "\n";
    log << std::string(78, '=');
    LOG_OUT << log.str();
}

void PrintHeading2(const std::string& heading) {
    std::ostringstream log;
    log << "\n" << heading << "\n";
    log << std::string(std::min<int>(heading.size(), 78), '-');
    LOG_OUT << log.str();
}

template <>
std::vector<std::string> CSVToVector(const std::string& csv) {
    auto elems = StringSplit(csv, ",;");
    std::vector<std::string> values;
    values.reserve(elems.size());
    for (auto& elem : elems) {
        StringTrim(&elem);
        if (elem.empty()) {
            continue;
        }
        values.push_back(elem);
    }
    return values;
}

template <>
std::vector<int> CSVToVector(const std::string& csv) {
    auto elems = StringSplit(csv, ",;");
    std::vector<int> values;
    values.reserve(elems.size());
    for (auto& elem : elems) {
        StringTrim(&elem);
        if (elem.empty()) {
            continue;
        }
        try {
            values.push_back(std::stoi(elem));
        }
        catch (const std::invalid_argument&) {
            return std::vector<int>(0);
        }
    }
    return values;
}

template <>
std::vector<float> CSVToVector(const std::string& csv) {
    auto elems = StringSplit(csv, ",;");
    std::vector<float> values;
    values.reserve(elems.size());
    for (auto& elem : elems) {
        StringTrim(&elem);
        if (elem.empty()) {
            continue;
        }
        try {
            values.push_back(std::stod(elem));
        }
        catch (const std::invalid_argument&) {
            return std::vector<float>(0);
        }
    }
    return values;
}

template <>
std::vector<double> CSVToVector(const std::string& csv) {
    auto elems = StringSplit(csv, ",;");
    std::vector<double> values;
    values.reserve(elems.size());
    for (auto& elem : elems) {
        StringTrim(&elem);
        if (elem.empty()) {
            continue;
        }
        try {
            values.push_back(std::stold(elem));
        }
        catch (const std::invalid_argument&) {
            return std::vector<double>(0);
        }
    }
    return values;
}

void RemoveCommandLineArgument(const std::string& arg, int* argc, char** argv) {
    for (int i = 0; i < *argc; ++i) {
        if (argv[i] == arg) {
            for (int j = i + 1; j < *argc; ++j) {
                argv[i] = argv[j];
            }
            *argc -= 1;
            break;
        }
    }
}

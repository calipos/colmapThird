#ifndef _MISC_H_
#define _MISC_H_

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef STRINGIFY
#define STRINGIFY(s) STRINGIFY_(s)
#define STRINGIFY_(s) #s
#endif  // STRINGIFY

// Log first-order heading with over- and underscores.
void PrintHeading1(const std::string& heading);

// Log second-order heading with underscores.
void PrintHeading2(const std::string& heading);

// Check if vector contains elements.
template <typename T>
bool VectorContainsValue(const std::vector<T>& vector, T value);

template <typename T>
bool VectorContainsDuplicateValues(const std::vector<T>& vector);

// Parse CSV line to a list of values.
template <typename T>
std::vector<T> CSVToVector(const std::string& csv);

// Concatenate values in list to comma-separated list.
template <typename T>
std::string VectorToCSV(const std::vector<T>& values);

// Remove an argument from the list of command-line arguments.
void RemoveCommandLineArgument(const std::string& arg, int* argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool VectorContainsValue(const std::vector<T>& vector, const T value) {
    return std::find_if(vector.begin(), vector.end(), [value](const T element) {
        return element == value;
        }) != vector.end();
}

template <typename T>
bool VectorContainsDuplicateValues(const std::vector<T>& vector) {
    std::vector<T> unique_vector = vector;
    return std::unique(unique_vector.begin(), unique_vector.end()) !=
        unique_vector.end();
}

template <typename T>
std::string VectorToCSV(const std::vector<T>& values) {
    if (values.empty()) {
        return "";
    }

    std::ostringstream stream;
    for (const T& value : values) {
        stream << value << ", ";
    }
    std::string buf = stream.str();
    buf.resize(buf.size() - 2);
    return buf;
}
#endif // !_MISC_H_

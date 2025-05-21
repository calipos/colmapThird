#ifndef  _OBJECT_POINTS_H_
#define _OBJECT_POINTS_H_
#include <filesystem>
#include <map>
#include <array>
std::map<int, std::array<double, 3>> readPoints3DFromTXT(const std::filesystem::path& points3DTXT);
#endif // ! _OBJECT_POINTS_H_

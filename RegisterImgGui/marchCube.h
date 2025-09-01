#ifndef _MARCHING_CUBE_H_
#define _MARCHING_CUBE_H_
#include <vector>
#include <filesystem>
#include "Eigen/Core"
namespace mc
{
	struct Mesh
	{
		std::list<Eigen::Vector3f> vertices;
		std::list<Eigen::Vector3i> triangles;
		bool saveMesh(const std::filesystem::path& path);
	};
	Mesh marchcube(const Eigen::Matrix4Xf& grid, const std::vector<float>& gridSdf,
		const int& numX, const int& numY, const int& numZ,
		const float& gridUnit,
		const float& isoLevel);
}
#endif // !_MARCHING_CUBE_H_

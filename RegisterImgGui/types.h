#ifndef _TYPE_H_
#define _TYPE_H_
#ifdef _MSC_VER
#if _MSC_VER >= 1600
#include <cstdint>
#else
typedef __int8 int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned __int8 uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#endif
#elif __GNUC__ >= 3
#include <cstdint>
#endif
#include <Eigen/Core>
namespace Eigen {

	using Matrix3x4f = Matrix<float, 3, 4>;
	using Matrix3x4d = Matrix<double, 3, 4>;
	using Matrix6d = Matrix<double, 6, 6>;
	using Vector3ub = Matrix<uint8_t, 3, 1>;
	using Vector4ub = Matrix<uint8_t, 4, 1>;
	using Vector6d = Matrix<double, 6, 1>;
	using RowMajorMatrixXi = Matrix<int, Dynamic, Dynamic, RowMajor>;

}  // namespace Eigen


// Unique identifier for cameras.
typedef uint32_t camera_t;

// Unique identifier for images.
typedef uint32_t image_t;

// Each image pair gets a unique ID, see `Database::ImagePairToPairId`.
typedef uint64_t image_pair_t;

// Index per image, i.e. determines maximum number of 2D points per image.
typedef uint32_t point2D_t;

// Unique identifier per added 3D point. Since we add many 3D points,
// delete them, and possibly re-add them again, the maximum number of allowed
// unique indices should be large.
typedef uint64_t point3D_t;

// Values for invalid identifiers or indices.
const camera_t kInvalidCameraId = std::numeric_limits<camera_t>::max();
const image_t kInvalidImageId = std::numeric_limits<image_t>::max();
const image_pair_t kInvalidImagePairId =
std::numeric_limits<image_pair_t>::max();
const point2D_t kInvalidPoint2DIdx = std::numeric_limits<point2D_t>::max();
const point3D_t kInvalidPoint3DId = std::numeric_limits<point3D_t>::max();


// Simple implementation of C++20's std::span, as Ubuntu 20.04's default GCC
// version does not come with full C++20 and we still want to support it.
template <typename T>
class span {
	T* ptr_;
	const size_t size_;

public:
	span(T* ptr, size_t len) noexcept : ptr_{ ptr }, size_{ len } {}

	T& operator[](size_t i) noexcept { return ptr_[i]; }
	T const& operator[](size_t i) const noexcept { return ptr_[i]; }

	size_t size() const noexcept { return size_; }

	T* begin() noexcept { return ptr_; }
	T* end() noexcept { return ptr_ + size_; }
	const T* begin() const noexcept { return ptr_; }
	const T* end() const noexcept { return ptr_ + size_; }
};


struct Point2D {
	// The image coordinates in pixels, starting at upper left corner with 0.
	Eigen::Vector2d xy = Eigen::Vector2d::Zero();

	// The identifier of the 3D point. If the 2D point is not part of a 3D point
	// track the identifier is `kInvalidPoint3DId` and `HasPoint3D() = false`.
	point3D_t point3D_id = kInvalidPoint3DId;

	// Determin whether the 2D point observes a 3D point.
	inline bool HasPoint3D() const;

	inline bool operator==(const Point2D& other) const;
	inline bool operator!=(const Point2D& other) const;
};
std::ostream& operator<<(std::ostream& stream, const Point2D& point2D);
bool Point2D::HasPoint3D() const { return point3D_id != kInvalidPoint3DId; }
bool Point2D::operator==(const Point2D& other) const {
	return xy == other.xy && point3D_id == other.point3D_id;
}
bool Point2D::operator!=(const Point2D& other) const {
	return !(*this == other);
}
#endif // !_TYPE_H_

#ifndef __INS_MESH_TYPE_DEF_H__
#define __INS_MESH_TYPE_DEF_H__
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include <cstdint>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <atomic>
#define SINGLE_PRECISION
#define GRAIN_SIZE 1024
#if defined(SINGLE_PRECISION)
typedef float Float;
#else
typedef double Float;
#endif
namespace Eigen
{
	typedef Eigen::Matrix<int32_t, 2, 1>                            Vector2i;
	typedef Eigen::Matrix<int32_t, 3, 1>                            Vector3i;
	typedef Eigen::Matrix<int32_t, 4, 1>                            Vector4i;
	typedef Eigen::Matrix<uint32_t, 2, 1>                           Vector2u;
	typedef Eigen::Matrix<uint32_t, 3, 1>                           Vector3u;
	typedef Eigen::Matrix<uint32_t, 4, 1>                           Vector4u;
	typedef Eigen::Matrix<uint8_t, 4, 1>                            Vector4u8;
	typedef Eigen::Matrix<Float, 2, 1>                              Vector2f;
	typedef Eigen::Matrix<Float, 3, 1>                              Vector3f;
	typedef Eigen::Matrix<Float, 4, 1>                              Vector4f;
	typedef Eigen::Matrix<int32_t, Eigen::Dynamic, 1>               VectorXi;
	typedef Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>              VectorXu;
	typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>               VectorXu8;
	typedef Eigen::Matrix<bool, Eigen::Dynamic, 1>                  VectorXb;
	typedef Eigen::Matrix<Float, Eigen::Dynamic, 1>                 VectorXf;
	typedef Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>  MatrixXi;
	typedef Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXu;
	typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>  MatrixXu8;
	typedef Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>    MatrixXf;
	typedef Eigen::Matrix<Float, 2, 2>                              Matrix2f;
	typedef Eigen::Matrix<Float, 3, 3>                              Matrix3f;
	typedef Eigen::Matrix<Float, 4, 4>                              Matrix4f;
}

#define PROGRESS_BLKSIZE (1 << 18)
#define SHOW_PROGRESS(i, maxval, text) 

#define PROGRESS_SHIFT 18u
#define SHOW_PROGRESS_RANGE(range, maxval, text) 


#if defined(_WIN32)
#define RCPOVERFLOW_FLT   2.93873587705571876e-39f
#define RCPOVERFLOW_DBL   5.56268464626800345e-309
#else
#define RCPOVERFLOW_FLT   0x1p-128f
#define RCPOVERFLOW_DBL   0x1p-1024
#endif

#if defined(SINGLE_PRECISION)
#define RCPOVERFLOW RCPOVERFLOW_FLT
#else
#define RCPOVERFLOW RCPOVERFLOW_DBL
#endif
#define DEG_TO_RAG 0.01745329251994329576923690768489
#define RAD_TO_DEG 57.295779513082320876798154814105
#define PI 3.1415926535897932384626433832795
#define PI2 6.283185307179586476925286766559

inline std::string str_tolower(std::string str) {
	std::transform(str.begin(), str.end(), str.begin(), ::tolower);
	return str;
}
inline std::vector<std::string>& str_tokenize(const std::string& s, char delim, std::vector<std::string>& elems, bool include_empty = false) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim))
		if (!item.empty() || include_empty)
			elems.push_back(item);
	return elems;
}
inline std::vector<std::string> str_tokenize(const std::string& s, char delim, bool include_empty) {
	std::vector<std::string> elems;
	str_tokenize(s, delim, elems, include_empty);
	return elems;
}
inline uint32_t str_to_uint32_t(const std::string& str) {
	char* end_ptr = nullptr;
	uint32_t result = (uint32_t)strtoul(str.c_str(), &end_ptr, 10);
	if (*end_ptr != '\0')
		throw std::runtime_error("Could not parse unsigned integer \"" + str + "\"");
	return result;
}
inline bool atomicCompareAndExchange(volatile uint32_t* v, uint32_t newValue, uint32_t oldValue) {
#if defined(_WIN32)
	return _InterlockedCompareExchange(
		reinterpret_cast<volatile long*>(v), (long)newValue, (long)oldValue) == (long)oldValue;
#else
	return __sync_bool_compare_and_swap(v, oldValue, newValue);
#endif
}

inline uint32_t atomicAdd(volatile uint32_t* dst, uint32_t delta) {
#if defined(_MSC_VER)
	return _InterlockedExchangeAdd(reinterpret_cast<volatile long*>(dst), delta) + delta;
#else
	return __sync_add_and_fetch(dst, delta);
#endif
}

inline float atomicAdd(volatile float* dst, float delta) {
	union bits { float f; uint32_t i; };
	bits oldVal, newVal;
	do {
#if defined(__i386__) || defined(__amd64__)
		__asm__ __volatile__("pause\n");
#endif
		oldVal.f = *dst;
		newVal.f = oldVal.f + delta;
	} while (!atomicCompareAndExchange((volatile uint32_t*)dst, newVal.i, oldVal.i));
	return newVal.f;
}

/// Always-positive modulo function, Float precision version (assumes b > 0)
inline Float modulo(Float a, Float b) {
	Float r = std::fmod(a, b);
	return (r < 0.0) ? r + b : r;
}

/// Always-positive modulo function (assumes b > 0)
inline int32_t modulo(int32_t a, int32_t b) {
	int32_t r = a % b;
	return (r < 0) ? r + b : r;
}

inline float fast_acos(float x) {
	float negate = float(x < 0.0f);
	x = std::abs(x);
	float ret = -0.0187293f;
	ret *= x; ret = ret + 0.0742610f;
	ret *= x; ret = ret - 0.2121144f;
	ret *= x; ret = ret + 1.5707288f;
	ret = ret * std::sqrt(1.0f - x);
	ret = ret - 2.0f * negate * ret;
	return negate * (float)PI + ret;
}

template<typename T, typename U> inline T union_cast(const U& val) {
	union { U u; T t; } tmp = { val };
	return tmp.t;
}

inline Float signum(Float value) {
	return std::copysign((Float)1, value);
}

inline void coordinate_system(const Eigen::Vector3f& a, Eigen::Vector3f& b, Eigen::Vector3f& c) {
	if (std::abs(a.x()) > std::abs(a.y())) {
		Float invLen = 1.0f / std::sqrt(a.x() * a.x() + a.z() * a.z());
		c = Eigen::Vector3f(a.z() * invLen, 0.0f, -a.x() * invLen);
	}
	else {
		Float invLen = 1.0f / std::sqrt(a.y() * a.y() + a.z() * a.z());
		c = Eigen::Vector3f(0.0f, a.z() * invLen, -a.y() * invLen);
	}
	b = c.cross(a);
}
template <typename TimeT = std::chrono::milliseconds> class Timer {
public:
	Timer() {
		start = std::chrono::system_clock::now();
	}

	size_t value() const {
		auto now = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<TimeT>(now - start);
		return (size_t)duration.count();
	}

	size_t reset() {
		auto now = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<TimeT>(now - start);
		start = now;
		return (size_t)duration.count();
	}
private:
	std::chrono::system_clock::time_point start;
};
inline std::string timeString(double time, bool precise = false) {
	if (std::isnan(time) || std::isinf(time))
		return "inf";

	std::string suffix = "ms";
	if (time > 1000) {
		time /= 1000; suffix = "s";
		if (time > 60) {
			time /= 60; suffix = "m";
			if (time > 60) {
				time /= 60; suffix = "h";
				if (time > 12) {
					time /= 12; suffix = "d";
				}
			}
		}
	}

	std::ostringstream os;
	os << std::setprecision(precise ? 4 : 1)
		<< std::fixed << time << suffix;

	return os.str();
}

inline std::string memString(size_t size, bool precise = false) {
	double value = (double)size;
	const char* suffixes[] = {
		"B", "KiB", "MiB", "GiB", "TiB", "PiB"
	};
	int suffix = 0;
	while (suffix < 5 && value > 1024.0f) {
		value /= 1024.0f; ++suffix;
	}

	std::ostringstream os;
	os << std::setprecision(suffix == 0 ? 0 : (precise ? 4 : 1))
		<< std::fixed << value << " " << suffixes[suffix];

	return os.str();
}

template <typename Matrix> inline size_t sizeInBytes(const Matrix& matrix) {
	return matrix.size() * sizeof(typename Matrix::Scalar);
}

class ordered_lock {
public:
	ordered_lock() : next_ticket(0), counter(0) {}
	void lock() {
		std::unique_lock<std::mutex> acquire(cvar_lock);
		unsigned int ticket = next_ticket++;
		while (ticket != counter)
			cvar.wait(acquire);
	}
	void unlock() {
		std::unique_lock<std::mutex> acquire(cvar_lock);
		counter++;
		cvar.notify_all();
	}
protected:
	std::condition_variable  cvar;
	std::mutex               cvar_lock;
	unsigned int             next_ticket, counter;
};

#endif // !__INS_MESH_TYPE_DEF_H__

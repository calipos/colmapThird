#pragma once
#include "insmeshcomm.h"

struct Ray {
    Eigen::Vector3f o, d;
    Float mint, maxt;

    Ray(const Eigen::Vector3f& o, const Eigen::Vector3f& d) :
        o(o), d(d), mint(0), maxt(std::numeric_limits<Float>::infinity()) { }

    Ray(const Eigen::Vector3f& o, const Eigen::Vector3f& d, Float mint, Float maxt) :
        o(o), d(d), mint(mint), maxt(maxt) { }

    Eigen::Vector3f operator()(Float t) const { return o + t * d; }
};

struct AABB {
    Eigen::Vector3f minAABB, maxAABB;

    AABB() { clear(); }

    AABB(const Eigen::Vector3f& minAABB, const Eigen::Vector3f& maxAABB) : minAABB(minAABB), maxAABB(maxAABB) {}

    void clear() {
        const Float inf = std::numeric_limits<Float>::infinity();
        minAABB.setConstant(inf);
        maxAABB.setConstant(-inf);
    }

    void expandBy(const Eigen::Vector3f& p) {
        minAABB = minAABB.cwiseMin(p);
        maxAABB = maxAABB.cwiseMax(p);
    }

    void expandBy(const AABB& aabb) {
        minAABB = minAABB.cwiseMin(aabb.minAABB);
        maxAABB = maxAABB.cwiseMax(aabb.maxAABB);
    }

    bool contains(const Eigen::Vector3f& p) {
        return (p.array() >= minAABB.array()).all() &&
            (p.array() <= maxAABB.array()).all();
    }

    bool rayIntersect(const Ray& ray) const {
        Float nearT = -std::numeric_limits<Float>::infinity();
        Float farT = std::numeric_limits<Float>::infinity();

        for (int i = 0; i < 3; i++) {
            Float origin = ray.o[i];
            Float minVal = minAABB[i], maxVal = maxAABB[i];

            if (ray.d[i] == 0) {
                if (origin < minVal || origin > maxVal)
                    return false;
            }
            else {
                Float t1 = (minVal - origin) / ray.d[i];
                Float t2 = (maxVal - origin) / ray.d[i];

                if (t1 > t2)
                    std::swap(t1, t2);

                nearT = (std::max)(t1, nearT);
                farT = (std::min)(t2, farT);

                if (!(nearT <= farT))
                    return false;
            }
        }

        return ray.mint <= farT && nearT <= ray.maxt;
    }

    Float squaredDistanceTo(const Eigen::Vector3f& p) const {
        Float result = 0;
        for (int i = 0; i < 3; ++i) {
            Float value = 0;
            if (p[i] < minAABB[i])
                value = minAABB[i] - p[i];
            else if (p[i] > maxAABB[i])
                value = p[i] - maxAABB[i];
            result += value * value;
        }
        return result;
    }

    int largestAxis() const {
        Eigen::Vector3f extents = maxAABB - minAABB;

        if (extents[0] >= extents[1] && extents[0] >= extents[2])
            return 0;
        else if (extents[1] >= extents[0] && extents[1] >= extents[2])
            return 1;
        else
            return 2;
    }

    Float surfaceArea() const {
        Eigen::Vector3f d = maxAABB - minAABB;
        return 2.0f * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2]);
    }

    Eigen::Vector3f center() const {
        return 0.5f * (minAABB + maxAABB);
    }

    static AABB merge(const AABB& aabb1, const AABB& aabb2) {
        return AABB(aabb1.minAABB.cwiseMin(aabb2.minAABB), aabb1.maxAABB.cwiseMax(aabb2.maxAABB));
    }
};

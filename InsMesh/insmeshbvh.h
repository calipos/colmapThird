#include "insmeshaabb.h"

/* BVH node in 32 bytes */
struct BVHNode {
    union {
        struct {
            unsigned flag : 1;
            std::uint32_t size : 31;
            std::uint32_t start;
        } leaf;

        struct {
            std::uint32_t unused;
            std::uint32_t rightChild;
        } inner;
    };
    AABB aabb;

    inline bool isLeaf() const {
        return leaf.flag == 1;
    }

    inline bool isInner() const {
        return leaf.flag == 0;
    }

    inline bool isUnused() const {
        return inner.unused == 0 && inner.rightChild == 0;
    }

    inline std::uint32_t start() const {
        return leaf.start;
    }

    inline std::uint32_t end() const {
        return leaf.start + leaf.size;
    }
};

class BVH {
    friend struct BVHBuildTask;
    /* Cost values for BVH surface area heuristic */
    enum { T_aabb = 1, T_tri = 1 };
public:
    BVH(const Eigen::MatrixXu* F, const Eigen::MatrixXf* V, const Eigen::MatrixXf* N, const AABB& aabb);

    ~BVH();

    void setData(const Eigen::MatrixXu* F, const Eigen::MatrixXf* V, const Eigen::MatrixXf* N) { mF = F; mV = V; mN = N; }

    const Eigen::MatrixXu* F() const { return mF; }
    const Eigen::MatrixXf* V() const { return mV; }
    const Eigen::MatrixXf* N() const { return mN; }
    Float diskRadius() const { return mDiskRadius; }

    void build();

    void printStatistics() const;

    bool rayIntersect(Ray ray) const;

    bool rayIntersect(Ray ray, std::uint32_t& idx, Float& t, Eigen::Vector2f* uv = nullptr) const;

    void findNearestWithRadius(const Eigen::Vector3f& p, Float radius,
        std::vector<std::uint32_t>& result,
        bool includeSelf = false) const;

    std::uint32_t findNearest(const Eigen::Vector3f& p, Float& radius, bool includeSelf = false) const;

    void findKNearest(const Eigen::Vector3f& p, std::uint32_t k, Float& radius,
        std::vector<std::pair<Float, std::uint32_t> >& result,
        bool includeSelf = false) const;

    void findKNearest(const Eigen::Vector3f& p, const Eigen::Vector3f& N, std::uint32_t k,
        Float& radius,
        std::vector<std::pair<Float, std::uint32_t> >& result,
        Float angleThresh = 30,
        bool includeSelf = false) const;

protected:
    bool rayIntersectTri(const Ray& ray, std::uint32_t i, Float& t, Eigen::Vector2f& uv) const;
    bool rayIntersectDisk(const Ray& ray, std::uint32_t i, Float& t) const;
    void refitBoundingBoxes(std::uint32_t node_idx = 0);
    std::pair<Float, std::uint32_t> statistics(std::uint32_t node_idx = 0) const;

protected:
    std::vector<BVHNode> mNodes;
    std::uint32_t* mIndices;
    const Eigen::MatrixXu* mF;
    const Eigen::MatrixXf* mV, * mN;
    Float mDiskRadius;
};
#pragma once

#pragma once
#include "insmeshaabb.h"
struct MeshStats {
    AABB mAABB;
    Eigen::Vector3f mWeightedCenter;
    double mAverageEdgeLength;
    double mMaximumEdgeLength;
    double mSurfaceArea;

    MeshStats() :
        mWeightedCenter(Eigen::Vector3f::Zero()),
        mAverageEdgeLength(0.0f),
        mMaximumEdgeLength(0.0f),
        mSurfaceArea(0.0f) { }
};

extern MeshStats
compute_mesh_stats(const Eigen::MatrixXu& F, const Eigen::MatrixXf& V,
    bool deterministic = false);

void compute_dual_vertex_areas(
    const Eigen::MatrixXu& F, const Eigen::MatrixXf& V, const Eigen::VectorXu& V2E,
    const Eigen::VectorXu& E2E, const Eigen::VectorXb& nonManifold, Eigen::VectorXf& A);

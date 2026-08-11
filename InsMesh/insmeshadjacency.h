#pragma once
#include "insmeshcomm.h"
/* Stores integer jumps between nodes of the adjacency matrix */
struct IntegerVariable {
    unsigned short rot : 2;
    signed   short translate_u : 7;
    signed   short translate_v : 7;

    Eigen::Vector2i shift() const {
        return Eigen::Vector2i(translate_u, translate_v);
    }

    void setShift(Eigen::Vector2i& v) {
        translate_u = v.x();
        translate_v = v.y();
    }
};

/* Stores a weighted adjacency matrix entry together with integer variables */
struct Link {
    uint32_t id;
    float weight;
    union {
        IntegerVariable ivar[2];
        uint32_t ivar_uint32;
    };

    inline Link() { }
    inline Link(uint32_t id) : id(id), weight(1.0f), ivar_uint32(0u) { }
    inline Link(uint32_t id, float weight) : id(id), weight(weight), ivar_uint32(0u) { }

    inline bool operator<(const Link& link) const { return id < link.id; }
};

typedef Link** AdjacencyMatrix;

extern AdjacencyMatrix generate_adjacency_matrix_uniform(
    const Eigen::MatrixXu& F, const Eigen::VectorXu& V2E,
    const Eigen::VectorXu& E2E, const Eigen::VectorXb& nonManifold);

extern AdjacencyMatrix generate_adjacency_matrix_cotan(
    const Eigen::MatrixXu& F, const Eigen::MatrixXf& V, const Eigen::VectorXu& V2E,
    const Eigen::VectorXu& E2E, const Eigen::VectorXb& nonManifold);

inline Link& search_adjacency(AdjacencyMatrix& adj, uint32_t i, uint32_t j) {
    for (Link* l = adj[i]; l != adj[i + 1]; ++l)
        if (l->id == j)
            return *l;
    throw std::runtime_error("search_adjacency: failure!");
}

class BVH;
struct MeshStats;

extern AdjacencyMatrix generate_adjacency_matrix_pointcloud(
    Eigen::MatrixXf& V, Eigen::MatrixXf& N, const BVH* bvh, MeshStats& stats,
    uint32_t knn_points, bool deterministic = false);

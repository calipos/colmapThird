#pragma once

#include "insmeshcomm.h"

static const uint32_t INVALID = (uint32_t)-1;

inline uint32_t dedge_prev_3(uint32_t e) { return (e % 3 == 0) ? e + 2 : e - 1; }
inline uint32_t dedge_next_3(uint32_t e) { return (e % 3 == 2) ? e - 2 : e + 1; }
inline uint32_t dedge_prev_4(uint32_t e) { return (e % 4 == 0) ? e + 3 : e - 1; }
inline uint32_t dedge_next_4(uint32_t e) { return (e % 4 == 3) ? e - 3 : e + 1; }

inline uint32_t dedge_prev(uint32_t e, uint32_t deg) { return (e % deg == 0u) ? e + (deg - 1) : e - 1; }
inline uint32_t dedge_next(uint32_t e, uint32_t deg) { return (e % deg == deg - 1) ? e - (deg - 1) : e + 1; }

void build_dedge(const Eigen::MatrixXu& F, const Eigen::MatrixXf& V, Eigen::VectorXu& V2E,
    Eigen::VectorXu& E2E, Eigen::VectorXb& boundary, Eigen::VectorXb& nonManifold,
    bool quiet = false);

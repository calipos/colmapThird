#pragma once
#include "insmeshcomm.h"

extern void subdivide(Eigen::MatrixXu& F, Eigen::MatrixXf& V, Eigen::VectorXu& V2E, Eigen::VectorXu& E2E,
    Eigen::VectorXb& boundary, Eigen::VectorXb& nonmanifold,
    Float maxLength, bool deterministic = false);

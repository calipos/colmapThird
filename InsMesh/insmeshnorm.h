/*
    normal.h: Helper routines for computing vertex normals

    This file is part of the implementation of

        Instant Field-Aligned Meshes
        Wenzel Jakob, Daniele Panozzo, Marco Tarini, and Olga Sorkine-Hornung
        In ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2015)

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "insmeshcomm.h"
#include <map>
#include <set>

extern void
generate_smooth_normals(const Eigen::MatrixXu& F, const Eigen::MatrixXf& V, Eigen::MatrixXf& N, bool deterministic);

extern void
generate_smooth_normals(const Eigen::MatrixXu& F, const Eigen::MatrixXf& V,
    const Eigen::VectorXu& V2E, const Eigen::VectorXu& E2E,
    const Eigen::VectorXb& nonManifold, Eigen::MatrixXf& N);

extern void
generate_crease_normals(Eigen::MatrixXu& F, Eigen::MatrixXf& V, const Eigen::VectorXu& V2E,
    const Eigen::VectorXu& E2E, const Eigen::VectorXb boundary,
    const Eigen::VectorXb& nonManifold, Float angleThreshold,
    Eigen::MatrixXf& N, std::map<uint32_t, uint32_t>& creases);

extern void
generate_crease_normals(
    const Eigen::MatrixXu& F, const Eigen::MatrixXf& V, const Eigen::VectorXu& V2E, const Eigen::VectorXu& E2E,
    const Eigen::VectorXb boundary, const Eigen::VectorXb& nonManifold, Float angleThreshold,
    Eigen::MatrixXf& N, std::set<uint32_t>& creases);

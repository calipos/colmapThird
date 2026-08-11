#pragma once
#include "insmeshcomm.h"
extern void reorder_mesh(Eigen::MatrixXu& F, std::vector<Eigen::MatrixXf>& V_vec, std::vector<Eigen::MatrixXf>& F_vec);

extern void replicate_vertices(Eigen::MatrixXu& F, std::vector<Eigen::MatrixXf>& V);

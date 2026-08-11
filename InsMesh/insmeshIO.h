#pragma once
#include <string>
#include "insmeshcomm.h"

extern void
load_mesh_or_pointcloud(const std::string& filename, Eigen::MatrixXu& F,
    Eigen::MatrixXf& V, Eigen::MatrixXf& N);

extern void load_obj(const std::string& filename, Eigen::MatrixXu& F, Eigen::MatrixXf& V);

extern void load_ply(const std::string& filename, Eigen::MatrixXu& F, Eigen::MatrixXf& V,
    Eigen::MatrixXf& N, bool pointcloud = false);

extern void
load_pointcloud(const std::string& filename, Eigen::MatrixXf& V, Eigen::MatrixXf& N);

extern void write_mesh(const std::string& filename, const Eigen::MatrixXu& F,
    const Eigen::MatrixXf& V,
    const Eigen::MatrixXf& N = Eigen::MatrixXf(),
    const Eigen::MatrixXf& Nf = Eigen::MatrixXf(),
    const Eigen::MatrixXf& UV = Eigen::MatrixXf(),
    const Eigen::MatrixXf& C = Eigen::MatrixXf());

extern void write_obj(const std::string& filename, const Eigen::MatrixXu& F,
    const Eigen::MatrixXf& V,
    const Eigen::MatrixXf& N = Eigen::MatrixXf(),
    const Eigen::MatrixXf& Nf = Eigen::MatrixXf(),
    const Eigen::MatrixXf& UV = Eigen::MatrixXf(),
    const Eigen::MatrixXf& C = Eigen::MatrixXf());

extern void write_ply(const std::string& filename, const Eigen::MatrixXu& F,
    const Eigen::MatrixXf& V,
    const Eigen::MatrixXf& N = Eigen::MatrixXf(),
    const Eigen::MatrixXf& Nf = Eigen::MatrixXf(),
    const Eigen::MatrixXf& UV = Eigen::MatrixXf(),
    const Eigen::MatrixXf& C = Eigen::MatrixXf());

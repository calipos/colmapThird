#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <unordered_set>
#include <Eigen/Core>
#include "ceres/ceres.h"
struct ExponentialResidual {
    ExponentialResidual(double x, double y) : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T* const ABC, T* residual) const
    {
        residual[0] = ABC[0]*exp(ABC[1] * x_  ) + ABC[2] - y_;
        //T diff = ABC[0]*exp(ABC[1] * x_ ) + ABC[2] - y_;
        //residual[0] = diff * diff;
        return true;
    }
private:
    const double x_;
    const double y_;
};
struct ExponentialResidual2 :public ceres::SizedCostFunction<1, 3 >
{
    ExponentialResidual2(double x, double y) : x_(x), y_(y) {}
    virtual bool Evaluate(double const* const* ABC,
        double* residuals,
        double** jacobians) const
    {
        residuals[0] = ABC[0][0]*exp(ABC[0][1] * x_) + ABC[0][2] - y_;
        if (jacobians != NULL && jacobians[0] != NULL) {
            jacobians[0][0] = exp(ABC[0][1] * x_);
            jacobians[0][1] = ABC[0][0] * exp(ABC[0][1] * x_)*x_;
            jacobians[0][2] = 1;
        }
        return true;
    }
private:
    const double x_;
    const double y_;
};

int test_curve_fit()
{
    std::fstream fin("C:/Users/Administrator/Downloads/xt_y_local.txt", std::ios::in);
    std::string aline;
    std::getline(fin, aline);
	std::vector<double>xs,ys;
    while (std::getline(fin, aline))
    {
        std::stringstream ss(aline);
        float x, y;
        ss >> x >> y;
		xs.emplace_back(x);
		ys.emplace_back(y);
    }




    double* ABC = new double[3];
    ABC[0] = 248.4540321943;
    ABC[1] = -1e-3;
    ABC[2] = 2815.2785291575;

    ceres::Problem problem;
	for (int i = 0; i < xs.size(); i++)
	{ 
        //problem.AddResidualBlock(
        //    new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 3>(
        //        new ExponentialResidual(xs[i], ys[i])),
        //    NULL,
        //    &ABC[0]);

        ceres::CostFunction* cost_function = new ExponentialResidual2(xs[i], ys[i]);
        problem.AddResidualBlock(cost_function, NULL, ABC); 
	}

    ceres::Solver::Options options;
    options.max_num_iterations = 2500;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n"; 
    std::cout << "Final   ABC: " << ABC[0] << "   " << ABC[1] << "   " << ABC[2] << "\n";

    return 0;
}
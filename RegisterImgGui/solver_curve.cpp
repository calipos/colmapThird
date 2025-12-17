#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <unordered_set>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "ceres/ceres.h"
#define _USE_ERR_SQARE_ 0
#define _USE_AUTO_GRAD_ 0
struct ExponentialResidual {
    ExponentialResidual(double x, double y) : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T* const ABC, T* residual) const
    {
#if _USE_ERR_SQARE_==0
        residual[0] = ABC[0]*exp(ABC[1] * x_  ) + ABC[2] - y_;
#else
        T diff = ABC[0]*exp(ABC[1] * x_ ) + ABC[2] - y_;
        residual[0] = diff * diff;
#endif
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
#if _USE_ERR_SQARE_==0
        residuals[0] = ABC[0][0]*exp(ABC[0][1] * x_) + ABC[0][2] - y_;
        if (jacobians != NULL && jacobians[0] != NULL) {
            jacobians[0][0] = exp(ABC[0][1] * x_);
            jacobians[0][1] = ABC[0][0] * exp(ABC[0][1] * x_)*x_;
            jacobians[0][2] = 1;
        }
#else
        double expValue = exp(ABC[0][1] * x_);
        double diff = ABC[0][0] * expValue + ABC[0][2] - y_;
        residuals[0] = diff* diff;
        if (jacobians != NULL && jacobians[0] != NULL) {
            jacobians[0][0] = 2 * diff * expValue;
            jacobians[0][1] = 2 * diff * ABC[0][0] * expValue * x_;
            jacobians[0][2] = 2 * diff;
        }
#endif // SQARE__==0

        return true;
    }
private:
    const double x_;
    const double y_;
};
double verify(const std::vector<double>& xs, const std::vector<double>& ys, const double* const ABC)
{
    std::vector<double> err(xs.size());
    for (int i = 0; i < xs.size(); i++)
    {
        err[i] = ABC[0] * exp(xs[i] * ABC[1]) + ABC[2] - ys[i];
        err[i] = (err[i] * err[i]);
    }
    return std::accumulate(err.begin(), err.end(), 0.) / xs.size();
}
int figure_init(const std::vector<double>& xs, const std::vector<double>& ys, double& A, double& B, double& C)
{
    A = 0;
    B = 0;
    C = 0;
    //{//Aexp(Bx)+C=y =>  exp(Bx+A`)+C=y  =>  A`+Bx+C+1=y
    //    Eigen::MatrixXd left(xs.size(),2);
    //    Eigen::VectorXd right(ys.size());
    //    for (int i = 0; i < xs.size(); i++)
    //    {
    //        left(i, 0) = xs[0]; left(i, 1) = 1;
    //        right[i] = ys[i];
    //    } 
    //    Eigen::VectorXd BD1 = left.colPivHouseholderQr().solve(right);
    //    C = BD1[1];
    //    std::cout << BD1 << std::endl;
    //}
    //{
    //    //Aexp(Bx)+C=y =>  exp(Bx+A`)+C=y  =>  A`+Bx=y-C-1
    //    Eigen::MatrixXd left(xs.size(), 2);
    //    Eigen::VectorXd right(ys.size());
    //    for (int i = 0; i < xs.size(); i++)
    //    {
    //        left(i, 0) = 1; left(i, 1) = xs[0];
    //        right[i] = ys[i]-C-1;
    //    }
    //    Eigen::VectorXd AB = left.colPivHouseholderQr().solve(right); 
    //    std::cout << AB << std::endl;
    //}
    {//Aexp(Bx)+C=y =>  AX+C=y  =>  A`+Bx+C+1=y
        Eigen::MatrixXd left(xs.size(), 2);
        Eigen::VectorXd right(ys.size());
        for (int i = 0; i < xs.size(); i++)
        {
            left(i, 0) = xs[i]; left(i, 1) = 1;
            right[i] = ys[i];
        }
        Eigen::BDCSVD<Eigen::MatrixXd> svd(left, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd AC = svd.solve(right);
        //Eigen::VectorXd AC = left.colPivHouseholderQr().solve(right);
        A = AC[0];
        B = -0.001;
        C = AC[1];
        std::cout << AC << std::endl;
    } 
    return 0;
}
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
     

     
    figure_init(xs, ys, ABC[0], ABC[1], ABC[2]);

    std::cout << "init   ABC: "  <<std::fixed << std::setprecision(4) << ABC[0] << "   " << ABC[1] << "   " << ABC[2] << "\n";
    std::cout << "ERR = " << verify(xs, ys, ABC) << std::endl;;
    ceres::Problem problem;
	for (int i = 0; i < xs.size(); i++)
	{ 
#if _USE_AUTO_GRAD_>0
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 3>(
                new ExponentialResidual(xs[i], ys[i])),
            NULL,
            &ABC[0]);
#else
        ceres::CostFunction* cost_function = new ExponentialResidual2(xs[i], ys[i]);
        problem.AddResidualBlock(cost_function, NULL, ABC);
#endif // _USE_AUTO_GRAD_>0
	}

    ceres::Solver::Options options;
    options.max_num_iterations = 25000;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n"; 
    std::cout << "Final   ABC: " << std::fixed << std::setprecision(8) << ABC[0] << "   " << ABC[1] << "   " << ABC[2] << "\n";
    std::cout << "ERR = " << verify(xs, ys, ABC) << std::endl;;
    return 0;
}
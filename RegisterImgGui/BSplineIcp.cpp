#include <filesystem>
#include <fstream>
#include <string>
#include "bfm.h"
#include "meshDraw.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"
#include "Eigen/Core"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "log.h"
namespace bs
{
    template <typename Dtype>
    class BSplineBasis
    {
    public:
        // 均匀节点向量：degree p，控制点数量 n+1
        // 节点数 = n + p + 2
        static std::vector<Dtype> uniformKnots(int numCtrl, int degree)
        {
            int m = numCtrl + degree + 1; // 节点数
            std::vector<Dtype> knots(m);
            for (int i = 0; i < m; ++i)
            {
                if (i <= degree)
                    knots[i] = 0.0;
                else if (i >= numCtrl)
                    knots[i] = 1.0;
                else
                    knots[i] = Dtype(i - degree) / Dtype(numCtrl - degree);
            }
            return knots;
        }

        // 计算第 i 个基函数在参数 t 处的值
        static Dtype basis(int i, int degree, Dtype t,
            const std::vector<Dtype>& knots)
        {
            if (degree == 0)
            {
                if ((knots[i] <= t && t < knots[i + 1]) ||
                    (t == 1.0 && knots[i + 1] == 1.0 && knots[i] < 1.0))
                {
                    return 1.0;
                }
                return 0.0;
            }

            Dtype left = 0.0, right = 0.0;
            Dtype d1 = knots[i + degree] - knots[i];
            Dtype d2 = knots[i + degree + 1] - knots[i + 1];

            if (d1 > 1e-12)
                left = (t - knots[i]) / d1 * basis(i, degree - 1, t, knots);
            if (d2 > 1e-12)
                right =
                (knots[i + degree + 1] - t) / d2 * basis(i + 1, degree - 1, t, knots);

            return left + right;
        }

        static Dtype basisDeriv(int i, int degree, Dtype t,
            const std::vector<Dtype>& knots)
        {
            if (degree == 0)
            {
                // 0 次基函数的导数为 0（分段常数）
                return 0.0;
            }

            Dtype left = 0.0, right = 0.0;

            Dtype d1 = knots[i + degree] - knots[i];
            Dtype d2 = knots[i + degree + 1] - knots[i + 1];

            if (d1 > 1e-12)
                left = static_cast<Dtype>(degree) / d1 * basis(i, degree - 1, t, knots);

            if (d2 > 1e-12)
                right = static_cast<Dtype>(degree) / d2 *
                basis(i + 1, degree - 1, t, knots);

            return left - right;
        }

        // 计算所有非零基函数（数量 = degree+1）
        static std::vector<std::pair<int, Dtype>>
            allBasis(int numCtrl, int degree, Dtype t, const std::vector<Dtype>& knots)
        {
            std::vector<std::pair<int, Dtype>> result;
            for (int i = 0; i < numCtrl; ++i)
            {
                Dtype b = basis(i, degree, t, knots);
                if (b > 1e-12)
                    result.emplace_back(i, b);
            }
            return result;
        }
    };
    template <typename Dtype>
    class FFDGrid
    {
    public:
        int nx, ny, nz; // 控制点数量
        int degree;     // B 样条阶数（通常 3）
        std::vector<Eigen::Matrix<Dtype, 3, 1>>
            ctrl; // 控制点，索引 = (i*ny + j)*nz + k

        std::vector<Dtype> knotX, knotY, knotZ;

        FFDGrid(int nx_, int ny_, int nz_, int degree_ = 3)
            : nx(nx_), ny(ny_), nz(nz_), degree(degree_)
        {
            ctrl.resize(nx * ny * nz);
            knotX = BSplineBasis<Dtype>::uniformKnots(nx, degree);
            knotY = BSplineBasis<Dtype>::uniformKnots(ny, degree);
            knotZ = BSplineBasis<Dtype>::uniformKnots(nz, degree);
        }

        int index(int i, int j, int k) const { return (i * ny + j) * nz + k; }

        Eigen::Matrix<Dtype, 3, 1>& at(int i, int j, int k)
        {
            return ctrl[index(i, j, k)];
        }
        const Eigen::Matrix<Dtype, 3, 1>& at(int i, int j, int k) const
        {
            return ctrl[index(i, j, k)];
        }

        // 初始化：把控制点摆成单位立方体
        void initUnitCube()
        {
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                    for (int k = 0; k < nz; ++k)
                    {
                        at(i, j, k) = { Dtype(i) / Dtype(nx - 1), Dtype(j) / Dtype(ny - 1),
                                       Dtype(k) / Dtype(nz - 1) };
                    }
        }
        void jacobian(const Dtype& s, const Dtype& t, const Dtype& u, Eigen::Matrix<Dtype, 3, 1>& dXds, Eigen::Matrix<Dtype, 3, 1>& dXdt,
            Eigen::Matrix<Dtype, 3, 1>& dXdu) const
        {
            dXds = Eigen::Matrix<Dtype, 3, 1>(0, 0, 0);
            dXdt = Eigen::Matrix<Dtype, 3, 1>(0, 0, 0);
            dXdu = Eigen::Matrix<Dtype, 3, 1>(0, 0, 0);

            for (int i = 0; i < nx; ++i)
            {
                Dtype Bu = BSplineBasis<Dtype>::basis(i, degree, s, knotX);
                Dtype dBu = BSplineBasis<Dtype>::basisDeriv(i, degree, s, knotX);

                for (int j = 0; j < ny; ++j)
                {
                    Dtype Bv = BSplineBasis<Dtype>::basis(j, degree, t, knotY);
                    Dtype dBv = BSplineBasis<Dtype>::basisDeriv(j, degree, t, knotY);

                    for (int k = 0; k < nz; ++k)
                    {
                        Dtype Bw = BSplineBasis<Dtype>::basis(k, degree, u, knotZ);
                        Dtype dBw = BSplineBasis<Dtype>::basisDeriv(k, degree, u, knotZ);

                        const Eigen::Matrix<Dtype, 3, 1>& P = at(i, j, k);

                        dXds += P * (dBu * Bv * Bw);
                        dXdt += P * (Bu * dBv * Bw);
                        dXdu += P * (Bu * Bv * dBw);
                    }
                }
            }
        }
    };
    template <typename Dtype>
    class FFD
    {
    public:
        FFDGrid<Dtype> grid;

        FFD(int nx, int ny, int nz, int degree = 3) : grid(nx, ny, nz, degree) {}

        Eigen::Matrix<Dtype, 3, 1>
            computeParamNaive(const Eigen::Matrix<Dtype, 3, 1>& p) const
        {
            Eigen::Matrix<Dtype, 3, 1> pmin(1e30, 1e30, 1e30),
                pmax(-1e30, -1e30, -1e30);
            for (const auto& c : grid.ctrl)
            {
                pmin.x() = std::min(pmin.x(), c.x());
                pmax.x() = std::max(pmax.x(), c.x());
                pmin.y() = std::min(pmin.y(), c.y());
                pmax.y() = std::max(pmax.y(), c.y());
                pmin.z() = std::min(pmin.z(), c.z());
                pmax.z() = std::max(pmax.z(), c.z());
            }
            Eigen::Matrix<Dtype, 3, 1> s;
            s.x() = (pmax.x() - pmin.x() > 1e-12)
                ? (p.x() - pmin.x()) / (pmax.x() - pmin.x())
                : 0.0;
            s.y() = (pmax.y() - pmin.y() > 1e-12)
                ? (p.y() - pmin.y()) / (pmax.y() - pmin.y())
                : 0.0;
            s.z() = (pmax.z() - pmin.z() > 1e-12)
                ? (p.z() - pmin.z()) / (pmax.z() - pmin.z())
                : 0.0;
            // 夹紧到 [0,1]
            s.x() = std::min((Dtype)1.0, std::max((Dtype)0.0, s.x()));
            s.y() = std::min((Dtype)1.0, std::max((Dtype)0.0, s.y()));
            s.z() = std::min((Dtype)1.0, std::max((Dtype)0.0, s.z()));
            return s;
        }
        Eigen::Matrix<Dtype, 3, 1>
            computeParam(const Eigen::Matrix<Dtype, 3, 1>& p) const
        {
            auto clamp01 = [](Dtype v)
            { return std::max(static_cast<Dtype>(0), std::min(static_cast<Dtype>(1), v)); };

            Eigen::Matrix<Dtype, 3, 1> stu = p;
            stu.x() = clamp01(stu.x());
            stu.y() = clamp01(stu.y());
            stu.z() = clamp01(stu.z());
            for (int iter = 0; iter < 10; iter++)
            {
                Eigen::Matrix<Dtype, 3, 1> physicalPt = deformWithParam(stu);
                Eigen::Matrix<Dtype, 3, 1> F = physicalPt - p;
                if (F.norm() < 1e-6)
                    break;

                Eigen::Matrix<Dtype, 3, 1> dXds, dXdt, dXdu;
                grid.jacobian(stu.x(), stu.y(), stu.z(), dXds, dXdt, dXdu);

                // J * delta = -F
                Dtype a11 = dXds.x(), a12 = dXdt.x(), a13 = dXdu.x();
                Dtype a21 = dXds.y(), a22 = dXdt.y(), a23 = dXdu.y();
                Dtype a31 = dXds.z(), a32 = dXdt.z(), a33 = dXdu.z();

                Dtype det = a11 * (a22 * a33 - a23 * a32) -
                    a12 * (a21 * a33 - a23 * a31) +
                    a13 * (a21 * a32 - a22 * a31);

                if (std::abs(det) < 1e-14)
                {
                    std::cout << "det< 1e-14\n";
                    break;
                }
                Dtype invDet = 1.0 / det;

                Dtype b11 = (a22 * a33 - a23 * a32) * invDet;
                Dtype b12 = -(a12 * a33 - a13 * a32) * invDet;
                Dtype b13 = (a12 * a23 - a13 * a22) * invDet;

                Dtype b21 = -(a21 * a33 - a23 * a31) * invDet;
                Dtype b22 = (a11 * a33 - a13 * a31) * invDet;
                Dtype b23 = -(a11 * a23 - a13 * a21) * invDet;

                Dtype b31 = (a21 * a32 - a22 * a31) * invDet;
                Dtype b32 = -(a11 * a32 - a12 * a31) * invDet;
                Dtype b33 = (a11 * a22 - a12 * a21) * invDet;

                // delta = -J^{-1} F
                Dtype ds = -(b11 * F.x() + b12 * F.y() + b13 * F.z());
                Dtype dt = -(b21 * F.x() + b22 * F.y() + b23 * F.z());
                Dtype du = -(b31 * F.x() + b32 * F.y() + b33 * F.z());

                Dtype step = 1.0;
                Dtype ns = stu.x() + step * ds;
                Dtype nt = stu.y() + step * dt;
                Dtype nu_ = stu.z() + step * du;

                int guard = 0;
                while ((ns < -0.01 || ns > 1.01 || nt < -0.01 || nt > 1.01 ||
                    nu_ < -0.01 || nu_ > 1.01) &&
                    guard < 10)
                {
                    step *= 0.5;
                    ns = stu.x() + step * ds;
                    nt = stu.y() + step * dt;
                    nu_ = stu.z() + step * du;
                    ++guard;
                }

                stu.x() = clamp01(ns);
                stu.y() = clamp01(nt);
                stu.z() = clamp01(nu_);
            }

            return stu;
        }



        // 用给定参数直接求位置
        Eigen::Matrix<Dtype, 3, 1>
            deformWithParam(const Eigen::Matrix<Dtype, 3, 1>& stu) const
        {
            auto bx =
                BSplineBasis<Dtype>::allBasis(grid.nx, grid.degree, stu.x(), grid.knotX);
            auto by =
                BSplineBasis<Dtype>::allBasis(grid.ny, grid.degree, stu.y(), grid.knotY);
            auto bz =
                BSplineBasis<Dtype>::allBasis(grid.nz, grid.degree, stu.z(), grid.knotZ);

            Eigen::Matrix<Dtype, 3, 1> result(0, 0, 0);
            for (auto& [i, wi] : bx)
                for (auto& [j, wj] : by)
                    for (auto& [k, wk] : bz)
                    {
                        double w = wi * wj * wk;
                        result += grid.at(i, j, k) * w;
                    }
            return result;
        }

        std::vector<std::pair<int, double>>
            computeWeightsWithParam(const Eigen::Matrix<Dtype, 3, 1>& s) const
        {
            auto bx =
                BSplineBasis<Dtype>::allBasis(grid.nx, grid.degree, s.x(), grid.knotX);
            auto by =
                BSplineBasis<Dtype>::allBasis(grid.ny, grid.degree, s.y(), grid.knotY);
            auto bz =
                BSplineBasis<Dtype>::allBasis(grid.nz, grid.degree, s.z(), grid.knotZ);

            std::vector<std::pair<int, double>> weights;
            for (auto& [i, wi] : bx)
                for (auto& [j, wj] : by)
                    for (auto& [k, wk] : bz)
                    {
                        double w = wi * wj * wk;
                        weights.emplace_back(grid.index(i, j, k), w);
                    }
            return weights;
        }

        void moveStuTo(const Eigen::Matrix<Dtype, 3, 1>& stu,
            const Eigen::Matrix<Dtype, 3, 1>& targetStu)
        {
            Eigen::Matrix<Dtype, 3, 1> delta = targetStu - stu;

            auto weights = computeWeightsWithParam(stu);
            double sumW2 = 0.0;
            for (auto& [idx, w] : weights)
                sumW2 += w * w;
            if (sumW2 < 1e-12)
                return;

            for (auto& [idx, w] : weights)
            {
                // 最小二乘意义下把位移分配到控制点
                grid.ctrl[idx] += delta * (w / sumW2);
            }
            return;
        }


    };
    void figureMeshPtsBnd(Eigen::RowVector3f& minV, Eigen::RowVector3f& maxV)
    {
        Eigen::RowVector3f range = maxV - minV;
        minV = minV - 0.05 * range;
        maxV = maxV + 0.05 * range;
        return;
    }
}
int readplyXyzi(const std::filesystem::path& path,cv::Mat&pts,std::vector<float>&intensity)
{
	if (!std::filesystem::exists(path))
	{
		LOG_ERR_OUT << "not exists : " << path;
		return -1;
	}
	{
		std::fstream fin(path, std::ios::in | std::ios::binary);
		std::string aline;
		std::getline(fin, aline);
		std::getline(fin, aline);
		std::getline(fin, aline);
		std::vector<std::string> segs;
		{ 
			std::stringstream ss(aline);
			std::string token;
			while (ss >> token) {           
				segs.push_back(token);
			}
		}
		if (segs.size() != 3)
		{
			LOG_ERR_OUT << "file format error! : " << path;
			return -1; 
		}
		int ptsCnt = -1;
		std::stringstream ss;
		ss << segs[2];
		ss >> ptsCnt;
		if (ptsCnt < 10)
		{
			LOG_ERR_OUT << "file format error! : " << path;
			return -1;
		}
		pts=cv::Mat(ptsCnt,3,CV_32FC1);
		intensity.resize(ptsCnt);
		std::getline(fin, aline);
		std::getline(fin, aline);
		std::getline(fin, aline);
		std::getline(fin, aline);
		std::getline(fin, aline);
		for (int i = 0; i < ptsCnt; i++)
		{
			fin.read((char*)&pts.ptr<float>(i)[0], sizeof(float));
			fin.read((char*)&pts.ptr<float>(i)[1], sizeof(float));
			fin.read((char*)&pts.ptr<float>(i)[2], sizeof(float));
			fin.read((char*)&intensity[i], sizeof(float));
		}
	}
	return 0;
}
int test_bsIcp()
{
	std::filesystem::path mvsResultDir("D:/repo/colmapThird/data/a/result");
	Eigen::MatrixX3f V;
	Eigen::MatrixX3i F;
	Eigen::MatrixX3f C;

	cv::Mat ptsTgt;
	std::vector<float>intensityTgt;
	if (0!=readplyXyzi("D:/repo/colmapThird/data/a/result/c.ply", ptsTgt, intensityTgt))
	{
		return -1;
	}

	bfm::Bfm2019*bfmIns = new bfm::Bfm2019("D:/repo/colmapThird/models/model2019_face12.h5");
	bfmIns->generateRandomFace(V, C);
    {
        Eigen::RowVector3f colMin = V.colwise().minCoeff();
        Eigen::RowVector3f colMax = V.colwise().maxCoeff();
        Eigen::RowVector3f center = (colMin + colMax) * 0.5;
        V = V.rowwise() - center;
        colMin = V.colwise().minCoeff();
        colMax = V.colwise().maxCoeff();
        bs::figureMeshPtsBnd(colMin, colMax);
        float scale = 2 * (std::max)(colMax.cwiseAbs().maxCoeff(), colMin.cwiseAbs().maxCoeff());
        float scale_inv = 1.f / scale;
        V = V * scale_inv;
        V = V.rowwise() + Eigen::RowVector3f(0.5, 0.5, 0.5);
    }
	F = bfmIns->F;


 //   meshdraw::utils::savePts(mvsResultDir / "V.txt", V);
	//Eigen::MatrixX3f meshVRotated = (V * bfm_R.transpose() * bfm_scale).rowwise() + bfm_t;
 //   meshdraw::utils::savePts(mvsResultDir / "meshVRotated.txt", meshVRotated);
    //iter
    {	
        float bfm_scale;
        Eigen::Matrix3f bfm_R;
        Eigen::RowVector3f bfm_t;
        {
            Eigen::Matrix3Xf ptsTgt(3, 3);
            Eigen::Matrix3Xf ptsSrc(3, 3);
            ptsTgt(0, 0) = -0.026656504137979587;
            ptsTgt(1, 0) = -0.043014894341780643;
            ptsTgt(2, 0) = 1.0267763105463232;
            ptsTgt(0, 1) = -0.11071509463256694;
            ptsTgt(1, 1) = -0.043265088116702488;
            ptsTgt(2, 1) = 1.0368117383245794;
            ptsTgt(0, 2) = -0.070484715500162845;
            ptsTgt(1, 2) = 0.092090697929671381;
            ptsTgt(2, 2) = 0.98627442545928867;
            ptsSrc.col(0) = V.row(9440);
            ptsSrc.col(1) = V.row(22884);
            ptsSrc.col(2) = V.row(26877);

            Eigen::Matrix4f T = Eigen::umeyama(ptsSrc, ptsTgt, true);
            Eigen::Matrix3f sR = T.block<3, 3>(0, 0);
            bfm_scale = std::cbrt(sR.determinant());
            bfm_R = sR / bfm_scale;
            bfm_t = T.block<3, 1>(0, 3);
        }
        Eigen::MatrixX3f meshVRotated = (V * bfm_R.transpose() * bfm_scale).rowwise() + bfm_t;
        cv::Mat meshVRotatedCv; 
        cv::eigen2cv(meshVRotated, meshVRotatedCv);


        int k = 1;
        cv::flann::KDTreeIndexParams indexParams(4);
        cv::flann::Index tree(meshVRotatedCv, indexParams);//此处用target构建k-d树
        cv::Mat ldIdx;
        cv::Mat ldDists;
        tree.knnSearch(ptsTgt, ldIdx, ldDists, k, cv::flann::SearchParams(32));


        bs::FFD<float> ffd(4, 4, 4, 3);
        ffd.grid.initUnitCube();
        Eigen::Matrix3Xf stu(3, V.rows());



        for (int j = 0; j < ptsTgt.rows; j++)
        {
            const int& targetIdx = j;
            const int& sourceIdx = ldIdx.ptr<int>(j)[0];
            Eigen::Vector3f targetStu = ffd.computeParam(Eigen::Vector3f(ptsTgt.ptr<float>(targetIdx)[0], ptsTgt.ptr<float>(targetIdx)[1], ptsTgt.ptr<float>(targetIdx)[2]));
            Eigen::Vector3f sourceStu = ffd.computeParam(Eigen::Vector3f(ptsTgt.ptr<float>(targetIdx)[0], ptsTgt.ptr<float>(targetIdx)[1], ptsTgt.ptr<float>(targetIdx)[2]));
        }
        for (size_t i = 0; i < V.rows(); i++)
        {
            stu.col(i) = ffd.computeParamNaive(V.row(i));


            Eigen::Vector3f scrStu = ffd.computeParam(pts.col(srcIdx));
            Eigen::Vector3f targetStu = ffd.computeParam(targetPos);
            ffd.moveStuTo(scrStu, targetStu);
            for (size_t i = 0; i < pts.cols(); i++)
            {
                pts.col(i) = ffd.deformWithParam(stu.col(i));
            }
        }

    

        LOG_OUT;
    }



	return 0;
}
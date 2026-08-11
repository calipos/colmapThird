#pragma once
#include "insmeshcomm.h"
#include "insmeshadjacency.h"
class Serializer;

extern AdjacencyMatrix
downsample_graph(const AdjacencyMatrix adj, const Eigen::MatrixXf& V,
    const Eigen::MatrixXf& N, const Eigen::VectorXf& areas, Eigen::MatrixXf& V_p,
    Eigen::MatrixXf& V_n, Eigen::VectorXf& areas_p, Eigen::MatrixXu& to_upper,
    Eigen::VectorXu& to_lower, bool deterministic = false);

struct MultiResolutionHierarchy {
    enum { MAX_DEPTH = 25 };
public:
    MultiResolutionHierarchy();
    void free();
    void save(Serializer& state);
    void load(const Serializer& state);

    int levels() const { return (int)mV.size(); }

    void build(bool deterministic = false);

    void printStatistics() const;
    void resetSolution();

    inline ordered_lock& mutex() { return mMutex; }

    inline const std::vector<std::vector<uint32_t>>& phases(int level) const { return mPhases[level]; }
    inline const AdjacencyMatrix& adj(int level = 0) const { return mAdj[level]; }
    inline AdjacencyMatrix& adj(int level = 0) { return mAdj[level]; }
    inline const Eigen::MatrixXf& V(int level = 0) const { return mV[level]; }
    inline const Eigen::MatrixXf& N(int level = 0) const { return mN[level]; }
    inline const Eigen::VectorXf& A(int level = 0) const { return mA[level]; }
    inline const Eigen::MatrixXu& toUpper(int level) const { return mToUpper[level]; }
    inline const Eigen::VectorXu& toLower(int level) const { return mToLower[level]; }
    inline const Eigen::MatrixXf& Q(int level = 0) const { return mQ[level]; }
    inline const Eigen::MatrixXf& O(int level = 0) const { return mO[level]; }
    inline const Eigen::MatrixXf& CQ(int level = 0) const { return mCQ[level]; }
    inline const Eigen::MatrixXf& CO(int level = 0) const { return mCO[level]; }
    inline const Eigen::VectorXf& CQw(int level = 0) const { return mCQw[level]; }
    inline const Eigen::VectorXf& COw(int level = 0) const { return mCOw[level]; }
    inline const Eigen::MatrixXu& F() const { return mF; }
    inline const Eigen::VectorXu& E2E() const { return mE2E; }
    inline Eigen::MatrixXf& Q(int level = 0) { return mQ[level]; }
    inline Eigen::MatrixXf& O(int level = 0) { return mO[level]; }
    inline Eigen::MatrixXf& CQ(int level = 0) { return mCQ[level]; }
    inline Eigen::MatrixXf& CO(int level = 0) { return mCO[level]; }
    inline Eigen::VectorXf& CQw(int level = 0) { return mCQw[level]; }
    inline Eigen::VectorXf& COw(int level = 0) { return mCOw[level]; }

    inline void setF(Eigen::MatrixXu&& F) { mF = std::move(F); }
    inline void setE2E(Eigen::VectorXu&& E2E) { mE2E = std::move(E2E); }
    inline void setV(Eigen::MatrixXf&& V) { mV.clear(); mV.push_back(std::move(V)); }
    inline void setN(Eigen::MatrixXf&& N) { mN.clear(); mN.push_back(std::move(N)); }
    inline void setA(Eigen::MatrixXf&& A) { mA.clear(); mA.push_back(std::move(A)); }
    inline void setAdj(AdjacencyMatrix&& adj) { mAdj.clear(); mAdj.push_back(std::move(adj)); }

    inline uint32_t size(int level = 0) const { return mV[level].cols(); }

    inline Float scale() const { return mScale; }
    inline void setScale(Float scale) { mScale = scale; }
    inline int iterationsQ() const { return mIterationsQ; }
    inline void setIterationsQ(int iterationsQ) { mIterationsQ = iterationsQ; }
    inline int iterationsO() const { return mIterationsO; }
    inline void setIterationsO(int iterationsO) { mIterationsO = iterationsO; }
    inline size_t totalSize() const { return mTotalSize; }

    void clearConstraints();
    void propagateConstraints(int rosy, int posy);
    void propagateSolution(int rosy);

    inline Eigen::Vector3f faceCenter(uint32_t idx) const {
        Eigen::Vector3f pos = Eigen::Vector3f::Zero();
        for (int i = 0; i < 3; ++i)
            pos += mV[0].col(mF(i, idx));
        return pos * (1.0f / 3.0f);
    }

    inline Eigen::Vector3f faceNormal(uint32_t idx) const {
        Eigen::Vector3f p0 = mV[0].col(mF(0, idx)),
            p1 = mV[0].col(mF(1, idx)),
            p2 = mV[0].col(mF(2, idx));
        return (p1 - p0).cross(p2 - p0).normalized();
    }

    /* Flags which indicate whether the integer variables are froen */
    bool frozenQ() const { return mFrozenQ; }
    bool frozenO() const { return mFrozenO; }
    void setFrozenQ(bool frozen) { mFrozenQ = frozen; }
    void setFrozenO(bool frozen) { mFrozenO = frozen; }
public:
    Eigen::MatrixXu mF;
    Eigen::VectorXu mE2E;
    std::vector<std::vector<std::vector<uint32_t>>> mPhases;
    std::vector<AdjacencyMatrix> mAdj;
    std::vector<Eigen::MatrixXf> mV;
    std::vector<Eigen::MatrixXf> mN;
    std::vector<Eigen::VectorXf> mA;
    std::vector<Eigen::VectorXu> mToLower;
    std::vector<Eigen::MatrixXu> mToUpper;
    std::vector<Eigen::MatrixXf> mO;
    std::vector<Eigen::MatrixXf> mQ;
    std::vector<Eigen::MatrixXf> mCQ;
    std::vector<Eigen::MatrixXf> mCO;
    std::vector<Eigen::VectorXf> mCQw;
    std::vector<Eigen::VectorXf> mCOw;
    bool mFrozenQ, mFrozenO;
    ordered_lock mMutex;
    Float mScale;
    int mIterationsQ;
    int mIterationsO;
    uint32_t mTotalSize;
};

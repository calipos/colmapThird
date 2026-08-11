#include <iostream>
#include "tbb/tbb.h"
#include "insmeshDirectEdge.h"

void build_dedge(const Eigen::MatrixXu& F, const Eigen::MatrixXf& V, Eigen::VectorXu& V2E,
    Eigen::VectorXu& E2E, Eigen::VectorXb& boundary, Eigen::VectorXb& nonManifold, bool quiet) {
    if (!quiet) {
        std::cout << "Building a directed edge data structure .. ";
        std::cout.flush();
    }
    Timer<> timer;


    V2E.resize(V.cols());
    V2E.setConstant(INVALID);

    uint32_t deg = F.rows();
    std::vector<std::pair<uint32_t, uint32_t>> tmp(F.size());

    tbb::parallel_for(
        tbb::blocked_range<uint32_t>(0u, (uint32_t)F.cols(), GRAIN_SIZE),
        [&](const tbb::blocked_range<uint32_t>& range) {
            for (uint32_t f = range.begin(); f != range.end(); ++f) {
                for (uint32_t i = 0; i < deg; ++i) {
                    /// 顶点Id
                    uint32_t idx_cur = F(i, f);
                    /// 下一个 顶点Id
                    uint32_t idx_next = F((i + 1) % deg, f);
                    /// 半边在facets的位置索引
                    uint32_t edge_id = deg * f + i;
                    if (idx_cur >= V.cols() || idx_next >= V.cols())
                        throw std::runtime_error("Mesh data contains an out-of-bounds vertex reference!");
                    if (idx_cur == idx_next)
                        continue;
                    ///给V2E(顶点序)分配半边索引(facets的位置索引),先到先得.共边的相邻面的相同顶点,也触发了改条件
                    ///tmp[]大小是3*F,存每一个半边信息,其元素是{下一个点Id,半边索引}
                 

                    tmp[edge_id] = std::make_pair(idx_next, INVALID);
                    if (!atomicCompareAndExchange(&V2E[idx_cur], edge_id, INVALID)) {
                        uint32_t idx = V2E[idx_cur];
                        while (!atomicCompareAndExchange(&tmp[idx].second, edge_id, INVALID))
                            idx = tmp[idx].second;
                    }
                }
            }
        }
    );


    nonManifold.resize(V.cols());
    nonManifold.setConstant(false);

    E2E.resize(F.cols() * deg);
    E2E.setConstant(INVALID);

    tbb::parallel_for(
        tbb::blocked_range<uint32_t>(0u, (uint32_t)F.cols(), GRAIN_SIZE),
        [&](const tbb::blocked_range<uint32_t>& range) {
            for (uint32_t f = range.begin(); f != range.end(); ++f) {
                for (uint32_t i = 0; i < deg; ++i) {
                    uint32_t idx_cur = F(i, f),
                        idx_next = F((i + 1) % deg, f),
                        edge_id_cur = deg * f + i;

                    if (idx_cur == idx_next)
                        continue;

                    uint32_t it = V2E[idx_next], edge_id_opp = INVALID;
                    while (it != INVALID) {
                        if (tmp[it].first == idx_cur) {
                            if (edge_id_opp == INVALID) {
                                edge_id_opp = it;
                            }
                            else {
                                nonManifold[idx_cur] = true;
                                nonManifold[idx_next] = true;
                                edge_id_opp = INVALID;
                                break;
                            }
                        }
                        it = tmp[it].second;
                    }

                    if (edge_id_opp != INVALID && edge_id_cur < edge_id_opp) {
                        E2E[edge_id_cur] = edge_id_opp;
                        E2E[edge_id_opp] = edge_id_cur;
                    }
                }
            }
            if (!quiet)
                SHOW_PROGRESS_RANGE(range, F.cols(), "Building directed edge data structure (2/3)");
        }
    );

    std::atomic<uint32_t> nonManifoldCounter(0), boundaryCounter(0), isolatedCounter(0);

    boundary.resize(V.cols());
    boundary.setConstant(false);

    /* Detect boundary regions of the mesh and adjust vertex->edge pointers*/
    tbb::parallel_for(
        tbb::blocked_range<uint32_t>(0u, (uint32_t)V.cols(), GRAIN_SIZE),
        [&](const tbb::blocked_range<uint32_t>& range) {
            for (uint32_t i = range.begin(); i != range.end(); ++i) {
                uint32_t edge = V2E[i];
                if (edge == INVALID) {
                    isolatedCounter++;
                    continue;
                }
                if (nonManifold[i]) {
                    nonManifoldCounter++;
                    V2E[i] = INVALID;
                    continue;
                }

                /* Walk backwards to the first boundary edge (if any) */
                uint32_t start = edge, v2e = INVALID;
                do {
                    v2e = (std::min)(v2e, edge);
                    uint32_t prevEdge = E2E[dedge_prev(edge, deg)];
                    if (prevEdge == INVALID) {
                        /* Reached boundary -- update the vertex->edge link */
                        v2e = edge;
                        boundary[i] = true;
                        boundaryCounter++;
                        break;
                    }
                    edge = prevEdge;
                } while (edge != start);
                V2E[i] = v2e;
            }
            if (!quiet)
                SHOW_PROGRESS_RANGE(range, V.cols(), "Building directed edge data structure (3/3)");
        }
    );

    if (!quiet) {
        std::cout << "done. (";
        if (nonManifoldCounter)
            std::cout << nonManifoldCounter << " non-manifold vertices, ";
        if (boundaryCounter)
            std::cout << boundaryCounter << " boundary vertices, ";
        if (isolatedCounter)
            std::cout << isolatedCounter << " isolated vertices, ";
        std::cout << "took " << timeString(timer.value()) << ")" << std::endl;
    }


    //std::cout << V << std::endl;
    //std::cout << F << std::endl;
    //std::cout << "V2E\n"<<V2E << std::endl;
    //std::cout << "E2E\n"<<E2E << std::endl;
    //std::cout << "boundary\n"<<boundary << std::endl;
    //std::cout << "nonManifold\n"<<nonManifold << std::endl;
    return;
}

#include <iostream>
#include <set>
#include <cmath>
#include "insmeshcomm.h"
#include "insmeshstate.h"
#include "insmeshIO.h"
#include "insmeshhierarchy.h"
#include "insmeshDirectEdge.h"
#include "insmeshsubdivide.h"
#include "insmeshbvh.h"
#include "insmeshnorm.h"
#include "insmodfield.h"
#include "insmeshadjacency.h"
#include "insmeshreorder.h"
#include "insmeshextract.h"
int nprocs = -1;
void batch_process(const std::string& input, const std::string& output,
    int rosy, int posy, Float scale, int face_count,
    int vertex_count, Float creaseAngle, bool extrinsic,
    bool align_to_boundaries, int smooth_iter, int knn_points,
    bool pure_quad, bool deterministic)
{
    std::cout << std::endl;
    std::cout << "Running in batch mode:" << std::endl;
    std::cout << "   Input file             = " << input << std::endl;
    std::cout << "   Output file            = " << output << std::endl;
    std::cout << "   Rotation symmetry type = " << rosy << std::endl;
    std::cout << "   Position symmetry type = " << (posy == 3 ? 6 : posy) << std::endl;
    std::cout << "   Crease angle threshold = ";
    if (creaseAngle > 0)
        std::cout << creaseAngle << std::endl;
    else
        std::cout << "disabled" << std::endl;
    std::cout << "   Extrinsic mode         = " << (extrinsic ? "enabled" : "disabled") << std::endl;
    std::cout << "   Align to boundaries    = " << (align_to_boundaries ? "yes" : "no") << std::endl;
    std::cout << "   kNN points             = " << knn_points << " (only applies to point clouds)" << std::endl;
    std::cout << "   Fully deterministic    = " << (deterministic ? "yes" : "no") << std::endl;
    if (posy == 4)
        std::cout << "   Output mode            = " << (pure_quad ? "pure quad mesh" : "quad-dominant mesh") << std::endl;
    std::cout << std::endl;

    Eigen::MatrixXu F;
    Eigen::MatrixXf V, N;
    Eigen::VectorXf A;
    std::set<uint32_t> crease_in, crease_out;
    BVH* bvh = nullptr;
    AdjacencyMatrix adj = nullptr;
    load_mesh_or_pointcloud(input, F, V, N);
    bool pointcloud = F.size() == 0;
    Timer<> timer;
    MeshStats stats = compute_mesh_stats(F, V, deterministic);
    if (pointcloud) {
        bvh = new BVH(&F, &V, &N, stats.mAABB);
        bvh->build();
        adj = generate_adjacency_matrix_pointcloud(V, N, bvh, stats, knn_points, deterministic);
        A.resize(V.cols());
        A.setConstant(1.0f);
    }
    if (scale < 0 && vertex_count < 0 && face_count < 0) {
        std::cout << "No target vertex count/face count/scale argument provided. "
            "Setting to the default of 1/16 * input vertex count." << std::endl;
        vertex_count = V.cols() / 2;
    }

    if (scale > 0) {
        Float face_area = posy == 4 ? (scale * scale) : (std::sqrt(3.f) / 4.f * scale * scale);
        face_count = stats.mSurfaceArea / face_area;
        vertex_count = posy == 4 ? face_count : (face_count / 2);
    }
    else if (face_count > 0) {
        Float face_area = stats.mSurfaceArea / face_count;
        vertex_count = posy == 4 ? face_count : (face_count / 2);
        scale = posy == 4 ? std::sqrt(face_area) : (2 * std::sqrt(face_area * std::sqrt(1.f / 3.f)));
    }
    else if (vertex_count > 0) {
        face_count = posy == 4 ? vertex_count : (vertex_count * 2);
        Float face_area = stats.mSurfaceArea / face_count;
        scale = posy == 4 ? std::sqrt(face_area) : (2 * std::sqrt(face_area * std::sqrt(1.f / 3.f)));
    }

    std::cout << "Output mesh goals (approximate)" << std::endl;
    std::cout << "   Vertex count           = " << vertex_count << std::endl;
    std::cout << "   Face count             = " << face_count << std::endl;
    std::cout << "   Edge length            = " << scale << std::endl;
    MultiResolutionHierarchy mRes;

    if (!pointcloud) {
        /* Subdivide the mesh if necessary */
        Eigen::VectorXu V2E, E2E;
        Eigen::VectorXb boundary, nonManifold;
        if (stats.mMaximumEdgeLength * 2 > scale || stats.mMaximumEdgeLength > stats.mAverageEdgeLength * 2) {
            std::cout << "Input mesh is too coarse for the desired output edge length "
                "(max input mesh edge length=" << stats.mMaximumEdgeLength
                << "), subdividing .." << std::endl;
            build_dedge(F, V, V2E, E2E, boundary, nonManifold);
            subdivide(F, V, V2E, E2E, boundary, nonManifold, std::min(scale / 2, (Float)stats.mAverageEdgeLength * 2), deterministic);
        }

        /* Compute a directed edge data structure */
        build_dedge(F, V, V2E, E2E, boundary, nonManifold);

        /* Compute adjacency matrix */
        adj = generate_adjacency_matrix_uniform(F, V2E, E2E, nonManifold);

        /* Compute vertex/crease normals */
        if (creaseAngle >= 0)
            generate_crease_normals(F, V, V2E, E2E, boundary, nonManifold, creaseAngle, N, crease_in);
        else
            generate_smooth_normals(F, V, V2E, E2E, nonManifold, N);

        /* Compute dual vertex areas */
        compute_dual_vertex_areas(F, V, V2E, E2E, nonManifold, A);

        mRes.setE2E(std::move(E2E));
    }
    /* Build multi-resolution hierarrchy */
    mRes.setAdj(std::move(adj));
    mRes.setF(std::move(F));
    mRes.setV(std::move(V));
    mRes.setA(std::move(A));
    mRes.setN(std::move(N));
    mRes.setScale(scale);
    mRes.build(deterministic);
    mRes.resetSolution();

    if (align_to_boundaries && !pointcloud) {
        mRes.clearConstraints();
        for (uint32_t i = 0; i < 3 * mRes.F().cols(); ++i) {
            if (mRes.E2E()[i] == INVALID) {
                uint32_t i0 = mRes.F()(i % 3, i / 3);
                uint32_t i1 = mRes.F()((i + 1) % 3, i / 3);
                Eigen::Vector3f p0 = mRes.V().col(i0), p1 = mRes.V().col(i1);
                Eigen::Vector3f edge = p1 - p0;
                if (edge.squaredNorm() > 0) {
                    edge.normalize();
                    mRes.CO().col(i0) = p0;
                    mRes.CO().col(i1) = p1;
                    mRes.CQ().col(i0) = mRes.CQ().col(i1) = edge;
                    mRes.CQw()[i0] = mRes.CQw()[i1] = mRes.COw()[i0] =
                        mRes.COw()[i1] = 1.0f;
                }
            }
        }
        mRes.propagateConstraints(rosy, posy);
    }

    if (bvh) {
        bvh->setData(&mRes.F(), &mRes.V(), &mRes.N());
    }
    else if (smooth_iter > 0) {
        bvh = new BVH(&mRes.F(), &mRes.V(), &mRes.N(), stats.mAABB);
        bvh->build();
    }

    std::cout << "Preprocessing is done. (total time excluding file I/O: "
        << timeString(timer.reset()) << ")" << std::endl;

    Optimizer optimizer(mRes, false);
    optimizer.setRoSy(rosy);
    optimizer.setPoSy(posy);
    optimizer.setExtrinsic(extrinsic);

    std::cout << "Optimizing orientation field .. ";
    std::cout.flush();
    optimizer.optimizeOrientations(-1);
    optimizer.notify();
    optimizer.wait();
    std::cout << "done. (took " << timeString(timer.reset()) << ")" << std::endl;

    std::map<uint32_t, uint32_t> sing;
    compute_orientation_singularities(mRes, sing, extrinsic, rosy);
    std::cout << "Orientation field has " << sing.size() << " singularities." << std::endl;
    timer.reset();

    std::cout << "Optimizing position field .. ";
    std::cout.flush();
    optimizer.optimizePositions(-1);
    optimizer.notify();
    optimizer.wait();
    std::cout << "done. (took " << timeString(timer.reset()) << ")" << std::endl;

    //std::map<uint32_t, Vector2i> pos_sing;
    //compute_position_singularities(mRes, sing, pos_sing, extrinsic, rosy, posy);
    //cout << "Position field has " << pos_sing.size() << " singularities." << endl;
    //timer.reset();

    optimizer.shutdown();

    Eigen::MatrixXf O_extr, N_extr, Nf_extr;
    std::vector<std::vector<TaggedLink>> adj_extr;
    extract_graph(mRes, extrinsic, rosy, posy, adj_extr, O_extr, N_extr,
        crease_in, crease_out, deterministic);

    Eigen::MatrixXu F_extr;
    extract_faces(adj_extr, O_extr, N_extr, Nf_extr, F_extr, posy,
        mRes.scale(), crease_out, true, pure_quad, bvh, smooth_iter);
    std::cout << "Extraction is done. (total time: " << timeString(timer.reset()) << ")" << std::endl;

    write_mesh(output, F_extr, O_extr, Eigen::MatrixXf(), Nf_extr);
    if (bvh)
        delete bvh;
}

int main()
{
    bool extrinsic = true, dominant = false, align_to_boundaries = true;
    bool fullscreen = false, help = false, deterministic = false, compat = false;
    int rosy = 4, posy = 4, face_count = -1, vertex_count = -1;
    uint32_t knn_points = 10, smooth_iter = 2;
    Float crease_angle = -1, scale = -1;
    bool pure_quad = false;
    std::string infile = "../data/a/result/dense.obj";
    std::string outfile = "out2.obj";
    batch_process(infile, outfile, rosy,posy, scale, face_count, vertex_count, crease_angle, extrinsic, align_to_boundaries, smooth_iter, knn_points, pure_quad, deterministic);



	return 0;
}
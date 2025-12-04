/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once
#include <vector>
#include <memory>
#include <vector>
#include "opencv2/opencv.hpp"
#include "Eigen/Core"
#include "Eigen/Geometry"
#include <assert.h>
namespace currender
{
    template <typename T>
    using Image = cv::Mat_<T>;
    using Image1b = cv::Mat1b;
    using Image3b = cv::Mat3b;
    using Image1w = cv::Mat1w;
    using Image1i = cv::Mat1i;
    using Image1f = cv::Mat1f;
    using Image3f = cv::Mat3f;


    template <typename T>
    inline bool imwrite(const std::string& filename, const T& img,
        const std::vector<int>& params = std::vector<int>()) {
        return cv::imwrite(filename, img, params);
    }


    template <typename T, typename TT>
    inline void Init(Image<T>* image, int width, int height, TT val) {
        if (image->cols == width && image->rows == height) {
            image->setTo(val);
        }
        else {
            if (val == TT(0)) {
                *image = Image<T>::zeros(height, width);
            }
            else {
                *image = Image<T>::ones(height, width) * val;
            }
        }
    }

    template <typename T, typename TT>
    bool ConvertTo(const Image<T>& src, Image<TT>* dst, float scale = 1.0f) {
        src.convertTo(*dst, dst->type(), scale);

        return true;
    }
    struct MeshStats {
        Eigen::Vector3f center;
        Eigen::Vector3f bb_min;
        Eigen::Vector3f bb_max;
    };

    // partial copy of tinyobj::material_t
    struct ObjMaterial {
        std::string name;

        // same to bunny.mtl
        std::array<float, 3> ambient{ 0.117647f, 0.117647f, 0.117647f };   // Ka
        std::array<float, 3> diffuse{ 0.752941f, 0.752941f, 0.752941f };   // Kd
        std::array<float, 3> specular{ 0.752941f, 0.752941f, 0.752941f };  // Ks
        float shininess{ 8.0f };                                           // Ns
        float dissolve{
            1.0f };  // 1 == opaque; 0 == fully transparent, (inverted: Tr = 1 - d)
        // illumination model (see http://www.fileformat.info/format/material/)
        int illum{ 1 };

        std::string diffuse_texname;
        std::string diffuse_texpath;
        Image3b diffuse_tex;

        std::string ToString() const;
    };
    struct Mesh
    {
        std::vector<Eigen::Vector3f> vertices_;
        std::vector<Eigen::Vector3f> vertex_colors_;   // optional, RGB order
        std::vector<Eigen::Vector3i> vertex_indices_;  // face

        std::vector<Eigen::Vector3f> normals_;       // normal per vertex
        std::vector<Eigen::Vector3f> face_normals_;  // normal per face
        std::vector<Eigen::Vector3i> normal_indices_;

        std::vector<Eigen::Vector2f> uv_;
        std::vector<Eigen::Vector3i> uv_indices_;

        std::vector<ObjMaterial> materials_;

        // material_ids_[i]: face index i's material id.
        // This is used to access materials_.
        std::vector<int> material_ids_;

        // face_indices_per_material_[i]: the vector of material i's face indices.
        std::vector<std::vector<int>> face_indices_per_material_;
        MeshStats stats_;

    public:
        Mesh();
        ~Mesh();
        Mesh(const Mesh& src);
        void Clear();

        // get average normal per vertex from face normal
        // caution: this does not work for cube with 8 vertices unless vertices are
        // splitted (24 vertices)
        void CalcNormal();

        void CalcFaceNormal();
        void CalcStats();
        void Rotate(const Eigen::Matrix3f& R);
        void Translate(const Eigen::Vector3f& t);
        void Transform(const Eigen::Matrix3f& R, const Eigen::Vector3f& t);
        void Scale(float scale);
        void Scale(float x_scale, float y_scale, float z_scale);
        const std::vector<Eigen::Vector3f>& vertices() const;
        const std::vector<Eigen::Vector3f>& vertex_colors() const;
        const std::vector<Eigen::Vector3i>& vertex_indices() const;
        const std::vector<Eigen::Vector3f>& normals() const;
        const std::vector<Eigen::Vector3f>& face_normals() const;
        const std::vector<Eigen::Vector3i>& normal_indices() const;
        const std::vector<Eigen::Vector2f>& uv() const;
        const std::vector<Eigen::Vector3i>& uv_indices() const;
        const MeshStats& stats() const;
        const std::vector<int>& material_ids() const;
        const std::vector<ObjMaterial>& materials() const;

        bool set_vertices(const std::vector<Eigen::Vector3f>& vertices);
        bool set_vertex_colors(const std::vector<Eigen::Vector3f>& vertex_colors);
        bool set_vertex_indices(const std::vector<Eigen::Vector3i>& vertex_indices);
        bool set_normals(const std::vector<Eigen::Vector3f>& normals);
        bool set_face_normals(const std::vector<Eigen::Vector3f>& face_normals);
        bool set_normal_indices(const std::vector<Eigen::Vector3i>& normal_indices);
        bool set_uv(const std::vector<Eigen::Vector2f>& uv);
        bool set_uv_indices(const std::vector<Eigen::Vector3i>& uv_indices);
        bool set_material_ids(const std::vector<int>& material_ids);
        bool set_materials(const std::vector<ObjMaterial>& materials);

        bool LoadObj(const std::string& obj_path, const std::string& mtl_dir);
        bool LoadPly(const std::string& ply_path);
        bool WritePly(const std::string& ply_path) const;
        // not const since this will update texture name and path
        bool WriteObj(const std::string& obj_dir, const std::string& obj_basename,
            const std::string& mtl_basename = "", bool write_obj = true,
            bool write_mtl = true, bool write_texture = true);
    };
    // borrow from glm
    // radians
    template <typename genType>
    genType radians(genType degrees) {
        // "'radians' only accept floating-point input"
        assert(std::numeric_limits<genType>::is_iec559);

        return degrees * static_cast<genType>(0.01745329251994329576923690768489);
    }
    // degrees
    template <typename genType>
    genType degrees(genType radians) {
        // "'degrees' only accept floating-point input"
        assert(std::numeric_limits<genType>::is_iec559);
        return radians * static_cast<genType>(57.295779513082320876798154814105);
    }
    class Camera {
    public:
        virtual ~Camera() {}

        virtual int width() const = 0;
        virtual int height() const = 0;
        virtual const Eigen::Affine3d& c2w() const = 0;
        virtual const Eigen::Affine3d& w2c() const = 0;
        virtual void set_size(int width, int height) = 0;
        virtual void set_c2w(const Eigen::Affine3d& c2w) = 0;

        // camera -> image conversion
        virtual void Project(const Eigen::Vector3f& camera_p,
            Eigen::Vector3f* image_p) const = 0;
        virtual void Project(const Eigen::Vector3f& camera_p,
            Eigen::Vector2f* image_p) const = 0;
        virtual void Project(const Eigen::Vector3f& camera_p,
            Eigen::Vector2f* image_p, float* d) const = 0;

        // image -> camera conversion
        // need depth value as input
        virtual void Unproject(const Eigen::Vector3f& image_p,
            Eigen::Vector3f* camera_p) const = 0;
        virtual void Unproject(const Eigen::Vector2f& image_p, float d,
            Eigen::Vector3f* camera_p) const = 0;

        // position emmiting ray
        virtual void org_ray_c(float x, float y, Eigen::Vector3f* org) const = 0;
        virtual void org_ray_w(float x, float y, Eigen::Vector3f* org) const = 0;
        virtual void org_ray_c(int x, int y, Eigen::Vector3f* org) const = 0;
        virtual void org_ray_w(int x, int y, Eigen::Vector3f* org) const = 0;

        // ray direction
        virtual void ray_c(
            float x, float y,
            Eigen::Vector3f* dir) const = 0;  // ray in camera coordinate
        virtual void ray_w(
            float x, float y,
            Eigen::Vector3f* dir) const = 0;  // ray in world coordinate
        virtual void ray_c(int x, int y, Eigen::Vector3f* dir) const = 0;
        virtual void ray_w(int x, int y, Eigen::Vector3f* dir) const = 0;
    };

    // Pinhole camera model with pixel-scale principal point and focal length
    // Widely used in computer vision community as perspective camera model
    // Valid only if FoV is much less than 180 deg.
    class PinholeCamera : public Camera {
        int width_;
        int height_;

        Eigen::Affine3d c2w_;  // camera -> world, sometimes called as "pose"
        Eigen::Affine3d w2c_;

        Eigen::Matrix3f c2w_R_f_;
        Eigen::Vector3f c2w_t_f_;
        Eigen::Vector3f x_direc_, y_direc_, z_direc_;

        Eigen::Matrix3f w2c_R_f_;
        Eigen::Vector3f w2c_t_f_;

        Eigen::Vector2f principal_point_;
        Eigen::Vector2f focal_length_;

        std::vector<Eigen::Vector3f> org_ray_c_table_;
        std::vector<Eigen::Vector3f> org_ray_w_table_;
        std::vector<Eigen::Vector3f> ray_c_table_;
        std::vector<Eigen::Vector3f> ray_w_table_;

        void InitRayTable();

        void set_size_no_raytable_update(int width, int height);
        void set_c2w_no_raytable_update(const Eigen::Affine3d& c2w);
        void set_fov_y_no_raytable_update(float fov_y_deg);

    public:
        PinholeCamera();
        ~PinholeCamera();
        PinholeCamera(int width, int height, float fov_y_deg);
        PinholeCamera(int width, int height, const Eigen::Affine3d& c2w,
            float fov_y_deg);
        PinholeCamera(int width, int height, const Eigen::Affine3d& c2w,
            const Eigen::Vector2f& principal_point,
            const Eigen::Vector2f& focal_length);

        int width() const override;
        int height() const override;
        const Eigen::Affine3d& c2w() const override;
        const Eigen::Affine3d& w2c() const override;
        void set_size(int width, int height) override;
        void set_c2w(const Eigen::Affine3d& c2w) override;

        // FoV (Field of View) in degree interface is provided for convenience
        float fov_x() const;
        float fov_y() const;
        void set_fov_x(float fov_x_deg);
        void set_fov_y(float fov_y_deg);

        // pixel-scale principal point and focal length
        const Eigen::Vector2f& principal_point() const;
        const Eigen::Vector2f& focal_length() const;
        void set_principal_point(const Eigen::Vector2f& principal_point);
        void set_focal_length(const Eigen::Vector2f& focal_length);

        void Project(const Eigen::Vector3f& camera_p,
            Eigen::Vector3f* image_p) const override;
        void Project(const Eigen::Vector3f& camera_p,
            Eigen::Vector2f* image_p) const override;
        void Project(const Eigen::Vector3f& camera_p, Eigen::Vector2f* image_p,
            float* d) const override;
        void Unproject(const Eigen::Vector3f& image_p,
            Eigen::Vector3f* camera_p) const override;
        void Unproject(const Eigen::Vector2f& image_p, float d,
            Eigen::Vector3f* camera_p) const override;
        void org_ray_c(float x, float y, Eigen::Vector3f* org) const override;
        void org_ray_w(float x, float y, Eigen::Vector3f* org) const override;
        void org_ray_c(int x, int y, Eigen::Vector3f* org) const override;
        void org_ray_w(int x, int y, Eigen::Vector3f* org) const override;

        void ray_c(float x, float y, Eigen::Vector3f* dir) const override;
        void ray_w(float x, float y, Eigen::Vector3f* dir) const override;
        void ray_c(int x, int y, Eigen::Vector3f* dir) const override;
        void ray_w(int x, int y, Eigen::Vector3f* dir) const override;
    };

    // Orthographic/orthogonal projection camera with no perspective
    // Image coordinate is translated camera coordinate
    // Different from pinhole camera in particular x and y coordinate in image
    class OrthoCamera : public Camera {
        int width_;
        int height_;

        Eigen::Affine3d c2w_;  // camera -> world, sometimes called as "pose"
        Eigen::Affine3d w2c_;

        Eigen::Matrix3f c2w_R_f_;
        Eigen::Vector3f c2w_t_f_;
        Eigen::Vector3f x_direc_, y_direc_, z_direc_;

        Eigen::Matrix3f w2c_R_f_;
        Eigen::Vector3f w2c_t_f_;

        std::vector<Eigen::Vector3f> org_ray_c_table_;
        std::vector<Eigen::Vector3f> org_ray_w_table_;
        std::vector<Eigen::Vector3f> ray_c_table_;
        std::vector<Eigen::Vector3f> ray_w_table_;

        void InitRayTable();

        void set_size_no_raytable_update(int width, int height);
        void set_c2w_no_raytable_update(const Eigen::Affine3d& c2w);

    public:
        OrthoCamera();
        ~OrthoCamera();
        OrthoCamera(int width, int height);
        OrthoCamera(int width, int height, const Eigen::Affine3d& c2w);

        void set_size(int width, int height) override;
        void set_c2w(const Eigen::Affine3d& c2w) override;

        void Project(const Eigen::Vector3f& camera_p,
            Eigen::Vector3f* image_p) const override;
        void Project(const Eigen::Vector3f& camera_p,
            Eigen::Vector2f* image_p) const override;
        void Project(const Eigen::Vector3f& camera_p, Eigen::Vector2f* image_p,
            float* d) const override;
        void Unproject(const Eigen::Vector3f& image_p,
            Eigen::Vector3f* camera_p) const override;
        void Unproject(const Eigen::Vector2f& image_p, float d,
            Eigen::Vector3f* camera_p) const override;
        void org_ray_c(float x, float y, Eigen::Vector3f* org) const override;
        void org_ray_w(float x, float y, Eigen::Vector3f* org) const override;
        void org_ray_c(int x, int y, Eigen::Vector3f* org) const override;
        void org_ray_w(int x, int y, Eigen::Vector3f* org) const override;

        void ray_c(float x, float y, Eigen::Vector3f* dir) const override;
        void ray_w(float x, float y, Eigen::Vector3f* dir) const override;
        void ray_c(int x, int y, Eigen::Vector3f* dir) const override;
        void ray_w(int x, int y, Eigen::Vector3f* dir) const override;
    };

    void WriteTumFormat(const std::vector<Eigen::Affine3d>& poses,
        const std::string& path);
    bool LoadTumFormat(const std::string& path,
        std::vector<Eigen::Affine3d>* poses);
    bool LoadTumFormat(const std::string& path,
        std::vector<std::pair<int, Eigen::Affine3d>>* poses);

    inline PinholeCamera::PinholeCamera()
        : principal_point_(-1, -1), focal_length_(-1, -1) {
        set_size_no_raytable_update(-1, -1);
        set_c2w_no_raytable_update(Eigen::Affine3d::Identity());
    }

    inline PinholeCamera::~PinholeCamera() {}

    inline int PinholeCamera::width() const { return width_; }

    inline int PinholeCamera::height() const { return height_; }

    inline const Eigen::Affine3d& PinholeCamera::c2w() const { return c2w_; }

    inline const Eigen::Affine3d& PinholeCamera::w2c() const { return w2c_; }

    inline PinholeCamera::PinholeCamera(int width, int height, float fov_y_deg) {
        set_size_no_raytable_update(width, height);

        principal_point_[0] = width_ * 0.5f - 0.5f;
        principal_point_[1] = height_ * 0.5f - 0.5f;

        set_fov_y_no_raytable_update(fov_y_deg);

        set_c2w_no_raytable_update(Eigen::Affine3d::Identity());

        InitRayTable();
    }

    inline PinholeCamera::PinholeCamera(int width, int height,
        const Eigen::Affine3d& c2w,
        float fov_y_deg) {
        set_size_no_raytable_update(width, height);

        set_c2w_no_raytable_update(c2w);

        principal_point_[0] = width_ * 0.5f - 0.5f;
        principal_point_[1] = height_ * 0.5f - 0.5f;

        set_fov_y_no_raytable_update(fov_y_deg);

        InitRayTable();
    }

    inline PinholeCamera::PinholeCamera(int width, int height,
        const Eigen::Affine3d& c2w,
        const Eigen::Vector2f& principal_point,
        const Eigen::Vector2f& focal_length)
        : principal_point_(principal_point), focal_length_(focal_length) {
        set_size_no_raytable_update(width, height);
        set_c2w_no_raytable_update(c2w);
        InitRayTable();
    }

    inline void PinholeCamera::set_size_no_raytable_update(int width, int height) {
        width_ = width;
        height_ = height;
    }

    inline void PinholeCamera::set_c2w_no_raytable_update(
        const Eigen::Affine3d& c2w) {
        c2w_ = c2w;
        w2c_ = c2w_.inverse();

        c2w_R_f_ = c2w_.matrix().block<3, 3>(0, 0).cast<float>();
        c2w_t_f_ = c2w_.matrix().block<3, 1>(0, 3).cast<float>();
        x_direc_ = c2w_.matrix().block<3, 3>(0, 0).col(0).cast<float>();
        y_direc_ = c2w_.matrix().block<3, 3>(0, 0).col(1).cast<float>();
        z_direc_ = c2w_.matrix().block<3, 3>(0, 0).col(2).cast<float>();

        w2c_R_f_ = w2c_.matrix().block<3, 3>(0, 0).cast<float>();
        w2c_t_f_ = w2c_.matrix().block<3, 1>(0, 3).cast<float>();
    }

    inline void PinholeCamera::set_fov_y_no_raytable_update(float fov_y_deg) {
        focal_length_[1] =
            height_ * 0.5f /
            static_cast<float>(std::tan(radians<float>(fov_y_deg) * 0.5));
        focal_length_[0] = focal_length_[1];
    }

    inline void PinholeCamera::set_size(int width, int height) {
        set_size_no_raytable_update(width, height);

        InitRayTable();
    }

    inline void PinholeCamera::set_c2w(const Eigen::Affine3d& c2w) {
        set_c2w_no_raytable_update(c2w);

        InitRayTable();
    }

    inline float PinholeCamera::fov_x() const {
        return degrees<float>(2 * std::atan(width_ * 0.5f / focal_length_[0]));
    }

    inline float PinholeCamera::fov_y() const {
        return degrees<float>(2 * std::atan(height_ * 0.5f / focal_length_[1]));
    }

    inline const Eigen::Vector2f& PinholeCamera::principal_point() const {
        return principal_point_;
    }

    inline const Eigen::Vector2f& PinholeCamera::focal_length() const {
        return focal_length_;
    }

    inline void PinholeCamera::set_principal_point(
        const Eigen::Vector2f& principal_point) {
        principal_point_ = principal_point;
        InitRayTable();
    }

    inline void PinholeCamera::set_focal_length(
        const Eigen::Vector2f& focal_length) {
        focal_length_ = focal_length;
        InitRayTable();
    }

    inline void PinholeCamera::set_fov_x(float fov_x_deg) {
        // same focal length per pixel for x and y
        focal_length_[0] =
            width_ * 0.5f /
            static_cast<float>(std::tan(radians<float>(fov_x_deg) * 0.5));
        focal_length_[1] = focal_length_[0];
        InitRayTable();
    }

    inline void PinholeCamera::set_fov_y(float fov_y_deg) {
        // same focal length per pixel for x and y

        set_fov_y_no_raytable_update(fov_y_deg);

        InitRayTable();
    }

    inline void PinholeCamera::Project(const Eigen::Vector3f& camera_p,
        Eigen::Vector3f* image_p) const {
        (*image_p)[0] =
            focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
        (*image_p)[1] =
            focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
        (*image_p)[2] = camera_p[2];
    }

    inline void PinholeCamera::Project(const Eigen::Vector3f& camera_p,
        Eigen::Vector2f* image_p) const {
        (*image_p)[0] =
            focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
        (*image_p)[1] =
            focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
    }

    inline void PinholeCamera::Project(const Eigen::Vector3f& camera_p,
        Eigen::Vector2f* image_p, float* d) const {
        (*image_p)[0] =
            focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
        (*image_p)[1] =
            focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
        *d = camera_p[2];
    }

    inline void PinholeCamera::Unproject(const Eigen::Vector3f& image_p,
        Eigen::Vector3f* camera_p) const {
        (*camera_p)[0] =
            (image_p[0] - principal_point_[0]) * image_p[2] / focal_length_[0];
        (*camera_p)[1] =
            (image_p[1] - principal_point_[1]) * image_p[2] / focal_length_[1];
        (*camera_p)[2] = image_p[2];
    }

    inline void PinholeCamera::Unproject(const Eigen::Vector2f& image_p, float d,
        Eigen::Vector3f* camera_p) const {
        (*camera_p)[0] = (image_p[0] - principal_point_[0]) * d / focal_length_[0];
        (*camera_p)[1] = (image_p[1] - principal_point_[1]) * d / focal_length_[1];
        (*camera_p)[2] = d;
    }

    inline void PinholeCamera::org_ray_c(float x, float y,
        Eigen::Vector3f* org) const {
        (void)x;
        (void)y;
        (*org)[0] = 0.0f;
        (*org)[1] = 0.0f;
        (*org)[2] = 0.0f;
    }

    inline void PinholeCamera::org_ray_w(float x, float y,
        Eigen::Vector3f* org) const {
        (void)x;
        (void)y;
        *org = c2w_t_f_;
    }

    inline void PinholeCamera::ray_c(float x, float y, Eigen::Vector3f* dir) const {
        (*dir)[0] = (x - principal_point_[0]) / focal_length_[0];
        (*dir)[1] = (y - principal_point_[1]) / focal_length_[1];
        (*dir)[2] = 1.0f;
        dir->normalize();
    }

    inline void PinholeCamera::ray_w(float x, float y, Eigen::Vector3f* dir) const {
        ray_c(x, y, dir);
        *dir = c2w_R_f_ * *dir;
    }

    inline void PinholeCamera::org_ray_c(int x, int y, Eigen::Vector3f* org) const {
        *org = org_ray_c_table_[y * width_ + x];
    }
    inline void PinholeCamera::org_ray_w(int x, int y, Eigen::Vector3f* org) const {
        *org = org_ray_w_table_[y * width_ + x];
    }

    inline void PinholeCamera::ray_c(int x, int y, Eigen::Vector3f* dir) const {
        *dir = ray_c_table_[y * width_ + x];
    }
    inline void PinholeCamera::ray_w(int x, int y, Eigen::Vector3f* dir) const {
        *dir = ray_w_table_[y * width_ + x];
    }

    inline void PinholeCamera::InitRayTable() {
        org_ray_c_table_.resize(width_ * height_);
        org_ray_w_table_.resize(width_ * height_);
        ray_c_table_.resize(width_ * height_);
        ray_w_table_.resize(width_ * height_);

        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                org_ray_c(static_cast<float>(x), static_cast<float>(y),
                    &org_ray_c_table_[y * width_ + x]);
                org_ray_w(static_cast<float>(x), static_cast<float>(y),
                    &org_ray_w_table_[y * width_ + x]);

                ray_c(static_cast<float>(x), static_cast<float>(y),
                    &ray_c_table_[y * width_ + x]);
                ray_w(static_cast<float>(x), static_cast<float>(y),
                    &ray_w_table_[y * width_ + x]);
            }
        }
    }

    inline OrthoCamera::OrthoCamera() {
        set_size_no_raytable_update(-1, -1);
        set_c2w_no_raytable_update(Eigen::Affine3d::Identity());
    }
    inline OrthoCamera::~OrthoCamera() {}
    inline OrthoCamera::OrthoCamera(int width, int height) {
        set_size_no_raytable_update(width, height);
        set_c2w_no_raytable_update(Eigen::Affine3d::Identity());

        InitRayTable();
    }
    inline OrthoCamera::OrthoCamera(int width, int height,
        const Eigen::Affine3d& c2w) {
        set_size_no_raytable_update(width, height);

        set_c2w_no_raytable_update(c2w);

        InitRayTable();
    }

    inline void OrthoCamera::set_size_no_raytable_update(int width, int height) {
        width_ = width;
        height_ = height;
    }

    inline void OrthoCamera::set_c2w_no_raytable_update(
        const Eigen::Affine3d& c2w) {
        c2w_ = c2w;
        w2c_ = c2w_.inverse();

        c2w_R_f_ = c2w_.matrix().block<3, 3>(0, 0).cast<float>();
        c2w_t_f_ = c2w_.matrix().block<3, 1>(0, 3).cast<float>();
        x_direc_ = c2w_.matrix().block<3, 3>(0, 0).col(0).cast<float>();
        y_direc_ = c2w_.matrix().block<3, 3>(0, 0).col(1).cast<float>();
        z_direc_ = c2w_.matrix().block<3, 3>(0, 0).col(2).cast<float>();

        w2c_R_f_ = w2c_.matrix().block<3, 3>(0, 0).cast<float>();
        w2c_t_f_ = w2c_.matrix().block<3, 1>(0, 3).cast<float>();
    }
    inline void OrthoCamera::set_size(int width, int height) {
        set_size_no_raytable_update(width, height);

        InitRayTable();
    }

    inline void OrthoCamera::set_c2w(const Eigen::Affine3d& c2w) {
        set_c2w_no_raytable_update(c2w);

        InitRayTable();
    }

    inline void OrthoCamera::Project(const Eigen::Vector3f& camera_p,
        Eigen::Vector3f* image_p) const {
        *image_p = camera_p;
    }

    inline void OrthoCamera::Project(const Eigen::Vector3f& camera_p,
        Eigen::Vector2f* image_p) const {
        (*image_p)[0] = camera_p[0];
        (*image_p)[1] = camera_p[1];
    }

    inline void OrthoCamera::Project(const Eigen::Vector3f& camera_p,
        Eigen::Vector2f* image_p, float* d) const {
        (*image_p)[0] = camera_p[0];
        (*image_p)[1] = camera_p[1];
        *d = camera_p[2];
    }

    inline void OrthoCamera::Unproject(const Eigen::Vector3f& image_p,
        Eigen::Vector3f* camera_p) const {
        *camera_p = image_p;
    }

    inline void OrthoCamera::Unproject(const Eigen::Vector2f& image_p, float d,
        Eigen::Vector3f* camera_p) const {
        (*camera_p)[0] = image_p[0];
        (*camera_p)[1] = image_p[1];
        (*camera_p)[2] = d;
    }

    inline void OrthoCamera::org_ray_c(float x, float y,
        Eigen::Vector3f* org) const {
        (*org)[0] = x - width_ / 2;
        (*org)[1] = y - height_ / 2;
        (*org)[2] = 0.0f;
    }

    inline void OrthoCamera::org_ray_w(float x, float y,
        Eigen::Vector3f* org) const {
        *org = c2w_t_f_;

        Eigen::Vector3f offset_x = (x - width_ * 0.5f) * x_direc_;
        Eigen::Vector3f offset_y = (y - height_ * 0.5f) * y_direc_;

        *org += offset_x;
        *org += offset_y;
    }

    inline void OrthoCamera::ray_c(float x, float y, Eigen::Vector3f* dir) const {
        (void)x;
        (void)y;
        // parallell ray along with z axis
        (*dir)[0] = 0.0f;
        (*dir)[1] = 0.0f;
        (*dir)[2] = 1.0f;
    }

    inline void OrthoCamera::ray_w(float x, float y, Eigen::Vector3f* dir) const {
        (void)x;
        (void)y;
        // extract z direction of camera pose
        *dir = z_direc_;
    }

    inline void OrthoCamera::org_ray_c(int x, int y, Eigen::Vector3f* org) const {
        *org = org_ray_c_table_[y * width_ + x];
    }

    inline void OrthoCamera::org_ray_w(int x, int y, Eigen::Vector3f* org) const {
        *org = org_ray_w_table_[y * width_ + x];
    }

    inline void OrthoCamera::ray_c(int x, int y, Eigen::Vector3f* dir) const {
        *dir = ray_c_table_[y * width_ + x];
    }
    inline void OrthoCamera::ray_w(int x, int y, Eigen::Vector3f* dir) const {
        *dir = ray_w_table_[y * width_ + x];
    }

    inline void OrthoCamera::InitRayTable() {
        org_ray_c_table_.resize(width_ * height_);
        org_ray_w_table_.resize(width_ * height_);
        ray_c_table_.resize(width_ * height_);
        ray_w_table_.resize(width_ * height_);

        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                org_ray_c(static_cast<float>(x), static_cast<float>(y),
                    &org_ray_c_table_[y * width_ + x]);
                org_ray_w(static_cast<float>(x), static_cast<float>(y),
                    &org_ray_w_table_[y * width_ + x]);

                ray_c(static_cast<float>(x), static_cast<float>(y),
                    &ray_c_table_[y * width_ + x]);
                ray_w(static_cast<float>(x), static_cast<float>(y),
                    &ray_w_table_[y * width_ + x]);
            }
        }
    }


}
namespace currender {
     

    // Diffuse color
    enum class DiffuseColor {
        kNone = 0,     // Default white color
        kTexture = 1,  // From diffuse uv texture
        kVertex = 2    // From vertex color
    };

    // Normal used for shading
    // Also returned as output normal
    enum class ShadingNormal {
        kFace = 0,   // Face normal
        kVertex = 1  // Vertex normal. Maybe average of face normals
    };

    // Diffuse shading
    // Light ray same to viewing ray is used for shading
    enum class DiffuseShading {
        kNone = 0,        // No shading
        kLambertian = 1,  // Lambertian reflectance model
        kOrenNayar =
        2  // Simplified Oren-Nayar reflectatnce model described in wikipedia
           // https://en.wikipedia.org/wiki/Oren%E2%80%93Nayar_reflectance_model
    };

    // Interpolation method in texture uv space
    // Meaningful only if DiffuseColor::kTexture is specified otherwise ignored
    enum class ColorInterpolation {
        kNn = 0,       // Nearest Neigbor
        kBilinear = 1  // Bilinear interpolation
    };

    struct RendererOption {
        DiffuseColor diffuse_color{ DiffuseColor::kNone };
        ColorInterpolation interp{ ColorInterpolation::kBilinear };
        ShadingNormal shading_normal{ ShadingNormal::kVertex };
        DiffuseShading diffuse_shading{ DiffuseShading::kNone };

        float depth_scale{ 1.0f };       // Multiplied to output depth
        bool backface_culling{ true };   // Back-face culling flag
        float oren_nayar_sigma{ 0.3f };  // Oren-Nayar's sigma

        RendererOption() {}
        ~RendererOption() {}
        void CopyTo(RendererOption* dst) const {
            dst->diffuse_color = diffuse_color;
            dst->depth_scale = depth_scale;
            dst->interp = interp;
            dst->shading_normal = shading_normal;
            dst->diffuse_shading = diffuse_shading;
            dst->backface_culling = backface_culling;
        }
    };

    // interface (pure abstract base class with no state or defined methods) for
    // renderer
    class Renderer {
    public:
        virtual ~Renderer() {}

        // Set option
        virtual void set_option(const RendererOption& option) = 0;

        // Set mesh
        virtual void set_mesh(std::shared_ptr<const Mesh> mesh) = 0;

        // Should call after set_mesh() and before Render()
        // Don't modify mesh outside after calling PrepareMesh()
        virtual bool PrepareMesh() = 0;

        // Set camera
        virtual void set_camera(std::shared_ptr<const Camera> camera) = 0;

        // Rendering all images
        // If you don't need some of them, pass nullptr
        virtual bool Render(Image3b* color, Image1f* depth, Image3f* normal,
            Image1b* mask, Image1i* face_id) const = 0;

        // Rendering a image
        virtual bool RenderColor(Image3b* color) const = 0;
        virtual bool RenderDepth(Image1f* depth) const = 0;
        virtual bool RenderNormal(Image3f* normal) const = 0;
        virtual bool RenderMask(Image1b* mask) const = 0;
        virtual bool RenderFaceId(Image1i* face_id) const = 0;

        // These Image1w* depth interfaces are prepared for widely used 16 bit
        // (unsigned short) and mm-scale depth image format
        virtual bool RenderW(Image3b* color, Image1w* depth, Image3f* normal,
            Image1b* mask, Image1i* face_id) const = 0;
        virtual bool RenderDepthW(Image1w* depth) const = 0;
    };

}  // namespace currender

namespace currender {

    class Rasterizer : public Renderer {
        class Impl;
        std::unique_ptr<Impl> pimpl_;

    public:
        Rasterizer();
        ~Rasterizer() override;

        // Set option
        explicit Rasterizer(const RendererOption& option);
        void set_option(const RendererOption& option) override;

        // Set mesh
        void set_mesh(std::shared_ptr<const Mesh> mesh) override;

        // Should call after set_mesh() and before Render()
        // Don't modify mesh outside after calling PrepareMesh()
        bool PrepareMesh() override;

        // Set camera
        void set_camera(std::shared_ptr<const Camera> camera) override;

        // Rendering all images
        // If you don't need some of them, pass nullptr
        bool Render(Image3b* color, Image1f* depth, Image3f* normal, Image1b* mask,
            Image1i* face_id) const override;

        // Rendering a image
        bool RenderColor(Image3b* color) const override;
        bool RenderDepth(Image1f* depth) const override;
        bool RenderNormal(Image3f* normal) const override;
        bool RenderMask(Image1b* mask) const override;
        bool RenderFaceId(Image1i* face_id) const override;

        // These Image1w* depth interfaces are prepared for widely used 16 bit
        // (unsigned short) and mm-scale depth image format
        bool RenderW(Image3b* color, Image1w* depth, Image3f* normal, Image1b* mask,
            Image1i* face_id) const override;
        bool RenderDepthW(Image1w* depth) const override;
    };
    bool ValidateAndInitBeforeRender(bool mesh_initialized,
        std::shared_ptr<const Camera> camera,
        std::shared_ptr<const Mesh> mesh,
        const RendererOption& option, Image3b* color,
        Image1f* depth, Image3f* normal, Image1b* mask,
        Image1i* face_id);
}  // namespace currender

namespace currender {

    struct OrenNayarParam {
    public:
        float sigma{ 0.0f };
        float A{ 0.0f };
        float B{ 0.0f };
        OrenNayarParam();
        explicit OrenNayarParam(float sigma);
        ~OrenNayarParam();
    };

    struct PixelShaderInput {
    public:
        PixelShaderInput() = delete;
        Image3b* color{ nullptr };
        int x;
        int y;
        float u;
        float v;
        uint32_t face_index;
        const Eigen::Vector3f* ray_w{ nullptr };
        const Eigen::Vector3f* light_dir{ nullptr };
        const Eigen::Vector3f* shading_normal{ nullptr };
        const OrenNayarParam* oren_nayar_param{ nullptr };
        std::shared_ptr<const Mesh> mesh{ nullptr };

        PixelShaderInput(Image3b* color, int x, int y, float u, float v,
            uint32_t face_index, const Eigen::Vector3f* ray_w,
            const Eigen::Vector3f* light_dir,
            const Eigen::Vector3f* shading_normal,
            const OrenNayarParam* oren_nayar_param,
            std::shared_ptr<const Mesh> mesh);
        ~PixelShaderInput();
    };

    class DiffuseColorizer {
    public:
        virtual ~DiffuseColorizer() {}
        virtual void Process(const PixelShaderInput& input) const = 0;
    };

    class DiffuseDefaultColorizer : public DiffuseColorizer {
    public:
        DiffuseDefaultColorizer();
        ~DiffuseDefaultColorizer();
        void Process(const PixelShaderInput& input) const override;
    };

    class DiffuseVertexColorColorizer : public DiffuseColorizer {
    public:
        DiffuseVertexColorColorizer();
        ~DiffuseVertexColorColorizer();
        void Process(const PixelShaderInput& input) const override;
    };

    class DiffuseTextureNnColorizer : public DiffuseColorizer {
    public:
        DiffuseTextureNnColorizer();
        ~DiffuseTextureNnColorizer();
        void Process(const PixelShaderInput& input) const override;
    };

    class DiffuseTextureBilinearColorizer : public DiffuseColorizer {
    public:
        DiffuseTextureBilinearColorizer();
        ~DiffuseTextureBilinearColorizer();
        void Process(const PixelShaderInput& input) const override;
    };

    class DiffuseShader {
    public:
        virtual ~DiffuseShader() {}
        virtual void Process(const PixelShaderInput& input) const = 0;
    };

    class DiffuseDefaultShader : public DiffuseShader {
    public:
        DiffuseDefaultShader();
        ~DiffuseDefaultShader();
        void Process(const PixelShaderInput& input) const override;
    };

    class DiffuseLambertianShader : public DiffuseShader {
    public:
        DiffuseLambertianShader();
        ~DiffuseLambertianShader();
        void Process(const PixelShaderInput& input) const override;
    };

    class DiffuseOrenNayarShader : public DiffuseShader {
    public:
        DiffuseOrenNayarShader();
        ~DiffuseOrenNayarShader();
        void Process(const PixelShaderInput& input) const override;
    };

    class PixelShader {
        std::unique_ptr<DiffuseColorizer> diffuse_colorizer_{ nullptr };
        std::unique_ptr<DiffuseShader> diffuse_shader_{ nullptr };
        PixelShader(std::unique_ptr<DiffuseColorizer>&& diffuse_colorizer,
            std::unique_ptr<DiffuseShader>&& diffuse_shader);

    public:
        PixelShader(const PixelShader&) = delete;
        PixelShader& operator=(const PixelShader&) = delete;
        PixelShader(PixelShader&&) = delete;
        PixelShader& operator=(PixelShader&&) = delete;
        friend class PixelShaderFactory;
        PixelShader();
        ~PixelShader();
        void Process(const PixelShaderInput& input) const;
    };

    class PixelShaderFactory {
        PixelShaderFactory();
        ~PixelShaderFactory();

    public:
        static std::unique_ptr<PixelShader> Create(DiffuseColor diffuse_color,
            ColorInterpolation interp,
            DiffuseShading diffuse_shading);
    };

    inline OrenNayarParam::OrenNayarParam() {}
    inline OrenNayarParam::OrenNayarParam(float sigma) : sigma(sigma) {
        assert(0 <= sigma);
        float sigma_2 = sigma * sigma;
        A = 1.0f - (0.5f * sigma_2 / (sigma_2 + 0.33f));
        B = 0.45f * sigma_2 / (sigma_2 + 0.09f);
    }
    inline OrenNayarParam::~OrenNayarParam() {}

    inline PixelShaderInput::~PixelShaderInput() {}
    inline PixelShaderInput::PixelShaderInput(
        Image3b* color, int x, int y, float u, float v, uint32_t face_index,
        const Eigen::Vector3f* ray_w, const Eigen::Vector3f* light_dir,
        const Eigen::Vector3f* shading_normal,
        const OrenNayarParam* oren_nayar_param, std::shared_ptr<const Mesh> mesh)
        : color(color),
        x(x),
        y(y),
        u(u),
        v(v),
        face_index(face_index),
        ray_w(ray_w),
        light_dir(light_dir),
        shading_normal(shading_normal),
        oren_nayar_param(oren_nayar_param),
        mesh(mesh) {}

    inline PixelShaderFactory::PixelShaderFactory() {}

    inline PixelShaderFactory::~PixelShaderFactory() {}

    inline std::unique_ptr<PixelShader> PixelShaderFactory::Create(
        DiffuseColor diffuse_color, ColorInterpolation interp,
        DiffuseShading diffuse_shading) {
        std::unique_ptr<DiffuseColorizer> colorizer;
        std::unique_ptr<DiffuseShader> shader;

        if (diffuse_color == DiffuseColor::kVertex) {
            colorizer.reset(new DiffuseVertexColorColorizer);
        }
        else if (diffuse_color == DiffuseColor::kTexture) {
            if (interp == ColorInterpolation::kNn) {
                colorizer.reset(new DiffuseTextureNnColorizer);
            }
            else if (interp == ColorInterpolation::kBilinear) {
                colorizer.reset(new DiffuseTextureBilinearColorizer);
            }
        }
        else if (diffuse_color == DiffuseColor::kNone) {
            colorizer.reset(new DiffuseDefaultColorizer);
        }
        assert(colorizer);

        if (diffuse_shading == DiffuseShading::kNone) {
            shader.reset(new DiffuseDefaultShader);
        }
        else if (diffuse_shading == DiffuseShading::kLambertian) {
            shader.reset(new DiffuseLambertianShader);
        }
        else if (diffuse_shading == DiffuseShading::kOrenNayar) {
            shader.reset(new DiffuseOrenNayarShader);
        }
        assert(shader);

        return std::unique_ptr<PixelShader>(
            new PixelShader(std::move(colorizer), std::move(shader)));
    }

    inline PixelShader::PixelShader() {}
    inline PixelShader::~PixelShader() {}

    inline PixelShader::PixelShader(
        std::unique_ptr<DiffuseColorizer>&& diffuse_colorizer,
        std::unique_ptr<DiffuseShader>&& diffuse_shader) {
        diffuse_colorizer_ = std::move(diffuse_colorizer);
        diffuse_shader_ = std::move(diffuse_shader);
    }

    inline void PixelShader::Process(const PixelShaderInput& input) const {
        diffuse_colorizer_->Process(input);
        diffuse_shader_->Process(input);
    }

    inline DiffuseDefaultColorizer::DiffuseDefaultColorizer() {}
    inline DiffuseDefaultColorizer::~DiffuseDefaultColorizer() {}
    inline void DiffuseDefaultColorizer::Process(
        const PixelShaderInput& input) const {
        Image3b* color = input.color;
        int x = input.x;
        int y = input.y;

        std::memset(&color->data[color->cols * 3 * y + x * 3], 255,
            sizeof(unsigned char) * 3);
    }

    inline DiffuseVertexColorColorizer::DiffuseVertexColorColorizer() {}
    inline DiffuseVertexColorColorizer::~DiffuseVertexColorColorizer() {}
    inline void DiffuseVertexColorColorizer::Process(
        const PixelShaderInput& input) const {
        Image3b* color = input.color;
        int x = input.x;
        int y = input.y;
        float u = input.u;
        float v = input.v;
        uint32_t face_index = input.face_index;
        std::shared_ptr<const Mesh> mesh = input.mesh;

        const auto& vertex_colors = mesh->vertex_colors();
        const auto& faces = mesh->vertex_indices();
        Eigen::Vector3f interp_color;
        // barycentric interpolation of vertex color
        interp_color = (1.0f - u - v) * vertex_colors[faces[face_index][0]] +
            u * vertex_colors[faces[face_index][1]] +
            v * vertex_colors[faces[face_index][2]];

        cv::Vec3b& c = color->at<cv::Vec3b>(y, x);
        for (int k = 0; k < 3; k++) {
            c[k] = static_cast<unsigned char>(interp_color[k]);
        }
    }

    inline DiffuseTextureNnColorizer::DiffuseTextureNnColorizer() {}
    inline DiffuseTextureNnColorizer::~DiffuseTextureNnColorizer() {}
    inline void DiffuseTextureNnColorizer::Process(
        const PixelShaderInput& input) const {
        Image3b* color = input.color;
        int x = input.x;
        int y = input.y;
        float u = input.u;
        float v = input.v;
        uint32_t face_index = input.face_index;
        std::shared_ptr<const Mesh> mesh = input.mesh;

        const auto& uv = mesh->uv();
        const auto& uv_indices = mesh->uv_indices();
        int material_index = mesh->material_ids()[face_index];
        const auto& diffuse_texture = mesh->materials()[material_index].diffuse_tex;

        Eigen::Vector3f interp_color;
        // barycentric interpolation of uv
        Eigen::Vector2f interp_uv = (1.0f - u - v) * uv[uv_indices[face_index][0]] +
            u * uv[uv_indices[face_index][1]] +
            v * uv[uv_indices[face_index][2]];
        float f_tex_pos[2];
        f_tex_pos[0] = interp_uv[0] * (diffuse_texture.cols - 1);
        f_tex_pos[1] = (1.0f - interp_uv[1]) * (diffuse_texture.rows - 1);

        int tex_pos[2] = { 0, 0 };
        // get nearest integer index by round
        tex_pos[0] = static_cast<int>(std::round(f_tex_pos[0]));
        tex_pos[1] = static_cast<int>(std::round(f_tex_pos[1]));

        const cv::Vec3b& dt = diffuse_texture.at<cv::Vec3b>(tex_pos[1], tex_pos[0]);
        for (int k = 0; k < 3; k++) {
            interp_color[k] = dt[k];
        }

        cv::Vec3b& c = color->at<cv::Vec3b>(y, x);
        for (int k = 0; k < 3; k++) {
            c[k] = static_cast<unsigned char>(interp_color[k]);
        }
    }

    inline DiffuseTextureBilinearColorizer::DiffuseTextureBilinearColorizer() {}
    inline DiffuseTextureBilinearColorizer::~DiffuseTextureBilinearColorizer() {}
    inline void DiffuseTextureBilinearColorizer::Process(
        const PixelShaderInput& input) const {
        Image3b* color = input.color;
        int x = input.x;
        int y = input.y;
        float u = input.u;
        float v = input.v;
        uint32_t face_index = input.face_index;
        std::shared_ptr<const Mesh> mesh = input.mesh;

        const auto& uv = mesh->uv();
        const auto& uv_indices = mesh->uv_indices();
        int material_index = mesh->material_ids()[face_index];
        const auto& diffuse_texture = mesh->materials()[material_index].diffuse_tex;

        Eigen::Vector3f interp_color;

        // barycentric interpolation of uv
        Eigen::Vector2f interp_uv = (1.0f - u - v) * uv[uv_indices[face_index][0]] +
            u * uv[uv_indices[face_index][1]] +
            v * uv[uv_indices[face_index][2]];
        float f_tex_pos[2];
        f_tex_pos[0] = interp_uv[0] * (diffuse_texture.cols - 1);
        f_tex_pos[1] = (1.0f - interp_uv[1]) * (diffuse_texture.rows - 1);

        int tex_pos_min[2] = { 0, 0 };
        int tex_pos_max[2] = { 0, 0 };
        tex_pos_min[0] = static_cast<int>(std::floor(f_tex_pos[0]));
        tex_pos_min[1] = static_cast<int>(std::floor(f_tex_pos[1]));
        tex_pos_max[0] = tex_pos_min[0] + 1;
        tex_pos_max[1] = tex_pos_min[1] + 1;

        float local_u = f_tex_pos[0] - tex_pos_min[0];
        float local_v = f_tex_pos[1] - tex_pos_min[1];

        const cv::Vec3b& dt_minmin =
            diffuse_texture.at<cv::Vec3b>(tex_pos_min[1], tex_pos_min[0]);
        const cv::Vec3b& dt_maxmin =
            diffuse_texture.at<cv::Vec3b>(tex_pos_min[1], tex_pos_max[0]);
        const cv::Vec3b& dt_minmax =
            diffuse_texture.at<cv::Vec3b>(tex_pos_max[1], tex_pos_min[0]);
        const cv::Vec3b& dt_maxmax =
            diffuse_texture.at<cv::Vec3b>(tex_pos_max[1], tex_pos_max[0]);
        for (int k = 0; k < 3; k++) {
            // bilinear interpolation of pixel color
            interp_color[k] = (1.0f - local_u) * (1.0f - local_v) * dt_minmin[k] +
                local_u * (1.0f - local_v) * dt_maxmin[k] +
                (1.0f - local_u) * local_v * dt_minmax[k] +
                local_u * local_v * dt_maxmax[k];

            assert(0.0f <= interp_color[k] && interp_color[k] <= 255.0f);
        }

        cv::Vec3b& c = color->at<cv::Vec3b>(y, x);
        for (int k = 0; k < 3; k++) {
            c[k] = static_cast<unsigned char>(interp_color[k]);
        }
    }

    inline DiffuseDefaultShader::DiffuseDefaultShader() {}
    inline DiffuseDefaultShader::~DiffuseDefaultShader() {}
    inline void DiffuseDefaultShader::Process(const PixelShaderInput& input) const {
        // do nothing.
        (void)input;
    }

    inline DiffuseLambertianShader::DiffuseLambertianShader() {}
    inline DiffuseLambertianShader::~DiffuseLambertianShader() {}
    inline void DiffuseLambertianShader::Process(
        const PixelShaderInput& input) const {
        Image3b* color = input.color;
        int x = input.x;
        int y = input.y;

        // dot product of normal and inverse light direction
        float coeff = -input.light_dir->dot(*input.shading_normal);

        // if negative (may happen at back-face or occluding boundary), bound to 0
        if (coeff < 0.0f) {
            coeff = 0.0f;
        }

        cv::Vec3b& c = color->at<cv::Vec3b>(y, x);
        for (int k = 0; k < 3; k++) {
            c[k] = static_cast<uint8_t>(coeff * c[k]);
        }
    }

    inline DiffuseOrenNayarShader::DiffuseOrenNayarShader() {}
    inline DiffuseOrenNayarShader::~DiffuseOrenNayarShader() {}
    inline void DiffuseOrenNayarShader::Process(
        const PixelShaderInput& input) const {
        // angle against normal
        float dot_light = -input.light_dir->dot(*input.shading_normal);
        float theta_i = std::acos(dot_light);
        float dot_ray = -input.ray_w->dot(*input.shading_normal);
        float theta_r = std::acos(dot_ray);

        // angle against binormal (perpendicular to normal)
        Eigen::Vector3f binormal_light =
            -*input.shading_normal * dot_light - *input.light_dir;
        Eigen::Vector3f binormal_ray =
            -*input.shading_normal * dot_light - *input.ray_w;
        float phi_diff_cos = std::max(0.0f, binormal_light.dot(binormal_ray));

        float alpha = std::max(theta_i, theta_r);
        float beta = std::min(theta_i, theta_r);

        float A = input.oren_nayar_param->A;
        float B = input.oren_nayar_param->B;
        float coeff = std::max(0.0f, dot_light) *
            (A + (B * phi_diff_cos * std::sin(alpha) * std::tan(beta)));

        Image3b* color = input.color;
        int x = input.x;
        int y = input.y;
        cv::Vec3b& c = color->at<cv::Vec3b>(y, x);
        for (int k = 0; k < 3; k++) {
            c[k] = static_cast<uint8_t>(coeff * c[k]);
        }
    }

}  // namespace currender

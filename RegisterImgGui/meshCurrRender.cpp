/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */
#include <numeric>
#include <filesystem>
#include <memory>
#include <fstream>
#include <list>
#include "meshCurrRender.h"
#include "Eigen/Core"
#include <cassert>
#include "log.h"
#include "igl/per_face_normals.h"
#include "igl/per_vertex_normals.h"
#include "igl/centroid.h"
//#include "src/pixel_shader.h"
//#include "src/util_private.h"
 

namespace currender {
    template <typename T>
    void Argsort(const std::vector<T>& data, std::vector<size_t>* indices) {
        indices->resize(data.size());
        std::iota(indices->begin(), indices->end(), 0);
        std::sort(indices->begin(), indices->end(),
            [&data](size_t i1, size_t i2) { return data[i1] < data[i2]; });
    }
    inline float EdgeFunction(const Eigen::Vector3f& a, const Eigen::Vector3f& b,
        const Eigen::Vector3f& c) {
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]);
    }
    template <typename T>
    bool isEmpty(const Eigen::MatrixBase<T>& mat) {
        if (mat.rows() == 0 || mat.cols() == 0) {
            return true;
        }
        return false;
    }
    template <typename T>
    void CopyVec(const std::vector<T>& src, std::vector<T>* dst) {
        dst->clear();
        std::copy(src.begin(), src.end(), std::back_inserter(*dst));
    }
    std::vector<std::string> Split(const std::string& s, char delim) {
        std::vector<std::string> elems;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) {
            if (!item.empty()) {
                elems.push_back(item);
            }
        }
        return elems;
    }


    bool readFromSimpleObj(const std::filesystem::path& path,
        Eigen::MatrixX3f& pts,
        Eigen::MatrixX3i& faces)
    {
        std::list<cv::Point3f>pts_;
        std::list<cv::Point3i>faces_;
        std::fstream fin(path, std::ios::in);
        std::string aline;
        std::string flag = "";
        while (std::getline(fin, aline))
        {
            if (aline.length() < 4)continue;
            if (aline[0] == '#' && aline[1] == ' ')continue;

            if (aline[0] == 'v' && aline[1] == ' ')
            {
                std::stringstream ss(aline);
                float x, y, z;
                ss >> flag >> x >> y >> z;
                pts_.emplace_back(x, y, z);
            } 
            else if (aline[0] == 'f' && aline[1] == ' ')
            {
                std::stringstream ss(aline);
                int face_x, face_y, face_z;
                ss >> flag >> face_x >> face_y >> face_z;
                face_x -= 1; face_y -= 1; face_z -= 1;
                faces_.emplace_back(face_x, face_y, face_z);
            } 
        }
        fin.close();
        if (pts_.size() == 0 || faces_.size() == 0)
        {
            LOG_ERR_OUT << "pts_.size=0 OR faces_.size=0: " << pts_.size() << "; " << faces_.size();
            return false;
        }

        pts = Eigen::MatrixX3f(pts_.size(), 3);
        faces = Eigen::MatrixX3i(faces_.size(), 3);
        int i = 0;
        for (const auto& d : pts_)
        {
            pts(i, 0) = d.x;
            pts(i, 1) = d.y;
            pts(i, 2) = d.z;
            i += 1;
        }
        i = 0;
        for (const auto& d : faces_)
        {
            faces(i, 0) = d.x;
            faces(i, 1) = d.y;
            faces(i, 2) = d.z;
            i += 1;
        } 
        return true;
    }


}  // namespace

namespace currender {
std::string ObjMaterial::ToString() const {
    std::stringstream ss;

    ss << "newmtl " << name << '\n'
        << "Ka " << ambient[0] << " " << ambient[1] << " " << ambient[2] << '\n'
        << "Kd " << diffuse[0] << " " << diffuse[1] << " " << diffuse[2] << '\n'
        << "Ks " << specular[0] << " " << specular[1] << " " << specular[2] << '\n'
        << "Tr " << 1.0f - dissolve << '\n'
        << "illum " << illum << '\n'
        << "Ns " << shininess << '\n';
    if (diffuse_texname.length()>0) {
        ss << "map_Kd " << diffuse_texname << "\n";
    }
    ss.flush();

    return ss.str();
}
}
//mesh
namespace currender {

    Mesh::Mesh() {}
    Mesh::Mesh(const Mesh& src) {
        vertices_ = src.vertices_;
        vertex_colors_ = src.vertex_colors_;
        vertex_indices_ = src.vertex_indices_;        
        normals_ = src.normals_;
        face_normals_ = src.face_normals_;
        normal_indices_ = src.normal_indices_;        
        uv_ = src.uv_;
        uv_indices_ = src.uv_indices_;        
        materials_ = src.materials_;
        material_ids_ = src.material_ids_;
        face_indices_per_material_ = src.face_indices_per_material_;
        stats_ = src.stats_;
    }
    Mesh::~Mesh() {}

    void Mesh::Rotate(const Eigen::Matrix3f& R) 
    {//vertices_ normals_ face_normals_ is Eigen::MatrixX3f 
        Eigen::Matrix3f R_T = R.transpose();
        vertices_ = vertices_ * R_T;
        normals_ = normals_ * R_T;
        face_normals_ = face_normals_ * R_T; 
        CalcStats();
    }

    void Mesh::Translate(const Eigen::Vector3f& t) {
        vertices_ = vertices_.colwise() + t;
        CalcStats();
    }

    void Mesh::Transform(const Eigen::Matrix3f& R, const Eigen::Vector3f& t) {
        Rotate(R);
        Translate(t);
    }

    void Mesh::Scale(float scale) { Scale(scale, scale, scale); }

    void Mesh::Scale(float x_scale, float y_scale, float z_scale) {
        //bailing vertices_ = vertices_.colwise() * Eigen::Vector3f(x_scale, y_scale, z_scale);
    }

    void Mesh::Clear() {
        vertices_.conservativeResize(0,0);
        vertex_colors_.conservativeResize(0, 0);
        vertex_indices_.conservativeResize(0, 0);  // face
        normals_.conservativeResize(0, 0);
        normal_indices_.conservativeResize(0, 0);
        uv_.conservativeResize(0, 0);
        uv_indices_.conservativeResize(0, 0);
        materials_.clear();
        material_ids_.clear();
        face_indices_per_material_.clear();
    }
    void Mesh::CalcNormal() 
    {
        CalcFaceNormal();
        igl::per_vertex_normals(vertices_, vertex_indices_, normals_);
    }
    void Mesh::CalcStats() {
        if (isEmpty(vertex_indices_)) {
            return;
        }
        stats_.bb_min = vertices_.colwise().minCoeff();
        stats_.bb_max = vertices_.colwise().maxCoeff();
        igl::centroid(vertices_, vertex_indices_, stats_.center);
    }
    void Mesh::CalcFaceNormal() 
    {
        igl::per_face_normals(vertices_, vertex_indices_, face_normals_);
    }
    const std::vector<ObjMaterial>& Mesh::materials() const { return materials_; }
    const std::vector<int>& Mesh::material_ids() const { return material_ids_; }

    const Eigen::MatrixX3f& Mesh::vertices() const { return vertices_; }
    const Eigen::MatrixX3f& Mesh::vertex_colors() const {
        return vertex_colors_;
    }
    const Eigen::MatrixX3i& Mesh::vertex_indices() const {
        return vertex_indices_;
    }

    const Eigen::MatrixX3f& Mesh::normals() const { return normals_; }
    const Eigen::MatrixX3f& Mesh::face_normals() const {
        return face_normals_;
    }
    const Eigen::MatrixX3i& Mesh::normal_indices() const {
        return normal_indices_;
    }

    const Eigen::MatrixX2f& Mesh::uv() const { return uv_; }
    const Eigen::MatrixX3i& Mesh::uv_indices() const {
        return uv_indices_;
    }

    bool Mesh::set_vertices(const Eigen::MatrixX3f& vertices) {
        vertices_ = vertices;
        return true;
    }

    bool Mesh::set_vertex_colors(const Eigen::MatrixX3f& vertex_colors) {
        vertex_colors_ = vertex_colors;
        return true;
    }

    bool Mesh::set_vertex_indices(const Eigen::MatrixX3i& vertex_indices) {
        vertex_indices_ = vertex_indices;
        return true;
    }

    bool Mesh::set_normals(const Eigen::MatrixX3f& normals) {
        normals_ = normals;
        return true;
    }

    bool Mesh::set_face_normals(const Eigen::MatrixX3f& face_normals) {
        face_normals_ = face_normals;
        return true;
    }

    bool Mesh::set_normal_indices(const Eigen::MatrixX3i& normal_indices) {
        normal_indices_ = normal_indices;
        return true;
    }

    bool Mesh::set_uv(const Eigen::MatrixX2f& uv) {
        uv_ = uv;
        return true;
    }

    bool Mesh::set_uv_indices(const Eigen::MatrixX3i& uv_indices) {
        uv_indices_ = uv_indices;
        return true;
    }

    
    bool Mesh::set_material_ids(const std::vector<int>& material_ids) {
        int max_id = *std::max_element(material_ids.begin(), material_ids.end());
        if (max_id < 0) {
            LOG_ERR_OUT<<("material id must be positive\n");
            return false;
        }

        CopyVec(material_ids, &material_ids_);

        face_indices_per_material_.resize(max_id + 1);
        for (int i = 0; i < static_cast<int>(material_ids_.size()); i++) {
            face_indices_per_material_[material_ids_[i]].push_back(i);
        }

        return true;
    }

    bool Mesh::set_materials(const std::vector<ObjMaterial>& materials) {
        CopyVec(materials, &materials_);
        return true;
    }
    const MeshStats& Mesh::stats() const { return stats_; }
    
    bool WriteMtl(const std::string& path,
        const std::vector<ObjMaterial>& materials) {
        std::ofstream ofs(path);
        if (ofs.fail()) {
            LOG_ERR_OUT<<("couldn't open mtl path: %s\n", path.c_str());
            return false;
        }

        for (size_t i = 0; i < materials.size(); i++) {
            const ObjMaterial& material = materials[i];
            ofs << material.ToString();
            if (i != materials.size() - 1) {
                ofs << '\n';
            }
        }
        ofs.close();

        return true;
    }

    bool WriteTexture(const std::vector<ObjMaterial>& materials) {
        // write texture
        bool ret{ true };
        for (size_t i = 0; i < materials.size(); i++) {
            const ObjMaterial& material = materials[i];
            bool ret_write = imwrite(material.diffuse_texpath, material.diffuse_tex);
            if (ret) {
                ret = ret_write;
            }
        }

        return ret;
    }
    bool Mesh::LoadPly(const std::string& ply_path) {
        std::ifstream ifs(ply_path);
        std::string str;
        if (ifs.fail()) {
            LOG_ERR_OUT<<("couldn't open ply: %s\n", ply_path.c_str());
            return false;
        }

        getline(ifs, str);
        if (str != "ply") {
            LOG_ERR_OUT<<("ply first line is wrong: %s\n", str.c_str());
            return false;
        }
        getline(ifs, str);
        if (str.find("ascii") == std::string::npos) {
            LOG_ERR_OUT<<("only ascii ply is supported: %s\n", str.c_str());
            return false;
        }

        bool ret = false;
        std::int64_t vertex_num = 0;
        while (getline(ifs, str)) {
            if (str.find("element vertex") != std::string::npos) {
                std::vector<std::string> splitted = Split(str, ' ');
                if (splitted.size() == 3) {
                    vertex_num = std::atol(splitted[2].c_str());
                    ret = true;
                    break;
                }
            }
        }
        if (!ret) {
            LOG_ERR_OUT<<("couldn't find element vertex\n");
            return false;
        }
        if (vertex_num > std::numeric_limits<int>::max()) {
            LOG_ERR_OUT<<("The number of vertices exceeds the maximum: %d\n",
                std::numeric_limits<int>::max());
            return false;
        }

        ret = false;
        std::int64_t face_num = 0;
        while (getline(ifs, str)) {
            if (str.find("element face") != std::string::npos) {
                std::vector<std::string> splitted = Split(str, ' ');
                if (splitted.size() == 3) {
                    face_num = std::atol(splitted[2].c_str());
                    ret = true;
                    break;
                }
            }
        }
        if (!ret) {
            LOG_ERR_OUT<<("couldn't find element face\n");
            return false;
        }
        if (face_num > std::numeric_limits<int>::max()) {
            LOG_ERR_OUT<<("The number of faces exceeds the maximum: %d\n",
                std::numeric_limits<int>::max());
            return false;
        }

        while (getline(ifs, str)) {
            if (str.find("end_header") != std::string::npos) {
                break;
            }
        }

        vertices_.resize(vertex_num,3);
        int vertex_count = 0;
        while (getline(ifs, str)) {
            std::vector<std::string> splitted = Split(str, ' ');
            vertices_(vertex_count,0) =
                static_cast<float>(std::atof(splitted[0].c_str()));
            vertices_(vertex_count,1) =
                static_cast<float>(std::atof(splitted[1].c_str()));
            vertices_(vertex_count,2) =
                static_cast<float>(std::atof(splitted[2].c_str()));
            vertex_count++;
            if (vertex_count >= vertex_num) {
                break;
            }
        }

        vertex_indices_.resize(face_num,3);
        int face_count = 0;
        while (getline(ifs, str)) {
            std::vector<std::string> splitted = Split(str, ' ');
            vertex_indices_(face_count,0) = std::atoi(splitted[1].c_str());
            vertex_indices_(face_count,1) = std::atoi(splitted[2].c_str());
            vertex_indices_(face_count,2) = std::atoi(splitted[3].c_str());

            face_count++;
            if (face_count >= face_num) {
                break;
            }
        }

        ifs.close();

        CalcNormal();

        CalcStats();

        return true;
    }

    bool Mesh::WritePly(const std::string& ply_path) const {
        std::ofstream ofs(ply_path);
        std::string str;
        if (ofs.fail()) {
            LOG_ERR_OUT<<("couldn't open ply: %s\n", ply_path.c_str());
            return false;
        }
        
        bool has_vertex_normal = !isEmpty(normals_);
        if (has_vertex_normal) {
            assert(vertices_.size() == normals_.size());
        }
        bool has_vertex_color = !isEmpty(vertex_colors_);
        if (has_vertex_color) {
            assert(vertices_.size() == vertex_colors_.size());
        }

        ofs << "ply"
            << "\n";
        ofs << "format ascii 1.0"
            << "\n";
        ofs << "element vertex " + std::to_string(vertices_.size()) << "\n";
        ofs << "property float x\n"
            "property float y\n"
            "property float z\n";
        if (has_vertex_normal) {
            ofs << "property float nx\n"
                "property float ny\n"
                "property float nz\n";
        }
        if (has_vertex_color) {
            ofs << "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "property uchar alpha\n";
        }
        ofs << "element face " + std::to_string(vertex_indices_.size()) << "\n";
        ofs << "property list uchar int vertex_indices"
            << "\n";
        ofs << "end_header"
            << "\n";

        for (size_t i = 0; i < vertices_.size(); i++) {
            ofs << vertices_(i,0) << " " << vertices_(i,1) << " " << vertices_(i,2)
                << " ";
            if (has_vertex_normal) {
                ofs << normals_(i,0) << " " << normals_(i,1) << " " << normals_(i,2)
                    << " ";
            }
            if (has_vertex_color) {
                ofs << static_cast<int>(std::round(vertex_colors_(i,0))) << " "
                    << static_cast<int>(std::round(vertex_colors_(i,1))) << " "
                    << static_cast<int>(std::round(vertex_colors_(i,2))) << " 255 ";
            }
            ofs << "\n";
        }

        for (size_t i = 0; i < vertex_indices_.size(); i++) {
            ofs << "3 " << vertex_indices_(i,0) << " " << vertex_indices_(i,1) << " "
                << vertex_indices_(i,2) << " "
                << "\n";
        }

        ofs.close();

        return true;
    }

    bool Mesh::WriteObj(const std::string& obj_dir, const std::string& obj_basename,
        const std::string& mtl_basename, bool write_obj,
        bool write_mtl, bool write_texture) {
        bool ret{ true };
        std::string mtl_name = mtl_basename + ".mtl";
        if (mtl_basename.empty()) {
            mtl_name = obj_basename + ".mtl";
        }
        std::string mtl_path = obj_dir + "/" + mtl_name;

        std::string obj_path = obj_dir + "/" + obj_basename + ".obj";

        // write obj
        if (write_obj) {
            std::ofstream ofs(obj_path);
            if (ofs.fail()) {
                LOG_ERR_OUT<<("couldn't open obj path: %s\n", obj_path.c_str());
                return false;
            }

            ofs << "mtllib " << mtl_name << "\n"
                << "\n";
            // vertices
            for (int i = 0; i < vertices_.rows(); i++)
            {
                ofs << "v " << vertices_(i,0) << " " << vertices_(i, 1) << " " << vertices_(i, 2) << " 1.0\n";
            }

            // uv
            for (int i = 0; i < uv_.rows(); i++)
            {
                ofs << "vt " << uv_(i, 0) << " " << uv_(i, 1) << " 0\n";
            } 

            // vertex normals
            for (int i = 0; i < normals_.rows(); i++)
            {
                ofs << "vn " << normals_(i, 0) << " " << normals_(i, 1) << " " << normals_(i, 2) << "\n";
            }

            // indices by material (group)
            // CAUTION: This breaks original face indices
            
            bool write_uv_indices = !isEmpty(uv_indices_);
            bool write_normal_indices = !isEmpty(normal_indices_);
            for (size_t k = 0; k < face_indices_per_material_.size(); k++) {
                ofs << "usemtl " << materials_[k].name << "\n";
                for (size_t i = 0; i < face_indices_per_material_[k].size(); i++) {
                    int f_idx = face_indices_per_material_[k][i];
                    ofs << "f";
                    for (int j = 0; j < 3; j++) {
                        ofs << " " << std::to_string(vertex_indices_(f_idx,j) + 1);
                        if (!write_uv_indices && !write_normal_indices) {
                            continue;
                        }
                        ofs << "/";
                        if (write_uv_indices) {
                            ofs << std::to_string(uv_indices_(f_idx,j) + 1);
                        }
                        if (write_normal_indices) {
                            ofs << "/" << std::to_string(normal_indices_(f_idx,j) + 1);
                        }
                    }
                    ofs << "\n";
                }
            }
            ofs.close();
        }

        // update texture path
        for (auto& material : materials_) {
            if (material.diffuse_texname.empty()) {
                // default name
                material.diffuse_texname = obj_basename + ".png";
            }

            // update path
            material.diffuse_texpath = obj_dir + "/" + material.diffuse_texname;
        }

        // write mtl
        if (write_mtl) {
            ret = WriteMtl(mtl_path, materials_);
        }

        if (write_texture) {
            ret = WriteTexture(materials_);
        }

        return ret;
    }
    
}

namespace currender {

    // Rasterizer::Impl implementation
    class Rasterizer::Impl {
        bool mesh_initialized_{ false };
        std::shared_ptr<const Camera> camera_{ nullptr };
        std::shared_ptr<const Mesh> mesh_{ nullptr };
        RendererOption option_;

    public:
        Impl();
        ~Impl();

        explicit Impl(const RendererOption& option);
        void set_option(const RendererOption& option);

        void set_mesh(std::shared_ptr<const Mesh> mesh);

        bool PrepareMesh();

        void set_camera(std::shared_ptr<const Camera> camera);

        bool Render(Image3b* color, Image1f* depth, Image3f* normal, Image1b* mask,
            Image1i* face_id) const;

        bool RenderColor(Image3b* color) const;
        bool RenderDepth(Image1f* depth) const;
        bool RenderNormal(Image3f* normal) const;
        bool RenderMask(Image1b* mask) const;
        bool RenderFaceId(Image1i* face_id) const;

        bool RenderW(Image3b* color, Image1w* depth, Image3f* normal, Image1b* mask,
            Image1i* face_id) const;
        bool RenderDepthW(Image1w* depth) const;
    };

    Rasterizer::Impl::Impl() {}
    Rasterizer::Impl::~Impl() {}

    Rasterizer::Impl::Impl(const RendererOption& option) { set_option(option); }

    void Rasterizer::Impl::set_option(const RendererOption& option) {
        option.CopyTo(&option_);
    }

    void Rasterizer::Impl::set_mesh(std::shared_ptr<const Mesh> mesh) {
        mesh_initialized_ = false;
        mesh_ = mesh;
        if (isEmpty(mesh_->face_normals())) {
            LOG_WARN_OUT<<("face normal is empty. culling and shading may not work\n");
        }
        if (isEmpty(mesh_->normals())) {
            LOG_WARN_OUT << ("vertex normal is empty. shading may not work\n");
        }
    }

    bool Rasterizer::Impl::PrepareMesh() {
        if (mesh_ == nullptr) {
            LOG_ERR_OUT<<("mesh has not been set\n");
            return false;
        }
        mesh_initialized_ = true;
        return true;
    }

    void Rasterizer::Impl::set_camera(std::shared_ptr<const Camera> camera) {
        camera_ = camera;
    }

    bool Rasterizer::Impl::Render(Image3b* color, Image1f* depth, Image3f* normal,
        Image1b* mask, Image1i* face_id) const {
        if (!ValidateAndInitBeforeRender(mesh_initialized_, camera_, mesh_, option_,
            color, depth, normal, mask, face_id)) {
            return false;
        }

        // make pixel shader
        std::unique_ptr<PixelShader> pixel_shader = PixelShaderFactory::Create(
            option_.diffuse_color, option_.interp, option_.diffuse_shading);

        OrenNayarParam oren_nayar_param(option_.oren_nayar_sigma);

        const Eigen::Matrix3f w2c_R = camera_->w2c().rotation().cast<float>();
        const Eigen::Vector3f w2c_t = camera_->w2c().translation().cast<float>();
        const Eigen::Matrix3f w2c_R_T = w2c_R.transpose();
 

        // project face to 2d (fully parallel)
        //std::vector<Eigen::Vector3f> camera_vertices(mesh_->vertices().size());
        //std::vector<Eigen::Vector3f> camera_normals(mesh_->vertices().size());
        //std::vector<float> camera_depth_list(mesh_->vertices().size());
        Eigen::MatrixX3f  image_vertices(mesh_->vertices().rows(),3);


        Eigen::MatrixX3f camera_vertices = mesh_->vertices() * w2c_R_T;
        camera_vertices = camera_vertices.rowwise() + w2c_t.transpose();
        Eigen::MatrixX3f camera_normals = (mesh_->normals() * w2c_R_T);
        Eigen::VectorXf camera_depth_list = camera_vertices.col(2);
        camera_->Project(camera_vertices, image_vertices); 

        Image1f depth_internal;
        Image1f* depth_{ depth };
        if (depth_ == nullptr) {
            depth_ = &depth_internal;
        }
        Init(depth_, camera_->width(), camera_->height(), 0.0f);

        Image1i face_id_internal;
        Image1i* face_id_{ face_id };
        if (face_id_ == nullptr) {
            face_id_ = &face_id_internal;
        }
        Init(face_id_, camera_->width(), camera_->height(), -1);

        // 255: backface, 0:frontface
        Image1b backface_image;
        Init(&backface_image, camera_->width(), camera_->height(),
            static_cast<unsigned char>(0));

        // 0:(1 - u - v), 1:u, 2:v
        Image3f weight_image;
        Init(&weight_image, camera_->width(), camera_->height(), 0.0f);

        // make face id image by z-buffer method
        for (int i = 0; i < static_cast<int>(mesh_->vertex_indices().size()); i++) {
            const Eigen::Vector3i& face = mesh_->vertex_indices().row(i);
            const Eigen::Vector3f& v0_i = image_vertices.row(face[0]);
            const Eigen::Vector3f& v1_i = image_vertices.row(face[1]);
            const Eigen::Vector3f& v2_i = image_vertices.row(face[2]);

            // skip if a vertex is back of the camera
            // todo: add near and far plane
            if (v0_i.z() < 0.0f || v1_i.z() < 0.0f || v2_i.z() < 0.0f) {
                continue;
            }

            float xmin = std::min({ v0_i.x(), v1_i.x(), v2_i.x() });
            float ymin = std::min({ v0_i.y(), v1_i.y(), v2_i.y() });
            float xmax = std::max({ v0_i.x(), v1_i.x(), v2_i.x() });
            float ymax = std::max({ v0_i.y(), v1_i.y(), v2_i.y() });

            // the triangle is out of screen
            if (xmin > camera_->width() - 1 || xmax < 0 ||
                ymin > camera_->height() - 1 || ymax < 0) {
                continue;
            }

            uint32_t x0 = std::max(int32_t(0), (int32_t)(std::ceil(xmin)));
            uint32_t x1 = std::min(camera_->width() - 1, (int32_t)(std::floor(xmax)));
            uint32_t y0 = std::max(int32_t(0), (int32_t)(std::ceil(ymin)));
            uint32_t y1 = std::min(camera_->height() - 1, (int32_t)(std::floor(ymax)));

            float area = EdgeFunction(v0_i, v1_i, v2_i);
            if (std::abs(area) < std::numeric_limits<float>::min()) {
                continue;
            }
            for (uint32_t y = y0; y <= y1; ++y) {
                for (uint32_t x = x0; x <= x1; ++x) {
                    Eigen::Vector3f ray_w;
                    camera_->ray_w(static_cast<int>(x), static_cast<int>(y), &ray_w);
                    // even if back-face culling is enabled, dont' skip back-face
                    // need to update z-buffer to handle front-face occluded by back-face
                    bool backface = mesh_->face_normals().row(i).dot(ray_w) > 0;
                    Eigen::Vector3f pixel_sample(static_cast<float>(x),
                        static_cast<float>(y), 0.0f);
                    float w0 = EdgeFunction(v1_i, v2_i, pixel_sample);
                    float w1 = EdgeFunction(v2_i, v0_i, pixel_sample);
                    float w2 = EdgeFunction(v0_i, v1_i, pixel_sample);
                    if ((!backface && (w0 >= 0 && w1 >= 0 && w2 >= 0)) ||
                        (backface && (w0 <= 0 && w1 <= 0 && w2 <= 0))) {
                        w0 /= area;
                        w1 /= area;
                        w2 /= area;
#if 0
                        // original
                        pixel_sample.z() = w0 * v0_i.z() + w1 * v1_i.z() + w2 * v2_i.z();
#else
                        // Perspective-Correct Interpolation 
                        w0 /= v0_i.z();
                        w1 /= v1_i.z();
                        w2 /= v2_i.z();

                        pixel_sample.z() = 1.0f / (w0 + w1 + w2);

                        w0 = w0 * pixel_sample.z();
                        w1 = w1 * pixel_sample.z();
                        w2 = w2 * pixel_sample.z();
                        // Perspective-Correct Interpolation 
#endif

                        float& d = depth->at<float>(y, x);
                        if (d < std::numeric_limits<float>::min() || pixel_sample.z() < d) {
                            d = pixel_sample.z();
                            face_id->at<int>(y, x) = i;
                            cv::Vec3f& weight = weight_image.at<cv::Vec3f>(y, x);
                            weight[0] = w0;
                            weight[1] = w1;
                            weight[2] = w2;
                            backface_image.at<unsigned char>(y, x) = backface ? 255 : 0;
                        }
                    }
                }
            }
        }

        // make images by referring to face id image
        for (int y = 0; y < backface_image.rows; y++) {
            for (int x = 0; x < backface_image.cols; x++) {
                const unsigned char& bf = backface_image.at<unsigned char>(y, x);
                int& fid = face_id_->at<int>(y, x);
                if (option_.backface_culling && bf == 255) {
                    depth_->at<float>(y, x) = 0.0f;
                    fid = -1;
                    continue;
                }

                if (fid > 0) {
                    Eigen::Vector3f ray_w;
                    camera_->ray_w(x, y, &ray_w);

                    cv::Vec3f& weight = weight_image.at<cv::Vec3f>(y, x);
                    float w0 = weight[0];
                    float w1 = weight[1];
                    float w2 = weight[2];

                    // fill mask
                    if (mask != nullptr) {
                        mask->at<unsigned char>(y, x) = 255;
                    }

                    // calculate shading normal
                    Eigen::Vector3f shading_normal_w{ 0.0f, 0.0f, 0.0f };
                    if (option_.shading_normal == ShadingNormal::kFace) {
                        shading_normal_w = mesh_->face_normals().row(fid);
                    }
                    else if (option_.shading_normal == ShadingNormal::kVertex) {
                        // barycentric interpolation of normal
                        const auto& normals = mesh_->normals();
                        const auto& normal_indices = mesh_->normal_indices();
                        shading_normal_w = w0 * normals.row(normal_indices(fid,0)) +
                            w1 * normals.row(normal_indices(fid,1)) +
                            w2 * normals.row(normal_indices(fid, 2));
                    }

                    // set shading normal
                    if (normal != nullptr) {
                        Eigen::Vector3f shading_normal_c =
                            w2c_R * shading_normal_w;  // rotate to camera coordinate
                        cv::Vec3f& n = normal->at<cv::Vec3f>(y, x);
                        for (int k = 0; k < 3; k++) {
                            n[k] = shading_normal_c[k];
                        }
                    }

                    // delegate color calculation to pixel_shader
                    if (color != nullptr) {
                        Eigen::Vector3f light_dir = ray_w;  // emit light same as ray
                        PixelShaderInput pixel_shader_input(color, x, y, w1, w2, fid, &ray_w,
                            &light_dir, &shading_normal_w,
                            &oren_nayar_param, mesh_);
                        pixel_shader->Process(pixel_shader_input);
                    }
                }
            }
        }
         
        return true;
    }

    bool Rasterizer::Impl::RenderColor(Image3b* color) const {
        return Render(color, nullptr, nullptr, nullptr, nullptr);
    }

    bool Rasterizer::Impl::RenderDepth(Image1f* depth) const {
        return Render(nullptr, depth, nullptr, nullptr, nullptr);
    }

    bool Rasterizer::Impl::RenderNormal(Image3f* normal) const {
        return Render(nullptr, nullptr, normal, nullptr, nullptr);
    }

    bool Rasterizer::Impl::RenderMask(Image1b* mask) const {
        return Render(nullptr, nullptr, nullptr, mask, nullptr);
    }

    bool Rasterizer::Impl::RenderFaceId(Image1i* face_id) const {
        return Render(nullptr, nullptr, nullptr, nullptr, face_id);
    }

    bool Rasterizer::Impl::RenderW(Image3b* color, Image1w* depth, Image3f* normal,
        Image1b* mask, Image1i* face_id) const {
        if (depth == nullptr) {
            LOG_ERR_OUT<<"depth is nullptr";
            return false;
        }

        Image1f f_depth;
        bool org_ret = Render(color, &f_depth, normal, mask, face_id);

        if (org_ret) {
            ConvertTo(f_depth, depth);
        }

        return org_ret;
    }

    bool Rasterizer::Impl::RenderDepthW(Image1w* depth) const {
        return RenderW(nullptr, depth, nullptr, nullptr, nullptr);
    }

    // Renderer implementation
    Rasterizer::Rasterizer() : pimpl_(std::unique_ptr<Impl>(new Impl)) {}

    Rasterizer::~Rasterizer() {}

    Rasterizer::Rasterizer(const RendererOption& option)
        : pimpl_(std::unique_ptr<Impl>(new Impl(option))) {}

    void Rasterizer::set_option(const RendererOption& option) {
        pimpl_->set_option(option);
    }

    void Rasterizer::set_mesh(std::shared_ptr<const Mesh> mesh) {
        pimpl_->set_mesh(mesh);
    }

    bool Rasterizer::PrepareMesh() { return pimpl_->PrepareMesh(); }

    void Rasterizer::set_camera(std::shared_ptr<const Camera> camera) {
        pimpl_->set_camera(camera);
    }

    bool Rasterizer::Render(Image3b* color, Image1f* depth, Image3f* normal,
        Image1b* mask, Image1i* face_id) const {
        return pimpl_->Render(color, depth, normal, mask, face_id);
    }

    bool Rasterizer::RenderColor(Image3b* color) const {
        return pimpl_->RenderColor(color);
    }

    bool Rasterizer::RenderDepth(Image1f* depth) const {
        return pimpl_->RenderDepth(depth);
    }

    bool Rasterizer::RenderNormal(Image3f* normal) const {
        return pimpl_->RenderNormal(normal);
    }

    bool Rasterizer::RenderMask(Image1b* mask) const {
        return pimpl_->RenderMask(mask);
    }

    bool Rasterizer::RenderFaceId(Image1i* face_id) const {
        return pimpl_->RenderFaceId(face_id);
    }

    bool Rasterizer::RenderW(Image3b* color, Image1w* depth, Image3f* normal,
        Image1b* mask, Image1i* face_id) const {
        return pimpl_->RenderW(color, depth, normal, mask, face_id);
    }

    bool Rasterizer::RenderDepthW(Image1w* depth) const {
        return pimpl_->RenderDepthW(depth);
    }


    bool ValidateAndInitBeforeRender(bool mesh_initialized,
        std::shared_ptr<const Camera> camera,
        std::shared_ptr<const Mesh> mesh,
        const RendererOption& option, Image3b* color,
        Image1f* depth, Image3f* normal, Image1b* mask,
        Image1i* face_id) {
        if (camera == nullptr) {
            LOG_ERR_OUT << ("camera has not been set\n");
            return false;
        }
        if (!mesh_initialized) {
            LOG_ERR_OUT << ("mesh has not been initialized\n");
            return false;
        }
        
        if (option.backface_culling && isEmpty(mesh->face_normals())) {
            LOG_ERR_OUT << ("specified back-face culling but face normal is empty.\n");
            return false;
        }
        if (option.diffuse_color == DiffuseColor::kTexture &&
            mesh->materials().empty()) {
            LOG_ERR_OUT<<("specified texture as diffuse color but texture is empty.\n");
            return false;
        }
        if (option.diffuse_color == DiffuseColor::kTexture) {
            for (int i = 0; i < static_cast<int>(mesh->materials().size()); i++) {
                if (mesh->materials()[i].diffuse_tex.empty()) {
                    LOG_ERR_OUT<<("specified texture as diffuse color but %d th texture is empty.\n",
                        i);
                    return false;
                }
            }
        }
        if (option.diffuse_color == DiffuseColor::kVertex &&
            isEmpty(mesh->vertex_colors())) {
            LOG_ERR_OUT<<(
                "specified vertex color as diffuse color but vertex color is empty.\n");
            return false;
        }
        if (option.shading_normal == ShadingNormal::kFace &&
            isEmpty(mesh->face_normals())) {
            LOG_ERR_OUT<<("specified face normal as shading normal but face normal is empty.\n");
            return false;
        }
        if (option.shading_normal == ShadingNormal::kVertex &&
            isEmpty(mesh->normals())) {
            LOG_ERR_OUT<<(
                "specified vertex normal as shading normal but vertex normal is "
                "empty.\n");
            return false;
        }
        if (color == nullptr && depth == nullptr && normal == nullptr &&
            mask == nullptr && face_id == nullptr) {
            LOG_ERR_OUT<<("all arguments are nullptr. nothing to do\n");
            return false;
        }

        int width = camera->width();
        int height = camera->height();

        if (color != nullptr) {
            Init(color, width, height, unsigned char(0));
        }
        if (depth != nullptr) {
            Init(depth, width, height, 0.0f);
        }
        if (normal != nullptr) {
            Init(normal, width, height, 0.0f);
        }
        if (mask != nullptr) {
            Init(mask, width, height, unsigned char(0));
        }
        if (face_id != nullptr) {
            // initialize with -1 (no hit)
            Init(face_id, width, height, -1);
        }

        return true;
    }

}  // namespace currender

int test_render()
{ 
    using namespace currender;
    std::shared_ptr<Mesh>msh(new Mesh()); 
    Eigen::MatrixX3f V;
    Eigen::MatrixX3i F;
    bool readRet = readFromSimpleObj("D:/ucl360/UCL360Calib/CameraCalibGui/pro80/in/scan_1027-023915.pts.obj", V, F);
    if (!readRet)
    {
        return -1;
    }
    msh->set_vertices(V);
    msh->set_vertex_indices(F);
    msh->CalcNormal();
    
    
    std::shared_ptr<Camera>cam(new PinholeCamera(800, 600, Eigen::Affine3f::Identity(), Eigen::Vector2f(400, 300), Eigen::Vector2f(600.f, 600.f)));
    //cam->set_size(1200,800);
    //cam->set_c2w(Eigen::Affine3f::Identity());
    //std::dynamic_pointer_cast<PinholeCamera>(cam)->set_principal_point(Eigen::Vector2f(600.f, 400.f));
    //std::dynamic_pointer_cast<PinholeCamera>(cam)->set_focal_length(Eigen::Vector2f(600.f, 600.f));
    
    
    Rasterizer render;
    render.set_mesh(msh);
    render.set_camera(cam);
    render.PrepareMesh();
    Image3b color;
    Image1f depth;
    Image3f normal;
    Image1b mask;
    Image1i face_id;
    render.Render(&color, &depth, &normal, &mask, &face_id);
    return 0;
}
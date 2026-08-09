import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import numpy as np
class Silhouette3DReconstructor:
    """基于剪影的3D体素重建器"""
    
    def __init__(self, voxel_resolution: int = 100, world_size: float = 2.0):
        """
        初始化重建器
        
        Args:
            voxel_resolution: 体素网格分辨率（每个维度的体素数）
            world_size: 世界空间的大小（假设为立方体）
        """
        self.resolution = voxel_resolution
        self.world_size = world_size
        self.voxel_size = world_size / voxel_resolution
        
        # 初始化体素网格（全部为True表示初始都是实心）
        self.voxels = np.ones((voxel_resolution, voxel_resolution, voxel_resolution//2), dtype=bool)
        
        # 世界坐标系偏移（使世界中心在原点）
        self.offset = world_size / 2
        
    def world_to_voxel(self, points_3d: np.ndarray) -> np.ndarray:
        """将世界坐标转换为体素坐标"""
        # 将点从世界坐标映射到体素坐标
        voxel_coords = (points_3d + self.offset) / self.voxel_size
        return voxel_coords.astype(int)
    
    def voxel_to_world(self, voxel_coords: np.ndarray) -> np.ndarray:
        """将体素坐标转换为世界坐标"""
        world_coords = voxel_coords * self.voxel_size - np.array([self.offset,self.offset,-50])
        return world_coords
    
    def get_voxel_centers(self) -> np.ndarray:
        """获取所有体素的中心点在世界坐标系中的位置"""
        indices = np.indices((self.resolution, self.resolution, self.resolution//2))
        voxel_centers = self.voxel_to_world(indices.reshape(3, -1).T)
        return voxel_centers
    
    def project_points(self, points_3d: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        将3D点投影到图像平面
        
        Args:
            points_3d: (N, 3) 世界坐标系中的3D点
            K: (3, 3) 相机内参矩阵
            R: (3, 3) 旋转矩阵
            t: (3, 1) 平移向量
        
        Returns:
            (N, 2) 图像坐标系中的像素坐标
        """
        
        points_cam = (R.T @ points_3d.T  )/160*300
        
        return points_cam[:2, :].T
        # 透视投影到归一化平面
        with np.errstate(divide='ignore', invalid='ignore'):
            points_norm = points_cam[:, :2] / points_cam[:, 2:3]
        
        # 应用内参矩阵
        points_pixel = (K @ points_norm.T).T
        
        return points_pixel[:, :2]
    
    def is_inside_silhouette(self, points_pixel: np.ndarray, silhouette: np.ndarray) -> np.ndarray:
        """
        检查像素点是否在剪影区域内
        
        Args:
            points_pixel: (N, 2) 像素坐标
            silhouette: (H, W) 二值剪影图像
        
        Returns:
            (N,) boolean数组，表示每个点是否在剪影内
        """
        h, w = silhouette.shape
        # 四舍五入并转换为整数
        x = np.round(points_pixel[:, 0]+silhouette.shape[1]/2).astype(int)
        y = np.round(silhouette.shape[0]/2-points_pixel[:, 1]).astype(int)
        
        # 检查是否在图像范围内
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        
        # 初始化结果为False
        inside = np.zeros(len(points_pixel), dtype=bool)
        
        # 对有效的点检查剪影值
        if np.any(valid):
            valid_indices = np.where(valid)[0]
            inside[valid_indices] = silhouette[y[valid_indices], x[valid_indices]] > 127
        
        return inside
    
    def carve_voxels(self, silhouette: np.ndarray, R: np.ndarray):
        """
        使用单个视角的剪影进行体素雕刻
        
        Args:
            silhouette: (H, W) 二值剪影图像
            K: (3, 3) 相机内参矩阵
            R: (3, 3) 旋转矩阵
            t: (3, 1) 平移向量
        """
        # 获取所有体素中心点
        voxel_centers = self.get_voxel_centers()
        
        # 投影到图像平面
        points_pixel = self.project_points(voxel_centers, R)

        # p =points_pixel[inside].astype(np.int32)
        # img = np.zeros([600,600],dtype=np.uint8)
        # img[p[:, 0], p[:, 1]] = 255
        # Image.fromarray(img).save('bfmGan/output.png')

        
        # 检查哪些点在剪影内
        inside = self.is_inside_silhouette(points_pixel, silhouette)
        
        # 更新体素网格（只有在剪影外的体素才被移除）
        self.voxels = self.voxels.reshape(-1) & inside
        self.voxels = self.voxels.reshape(self.resolution, self.resolution, self.resolution//2)
    def reset(self):
        self.voxels[:] = 1
    def reconstruct(self, silhouettes: List[np.ndarray], R_list: List[np.ndarray]):
        """
        使用多个视角的剪影进行3D重建
        
        Args:
            silhouettes: 剪影图像列表
            K_list: 相机内参矩阵列表
            R_list: 旋转矩阵列表
            t_list: 平移向量列表
        """
        print(f"开始体素雕刻重建，体素分辨率: {self.resolution}x{self.resolution}x{self.resolution}")
        print(f"使用 {len(silhouettes)} 个视角")
        
        for i, (sil,  R) in enumerate(zip(silhouettes, R_list)):
            print(f"处理视角 {i+1}/{len(silhouettes)}...")
            self.carve_voxels(sil, R)
            
            # 可选：打印当前体素数
            voxel_count = np.sum(self.voxels)
            np.savetxt('bfmgan/data.txt', self.get_voxel_centers()[self.voxels.reshape(-1)])
            # print(f"  剩余体素数: {voxel_count}")
            # break
            
            
     
        
        print("重建完成！")
    
    def extract_mesh(self, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        从体素网格提取三角网格（使用Marching Cubes算法）
        
        Returns:
            vertices: (N, 3) 顶点坐标
            faces: (M, 3) 三角形索引
        """
        try:
            from skimage.measure import marching_cubes
            
            # 确保体素网格是二值的
            voxels_binary = self.voxels.astype(float)
            
            # 应用Marching Cubes
            verts, faces, _, _ = marching_cubes(voxels_binary, level=threshold)
            
            # 将顶点坐标从体素空间转换到世界空间
            verts_world = self.voxel_to_world(verts)
            faces[:, [0, 1]] = faces[:, [1, 0]]
            return verts_world, faces
            
        except ImportError:
            print("警告: scikit-image未安装，无法提取网格。请安装: pip install scikit-image")
            return np.array([]), np.array([])
    
    def visualize_voxels(self):
        """可视化体素模型"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 获取占用体素的坐标
        occupied = np.where(self.voxels)
        if len(occupied[0]) == 0:
            print("没有体素可显示")
            return
        
        # 转换为世界坐标
        points = self.voxel_to_world(np.array(occupied).T)
        
        # 绘制散点图
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c='blue', alpha=0.6, s=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D重建结果 (体素数: {len(points)})')
        
        # 设置等比例坐标轴
        max_range = self.world_size / 2
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        plt.show()


def create_sample_data():
    """
    创建示例数据用于测试
    
    模拟一个立方体物体，从不同角度拍摄
    """
    # 假设物体是一个边长为0.5的立方体，中心在原点
    cube_size = 0.5
    
    # 定义相机参数
    image_size = (480, 640)
    focal_length = 500
    
    # 相机内参
    K = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # 生成多个视角的相机位姿（围绕物体旋转）
    num_views = 1+7*2
    radius = 1.5
    
    K_list = []
    R_list = []
    t_list = []
    silhouettes = [] 
    
    cameraCnt=4
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 200])
    R_list .append(camera_pose[0:3,0:3])
    t_list .append(camera_pose[0:3,3].reshape(3, 1))
    silhouettes.append(np.array(Image.open(f'bfmGan/mask_{0:02d}.png')))
    for camera_i in range(1,cameraCnt):
        theta = 3.141592653589793*0.5/cameraCnt*camera_i 
        camera_pose = np.eye(4)
        camera_pose[0,0] = np.cos(theta)
        camera_pose[0,2] = np.sin(theta)
        camera_pose[2,0] = -np.sin(theta)
        camera_pose[2,2] = np.cos(theta)
        # camera_pose=camera_pose.T
        camera_pose[:3, 3] = np.array([200*np.sin(theta), 0, 200*np.cos(theta)])        
        R_list .append(camera_pose[0:3,0:3])
        t_list .append(camera_pose[0:3,3].reshape(3, 1))
        silhouettes.append(np.array(Image.open(f'bfmGan/mask_{camera_i:02d}.png')))
    for camera_i in range(1,cameraCnt):
        theta = -3.141592653589793*0.5/cameraCnt*camera_i  
        camera_pose = np.eye(4)
        camera_pose[0,0] = np.cos(theta)
        camera_pose[0,2] = np.sin(theta)
        camera_pose[2,0] = -np.sin(theta)
        camera_pose[2,2] = np.cos(theta)
        # camera_pose=camera_pose.T
        camera_pose[:3, 3] = np.array([200*np.sin(theta), 0, 200*np.cos(theta)])        
        R_list .append(camera_pose[0:3,0:3])
        t_list .append(camera_pose[0:3,3].reshape(3, 1))
        silhouettes.append(np.array(Image.open(f'bfmGan/mask_{camera_i+cameraCnt-1:02d}.png')))
        

    return silhouettes, K_list, R_list, t_list


 


def main():
    """主函数示例"""
    
    # 1. 创建示例数据（在实际使用时，需要替换为真实数据）
    print("生成示例数据...")
    silhouettes, K_list, R_list, t_list = create_sample_data()
    

    
    # 2. 创建重建器并执行重建
    reconstructor = Silhouette3DReconstructor(voxel_resolution=100, world_size=200.0)
    reconstructor.reconstruct(silhouettes, R_list)
     
    
    # 4. 提取网格（如果安装了scikit-image）
    try:
        verts, faces = reconstructor.extract_mesh()
        if len(verts) > 0:
            print(f"提取网格: {len(verts)} 个顶点, {len(faces)} 个三角面")
            
            # 可以保存为OBJ文件
            save_obj(verts, faces, 'bfmgan/reconstructed_model.obj')
    except Exception as e:
        print(f"网格提取失败: {e}")


def save_obj(vertices: np.ndarray, faces: np.ndarray, filename: str):
    """保存为OBJ文件格式"""
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        for face in faces + 1:  # OBJ格式索引从1开始
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print(f"网格已保存到 {filename}")


if __name__ == "__main__":
    main()
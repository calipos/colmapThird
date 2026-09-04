import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.ndimage import map_coordinates
import time
import cv2


class HybridHarmonicWarp:
    """
    混合方法：结合位移场和直接坐标方法
    优点：可以更容易控制变形幅度
    """

    def __init__(self, image):
        self.image = image
        self.height, self.width = image.shape[:2]
        self.is_color = len(image.shape) == 3

    def warp_with_scale(self, source_points, target_points, scale=1.0):
        """
        带缩放因子的变形
        scale: 变形强度 (0-1)
        """
        n = self.height * self.width

        # 构建系统（使用LIL格式）
        A = self._build_laplacian()
        b_x = np.zeros(n)
        b_y = np.zeros(n)

        # 边界条件
        for i in range(self.height):
            for j in range(self.width):
                idx = i * self.width + j
                if i == 0 or i == self.height-1 or j == 0 or j == self.width-1:
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b_x[idx] = j
                    b_y[idx] = i

        # 控制点约束（带缩放）
        for src, tgt in zip(source_points, target_points):
            x, y = int(round(src[0])), int(round(src[1]))
            if 0 <= x < self.width and 0 <= y < self.height:
                idx = y * self.width + x
                A[idx, :] = 0
                A[idx, idx] = 1
                # 混合：原始位置 + 缩放后的位移
                b_x[idx] = src[0] + scale * (tgt[0] - src[0])
                b_y[idx] = src[1] + scale * (tgt[1] - src[1])

        # 求解
        A_csr = csr_matrix(A)
        map_x = spsolve(A_csr, b_x).reshape((self.height, self.width))
        map_y = spsolve(A_csr, b_y).reshape((self.height, self.width))

        # 应用变形
        if self.is_color:
            warped = np.zeros_like(self.image)
            for c in range(self.image.shape[2]):
                warped[:, :, c] = map_coordinates(
                    self.image[:, :, c], [map_y, map_x], order=1
                )
        else:
            warped = map_coordinates(self.image, [map_y, map_x], order=1)

        return warped, map_x, map_y

    def _build_laplacian(self):
        """构建拉普拉斯矩阵（修复版）"""
        n = self.height * self.width
        A = lil_matrix((n, n), dtype=np.float64)

        # 填充拉普拉斯算子
        for i in range(self.height):
            for j in range(self.width):
                idx = i * self.width + j
                A[idx, idx] = -4

                if i > 0:
                    A[idx, (i-1) * self.width + j] = 1
                if i < self.height - 1:
                    A[idx, (i+1) * self.width + j] = 1
                if j > 0:
                    A[idx, i * self.width + (j-1)] = 1
                if j < self.width - 1:
                    A[idx, i * self.width + (j+1)] = 1

        return A


class ImageWarp:
    def __init__(self, image):
        self.image = image
        self.height, self.width = image.shape[:2]
        self.is_color = len(image.shape) == 3

    def warp_with_scale(self, source_points, target_points, scale=1.0):
        n = self.height * self.width

        A = self._build_laplacian()
        b_x = np.zeros(n, dtype=np.float32)
        b_y = np.zeros(n, dtype=np.float32)

        # for i in range(self.height):
        #     for j in range(self.width):
        #         idx = i * self.width + j
        #         if i == 0 or i == self.height-1 or j == 0 or j == self.width-1:
        #             A[idx, :] = 0
        #             A[idx, idx] = 1

        # 控制点约束（带缩放）
        for src, tgt in zip(source_points, target_points):
            x, y = int(round(src[0])), int(round(src[1]))
            if 0 <= x < self.width and 0 <= y < self.height:
                idx = y * self.width + x
                # A[idx, :] = 0
                # A[idx, idx] = scale
                b_x[idx] = scale * (tgt[0] - src[0])
                b_y[idx] = scale * (tgt[1] - src[1])

        
        A_csr = csr_matrix(A)
        map_x = spsolve(A_csr, b_x).reshape((self.height, self.width))
        map_y = spsolve(A_csr, b_y).reshape((self.height, self.width))

        y, x = np.indices((self.height, self.width), dtype=np.float32)
        if self.is_color:
            warped = np.zeros_like(self.image)
            for c in range(self.image.shape[2]):
                warped[:, :, c] = map_coordinates(
                    self.image[:, :, c], [y+map_y, x+map_x], order=1
                )
        else:
            warped = map_coordinates(self.image, [y+map_y, x+map_x], order=1)

        np.savetxt(f'bfmgan/map_x.txt', toMap3d(map_x), fmt='%d %d %.6f')
        np.savetxt(f'bfmgan/map_y.txt', toMap3d(map_y), fmt='%d %d %.6f')
        return warped, x+map_x, y+map_y

    def _build_laplacian(self):
        n = self.height * self.width
        A = lil_matrix((n, n), dtype=np.float32)
        # 填充拉普拉斯算子
        for i in range(self.height):
            for j in range(self.width):
                idx = i * self.width + j
                A[idx, idx] = -4
                if j > 0 and j < self.width - 1 and i > 0 and i < self.height - 1:
                    A[idx, idx-1] = 1
                    A[idx, idx+1] = 1
                    A[idx, idx-self.width] = 1
                    A[idx, idx+self.width] = 1
        i = 0
        for j in range(self.width):
            idx = j
            A[idx, idx] = -3
            if j > 0 and j < self.width - 1:
                A[idx, idx-1] = 1
                A[idx, idx+1] = 1
                A[idx, idx+self.width] = 1
        i = self.height - 1
        for j in range(self.width):
            idx = i * self.width + j
            A[idx, idx] = -3
            if j > 0 and j < self.width - 1:
                A[idx, idx-1] = 1
                A[idx, idx+1] = 1
                A[idx, idx-self.width] = 1
        j = 0
        for i in range(self.height):
            idx = i * self.width
            A[idx, idx] = -3
            if i > 0 and i < self.height - 1:
                A[idx, idx+1] = 1
                A[idx, idx-self.width] = 1
                A[idx, idx+self.width] = 1
        j = self.width-1
        for i in range(self.height):
            idx = i * self.width+j
            A[idx, idx] = -3
            if i > 0 and i < self.height - 1:
                A[idx, idx-1] = 1
                A[idx, idx-self.width] = 1
                A[idx, idx+self.width] = 1
        A[0, 0] = -2
        A[0, 1] = 1
        A[0, self.width] = 1
        idx = self.width-1
        A[idx, idx] = -2
        A[idx, idx-1] = 1
        A[idx, idx+self.width] = 1
        idx = (self.height-1)*self.width
        A[idx, idx] = -2
        A[idx, idx+1] = 1
        A[idx, idx-self.width] = 1
        idx = self.height*self.width-1
        A[idx, idx] = -2
        A[idx, idx-1] = 1
        A[idx, idx-self.width] = 1

        return A


def create_complex_image():
    """创建复杂测试图像"""
    size = 400
    image = np.zeros((size, size, 3), dtype=np.uint8)
    # 绘制棋盘格以便观察变形效果
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                cv2.rectangle(image,
                              (i*50, j*50),
                              ((i+1)*50, (j+1)*50),
                              (200, 200, 200), -1)

    return image


def toMap3d(data):
    h, w = data.shape
    y, x = np.indices(data.shape, dtype=np.float32)
    x = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=2)
    z = np.expand_dims(data, axis=2)
    points = np.concatenate((x, y, z), axis=2).reshape(-1, 3)
    return points
def demo_hybrid_method():

    # 创建图像
    size = 400
    image = create_complex_image()

    warper = ImageWarp(image)

    # 控制点：右下角拉动
    source_points = []
    target_points = []
    controlPtsCnt=8
    for i in range(controlPtsCnt):
        theta = 360/controlPtsCnt*i *np.pi/180
        source_points.append(
            (size*0.5+size*0.35*np.cos(theta), size*0.5+size*0.35*np.sin(theta)))
        add = 30*np.pi/180
        target_points.append(
            (size*0.5+size*0.35*np.cos(theta+add), size*0.5+size*0.35*np.sin(theta+add)))





    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for idx, scale in enumerate([.5, 1]):
        warped, map_x, map_y = warper.warp_with_scale(
            source_points, target_points, scale
        )

        # 显示结果
        axes[0, idx].imshow(image, cmap='gray')
        axes[0, idx].set_title(f'强度: {scale:.1f}')
        axes[0, idx].axis('off')

        # 显示变形网格
        x_grid, y_grid = np.meshgrid(np.arange(0, size, 10),
                                     np.arange(0, size, 10))
        # 确保索引在有效范围内
        yi = np.clip(y_grid.astype(int), 0, size-1)
        xi = np.clip(x_grid.astype(int), 0, size-1)
        mapped_x = map_x[yi, xi]
        mapped_y = map_y[yi, xi]

        axes[1, idx].imshow(warped, cmap='gray')
        axes[1, idx].plot(mapped_x, mapped_y, 'r-', alpha=0.3, linewidth=0.5)
        axes[1, idx].plot(mapped_x.T, mapped_y.T, 'r-',
                          alpha=0.3, linewidth=0.5)
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 演示直接坐标方法
    demo_hybrid_method()

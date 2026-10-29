import os
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_VERBOSE"] = "False"
# 禁用matplotlib的GUI后端
import matplotlib
matplotlib.use('Agg')

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import glob
import zipfile
from datetime import datetime
from ultralytics import SAM
import struct
class SAM2GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM2 交互式图像分割工具")
        self.root.geometry("1200x800")
        
        # 初始化变量
        self.model = None
        self.current_image = None
        self.current_image_path = None
        self.display_image = None
        self.image_list = []
        self.current_image_index = 0
        self.scale_factor = 1.0
        
        # 为每张图片保存独立的交互状态
        self.image_annotations = {}  # {image_path: {'positive_points': [], 'negative_points': [], 'bboxes': [], 'operation_history': []}}
        self.image_results = {}      # {image_path: masks}
        
        # 当前图片的交互数据
        self.positive_points = []
        self.negative_points = []
        self.bboxes = []
        self.temp_bbox = None
        self.bbox_start = None
        self.current_mode = "positive"  # positive, negative, bbox
        
        # 操作历史（用于撤销）
        self.operation_history = []
        
        self.setup_ui()
        self.load_model()
        
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.Frame(main_frame, width=250)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # 图片加载区域
        load_frame = ttk.LabelFrame(control_frame, text="图片加载", padding=10)
        load_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(load_frame, text="添加单张图片", 
                  command=self.add_single_image).pack(fill=tk.X, pady=2)
        ttk.Button(load_frame, text="上传图片文件夹", 
                  command=self.load_image_folder).pack(fill=tk.X, pady=2)
        ttk.Button(load_frame, text="清空图片列表", 
                  command=self.clear_image_list).pack(fill=tk.X, pady=2)
        
        # 图片导航
        nav_frame = ttk.LabelFrame(control_frame, text="图片导航", padding=10)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        nav_button_frame = ttk.Frame(nav_frame)
        nav_button_frame.pack(fill=tk.X)
        
        ttk.Button(nav_button_frame, text="上一张", 
                  command=self.prev_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(nav_button_frame, text="下一张", 
                  command=self.next_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        
        self.image_info_label = ttk.Label(nav_frame, text="未加载图片")
        self.image_info_label.pack(pady=5)
        
        # 交互模式选择
        mode_frame = ttk.LabelFrame(control_frame, text="交互模式", padding=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="positive")
        ttk.Radiobutton(mode_frame, text="正向点 (Positive)", 
                       variable=self.mode_var, value="positive",
                       command=self.change_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="负向点 (Negative)", 
                       variable=self.mode_var, value="negative",
                       command=self.change_mode).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="边界框 (BBox)", 
                       variable=self.mode_var, value="bbox",
                       command=self.change_mode).pack(anchor=tk.W)
        
        # 当前标注信息
        info_frame = ttk.LabelFrame(control_frame, text="当前标注", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.points_info_label = ttk.Label(info_frame, text="正向点: 0\n负向点: 0\n边界框: 0")
        self.points_info_label.pack()
        
        # 操作按钮
        action_frame = ttk.LabelFrame(control_frame, text="操作", padding=10)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(action_frame, text="撤销", 
                  command=self.undo_operation).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="清除当前图片标注", 
                  command=self.clear_current_annotations).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="运行当前图片推理", 
                  command=self.run_current_inference).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="批量推理所有图片", 
                  command=self.run_batch_inference).pack(fill=tk.X, pady=2)
        
        # 结果保存
        save_frame = ttk.LabelFrame(control_frame, text="保存结果", padding=10)
        save_frame.pack(fill=tk.X)
        
        ttk.Button(save_frame, text="保存当前掩码", 
                  command=self.save_current_mask).pack(fill=tk.X, pady=2)
        ttk.Button(save_frame, text="批量保存所有掩码", 
                  command=self.save_all_masks).pack(fill=tk.X, pady=2)
        
        # 右侧图像显示区域
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建画布和滚动条
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="white")
        
        # 添加滚动条
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪 - 请加载图片")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def load_model(self):
        """加载SAM2模型"""
        try:
            self.status_var.set("正在加载SAM2模型...")
            self.root.update()
            # 使用推理模式并禁用某些功能
            import torch
            torch.backends.cudnn.enabled = False  # 可能有帮助
            
            # 尝试不同的模型加载方式
            try:
                self.model = SAM("sam2.1_t.pt")
            except Exception as e1:
                raise e1
                    
            self.status_var.set("SAM2模型加载成功")
        except Exception as e:
            print(f"模型加载完整错误: {str(e)}")
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")
            self.status_var.set("模型加载失败")
            
    def add_single_image(self):
        """添加单张图片到列表"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        if file_path:
            # 检查是否已存在
            if file_path not in self.image_list:
                self.image_list.append(file_path)
                # 为新图片初始化标注状态
                self.image_annotations[file_path] = {
                    'positive_points': [],
                    'negative_points': [],
                    'bboxes': [],
                    'operation_history': []
                }
                
            # 跳转到该图片
            self.current_image_index = self.image_list.index(file_path)
            self.load_current_image()
            messagebox.showinfo("成功", f"图片已添加，当前列表共有 {len(self.image_list)} 张图片")
            
    def clear_image_list(self):
        """清空图片列表"""
        if messagebox.askyesno("确认", "确定要清空所有图片列表吗？这将清除所有标注数据。"):
            self.image_list = []
            self.image_annotations = {}
            self.image_results = {}
            self.current_image_index = 0
            self.current_image = None
            self.current_image_path = None
            self.clear_current_annotations()
            self.canvas.delete("all")
            self.update_image_info()
            self.status_var.set("图片列表已清空")
            
    def save_current_image_state(self):
        """保存当前图片的交互状态"""
        if self.current_image_path:
            self.image_annotations[self.current_image_path] = {
                'positive_points': self.positive_points.copy(),
                'negative_points': self.negative_points.copy(),
                'bboxes': self.bboxes.copy(),
                'operation_history': self.operation_history.copy()
            }
            
    def load_image_state(self, image_path):
        """加载指定图片的交互状态"""
        if image_path in self.image_annotations:
            state = self.image_annotations[image_path]
            self.positive_points = state['positive_points'].copy()
            self.negative_points = state['negative_points'].copy()
            self.bboxes = state['bboxes'].copy()
            self.operation_history = state['operation_history'].copy()
        else:
            # 为新图片初始化状态
            self.positive_points = []
            self.negative_points = []
            self.bboxes = []
            self.operation_history = []
            self.image_annotations[image_path] = {
                'positive_points': [],
                'negative_points': [],
                'bboxes': [],
                'operation_history': []
            }

    def load_single_image(self):
        """加载单张图片（保持向后兼容）"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        if file_path:
            self.image_list = [file_path]
            self.current_image_index = 0
            # 初始化标注状态
            self.image_annotations = {file_path: {
                'positive_points': [],
                'negative_points': [],
                'bboxes': [],
                'operation_history': []
            }}
            self.load_current_image()
            
    def load_image_folder(self):
        """加载图片文件夹"""
        folder_path = filedialog.askdirectory(title="选择图片文件夹")
        if folder_path:
            # 获取文件夹中的所有图片
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
            new_image_list = []
            for ext in image_extensions:
                new_image_list.extend(glob.glob(os.path.join(folder_path, ext)))
                new_image_list.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
            if new_image_list:
                new_image_list.sort()
                # 添加到现有列表，避免重复
                for img_path in new_image_list:
                    if img_path not in self.image_list:
                        self.image_list.append(img_path)
                        # 为新图片初始化标注状态
                        self.image_annotations[img_path] = {
                            'positive_points': [],
                            'negative_points': [],
                            'bboxes': [],
                            'operation_history': []
                        }
                
                self.current_image_index = 0
                self.load_current_image()
                messagebox.showinfo("成功", f"文件夹已加载，当前列表共有 {len(self.image_list)} 张图片")
            else:
                messagebox.showwarning("警告", "文件夹中未找到图片文件")
                
    def load_current_image(self):
        """加载当前图片"""
        if not self.image_list:
            return
            
        # 保存之前图片的状态
        if self.current_image_path:
            self.save_current_image_state()
            
        image_path = self.image_list[self.current_image_index]
        self.current_image_path = image_path
        
        try:
            # 读取图片
            self.current_image = cv2.imread(image_path)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # 加载该图片的交互状态
            self.load_image_state(image_path)
            
            # 显示图片
            self.display_current_image()
            
            # 更新信息
            self.update_image_info()
            self.update_points_info()
            self.status_var.set(f"已加载: {os.path.basename(image_path)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"图片加载失败: {str(e)}")
            
    def display_current_image(self):
        """在画布上显示当前图片"""
        if self.current_image is None:
            return
            
        # 创建显示图片的副本
        display_img = self.current_image.copy()
        
        # 如果有推理结果，先应用掩码
        if self.current_image_path in self.image_results:
            masks = self.image_results[self.current_image_path]
            # 应用掩码（半透明覆盖）
            for mask in masks:
                # 将掩码调整到图像大小
                if mask.shape != display_img.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), 
                                    (display_img.shape[1], display_img.shape[0]))
                
                # 创建彩色掩码
                color = np.array([255, 255, 0])  # 黄色
                colored_mask = np.zeros_like(display_img)
                colored_mask[mask > 0.5] = color
                
                # 半透明叠加
                alpha = 0.4
                display_img = cv2.addWeighted(display_img, 1-alpha, colored_mask, alpha, 0)
        
        # 绘制标注
        self.draw_annotations(display_img)
        
        # 转换为PIL图片
        pil_image = Image.fromarray(display_img)
        
        # 计算缩放比例以适应画布
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            img_width, img_height = pil_image.size
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            self.scale_factor = min(scale_w, scale_h, 1.0)  # 不放大，只缩小
            
            new_width = int(img_width * self.scale_factor)
            new_height = int(img_height * self.scale_factor)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 转换为Tkinter格式
        self.display_image = ImageTk.PhotoImage(pil_image)
        
        # 清除画布并显示图片
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
        
        # 更新画布滚动区域
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def draw_annotations(self, image):
        """在图片上绘制标注"""
        # 绘制正向点（绿色）
        for point in self.positive_points:
            cv2.circle(image, point, 5, (0, 255, 0), -1)
            cv2.circle(image, point, 7, (0, 0, 0), 2)
            
        # 绘制负向点（红色）
        for point in self.negative_points:
            cv2.circle(image, point, 5, (255, 0, 0), -1)
            cv2.circle(image, point, 7, (0, 0, 0), 2)
            
        # 绘制边界框（蓝色）
        for bbox in self.bboxes:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            
        # 绘制临时边界框
        if self.temp_bbox:
            cv2.rectangle(image, (self.temp_bbox[0], self.temp_bbox[1]), 
                         (self.temp_bbox[2], self.temp_bbox[3]), (255, 255, 0), 2)
            
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """将画布坐标转换为图片坐标"""
        if self.scale_factor <= 0:
            return int(canvas_x), int(canvas_y)
        return int(canvas_x / self.scale_factor), int(canvas_y / self.scale_factor)
        
    def on_canvas_click(self, event):
        """画布点击事件"""
        if self.current_image is None:
            return
            
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        # 确保坐标在图片范围内
        img_height, img_width = self.current_image.shape[:2]
        img_x = max(0, min(img_x, img_width - 1))
        img_y = max(0, min(img_y, img_height - 1))
        
        if self.mode_var.get() == "positive":
            self.add_positive_point(img_x, img_y)
        elif self.mode_var.get() == "negative":
            self.add_negative_point(img_x, img_y)
        elif self.mode_var.get() == "bbox":
            self.bbox_start = (img_x, img_y)
            
    def on_canvas_drag(self, event):
        """画布拖拽事件"""
        if self.mode_var.get() == "bbox" and self.bbox_start and self.current_image is not None:
            img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
            img_height, img_width = self.current_image.shape[:2]
            img_x = max(0, min(img_x, img_width - 1))
            img_y = max(0, min(img_y, img_height - 1))
            
            self.temp_bbox = (min(self.bbox_start[0], img_x), min(self.bbox_start[1], img_y),
                             max(self.bbox_start[0], img_x), max(self.bbox_start[1], img_y))
            self.display_current_image()
            
    def on_canvas_release(self, event):
        """画布释放事件"""
        if self.mode_var.get() == "bbox" and self.bbox_start and self.temp_bbox:
            self.add_bbox(self.temp_bbox)
            self.temp_bbox = None
            self.bbox_start = None
            
    def add_positive_point(self, x, y):
        """添加正向点"""
        self.positive_points.append((x, y))
        self.operation_history.append(("add_positive", (x, y)))
        self.display_current_image()
        self.update_points_info()
        # 保存状态
        if self.current_image_path:
            self.save_current_image_state()
        
    def add_negative_point(self, x, y):
        """添加负向点"""
        self.negative_points.append((x, y))
        self.operation_history.append(("add_negative", (x, y)))
        self.display_current_image()
        self.update_points_info()
        # 保存状态
        if self.current_image_path:
            self.save_current_image_state()
        
    def add_bbox(self, bbox):
        """添加边界框"""
        self.bboxes.append(bbox)
        self.operation_history.append(("add_bbox", bbox))
        self.display_current_image()
        self.update_points_info()
        # 保存状态
        if self.current_image_path:
            self.save_current_image_state()
        
    def change_mode(self):
        """改变交互模式"""
        self.current_mode = self.mode_var.get()
        
    def undo_operation(self):
        """撤销上一个操作"""
        if not self.operation_history:
            messagebox.showinfo("信息", "没有可撤销的操作")
            return
            
        last_op = self.operation_history.pop()
        op_type, data = last_op
        
        if op_type == "add_positive":
            self.positive_points.remove(data)
        elif op_type == "add_negative":
            self.negative_points.remove(data)
        elif op_type == "add_bbox":
            self.bboxes.remove(data)
            
        self.display_current_image()
        self.update_points_info()
        # 保存状态
        if self.current_image_path:
            self.save_current_image_state()
        
    def clear_current_annotations(self):
        """清除当前图片的所有标注"""
        self.positive_points = []
        self.negative_points = []
        self.bboxes = []
        self.operation_history = []
        self.temp_bbox = None
        self.bbox_start = None
        if self.current_image is not None:
            self.display_current_image()
        self.update_points_info()
        # 保存状态
        if self.current_image_path:
            self.save_current_image_state()
            
    def clear_all_annotations(self):
        """清除所有标注（向后兼容）"""
        self.clear_current_annotations()
        
    def update_points_info(self):
        """更新标注信息显示"""
        # 显示当前图片的标注信息
        annotated_count = sum(1 for path in self.image_annotations 
                            if (self.image_annotations[path]['positive_points'] or 
                                self.image_annotations[path]['negative_points'] or 
                                self.image_annotations[path]['bboxes']))
        
        self.points_info_label.config(
            text=f"当前图片:\n"
                 f"正向点: {len(self.positive_points)}\n"
                 f"负向点: {len(self.negative_points)}\n"
                 f"边界框: {len(self.bboxes)}\n\n"
                 f"已标注图片: {annotated_count}/{len(self.image_list)}"
        )
        
    def update_image_info(self):
        """更新图片信息显示"""
        if self.image_list and self.current_image_path:
            # 检查当前图片是否有推理结果
            has_result = self.current_image_path in self.image_results
            result_text = " (已推理)" if has_result else ""
            
            self.image_info_label.config(
                text=f"{self.current_image_index + 1}/{len(self.image_list)}\n"
                     f"{os.path.basename(self.current_image_path)}{result_text}"
            )
        else:
            self.image_info_label.config(text="未加载图片")
            
    def prev_image(self):
        """上一张图片"""
        if self.image_list and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            
    def next_image(self):
        """下一张图片"""
        if self.image_list and self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.load_current_image()
            
    def run_current_inference(self):
        """运行当前图片的SAM2推理"""
        if self.current_image is None:
            messagebox.showwarning("警告", "请先加载图片")
            return
            
        if self.model is None:
            messagebox.showerror("错误", "SAM2模型未加载")
            return
            
        # 检查是否有标注
        if not self.positive_points and not self.negative_points and not self.bboxes:
            messagebox.showwarning("警告", "请先添加一些标注（点或边界框）")
            return
            
        try:
            self.status_var.set("正在运行SAM2推理...")
            self.root.update()
            
            masks = self._run_inference_for_image(self.current_image_path)
            if masks is not None:
                self.image_results[self.current_image_path] = masks
                self.display_results_for_current_image(masks)
                self.update_image_info()
                self.status_var.set("SAM2推理完成")
            else:
                messagebox.showerror("错误", "推理失败")
                self.status_var.set("推理失败")
                
        except Exception as e:
            messagebox.showerror("错误", f"推理过程中出错: {str(e)}")
            self.status_var.set("推理失败")
            
    def run_batch_inference(self):
        """批量推理所有图片"""
        if not self.image_list:
            messagebox.showwarning("警告", "请先加载图片")
            return
            
        if self.model is None:
            messagebox.showerror("错误", "SAM2模型未加载")
            return
            
        # 检查有标注的图片
        annotated_images = []
        for img_path in self.image_list:
            if img_path in self.image_annotations:
                annotations = self.image_annotations[img_path]
                if (annotations['positive_points'] or 
                    annotations['negative_points'] or 
                    annotations['bboxes']):
                    annotated_images.append(img_path)
        
        if not annotated_images:
            messagebox.showwarning("警告", "没有已标注的图片，请先对图片进行标注")
            return
            
        # 确认批量推理
        if not messagebox.askyesno("确认", f"将对 {len(annotated_images)} 张已标注的图片进行批量推理，继续吗？"):
            return
            
        try:
            # 创建进度窗口
            progress_window = tk.Toplevel(self.root)
            progress_window.title("批量推理进度")
            progress_window.geometry("400x100")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            progress_label = ttk.Label(progress_window, text="正在进行批量推理...")
            progress_label.pack(pady=10)
            
            progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
            progress_bar.pack(pady=10)
            progress_bar['maximum'] = len(annotated_images)
            
            successful_count = 0
            failed_count = 0
            
            for i, img_path in enumerate(annotated_images):
                progress_label.config(text=f"正在处理: {os.path.basename(img_path)}")
                progress_bar['value'] = i
                progress_window.update()
                
                # 临时设置状态以运行推理
                original_state = self._save_temp_state()
                self._load_temp_state(img_path)
                
                try:
                    masks = self._run_inference_for_image(img_path)
                    if masks is not None:
                        self.image_results[img_path] = masks
                        successful_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"图片 {img_path} 推理失败: {e}")
                    failed_count += 1
                
                # 恢复原始状态
                self._restore_temp_state(original_state)
            
            progress_bar['value'] = len(annotated_images)
            progress_window.update()
            
            # 关闭进度窗口
            progress_window.destroy()
            
            # 显示结果
            messagebox.showinfo("批量推理完成", 
                              f"批量推理完成！\n成功: {successful_count} 张\n失败: {failed_count} 张")
            
            # 更新当前显示
            if self.current_image_path in self.image_results:
                self.display_results_for_current_image(self.image_results[self.current_image_path])
            self.update_image_info()
            self.status_var.set(f"批量推理完成 - 成功 {successful_count} 张")
            
        except Exception as e:
            messagebox.showerror("错误", f"批量推理过程中出错: {str(e)}")
            self.status_var.set("批量推理失败")
    
    def _save_temp_state(self):
        """保存临时状态"""
        return {
            'positive_points': self.positive_points.copy(),
            'negative_points': self.negative_points.copy(),
            'bboxes': self.bboxes.copy(),
            'operation_history': self.operation_history.copy()
        }
    
    def _load_temp_state(self, img_path):
        """加载临时状态"""
        if img_path in self.image_annotations:
            state = self.image_annotations[img_path]
            self.positive_points = state['positive_points'].copy()
            self.negative_points = state['negative_points'].copy()
            self.bboxes = state['bboxes'].copy()
            self.operation_history = state['operation_history'].copy()
    
    def _restore_temp_state(self, state):
        """恢复临时状态"""
        self.positive_points = state['positive_points']
        self.negative_points = state['negative_points']
        self.bboxes = state['bboxes']
        self.operation_history = state['operation_history']
    
    def _run_inference_for_image(self, img_path):
        """为指定图片运行推理"""
        try:
            # 设置推理参数来避免GUI冲突
            import warnings
            warnings.filterwarnings("ignore")
            
            results = None
            
            # 优先使用边界框
            if self.bboxes:
                bbox = self.bboxes[0]
                try:
                    results = self.model(
                        img_path, 
                        bboxes=[bbox[0], bbox[1], bbox[2], bbox[3]],
                        verbose=False,
                        show=False,
                        save=False
                    )
                except Exception as e:
                    print(f"边界框推理失败: {e}")
                    results = self.model.predict(
                        img_path,
                        bboxes=[bbox[0], bbox[1], bbox[2], bbox[3]]
                    )
            
            # 如果有点标注
            elif self.positive_points or self.negative_points:
                points = []
                labels = []
                
                # 添加正向点
                for point in self.positive_points:
                    points.append(point)
                    labels.append(1)
                    
                # 添加负向点
                for point in self.negative_points:
                    points.append(point)
                    labels.append(0)
                    
                if points:
                    try:
                        results = self.model(
                            img_path, 
                            points=points, 
                            labels=labels,
                            verbose=False,
                            show=False,
                            save=False
                        )
                    except Exception as e:
                        print(f"点标注推理失败: {e}")
                        results = self.model.predict(
                            img_path,
                            points=points,
                            labels=labels
                        )
            
            if results and hasattr(results[0], 'masks') and results[0].masks is not None:
                return results[0].masks.data.cpu().numpy()
            else:
                return None
                
        except Exception as e:
            print(f"推理失败: {e}")
            return None
    
    def run_sam2_inference(self):
        """运行SAM2推理（向后兼容）"""
        self.run_current_inference()

    def display_results_for_current_image(self, masks):
        """显示当前图片的推理结果"""
        if self.current_image is None:
            messagebox.showerror("错误", "当前没有加载的图片")
            return
            
        try:
            # 创建掩码覆盖图像
            display_img = self.current_image.copy()
            
            # 应用掩码（半透明覆盖）
            for i, mask in enumerate(masks):
                # 将掩码调整到图像大小
                if mask.shape != display_img.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), 
                                    (display_img.shape[1], display_img.shape[0]))
                
                # 创建彩色掩码
                color = np.array([255, 255, 0])  # 黄色
                colored_mask = np.zeros_like(display_img)
                colored_mask[mask > 0.5] = color
                
                # 半透明叠加
                alpha = 0.4
                display_img = cv2.addWeighted(display_img, 1-alpha, colored_mask, alpha, 0)
            
            # 绘制标注
            self.draw_annotations(display_img)
            
            # 显示结果
            pil_image = Image.fromarray(display_img)
            
            # 缩放
            if self.scale_factor < 1.0:
                new_width = int(pil_image.size[0] * self.scale_factor)
                new_height = int(pil_image.size[1] * self.scale_factor)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            self.display_image = ImageTk.PhotoImage(pil_image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            messagebox.showinfo("成功", f"检测到 {len(masks)} 个掩码区域")
        except Exception as e:
            messagebox.showerror("错误", f"结果显示失败: {str(e)}")
            
    def display_results(self, results):
        """显示推理结果（向后兼容）"""
        if hasattr(results[0], 'masks') and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            self.image_results[self.current_image_path] = masks
            self.display_results_for_current_image(masks)
        else:
            messagebox.showwarning("警告", "未检测到任何掩码")
            
    def _generate_mask_filename(self, original_path):
        """生成掩码文件名"""
        dir = os.path.dirname(original_path)
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        return f"{dir+'/mask_'+base_name}.dat"
    
    def save_current_mask(self):
        """保存当前图片的掩码"""
        if self.current_image_path not in self.image_results:
            messagebox.showwarning("警告", "当前图片没有推理结果，请先运行推理")
            return
            
        try:
            # 生成默认文件名
            save_path = self._generate_mask_filename(self.current_image_path)
            with open(save_path, 'wb') as f:                      
                masks = self.image_results[self.current_image_path]                
                masks = masks.astype(np.uint8)*255
                f.write(struct.pack('<i', masks.shape[1]))     
                f.write(struct.pack('<i', masks.shape[2]))   
                f.write(struct.pack('<i', masks.shape[1]*masks.shape[2]))     
                f.write(masks.tobytes()) 
                messagebox.showinfo("成功", f"掩码已保存到: {save_path}")
                
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
            
    def save_all_masks(self):
        """批量保存所有掩码"""
        if not self.image_results:
            messagebox.showwarning("警告", "没有推理结果，请先运行推理")
            return
            
        result_count = len(self.image_results)
        
        if result_count == 1:
            # 只有一张图片，直接保存
            self.save_current_mask()
            return
            
        # 多张图片，保存为压缩包
        try:
            save_path = filedialog.asksaveasfilename(
                title="保存所有掩码",
                initialvalue=f"masks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                defaultextension=".zip",
                filetypes=[("ZIP文件", "*.zip"), ("所有文件", "*.*")]
            )
            
            if save_path:
                # 创建临时目录保存掩码文件
                import tempfile
                import shutil
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    saved_files = []
                    
                    # 保存每个掩码到临时目录
                    for img_path, masks in self.image_results.items():
                        mask_filename = self._generate_mask_filename(img_path)
                        mask_path = os.path.join(temp_dir, mask_filename)
                        
                        self._save_single_mask(masks, mask_path)
                        saved_files.append((mask_path, mask_filename))
                    
                    # 创建ZIP文件
                    with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for file_path, archive_name in saved_files:
                            zipf.write(file_path, archive_name)
                    
                messagebox.showinfo("成功", f"已保存 {len(saved_files)} 个掩码文件到压缩包: {save_path}")
                
        except Exception as e:
            messagebox.showerror("错误", f"批量保存失败: {str(e)}")
            
    def _save_single_mask(self, masks, save_path):
        """保存单个掩码文件"""
        # 合并所有掩码
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            combined_mask = np.logical_or(combined_mask, mask > 0.5)
        
        # 保存为二值图像
        mask_image = (combined_mask * 255).astype(np.uint8)
        cv2.imwrite(save_path, mask_image)
        
    def save_mask(self):
        """保存掩码（向后兼容）"""
        self.save_current_mask()

def main():
    root = tk.Tk()
    app = SAM2GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
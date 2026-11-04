import struct
import numpy as np
import json
import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path='models/selfie_multiclass_256x256.tflite'),
    running_mode=VisionRunningMode.IMAGE,
    output_category_mask=True,      # 输出类别掩码（推荐用于替换背景等）
    output_confidence_masks=False   # 是否需要置信度掩码（浮点数）
)

def extrcat_skin(folder_path):
    with ImageSegmenter.create_from_options(options) as segmenter:    
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if not filename.lower().endswith('.json'):
                    continue            
                file_path = os.path.join(root, filename)            
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                    continue
                if not isinstance(data, dict):
                    continue

                image_path = data.get("imagePath")
                if image_path is None:
                    continue 
                Qt = data.get("Qt")
                if Qt is None:
                    continue 
                if os.path.exists(image_path):
                    mp_image = mp.Image.create_from_file(image_path) 
                    segmentation_result = segmenter.segment(mp_image) 
                    category_mask = segmentation_result.category_mask
                    mask_numpy = category_mask.numpy_view()
                    skin_mask =(( mask_numpy==3).squeeze()*255).astype(np.uint8)
                    img = mp_image.numpy_view().copy()
                    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if contours:
                        mask = np.zeros_like(skin_mask)                     
                        cv2.drawContours(mask, contours, -1, color=255, thickness=-1)   
                        cv2.drawContours(img, contours, -1, color=(0,0,255), thickness=2)   
                        cv2.imwrite(image_path+'.bmp',img)
                        shapes = []
                        for contour in contours:                            
                            points = []
                            for p in contour:
                                ptSerialize = p[0].tolist()
                                points.append(ptSerialize)
                            shape = {"label": "polygon", "points": points, "group_id": "",
                                        "description": "", "shape_type": "point", "flags": {}, "mask": ""}
                            shapes.append(shape)
                        data['version']='5.4.1'
                        data['flags']={}
                        data['imageData']=None
                        data['imageHeight']=data['height']
                        data['imageWidth']=data['width']
                        data['shapes']=shapes 
                        with open(file_path, 'w') as f:
                            json.dump(data, f)


                        dir = os.path.dirname(file_path)
                        base_name = os.path.splitext(os.path.basename(file_path))[0]
                        save_path = f"{dir+'/mask_'+base_name}.dat"
                        with open(save_path, 'wb') as f:              
                            f.write(struct.pack('<i', mask.shape[0]))     
                            f.write(struct.pack('<i', mask.shape[1]))   
                            f.write(struct.pack('<i', mask.shape[0]*mask.shape[1]))     
                            f.write(mask.tobytes()) 
    
def recheck_skin(folder_path):
    with ImageSegmenter.create_from_options(options) as segmenter:    
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if not filename.lower().endswith('.json'):
                    continue            
                file_path = os.path.join(root, filename)            
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                    continue
                if not isinstance(data, dict):
                    continue

                image_path = data.get("imagePath")
                if image_path is None:
                    continue 
                Qt = data.get("Qt")
                if Qt is None:
                    continue 
                if os.path.exists(image_path+'.bmp'):
                    img=cv2.imread(image_path)
                    img_mask=cv2.imread(image_path+'.bmp')
                    b_img_mask, g_img_mask, r_img_mask = cv2.split(img_mask)

                    # 按条件生成二值图：red > 230 且 green < 20 且 blue < 20
                    mask = (r_img_mask > 230) & (g_img_mask < 20) & (b_img_mask < 20)
 
                    binary = (mask.astype(np.uint8)) * 255
 
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if contours:
                        mask = np.zeros_like(mask).astype(np.uint8)                     
                        cv2.drawContours(mask, contours, -1, color=255, thickness=-1)   
                        cv2.drawContours(img, contours, -1, color=(0,0,255), thickness=2)   
                        cv2.imwrite(image_path+'.bmp',img)
                        shapes = []
                        for contour in contours:                            
                            points = []
                            for p in contour:
                                ptSerialize = p[0].tolist()
                                points.append(ptSerialize)
                            shape = {"label": "polygon", "points": points, "group_id": "",
                                        "description": "", "shape_type": "point", "flags": {}, "mask": ""}
                            shapes.append(shape)
                        data['version']='5.4.1'
                        data['flags']={}
                        data['imageData']=None
                        data['imageHeight']=data['height']
                        data['imageWidth']=data['width']
                        data['shapes']=shapes 
                        with open(file_path, 'w') as f:
                            json.dump(data, f)
    
                        dir = os.path.dirname(file_path)
                        base_name = os.path.splitext(os.path.basename(file_path))[0]
                        save_path = f"{dir+'/mask_'+base_name}.dat"
                        with open(save_path, 'wb') as f:              
                            f.write(struct.pack('<i', mask.shape[0]))     
                            f.write(struct.pack('<i', mask.shape[1]))   
                            f.write(struct.pack('<i', mask.shape[0]*mask.shape[1]))     
                            f.write(mask.tobytes()) 

 



if __name__ == "__main__":
    folder = "D:/repo/colmapThird/data/a/result"
    extrcat_skin(folder)
    # recheck_skin(folder)
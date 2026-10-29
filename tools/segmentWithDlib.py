
import os
from pathlib import Path
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import dlib 
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from transformers import AutoModelForImageSegmentation
def findMaxFace(dets):
    if len(dets) == 0:
        return None
    maxFaceArea = 0
    maxFacIdx = 0
    for i, d in enumerate(dets):
        width = abs(d.rect.right()-d.rect.left())
        height = abs(d.rect.top()-d.rect.bottom())
        area = height*width
        if area > maxFaceArea:
            maxFaceArea = area
            maxFacIdx = i
    return dets[maxFacIdx]

def extract_image_paths(folder_path):
    use_dlib=True
    use_parseNet=False 
    cnn_face_detector = dlib.cnn_face_detection_model_v1(
        'models/mmod_human_face_detector.dat')
    landmarkPredictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    if use_parseNet:parser = bisenet.BiSeNet("models/faceparsingrResnet34.onnx")
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
            if use_dlib and os.path.exists(image_path):
                img = dlib.load_rgb_image(image_path)
                dets = cnn_face_detector(img)
                maxFace = findMaxFace(dets)
                landmarks = landmarkPredictor(img, maxFace.rect)
                if landmarks.num_parts == 0:
                    continue
                shapes = []
                points = []
                landmarks68 = np.random.rand(68,2)
                for i in range(landmarks.num_parts):
                    landmarks68[i,0]=landmarks.part(i).x
                    landmarks68[i,1]=landmarks.part(i).y 
                hull = ConvexHull(landmarks68)                
                hull_indices = hull.vertices
                for i in hull_indices:
                    ptSerialize = np.array([landmarks.part(i).x, landmarks.part(i).y])
                    ptSerialize = ptSerialize.tolist()
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
            if use_parseNet and os.path.exists(image_path):
                image = cv2.imread(str(image_path)) 
                mask = parser.parse(image)                
                unique_classes = len(set(mask.flatten()))
                print(f'  Parsed with {unique_classes} unique classes')
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                vis_result = bisenet.vis_parsing_maps(image_rgb, mask, save_image=False) 
                cv2.imwrite(str(image_path)+'.png', vis_result)


if __name__ == "__main__":
    
    birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
    folder = "D:/repo/colmapThird/data/a/result"
    items = extract_image_paths(folder)
    
    print(f"共找到 {len(items)} 个带 imagePath 字段的 JSON：")
    for item in items:
        print(f"{item['json_file']}  ->  {item['imagePath']}")
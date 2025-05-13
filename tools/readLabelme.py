import json
import numpy as np
import cv2
def readLabelmeMask(maskFile):
    with open(maskFile, 'r') as mask_file:
        maskData = json.load(mask_file) 
        assert len(maskData['shapes'])==1
        hullPts = maskData['shapes'][0]['points']
        w = maskData['imageWidth']
        h = maskData['imageHeight']
        mask = np.zeros((h,w,1),np.uint8)
        cv2.fillPoly(mask, np.int32([hullPts]), (255))
        return mask

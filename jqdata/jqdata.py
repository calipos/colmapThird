import os
import time
import akshare as ak 
print(ak.__version__)
from PIL import Image
import io
import numpy as np 
import easyocr
import pyautogui
import pyperclip  
import pandas as pd
import time
import numpy as np
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import cv2
def compare_images(img1_path, img2_path, threshold=30):
    """
    比较两张图片，显示不同的地方
    
    Args:
        img1_path: 第一张图片路径
        img2_path: 第二张图片路径
        threshold: 差异阈值，默认为30
    """
    # 读取图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("错误：无法读取图片，请检查路径")
        return
    
    # 检查尺寸是否一致
    if img1.shape != img2.shape:
        print(f"图片尺寸不一致: {img1.shape} vs {img2.shape}")
        return
    
    # 转换为灰度图（可选），便于比较
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 方法1：计算绝对差
    diff = cv2.absdiff(gray1, gray2)
    
    # 方法2：使用阈值标记差异区域
    _, diff_thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # 方法3：计算差异百分比
    total_pixels = diff.size
    diff_pixels = np.sum(diff > threshold)
    diff_percentage = (diff_pixels / total_pixels) * 100
    
    # 创建差异标记图（在原图上标记不同区域）
    marked_img = img1.copy()
    
    # 找差异区域的轮廓并标记
    contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(marked_img, contours, -1, (0, 0, 255), 2)  # 红色边框标记
    
    # 创建差异热力图
    diff_heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    
    # 显示结果
    plt.figure(figsize=(15, 10))
    
    # 原图1
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('图片1')
    plt.axis('off')
    
    # 原图2
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('图片2')
    plt.axis('off')
    
    # 差异图（灰度）
    plt.subplot(2, 3, 3)
    plt.imshow(diff, cmap='gray')
    plt.title(f'差异图 (差异: {diff_percentage:.2f}%)')
    plt.axis('off')
    
    # 阈值后的差异（二值化）
    plt.subplot(2, 3, 4)
    plt.imshow(diff_thresh, cmap='gray')
    plt.title('二值化差异')
    plt.axis('off')
    
    # 标记差异的图片
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))
    plt.title('标记差异区域')
    plt.axis('off')
    
    # 差异热力图
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB))
    plt.title('差异热力图')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print(f"图片尺寸: {img1.shape}")
    print(f"差异像素数: {diff_pixels}")
    print(f"差异百分比: {diff_percentage:.2f}%")
    print(f"平均差异值: {np.mean(diff):.2f}")
    print(f"最大差异值: {np.max(diff)}")
    
    return diff, diff_thresh, marked_img

time.sleep(1)
reader = easyocr.Reader(['ch_sim', 'en']) 

def getTimeString():
    local_time = time.localtime() 
    formatted_time = time.strftime("@%Y-%m-%d-%H-%M-%S", local_time)
    return formatted_time
def compareTimeStampBGrateThanA(a,b,diff):
    aInt=-1
    bInt=-1
    if   type(a) is str and a.startswith('@'):  
        aInt =  int(''.join(a.split('@')[1].split('-')))
    elif type(a) is int:
        aInt=a
    else:assert(False)
    if   type(b) is str and b.startswith('@'):  
        bInt =  int(''.join(b.split('@')[1].split('-')))
    elif type(b) is int:
        bInt=b
    else:assert(False)
    if aInt<0:
        return True
    elif aInt>0 and bInt>0 and bInt>aInt+diff:
        return True
    else: return False

def findLastDat(fileheader):
    for item in os.listdir('.'):
        if os.path.isfile(item) and item.startswith(fileheader): 
            print(f"📄 文件: {item}")
            stamp = item.split('.')[0].split(fileheader)[1]
            return stamp, pd.read_csv(item, encoding="utf-8-sig")
    return -1,None
addressPos=[1660, 60]
def putInAddress(www):
    pyautogui.click(addressPos[0], addressPos[1], button='left')
    pyautogui.write(www,interval=0.05)
    # pyperclip.copy(www)
    # pyautogui.hotkey('ctrl', 'v') 
    pyautogui.press('enter')
    time.sleep(0.1)
    pyautogui.press('enter')
    time.sleep(0.1)
def putInSearch(code,x,y):
    pyautogui.click(x,y, button='left')
    pyautogui.write(code,interval=0.05)
    # pyperclip.copy(www)
    # pyautogui.hotkey('ctrl', 'v')  
    time.sleep(0.2)
    pyautogui.press('enter')
    time.sleep(2)

def recogv2(path):
    
    ocr = PaddleOCR(use_angle_cls=True, lang='ch',cpu_threads=4) 
    result = ocr.predict(path) 
    for detection in result:
        print(f"识别到的文字: {detection[1]}, 置信度: {detection[2]:.2f}")
def findCodeSearch(img):
    result = reader.readtext(img) 
    searchsPos=[]
    searchsComment=[]
    for detection in result:
        if detection[1].find('关键字')>=0:
            searchsPos.append(detection[0])
            searchsComment.append(detection[1])
    if len(searchsPos)==1:
        return np.sum(searchsPos[0],axis=0)//4
    else :return None
def urlBaseCode(code):
    if code.startswith("600") or code.startswith("601") or code.startswith("603") or code.startswith("605") :
        return  "https://quote.eastmoney.com/concept/sh"+code+".html#chart-k-cyq"
    elif code.startswith("000") or code.startswith("001") or code.startswith("002")or code.startswith("300") :
        return  "https://quote.eastmoney.com/concept/sz"+code+".html#chart-k-cyq"
    else:return None
if __name__ == "__main__":  



    currentTime = getTimeString()
    codeListVersion, codeList= findLastDat('codeList')
    if(compareTimeStampBGrateThanA(codeListVersion,currentTime,3000000)):
        codeList = ak.stock_info_a_code_name() 
        print(codeList.head()) 
        codeList.to_csv("codeList"+currentTime+".csv", index=False, encoding="utf-8-sig")


 

    codeNumberList = codeList["code"].astype(str).str.zfill(6)
    for code in codeNumberList:
        url = urlBaseCode(code)
        if None==url:
            continue
        putInAddress(url)
        time.sleep(2)

        # pyperclip.copy("筹码分布")
        # pyautogui.hotkey('ctrl', 'f')  
        # pyautogui.hotkey('ctrl', 'v')  
        # time.sleep(0.2)

        # screenSearchBefore = pyautogui.screenshot() 
        # screenSearchBefore.save('screenSearchBefore.png') 
        pyautogui.moveTo(addressPos[0],500)
        
        pyautogui.scroll(-401)   
        pyautogui.click(1141,528, button='left')
        pyautogui.click(1141,614, button='left') 

        for mouseX in  [int(1258-9.43*x) for x in range(0,66)]:
            pyautogui.moveTo(mouseX,888)
            time.sleep(0.3)
            
            screenSearchAfter = pyautogui.screenshot()
            screenSearchAfter.save('screenSearchBefore.png')
            dailyInfo = pyautogui.screenshot(region=(530, 686, 110, 209))
            dailyInfo.save('dailyInfo.png')
            result = reader.recognize(np.array(dailyInfo))
            exit(-1)
            print(result)
        pyautogui.dragTo(1258,888,duration=3)
        for mouseX in  [int(1258-9.43*x) for x in range(0,66)]:
            pyautogui.moveTo(mouseX,888)
            time.sleep(0.3) 
            dailyInfo = pyautogui.screenshot(region=(530, 686, 110, 209))
            result = reader.recognize(np.array(dailyInfo))
            print(result)

        
        screenSearchAfter = pyautogui.screenshot()
        screenSearchAfter.save('screenSearchBefore.png')

        exit(0)
        # searchPos = findCodeSearch('full_screenshot.png')
        # if searchPos is not None:
        #     putInSearch(code,searchPos[0],searchPos[1])
        #     exit(0)
        # else:assert(False)
        # result = reader.recognize(np.array(screenshot))
        # screenshot.save('full_screenshot.png')  # 保存为文件
    exit(0)
 




            #     img_np = np.array(img)  # 关键：这行转换
            # result = reader.readtext(img_np)
            # return result
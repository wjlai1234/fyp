from imutils import contours
from flask import Flask, request, jsonify
from os import listdir
from os.path import isfile, join
from pdf2image import convert_from_path
#from skimage.filters import threshold_local
#from operator import itemgetter, attrgetter
from PIL import Image, ImageChops
from pyzbar.pyzbar import decode
from StudentCardDB import binarization
from pyzbar.pyzbar import ZBarSymbol
from tf_model_helper import TFModel
#from skimage import color
#import datefinder
#import textdistance
import csv
import cv2
import ftfy
import imutils
import json
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
import numpy as np
import os
import pandas as pd
import pytesseract
import re
import string
import time
import pathlib
import shutil
from flask import Flask, render_template, redirect, url_for, request, redirect, jsonify, Response
import os
import io
import base64
from pathlib import Path
'''Folder Location'''
path = pathlib.Path().resolve()

#server path
#path = r"/home/hostinger/ftp/myfyp1/scOCRInterface"

pytesseract.pytesseract.tesseract_cmd = r"C:/Users/Z/Tesseract-OCR/tesseract.exe" #version 5.0.0
#pytesseract.pytesseract.tesseract_cmd = r"C:/Users/p/Tesseract-OCR/tesseract.exe" #version 5.0.0

STUDENTCARD_FOLDER_PATH = str(path).replace('\\', '/') + "/StudentCardDB"
FACE_XML_PATH = str(path).replace('\\', '/') + "/haarcascade_frontalface_alt.xml"
OUTPUT_FOLDER_PATH = STUDENTCARD_FOLDER_PATH + "/Output"
PROCESS_FOLDER_PATH = STUDENTCARD_FOLDER_PATH + "/Process"
UPLOAD_FOLDER_PATH = STUDENTCARD_FOLDER_PATH + "/upload"
SCAN_LOGO_PATH = STUDENTCARD_FOLDER_PATH + "/yolov7"
detectLogoPath = SCAN_LOGO_PATH + "/detectLogo.py"
bestLogoPath = SCAN_LOGO_PATH + "/bestLogo.pt"
scanLogoImgFolder = SCAN_LOGO_PATH + "/runs/detect/exp"


isDebug = True

'''Student Card OCR Functions'''
def ConvertFileToImage(filepath):
    file_name = os.path.basename(filepath)
    output_filepath = OUTPUT_FOLDER_PATH + "/" + file_name
    if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
        print("Reading image file: ", filepath)
        img = cv2.imread(filepath) #[...,::-1]

    elif filepath.lower().endswith(('.pdf')):
        print("Reading PDF file: ", filepath)
        pages = convert_from_path(filepath, dpi=200, poppler_path= str(path) +'\bin')
        file_name_no_ext = os.path.splitext(file_name)[0]
        file_name = file_name_no_ext + '.jpg'
        for page in pages:
            page.save(output_filepath, 'JPEG')
            break
        img = cv2.imread(output_filepath)
        img = cv2.resize(img, (900, 600))
    if img.all() == None:
        print("file cannot be read")
    if isDebug:
        #RawImg Printed
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig1, (ax1) = plt.subplots(1, 1, figsize=(25, 8))
        ax1.imshow(img_rgb.astype('uint8'),cmap=plt.cm.gray)
        ax1.set_title("Raw image")
        ax1.set_axis_off()
        plt.show()
        #process_filepath = PROCESS_FOLDER_PATH + "/raw_img_" + file_name
        #cv2.imwrite(process_filepath, img)
        #raw_name = os.path.basename(process_filepath)
        #Replacing filename
        if filepath.lower().endswith(('.pdf')):
            output_filepath = UPLOAD_FOLDER_PATH + "/" + file_name
            cv2.imwrite(output_filepath, img)
        else:
            output_filepath = filepath
    return img, output_filepath

'''No use'''
def RotateImage_FaceDetection(image):
    img = image
    faceCascade = cv2.CascadeClassifier(FACE_XML_PATH)
    faces = faceCascade.detectMultiScale(
                img,
                scaleFactor=1.5,
                minNeighbors=5,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
    if isDebug:
        print(len(faces), " face detected")
    if len(faces) < 1:
        for angle in range(90,271,90):
            rotate_img= imutils.rotate(image, angle)
            faces = faceCascade.detectMultiScale(rotate_img, scaleFactor=1.5, minNeighbors=5, minSize=(50, 50),
                                                 flags=cv2.CASCADE_SCALE_IMAGE)
            if len(faces) > 0:
                img = rotate_img
                face_detected = True
                break
    return img, faces

def RotateImage_TextConfidenceLevel(img):
    temp_conf_level=[]
    temp_img=[]
    img=cv2.resize(img,(1000,700))
    temp_conf_level.append(CalcConf_RotateImage(img))
    temp_img.append(img)
    for i in range(90,271,90):
        rotate_img= imutils.rotate(img, i)
        img_rgb = cv2.cvtColor(rotate_img,cv2.COLOR_BGR2RGB)
        fig1, (ax1) = plt.subplots(1, 1, figsize=(25, 8))
        ax1.imshow(img_rgb.astype('uint8'),cmap=plt.cm.gray)
        ax1.set_title(i)
        ax1.set_axis_off()
        plt.show()
        confidence_level=CalcConf_RotateImage(rotate_img)
        temp_conf_level.append(confidence_level)
        temp_img.append(rotate_img)
    #print(temp_conf_level)
    #print(temp_conf_level.index(max(temp_conf_level)))
    #k=temp_img[temp_conf_level.index(max(temp_conf_level))]
    return temp_conf_level, temp_img
def CalcConf_RotateImage(img):
    img_data = pytesseract.image_to_data(img,lang="ind",output_type="data.frame")
    img_data = img_data.dropna()
    #print(img_data)
    if(img_data.empty==False):
        n_boxes = len(img_data['text'])
        pd_index = []
        for i in range(n_boxes):
            pd_index.append(i)
        o = pd.Series(pd_index)
        img_data = img_data.set_index([o])
        img_data.head()
        if((len(img_data)==1 and img_data['text'][0]==' ') or (len(img_data)==2 and img_data['text'][0]==' ')):
            return 0
        else:
            total = img_data.sum(axis=0)
            count = img_data.count(axis=0)
            avgConf = total.conf / count.conf
            return avgConf
    else:
        return 0
def RetrieveImage_FaceRatio(image, faces):
    raw_width = 1600
    raw_height = 1000
    image_face = image.copy()
    x, y, w, h = faces[len(faces)-1]
    cv2.rectangle(image_face, (x, y), (x+w, y+h), (0, 255, 0), 20)
    # card_width = 8 face_width
    # card_height = 4 face_height
    # face_x = 5 face_width from left
    # face_y = 2 face_height from top
    start_x = 0 if x-(5*w) < 0 else x-(5*w)
    start_y = 0 if y-(2*h) < 0 else y-(2*h)
    end_x = image.shape[1] if x+(2*w) > image.shape[1] else x+(2*w)
    end_y = image.shape[0] if y+(3*h) > image.shape[0] else y+(3*h)
    if isDebug:
        print("from ", image.shape[1], image.shape[0], " crop to ", start_x, end_x, start_y, end_y)
    region_0 = image[start_y:end_y, start_x:end_x]
    image_resize = cv2.resize(region_0, (raw_width, raw_height))
    if isDebug:
        image_face_rgb = cv2.cvtColor(image_face,cv2.COLOR_BGR2RGB)
        image_resize_rgb = cv2.cvtColor(image_resize,cv2.COLOR_BGR2RGB)
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 8))
        ax1.imshow(image_face_rgb.astype('uint8'),cmap=plt.cm.gray)
        ax1.set_title("Detected Face")
        ax1.set_axis_off()
        ax2.imshow(image_resize_rgb.astype('uint8'),cmap=plt.cm.gray)
        ax2.set_title("Resized image")
        ax2.set_axis_off()
        plt.show()
    return image_resize, image_face
def SegmentImage_FaceRatio(image_gray, image_rgb):
    faceCascade = cv2.CascadeClassifier(FACE_XML_PATH)
    faces = faceCascade.detectMultiScale(
                image_rgb,
                scaleFactor=1.5,
                minNeighbors=5,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
    if isDebug:
        print(len(faces), " face is detected")
    x, y, w, h = faces[len(faces)-1]
    #cv2.rectangle(img_resize_rgb, (x, y), (x+w, y+h), (0, 255, 0), 20)
    region_1 = image_gray[0:image_gray.shape[0], 0:x-50]
    region_2 = image_gray[y+h+50:y+h+300, x-100:image_gray.shape[1]-50]
    region_3 = image_rgb[y-80:y+h+80, x-50:x+w+50]
    if isDebug:
        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 8))
        ax1.imshow(region_1.astype('uint8'),cmap=plt.cm.gray)
        ax1.set_title("Region 1 image")
        ax1.set_axis_off()
        ax2.imshow(region_2.astype('uint8'),cmap=plt.cm.gray)
        ax2.set_title("Region 2 image")
        ax2.set_axis_off()
        region_3_rgb = cv2.cvtColor(region_3, cv2.COLOR_BGR2RGB)
        ax3.imshow(region_3_rgb.astype('uint8'),cmap=plt.cm.gray)
        ax3.set_title("Region 3 image")
        ax3.set_axis_off()
        plt.show()
    return region_1, region_2, region_3
'''No use'''

def RetrieveImage_QRCodeRatio(image, image_qrcode, x, y, w, h, filepath):
    raw_width = 1000
    raw_height = 700
    cv2.rectangle(image_qrcode, (x, y), (x+w, y+h), (0, 255, 0), 20)
    # card_width = 6.5 face_width
    # card_height = 3.5 face_height
    # face_x = 5.5 face_width from left # 6.5 - 5.5 = 1.0
    # face_y = 1 face_height from top # 3.5 - 1 = 2.5
    start_x = 0 if int(x-(5.5*w)) < 0 else int(x-(5.5*w))
    start_y = 0 if int(y-(1.5*h)) < 0 else int(y-(1.5*h))
    end_x = image.shape[1] if int(x+(1.0*w)) > image.shape[1] else int(x+(1.0*w))
    end_y = image.shape[0] if int(y+(2.5*h)) > image.shape[0] else int(y+(2.5*h))
    if isDebug:
        print("from ", image.shape[1], image.shape[0], " crop to ", start_x, end_x, start_y, end_y)
    region_0 = image[start_y:end_y, start_x:end_x]
    image_resize = cv2.resize(region_0, (raw_width, raw_height))
    if isDebug:
        #DetectedQR and Resized Printed
        image_qrcode_rgb = cv2.cvtColor(image_qrcode,cv2.COLOR_BGR2RGB)
        image_resize_rgb = cv2.cvtColor(image_resize,cv2.COLOR_BGR2RGB)
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 8))
        ax1.imshow(image_qrcode_rgb.astype('uint8'),cmap=plt.cm.gray)
        ax1.set_title("Detected QR Code")
        ax1.set_axis_off()
        ax2.imshow(image_resize_rgb.astype('uint8'),cmap=plt.cm.gray)
        ax2.set_title("Resized image")
        ax2.set_axis_off()
        plt.show()

        file_name = os.path.basename(filepath)
        process_filepath = PROCESS_FOLDER_PATH + "/detected_QRCode_" + file_name
        if image_qrcode_rgb is not None:
            cv2.imwrite(process_filepath, image_qrcode_rgb)
        detected_QRCode_name = os.path.basename(process_filepath)
        process_filepath = PROCESS_FOLDER_PATH + "/resized_img_" + file_name
        if image_resize_rgb is not None:
            cv2.imwrite(process_filepath, image_resize_rgb)
        resized_img_name = os.path.basename(process_filepath)
    return image_resize, detected_QRCode_name, resized_img_name

def RemoveQRCode(image, filepath):
    processedImage = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours and filter for QR code
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if isDebug:
        print("Detected QR boxes number : ", len(cnts))
    largestArea = 0
    selectedRect = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        x,y,w,h = cv2.boundingRect(approx)
        area = cv2.contourArea(c)
        ar = w / float(h)
        if len(approx) == 4 and area > 5000 and (ar > .85 and ar < 1.3):
            if area > largestArea:
                selectedRect = [x,y,w,h]
                largestArea = area
    if len(cnts) > 0 and largestArea != 0:
        if isDebug:
            print("Approx : 4, SelectedArea : ", largestArea)
        [x,y,w,h] = selectedRect
        cv2.rectangle(processedImage, (x, y), (x + w, y + h), (255,255,255), -1)
        processedImage, detectedQRCode, resizedImg = RetrieveImage_QRCodeRatio(processedImage, image, x, y, w, h, filepath)
    else:
        detectedQRCode = ""
        resizedImg = ""
    return processedImage, detectedQRCode, resizedImg

def FilterImage_Threshold(image, value, displayDebugImage, filepath, saveImg):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, image = cv2.threshold(gray, value[0], 255, value[1])
    if isDebug and displayDebugImage:
        #ThresholdImg Printed
        fig1, (ax1) = plt.subplots(1, 1, figsize=(25, 8))
        ax1.imshow(image.astype('uint8'),cmap=plt.cm.gray)
        ax1.set_title(value)
        ax1.set_axis_off()
        plt.show()

        file_name = os.path.basename(filepath)
        process_filepath = PROCESS_FOLDER_PATH + "/threshold_img_" + file_name
        if image is not None and saveImg:
            cv2.imwrite(process_filepath, image)
        threshold_name = os.path.basename(process_filepath)
    else:
        threshold_name = ""
    return image, threshold_name

def ProcessTesseract(image, filepath):
    image_to_string_result = pytesseract.image_to_string(image, lang="eng",config="--psm 6")
    img_data = pytesseract.image_to_data(image,lang="eng",config="--psm 6", output_type='data.frame')
    result = Tesseract_CleanText_1(image_to_string_result)
    result = Tesseract_CleanText_2(result)
    image_to_string_result = result.upper()

    # loop over each of the individual text localizations
    boxes_text_name = ""
    for i in range(0, len(img_data["text"])):
        # extract the bounding box coordinates of the text region from
        # the current result
        x = img_data["left"][i]
        y = img_data["top"][i]
        w = img_data["width"][i]
        h = img_data["height"][i]

        # extract the OCR text itself along with the confidence of the
        # text localization
        text = img_data["text"][i]
        conf = int(img_data["conf"][i])

        # can be define by user
        min_conf = 60

        # filter out weak confidence text localizations
        if conf > min_conf:
            # strip out non-ASCII text so we can draw the text on the image
            # using OpenCV, then draw a bounding box around the text along
            # with the text itself
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # Text Detected Image with boxes Printed
            file_name = os.path.basename(filepath)
            process_filepath = PROCESS_FOLDER_PATH + "/boxes_text_img_" + file_name
            cv2.imwrite(process_filepath, image)
            boxes_text_name = os.path.basename(process_filepath)

    img_data = img_data.dropna()
    total = img_data.sum(axis=0)
    count = img_data.count(axis=0)
    avgConf = total.conf / count.conf
    avgConf_str = str(("%.2f" % avgConf) + " %")
    if avgConf < 50:
        print("=============================================================")
        print("===== THIS IMAGE IS STRAIGHTLY NOT ABLE TO BE PROCESSED =====")
        print("===== ERROR : EXTREMELY LOW CONFIDENCE LEVEL            =====")
        print("=============================================================")
    if isDebug:
        print("Result from IMG_TO_STRING + FILTER")
        print("==================================")
    result, validity_coordinate = Tesseract_Image_To_Data_Postprocess(img_data)
    result = Tesseract_CleanText_1(result)
    result = Tesseract_CleanText_2(result)
    image_to_data_result = result.upper()
    image_to_data_image = image
    if isDebug:
        print(image_to_data_result)
        #Results Img_To_String Printed
        result_img_to_string = open(f'StudentCardDB/Process/img_to_string.txt', 'w')
        if (image_to_data_result != ""):
            result_img_to_string.write(f'{image_to_data_result}\n')
        else:
            result_img_to_string.write(f'Unable to detect!')
        result_img_to_string.close()
    return avgConf_str, image_to_string_result, image_to_data_result, image_to_data_image, boxes_text_name, validity_coordinate

def ProcessTesseract_MaximumConfidenceLevel1(img, arrayThreshold):
    print("Checking maximum confidence level 1 on OCR procress...")
    #arrayThreshold=[[100,cv2.THRESH_TRUNC],[125,cv2.THRESH_TRUNC],[150,cv2.THRESH_TRUNC],
    #                [175,cv2.THRESH_TRUNC],[200,cv2.THRESH_TRUNC],[225,cv2.THRESH_TRUNC],
    #                [100,cv2.THRESH_OTSU],[125,cv2.THRESH_OTSU],[150,cv2.THRESH_OTSU],
    #                [175,cv2.THRESH_OTSU],[200,cv2.THRESH_OTSU],[225,cv2.THRESH_OTSU]]
    #[130,cv2.THRESH_TRUNC],[150,cv2.THRESH_TRUNC],[170,cv2.THRESH_TRUNC],[190,cv2.THRESH_TRUNC]
    #temp=[[50,cv2.THRESH_TRUNC],[70,cv2.THRESH_TRUNC],[90,cv2.THRESH_TRUNC],[110,cv2.THRESH_TRUNC]]

    row = math.ceil(len(arrayThreshold)/3)
    if isDebug:
        print("Row : ", row)
    temp_conf=[]
    temp_img=[]
    for i in arrayThreshold:
        threshold = FilterImage_Threshold(img, i, False)
        img_data = pytesseract.image_to_data(threshold,lang="ind",config="--psm 6", output_type='data.frame')
        img_data = img_data.dropna()
        total = img_data.sum(axis=0)
        count = img_data.count(axis=0)
        avgConf = total.conf / count.conf
        #print("Confidence Level with parameter ",i," : "+str(("%.2f" % avgConf) + " %"))
        temp_conf.append(avgConf)
        temp_img.append(threshold)
    u=temp_conf.index(max(temp_conf))
    image=temp_img[u]
    if isDebug:
        fig1, ax = plt.subplots(row, 3, figsize=(25, 20))
        count=0
        for i in range(row):
            for j in range(3):
                if(count<len(temp_conf)):
                    ax[i,j].imshow(cv2.cvtColor(temp_img[count],cv2.COLOR_BGR2RGB).astype('uint8'),cmap=plt.cm.gray)
                    tempStr = str(arrayThreshold[count]) + " " + str(np.round(temp_conf[count], 3)) + "%"
                    ax[i,j].set_title(tempStr)
                    ax[i,j].set_axis_off()
                    count+=1
                else:
                    break
        plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
        plt.show()
        print(arrayThreshold[u], temp_conf[u], "% is selected")
    return image, arrayThreshold[u], temp_conf[u]

def ProcessTesseract_MaximumConfidenceLevel2(img, parameter, lvl1Image, lvl1Conf):
    print("Checking maximum confidence level 2 on OCR procress...")
    x=2
    temp=[ [parameter[0],parameter[1]],

        [parameter[0]-(3*x),parameter[1]],
            [parameter[0]-(2*x),parameter[1]],
            [parameter[0]-x,parameter[1]],
            [parameter[0]+x,parameter[1]],
            [parameter[0]+(2*x),parameter[1]],
            [parameter[0]+(3*x),parameter[1]]]
    temp_conf=[]
    temp_img=[]
    temp_conf.append(lvl1Conf)
    temp_img.append(lvl1Image)
    for i in range(1,len(temp)):
        threshold = FilterImage_Threshold(img, temp[i], False)
        img_data = pytesseract.image_to_data(threshold,lang="ind",config="--psm 6", output_type='data.frame')
        img_data = img_data.dropna()
        total = img_data.sum(axis=0)
        count = img_data.count(axis=0)
        avgConf = total.conf / count.conf
        #print("Confidence Level with parameter ",i," : "+str(("%.2f" % avgConf) + " %"))
        temp_conf.append(avgConf)
        temp_img.append(threshold)
    u=temp_conf.index(max(temp_conf))
    image=temp_img[u]
    if isDebug:
        fig1, ax = plt.subplots(3, 3, figsize=(20, 15))
        count=1
        for i in range(3):
            for j in range(3):
                if(i!=1):
                    if(count<len(temp_conf)):
                        ax[i,j].imshow(cv2.cvtColor(temp_img[count],cv2.COLOR_BGR2RGB).astype('uint8'),cmap=plt.cm.gray)
                        tempStr = str(temp[count]) + " " + str(np.round(temp_conf[count], 3)) + "%"
                        ax[i,j].set_title(tempStr)
                        ax[i,j].set_axis_off()
                        count+=1
                    else:
                        break
                else:
                    continue
        ax[1,1].imshow(cv2.cvtColor(temp_img[0],cv2.COLOR_BGR2RGB).astype('uint8'),cmap=plt.cm.gray)
        tempStr = str(temp[0]) + " " + str(np.round(temp_conf[0], 3)) + "%"
        ax[1,1].set_title(tempStr)
        ax[1,1].set_axis_off()
        plt.subplots_adjust(hspace = 0.01, wspace = 0.01)
        fig1.delaxes(ax[1,0])
        fig1.delaxes(ax[1,2])
        plt.show()
        print(temp[u], temp_conf[u], "% is selected")
    return image,temp[u]

def Tesseract_Image_To_Data_Postprocess(img_data):
    print(img_data)
    img_data = img_data.dropna()
    validity_string = 'validity'
    validity_coordinate = []
    n_boxes = len(img_data['text'])
    pd_index = []
    for i in range(n_boxes):
        pd_index.append(i)
    s = pd.Series(pd_index)
    img_data = img_data.set_index([s])
    num = 2
    num2 = 2
    num3 = 2
    text = ""
    for i in range(n_boxes):
        if int(img_data.loc[i, 'block_num']) < num:
            if int(img_data.loc[i, 'line_num']) < num2:
                if int(img_data.loc[i, 'par_num']) < num3:
                    text += img_data.loc[i, 'text'] + " "
                else:
                    num3 += 1
                    text += "\n"
                    text += img_data.loc[i, 'text'] + " "
            else:
                num2 += 1
                text += "\n"
                text += img_data.loc[i, 'text'] + " "
        else:
            num += 1
            text += "\n"
            text += img_data.loc[i, 'text'] + " "

        text = img_data.loc[i, 'text'].lower()
        if text.find(validity_string) != -1:
            validity_coordinate.append(img_data["left"][i])
            validity_coordinate.append(img_data["top"][i])
            validity_coordinate.append(img_data["width"][i])
            validity_coordinate.append(img_data["height"][i])

    print(validity_coordinate)
    return text, validity_coordinate

def Tesseract_CleanText_1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('—', '-', text)
    text = re.sub('…', '', text)
    text = re.sub('“', '', text)
    remove = string.punctuation
    # don't remove hyphens
    remove = remove.replace("-", "")
    remove = remove.replace("/", "")
    pattern = r"[{}]".format(remove)  # create the pattern
    text = re.sub(pattern, "", text)
    #text = re.sub('\w*\d\w*', '', text)
    text = os.linesep.join([s for s in text.splitlines() if s])
    return text

def Tesseract_CleanText_2(text):
    text = ftfy.fix_text(text)
    text = ftfy.fix_encoding(text)
    return text

def ExtractValue_PredefinedForeList(detectedText, predefinedList):
    #print(detectedText)
    result = ["#####"] * len(predefinedList)
    textlines = detectedText.split("\n")
    print(textlines)
    for textline in textlines:
        for keyword in predefinedList:
            if result[predefinedList.index(keyword)] == "#####":
                if keyword in textline:
                    #result.append(predefinedText," : ")
                    try:
                        textline = textline.split(keyword)
                        textline = textline[-1].split(':')
                        textline = textline[-1].replace("-","")
                        result[predefinedList.index(keyword)] = textline
                        break
                    except:
                        print("Error occurred at ExtractValue_PredefinedForeList")
    if isDebug:
        print("=======================")
        print("### Extracted Value ###")
        print("=======================")
        print(result)
    return result

def ExtractValue_PredefinedForeBackList(detectedText, predefinedForeList, predefinedBackList):
    if isDebug:
        print("=======================")
        print("### Extracted Value ###")
        print("=======================")
    result = ["#####"] * len(predefinedForeList)
    textlines = detectedText.split("\n")
    for textline in textlines:
        for keyword in predefinedForeList:
            if result[predefinedForeList.index(keyword)] == "#####":
                if keyword in textline:
                    a = textline.find(keyword)
                    for gredword in predefinedBackList:
                        if gredword in textline:
                            ### RE, not work, to be done
                            ##reString =  keyword + '(*?)' + gredword
                            ##output = re.search(reString, textline).group(1)
                            b = textline.find(gredword)
                            output = textline[a+len(keyword):b]
                            ## finetune work ##
                            if "T" in output:
                                output = output.replace("T","+")
                            if len(output) < 6:
                                detected = True
                                textline = textline[b+len(gredword):]
                                result[predefinedForeList.index(keyword)] = output
                                if isDebug:
                                    print(keyword," : ", end=" ")
                                    print(output)
                                break
    return result

def ExtractValue_SpecificPattern(detectedText, regularExpression, returnGroupIndex, deliminator, isRemoveCRLF, isRemoveSpace):
    if isDebug:
        print("=======================")
        print("### Extracted Value ###")
        print("=======================")
    if isRemoveSpace:
        detectedText = detectedText.replace(" ", "")
    if isRemoveCRLF:
        detectedText = detectedText.replace("\n", " ")
    extractedValue = ""
    try:
        if len(deliminator) > 0:
            textlines = detectedText.split(deliminator)
            for textline in textlines:
                if isDebug:
                    print("textline : ", textline)
                result = re.search(regularExpression, textline)
                if result:
                    extractedValue = result.group(returnGroupIndex)
                    if isDebug:
                        print("extracted : ", extractedValue)
                    break
        else:
            result = re.search(regularExpression, detectedText)
            if result:
                extractedValue = result.group(returnGroupIndex)
                if isDebug:
                    print("extracted : ", extractedValue)
    except:
        if isDebug:
            print("Error occured in ExtractValue_SpecificPattern")
    return extractedValue

def ExtractValue_BetweenLines(textlines,KeywordBeforeLine,KeywordAfterLine,regularExpression):
    textlines=textlines.split("\n")
    a,b=0,0
    for i in range(len(textlines)):
        if(KeywordBeforeLine in textlines[i]):
            a=i
            break
    for i in range(len(textlines)):
        if(KeywordAfterLine in textlines[i]):
            b=i
            break
    value=""
    print(a,b)
    if(b-a==2 or (b==0 and a!=0)):
        for i in (range(len(textlines)-a)):
            result = re.search(regularExpression, textlines[a+i+1])
            if result:
                value = result.group(0)
                if isDebug:
                    print("extracted : ", value)
                break
    elif(b-a>2):
        for i in range(a+1,b):
            value+=textlines[i].strip()
    return value

def ExtractValue_LastLine(textlines):
    textlines=textlines.split("\n")
    return textlines[-1]

def Extract_OCRA(region):
    #Number=pytesseract.image_to_string(region, lang="num_4",config='--psm 6')
    Number=pytesseract.image_to_string(region, lang="eng",config='--psm 6')
    return Number

def Extract_Font(gray,location,t):
    Result=[]
    for (i, (x, y, w, h)) in enumerate(location):
        group = gray[y - 5:y +h + 5, x - 5:x + w + 5]
        #group = cv2.threshold(group, 100, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = imutils.grab_contours(digitCnts)
        digitCnts = contours.sort_contours(digitCnts,method="left-to-right")[0]
        for c in digitCnts:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            if(t==0):
                Number = Extract_OCRA(roi)
            elif(t==1):
                Number = pytesseract.image_to_string(roi, lang="ind",config='--psm 6')
            Result.append(str(Number))
    return Result

def Crop_OCRATYPEFONT(dir,n):
    RectangleKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    SquareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image = cv2.imread(dir)
    #print('shape',image.shape)
    if (image.shape[1] > 300):
        image = imutils.resize(image, width=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ImgTophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, RectangleKernel)
    gradient_X = cv2.Sobel(ImgTophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradient_X = np.absolute(gradient_X)
    (minimumvalue, maximumvalue) = (np.min(gradient_X), np.max(gradient_X))
    gradient_X = (255 * ((gradient_X - minimumvalue) / (maximumvalue - minimumvalue)))
    gradient_X = gradient_X.astype("uint8")
    gradient_X = cv2.morphologyEx(gradient_X, cv2.MORPH_CLOSE, RectangleKernel)
    threshold = cv2.threshold(gradient_X, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, SquareKernel)
    cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = imutils.grab_contours(cntrs)
    location = []
    for (i, c) in enumerate(cntrs):
        (x, y, w, h) = cv2.boundingRect(c)
        #bpjs num ,ktp, bpjs cardn num
        if ( (95>w>70 and 11>h>7 and x>18 and n==0) or ( 135>w>110 and 14>h>7 and n==1) or (38>w>21 and 14>h>8 and n==2)  ) : #bpjs num
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            location.append((x, y, w, h))
            print("Coordinate (x,y,w,h) : "+str(x)+" "+ str(y)+ " "+str(w)+" "+str(h))
    if isDebug:
        img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        fig1, (ax1) = plt.subplots(1, 1, figsize=(25, 8))
        ax1.imshow(img_rgb.astype('uint8'),cmap=plt.cm.gray)
        ax1.set_title("OCR A Region")
        ax1.set_axis_off()
        plt.show()
    location.sort(key=lambda x:x[0])
    number=Extract_Font(gray,location,0)
    number=''.join(number)
    if (n==0 and len(number)!=11) or (n==2 and len(number)!=16):
        number=Extract_Font(gray,location,1)
    return re.sub("[^0-9]","",''.join(number))

def FaceDetector(img, img_copy, img_path):
    faceDetected = False
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Loading the required haar-cascade xml classifier file
    haar_cascade = cv2.CascadeClassifier(FACE_XML_PATH)

    # Face Detection Method
    faces_rect = haar_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces_rect) <= 0:
        print("Face not recognized!")
        detectedFaceImg = ""
    else:
        # Iterating through rectangles of detected faces
        for (x, y, w, h) in faces_rect:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        file_name = os.path.basename(img_path)
        process_filepath = PROCESS_FOLDER_PATH + "/detected_face_" + file_name
        cv2.imwrite(process_filepath, img_copy)
        detectedFaceImg = os.path.basename(process_filepath)

        #removing face
        if x is not None:
            start_point = (x - 50, y - 60)
            end_point = (x + w + 50, y + h + 180)
            cv2.rectangle(img, start_point, end_point, (255, 255, 255), -1)
            faceDetected = True

    return detectedFaceImg, faceDetected, img

def BarcodePreprocessing(image):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray


def BarcodeReader(img_scanBC, img_copy, img_path):
    barcodeDetected = False
    # Decode the barcode image
    # preprocessedImg, threshold_filepath = FilterImage_Threshold(img_scanBC, [100,cv2.THRESH_TRUNC], True, img_path, False)
    if img_scanBC is not None:
        preprocessedImg = BarcodePreprocessing(img_scanBC)
    else:
        preprocessedImg = img_scanBC

    detectedBarcodes = decode(preprocessedImg, symbols=[ZBarSymbol.CODE128])

    if not detectedBarcodes:
        print("Barcode Not Detected!")
        detectedBarcodeImg = ""
    else:
        # Traverse through all the detected barcodes in image
        for barcode in detectedBarcodes:

            # Locate the barcode position in image
            (x, y, w, h) = barcode.rect

            # Put the rectangle in image using
            cv2.rectangle(img_copy, (x - 10, y - 10),
                          (x + w + 10, y + h + 10),
                          (0, 255, 0), 2)

            #save detected barcode in process
            file_name = os.path.basename(img_path)
            process_filepath = PROCESS_FOLDER_PATH + "/detected_barcode_" + file_name
            cv2.imwrite(process_filepath, img_copy)
            detectedBarcodeImg = os.path.basename(process_filepath)

            #Crop Img based on barcode coordinate
            # crop_img = img_copy[(y - h * 4 - 15):(y - 15), (x - 20):(x + w + h)]
            # croppedImg = 255 * np.ones_like(img_copy)
            # croppedImg[(y - h * 4 - 15):(y - 15), (x - 20):(x + w + h)] = crop_img

            #remove barcode
            start_point = (x - 13, y - 13)
            end_point = (x + w + 13, y + h + 13)
            cv2.rectangle(img_scanBC, start_point, end_point, (255, 255, 255), -1)

            #Validation Barcode
            if barcode.data != "":
                #print("Data & Type")
                barcodeOutput = barcode.data.decode()
                #print(barcodeOutput.strip())
                #print(barcode.type)
                if(len(barcodeOutput) == 16):
                    barcodeDetected = True

    return detectedBarcodeImg, barcodeDetected, img_scanBC


def LogoScanner(file_name, img_path):
    detectedlogo = False
    os.system("python " + detectLogoPath + " --weights " + bestLogoPath + " --source " + img_path + " --save-txt --save-conf --exist-ok")
    if os.path.isfile(PROCESS_FOLDER_PATH + "/scanned_logo_" + file_name):
        detectedLogo_path = "/scanned_logo_" + file_name
        detectedlogo = True
        # with open(PROCESS_FOLDER_PATH + '/scanned_logo_results/result.txt') as file:
        #     for line in file:
        #         data = line.split()
        # print(data)
    else:
        detectedLogo_path = ""
        data = ""

    return detectedLogo_path, detectedlogo


def Process_StudentID(img_path):
    start_time = time.time()

    img, output_filepath = ConvertFileToImage(img_path)

    img_path = output_filepath  #replacing pdf to jpg filepath

    #arrayThreshold=[[80,cv2.THRESH_TRUNC],[110,cv2.THRESH_TRUNC],[140,cv2.THRESH_TRUNC],
    #                [170,cv2.THRESH_TRUNC],[200,cv2.THRESH_TRUNC],[230,cv2.THRESH_TRUNC]]
    #lvl1img,parameter,lvl1conf = ProcessTesseract_MaximumConfidenceLevel1(img, arrayThreshold)
    #print( lvl1img,parameter,lvl1conf)
    #img,parameter = ProcessTesseract_MaximumConfidenceLevel2(img,parameter,lvl1img, lvl1conf)
    #print(img,parameter) #5,12,15 perfect (1,7,8,10,14,syarizal.pdf bpjs num dan card num perfect)  (11 salah smua)
    #dyan permadi namnaya gk klaur

    img_copy = img.copy()
    img_copy1 = img.copy()

    file_name = os.path.basename(img_path)
    process_filepath = PROCESS_FOLDER_PATH + "/raw_img_" + file_name
    cv2.imwrite(process_filepath, img_copy)
    raw_img_filepath = os.path.basename(process_filepath)

    #Detect & Remove Logo
    detectedLogo_path, logoDetected = LogoScanner(file_name, img_path)
    if logoDetected:
        img = cv2.imread(PROCESS_FOLDER_PATH + "/cropped_logo_" + file_name)

    #Detect & Remove Barcode
    detectedBarcode_path, barcodeDetected, croppedImg = BarcodeReader(img, img_copy, img_path)

    #Detect & Remove Face
    detectedFace_path, faceDetected, croppedImg = FaceDetector(croppedImg, img_copy1, img_path)

    if faceDetected or logoDetected or barcodeDetected:
        #Pre-processing filter
        img, threshold_filepath = FilterImage_Threshold(croppedImg, [100,cv2.THRESH_TRUNC], True, img_path, True)

        #OCR Process
        conf, result1, result2, image2, boxes_text_filepath, validity_coordinate = ProcessTesseract(img, img_path)
    else:
        conf, result1, result2, image2, boxes_text_filepath, validity_coordinate = ""

    #Start Detecting Card's Info
    user_defined_list = ["Name",
                         "Student ID",
                         "Validity",
                         "Time Taken",
                         "raw_img_filepath",
                         "threshold_filepath",
                         "boxes_text_filepath",
                         "barcodeDetected",
                         "detectedBarcode_path",
                         "faceDetected",
                         "detectedFace_path",
                         "logoDetected",
                         "detectedLogo_path"]
    result = ["#####"] * len(user_defined_list)
    Temp_BPJSNumber=''
    Temp_BPJSCardNumber=''

    # if (img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))):
    #     Temp_BPJSNumber = Crop_OCRATYPEFONT(img_path, 0)
    #     Temp_BPJSCardNumber = Crop_OCRATYPEFONT(img_path, 2)
    #     #print("hahahaha",len(Temp_BPJSCardNumber))
    # else:
    #     Temp_BPJSNumber = Crop_OCRATYPEFONT(output_filepath, 0)
    #     Temp_BPJSCardNumber = Crop_OCRATYPEFONT(output_filepath, 2)

    #Temp_BPJSNumber = Crop_OCRATYPEFONT(img_path, 0)
    #Temp_BPJSCardNumber = Crop_OCRATYPEFONT(img_path, 2)

    # if (len(Temp_BPJSCardNumber) == 16):
    #     cardNumber = Temp_BPJSCardNumber
    # else:
    #     cardNumber = ExtractValue_SpecificPattern(result1,
    #                                               r'\b[0-9]{16}\b',
    #                                               0, "", False, True)

    # if (len(Temp_BPJSNumber) == 11):
    #     bpjsNumber = Temp_BPJSNumber
    #     TempNama = ExtractValue_SpecificPattern(result1,
    #                                             r'\b[0-9]{10}\b|\b[0-9]{11}\b|\b[0-9]{12}\b|\b[0-9]{13}\b',
    #                                             0, "", True, False)
    #     name = ExtractValue_BetweenLines(result1, TempNama, "", "(.*[A-Z]){5,}")
    # else:
    #     bpjsNumber = ExtractValue_SpecificPattern(result1,
    #                                               r'\b[0-9]{10}\b|\b[0-9]{11}\b|\b[0-9]{12}\b|\b[0-9]{13}\b',
    #                                               0, "", True, False)
    #     name = ExtractValue_BetweenLines(result1, bpjsNumber, "", "(.*[A-Z]){5,}")

    studentID = ExtractValue_SpecificPattern(result1, r'[\d]{2}[A-Z]{3,4}[\d]{4,5}', 0, "", True, False)
    studentName = ExtractValue_SpecificPattern(result1, '(?:[A-Z]+){3,5} [A-Z]{2,5} [A-Z]{2,5}', 0, "", True, False)
    validity = ExtractValue_SpecificPattern(result1, '[\d]+\-[\d]+\-[\d]+ \- [\d]+\-[\d]+\-[\d]+', 0, "", True, False)

    if len(studentID) > 0:
        result[user_defined_list.index("Student ID")] = studentID
    else:
        result[user_defined_list.index("Student ID")] = "-"
    if len(studentName) > 0:
        result[user_defined_list.index("Name")] = re.sub('[0-9]+', '', studentName)
    else:
        result[user_defined_list.index("Name")] = "-"

    if len(validity) > 0:
        result[user_defined_list.index("Validity")] = validity
    else:
        result[user_defined_list.index("Validity")] = "-"

    time_in_sec = "{:.2f}".format(time.time() - start_time)
    result[user_defined_list.index("Time Taken")] = time_in_sec
    result[user_defined_list.index("raw_img_filepath")] = raw_img_filepath
    result[user_defined_list.index("threshold_filepath")] = threshold_filepath
    result[user_defined_list.index("boxes_text_filepath")] = boxes_text_filepath
    result[user_defined_list.index("barcodeDetected")] = barcodeDetected
    result[user_defined_list.index("detectedBarcode_path")] = detectedBarcode_path
    result[user_defined_list.index("faceDetected")] = faceDetected
    result[user_defined_list.index("detectedFace_path")] = detectedFace_path
    result[user_defined_list.index("logoDetected")] = logoDetected
    result[user_defined_list.index("detectedLogo_path")] = detectedLogo_path

    for i in range(len(result) - 1, -1, -1):
        if result[i] == "#####":
            del result[i]
            del user_defined_list[i]

    # for keyword in user_defined_list:
    #     user_defined_list[user_defined_list.index(keyword)] = str(user_defined_list.index(keyword)).zfill(2) + "," + keyword

    my_array = [user_defined_list, result]

    pairs = zip(my_array[0], my_array[1])
    json_values = ('"{}": "{}"'.format(label, value)
                   for label, value in pairs)
    my_string = '{' + ', '.join(json_values) + '}'
    export_data = json.loads(my_string)
    # print("\nExtraction Confidence Level: " + conf)
    json_formatted_str = json.dumps(export_data, indent=2)
    #print("json formatted string: ")
    #print(json_formatted_str)

    '''Saving final results'''
    final_results = open(f'StudentCardDB/Output/final_results.txt', 'w')
    if json_formatted_str is not None:
        json_final_result = json.loads(json_formatted_str)
        if json_final_result["Name"] is not None:
            card_name = json_final_result["Name"]
            final_results.write(f'Name: {card_name}\n')
        if json_final_result["Student ID"] is not None:
            student_id = json_final_result["Student ID"]
            final_results.write(f'Student ID: {student_id}\n')
        if json_final_result["Validity"] is not None:
            validity = json_final_result["Validity"]
            final_results.write(f'Validity: {validity}\n')

        final_results.write(f'Time Taken: {time_in_sec} seconds\n')

    else:
        final_results.write(f'Unable to detect!')
    final_results.close()
    return json_formatted_str

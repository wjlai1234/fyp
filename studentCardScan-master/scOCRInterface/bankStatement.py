from flask import Flask, render_template, redirect, url_for, request, redirect, jsonify, Response, json
import math
import numpy as np
import cv2
import ftfy
import matplotlib.pyplot as plt
import os
import pandas as pd
import pytesseract
import re
import string
import time
from textblob import TextBlob
import concurrent
from os import listdir
from os.path import isfile, join
from pdf2image import convert_from_path
import csv
import imutils
import time, datetime
import calendar
import sys
import json
from ftplib import FTP
import requests
from PIL import Image
from difflib import SequenceMatcher
from statistics import median
import statistics
import platform
from PyPDF2 import PdfFileReader, PdfFileWriter
import sys, subprocess
import pikepdf
import pathlib

path = pathlib.Path().resolve()
iPATH = str(path).replace('\\', '/') + "/BankStatement/"
iTESSERACT_PATH = pytesseract.pytesseract.tesseract_cmd
iPOPPLER_PATH = r"C:/Users/Z/poppler-0.68.0/bin"

iTVEXTRACT_FOLDER = iPATH
iPROCESS_NAME = ""
iWORKING_DIRECTORY = iPATH + "sample/"
iRAW_FOLDER_PATH = iTVEXTRACT_FOLDER + "raw/"
iPROCESS_FOLDER_PATH = iTVEXTRACT_FOLDER + "process/"
iLOG_FOLDER_PATH = iTVEXTRACT_FOLDER + "log/"
iTIME_FOLDER_PATH = iLOG_FOLDER_PATH + "time/"
iXML_FOLDER_PATH = iTVEXTRACT_FOLDER + "xml/"

iOUTPUT_FOLDER_PATH = iPATH + "output/"
iOUTPUT_XLS_FOLDER_PATH = iOUTPUT_FOLDER_PATH + "xls/"
iOUTPUT_JSN_FOLDER_PATH = iOUTPUT_FOLDER_PATH + "json/"
iOUTPUT_RECON_FOLDER_PATH = iOUTPUT_FOLDER_PATH + "recon/"

iIS_DEBUG = False
iIS_IPYNB = False
iIS_DEPLOY = False
iIS_OUTPUT_XLSX = True
iIS_OUTPUT_JSON = True
tve_common = None
isDebug = True
# app = Flask(__name__)

'''Code'''
def TVExtract_Extraction_setCommon(tvextract_common):
    global tve_common
    tve_common = tvextract_common
    #print(tve_common, "Created")

def processTesseract(image, language="eng", psm=6, oem=3, whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-.,:' '"):
    '''
    :param image:
    :param language:
    :param psm:         Page Segmentation Mode
    :                     0    Orientation and script detection (OSD) only.
    :                     1    Automatic page segmentation with OSD.
    :                     2    Automatic page segmentation, but no OSD, or OCR.
    :                     3    Fully automatic page segmentation, but no OSD. (Default)
    :                     4    Assume a single column of text of variable sizes.
    :                     5    Assume a single uniform block of vertically aligned text.
    :                     6    Assume a single uniform block of text.
    :                     7    Treat the image as a single text line.
    :                     8    Treat the image as a single word.
    :                     9    Treat the image as a single word in a circle.
    :                     10   Treat the image as a single character.
    :                     11   Sparse text. Find as much text as possible in no particular order.
    :                     12   Sparse text with OSD.
    :                     13   Raw line. Treat the image as a single text line,
    :                          bypassing hacks that are Tesseract-specific.
    :param oem:         OCR Engine Mode
    :                     0    Legacy engine only.
    :                     1    Neural nets LSTM engine only.
    :                     2    Legacy + LSTM engines.
    :                     3    Default, based on what is available.
    :param whitelist:
    :return:
    '''
    conf = "--psm "+ str(psm) + " --oem " + str(oem) + " -c tessedit_char_whitelist=" + str(whitelist)
    image_to_data_result = pytesseract.image_to_data(image, lang=language, output_type='data.frame', config=conf)

    if tve_common.IS_DEBUG:
        tve_common.printLog("Image_to_Data RAW output:")
        tve_common.printLog(image_to_data_result)
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "processTesseract")
    return image_to_data_result

def tesseractToDataSingleThresh(argv):
    img, threshold,z,language, psm, oem, whitelist= argv[0],argv[1],argv[2],argv[3],argv[4],argv[5],argv[6]

    threshImg = filterImageThreshold(img, threshold, False)
    if z==1:
        img_data = pytesseract.image_to_data(threshImg, lang="ind",config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-.,:' '", output_type='data.frame')
    elif z==2:
        img_data = pytesseract.image_to_data(threshImg, lang="ind",config="--psm 3 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-.,:' '", output_type='data.frame')
    elif z==3:
        #BNI TYPE 1
        img_data = pytesseract.image_to_data(threshImg, lang="ind_w",config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-.,:' '", output_type='data.frame')
    elif z==4:
        #BCA
        #BNI TYPE 2 (to get the date and description properly)
        img_data = pytesseract.image_to_data(threshImg, lang="ind+eng",config="--psm 4 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-.,:' '", output_type='data.frame')
    elif z== 5:
        #BNI TYPE 2 (to get the mutation, debt/credit desc, and balance properly)
        img_data = pytesseract.image_to_data(threshImg, lang="ind+eng",config="--psm 3 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-.,:' '", output_type='data.frame')
    else:
        img_data = processTesseract(threshImg, language, psm, oem, whitelist)
    img_data = img_data.dropna()
    total = img_data.sum(axis=0)
    count = img_data.count(axis=0)
    avgConf = total.conf / count.conf
    return avgConf, threshImg, img_data

def processTesseractMaximumConfidenceLevel1(img, arrayThreshold, z=0, language="msa+eng", psm=6, oem=3, whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-.,:' '"):
    """Retrtieve image and configuration to retrieve text as a list,

    Args:
        img (OpenCV Greyscale Image): Input image for OCR
        arrayThreshold (_type_): Image filtering threshold (refer to filterImageThreshold function)
        z (int, optional): Tesseract image to string configuration modifier. Defaults to 0.
        language (str, optional): Tesseract OCR language settings. Defaults to "msa+eng".
        psm (int, optional): Tesseract Page Segmentation mode from 0 to 13. Defaults to 6.
        oem (int, optional): Tesseract OCR Engine Mode 0 to 3. Defaults to 3.
        whitelist (str, optional): _description_. Defaults to "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-.,:' '".

    Returns:
        OpenCV Greyscale Image: Thresholded image
        DataFrame: DataFrame of OCR with maximum confidence level
        List: Threshold with highest confidence
        Double: Double number with the highest confidence level
    """

    if tve_common.IS_DEBUG and tve_common.IS_IPYNB:
        img_rgb = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB)
        fig1, (ax1) = plt.subplots(1, 1, figsize=(25, 8))
        ax1.imshow(img_rgb.astype('uint8'),cmap=plt.cm.gray)
        ax1.set_title("Raw image")
        ax1.set_axis_off()
        plt.show()
    tve_common.printLog("Checking maximum confidence level 1 on OCR procress...")
    temp_conf =[]
    temp_img  =[]
    img_data_list  =[]
    THRESH_TYPE = ["cv.THRESH_BINARY", "cv.THRESH_BINARY_INV","cv.THRESH_TRUNC","cv.THRESH_TOZERO","cv.THRESH_TOZERO_INV"]

    args = ((img,arrThresh,z,language,psm,oem,whitelist) for arrThresh in arrayThreshold)
    with concurrent.futures.ThreadPoolExecutor() as executor: # concurrent.futures.ProcessPoolExecutor()
        for conf,outImg,data in executor.map(tesseractToDataSingleThresh,args):
            temp_conf.append(conf)
            temp_img.append(outImg)
            img_data_list.append(data)

    u = temp_conf.index(max(temp_conf))
    image = temp_img[u]
    image_data = img_data_list[u]
    if tve_common.IS_DEBUG:
        tve_common.printLog("image to data raw (droped NA) output:")
        tve_common.printLog(image_data)
    row = math.ceil(len(arrayThreshold)/3)
    if row==0:
        row+=1
    if tve_common.IS_IPYNB:
        fig1, ax = plt.subplots(row, 3, figsize=(25, 20))
        count=0
        for i in range(row):
            for j in range(3):
                if(count<len(temp_img)):
                    tempStr = str(arrayThreshold[count]) + " " + str(np.round(temp_conf[count], 3)) + "%"
                    if(row==1):
                        ax[j].imshow(cv2.cvtColor(temp_img[count],cv2.COLOR_BGR2RGB).astype('uint8'),cmap=plt.cm.gray)
                        ax[j].set_title(tempStr)
                        ax[j].set_axis_off()
                    else:
                        ax[i,j].imshow(cv2.cvtColor(temp_img[count],cv2.COLOR_BGR2RGB).astype('uint8'),cmap=plt.cm.gray)
                        ax[i,j].set_title(tempStr)
                        ax[i,j].set_axis_off()
                    count+=1
                else:
                    if(row==1):
                        ax[j].set_visible(False)
                    else:
                        ax[i,j].set_visible(False)
        plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
        plt.show()
        tve_common.printLog(str(arrayThreshold[u])+" " + str(temp_conf[u])+ " % is selected")
    if tve_common.IS_DEBUG:
        tve_common.printLog(str(arrayThreshold[u][0])+" "+(THRESH_TYPE[arrayThreshold[u][1]]) +" " + str(temp_conf[u])+ " % is selected")
        tve_common.timestamp_collector(time.time() - tve_common.start_time, "ProcessTesseract_MaximumConfidenceLevel1")
    return image, image_data, arrayThreshold[u], temp_conf[u]

def processTesseractPredefined(img,z):
    """OCR from image using pytesseract.image_to_string with a set of predefined parameter

    Args:
        img (OpenCV Greyscale Image): _description_
        z (int): type of predefined of parameter

    z Defined:
        1 - lang="ind",config="--psm 6 --oem 3\n
        2 - lang="ind",config="--psm 6 --oem 3\n
        3 - lang="ind_w",config="--psm 6 --oem 3\n
        4 - lang="ind+eng",config="--psm 4 --oem 3\n
        5 - lang="ind+eng",config="--psm 3 --oem 3

    Returns:
       String: image_to_string_result
    """
    if z==1:
        image_to_string_result = pytesseract.image_to_string(img, lang="ind",config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-.,:' '")
    elif z==2:
        image_to_string_result = pytesseract.image_to_string(img, lang="ind",config="--psm 3 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-.,:' '")
    elif z == 3 :
        #BNI TYPE 1
        image_to_string_result = pytesseract.image_to_string(img, lang="ind_w",config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-.,:' '")
    elif z == 4 :
        image_to_string_result = pytesseract.image_to_string(img, lang="ind+eng",config="--psm 4 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-.,:' '")
    elif z == 5:
        image_to_string_result = pytesseract.image_to_string(img, lang="ind+eng",config="--psm 3 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/-.,:' '")
    if tve_common.IS_DEBUG:
        tve_common.printLog("image to string raw output:")
        tve_common.printLog(image_to_string_result)
    image_to_string_result = getTesseractCleanedTextMain(image_to_string_result)
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "ProcessTesseract")
    return image_to_string_result

def TVExtract_PreProcessing_setCommon(tvextract_common):
    global tve_common
    tve_common = tvextract_common
    #print(tve_common, "Created")

def filterImageThreshold(image, value, displayDebugImage):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, image = cv2.threshold(gray, value[0], 255, value[1])

    if tve_common.IS_DEBUG and tve_common.IS_IPYNB and displayDebugImage:
        fig1, (ax1) = plt.subplots(1, 1, figsize=(25, 8))
        ax1.imshow(image.astype('uint8'),cmap=plt.cm.gray)
        ax1.set_title(value)
        ax1.set_axis_off()
        plt.show()
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "filterImageThreshold")
    return image

def TVExtract_PostProcessing_setCommon(tvextract_common):
    global tve_common
    tve_common = tvextract_common
    #print(tve_common, "Created")

def tesseractImageToDataPostprocess(img_data):
    img_data = img_data.dropna()
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
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "tesseractImageToDataPostprocess")
    return text

def getTesseractCleanedText1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('—', '-', text)
    text = re.sub('…', '', text)
    text = re.sub('“', '', text)
    remove = string.punctuation
    remove = remove.replace("-", "")
    remove = remove.replace("/", "")
    remove = remove.replace(".", "")
    remove = remove.replace(":", "")
    remove = remove.replace(",", "")

    remove = remove.replace("|", "")
    pattern = r"[{}]".format(remove)  # create the pattern
    text = re.sub(pattern, "", text)
    #text = re.sub('\w*\d\w*', '', text)
    text = os.linesep.join([s for s in text.splitlines() if s])
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "getTesseractCleanedText1")
    return text

def getTesseractCleanedText2(text):
    text = ftfy.fix_text(text)
    text = ftfy.fix_encoding(text)
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "getTesseractCleanedText2")
    return text

def extractValuePredefinedForeList(detectedText, predefinedList):
    #to get name of the ID card
    result = ["#####"] * len(predefinedList)
    textlines = detectedText.split("\n")
#     tve_common.printLog(textlines)
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
                        print("Error occurred at extractValuePredefinedForeList")
    if tve_common.IS_DEBUG:
        tve_common.printLog("### Extracted Value ### :")
        tve_common.printLog(result)
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "extractValuePredefinedForeList")
    return result

def extractValueSpecificPattern(detectedText, regularExpression, returnGroupIndex, deliminator, isRemoveCRLF, isRemoveSpace):
    #use to extract values for card and statement
    if tve_common.IS_DEBUG:
        tve_common.printLog("### Extracted Value ### :")
    if isRemoveSpace:
        detectedText = detectedText.replace(" ", "")
    if isRemoveCRLF:
        detectedText = detectedText.replace("\n", " ")
    extractedValue = ""
    try:
        if len(deliminator) > 0:
            textlines = detectedText.split(deliminator)
            for textline in textlines:
                if tve_common.IS_DEBUG:
                    tve_common.printLog("textline : "+ textline)
                result = re.search(regularExpression, textline)
                if result:
                    extractedValue = result.group(returnGroupIndex)
                    if tve_common.IS_DEBUG:
                        tve_common.printLog("extracted : "+ extractedValue)
                    break
        else:
            result = re.search(regularExpression, detectedText)
            if result:
                extractedValue = result.group(returnGroupIndex)
                if tve_common.IS_DEBUG:
                    tve_common.printLog("extracted : "+ extractedValue)
    except:
        if tve_common.IS_DEBUG:
            tve_common.printLog("Error occured in extractValueSpecificPattern")
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "extractValueSpecificPattern")
    return extractedValue

def getConfidenceLevel(X,IsNative):
    """Calculate the confidence level of the text

    Args:
        X (String): Cleaned texts from getTesseractCleanedText1() function
        IsNative (Boolean):


    Returns:
        Float: Confidence level
    """
    m,l=[],[]
    if IsNative is None:
        IsNative = False

    if IsNative :
        return 85

    conf=0
    try:
        if X!=[]:
            for c in X:
                total = c.sum(axis=0)
                count = c.count(axis=0)
                m.append(total.conf)
                l.append(count.conf)
            conf= sum(m)/sum(l)
        tve_common.timestamp_collector(time.time() - tve_common.start_time, "getConfidenceLevel")
    except Exception as e:
        tve_common.printLog("an error occurred when calculating confidence level")
        tve_common.printLog(e)
    return conf

def convertStringToDateTime(Date, dateFormat):
    """Convert string to data based on the specified format

    Args:
        Date (String): Date (in the form of string)
        dateFormat (Date format): Date format such as DD/MM/YYYY, DD-MM-YYYY, etc.

    Returns:
        Datetime: The datetime based on the format specified by the user
    """
    return datetime.datetime.strptime(Date,dateFormat)

def getCleanedCurrencyCode(temp):
    """Extract the cleaned currency code from the string input

    Args:
        temp (String): String to be extracted currency code

    Returns:
        String: Currency code
    """
    CurrencyCode = ['AED', 'AFN', 'ALL', 'AMD', 'ANG', 'AOA', 'ARS', 'AUD', 'AWG', 'AZN', 'BAM', 'BBD', 'BDT', 'BGN', 'BHD', 'BIF', 'BMD', 'BND', 'BOB', 'BOV', 'BRL', 'BSD', 'BTN', 'BWP', 'BYN', 'BZD', 'CAD', 'CDF', 'CHE', 'CHF', 'CHW', 'CLF', 'CLP', 'CNY', 'COP', 'COU', 'CRC', 'CUC', 'CUP', 'CVE', 'CZK', 'DJF', 'DKK', 'DOP', 'DZD', 'EGP',
                'ERN', 'ETB', 'EUR', 'FJD', 'GBP', 'GEL', 'GHS', 'GIP', 'GMD', 'GNF', 'GTQ', 'GYD', 'HKD', 'HNL', 'HRK', 'HTG', 'HUF', 'IDR', 'ILS', 'INR', 'IQD', 'IRR', 'ISK', 'JMD', 'JOD', 'JPY', 'KES', 'KGS', 'KHR', 'KMF', 'KPW', 'KYD', 'KZT', 'LAK', 'LBP', 'LKR', 'LRD', 'LSL', 'LYD', 'MAD', 'MDL', 'MGA', 'MKD', 'MMK', 'MNT', 'MOP',
                'MRU', 'MUR', 'MVR', 'MWK', 'MXN', 'MXV', 'MZN', 'NAD', 'NGN', 'NIO', 'NOK', 'NPR', 'NZD', 'OMR', 'PAB', 'PEN', 'PGK', 'PHP', 'PKR', 'PLN', 'PYG', 'QAR', 'RON', 'RSD', 'RUB', 'RWF', 'SAR', 'SBD', 'SCR', 'SDG', 'SEK', 'SGD', 'SHP', 'SLL', 'SOS', 'SRD', 'SSP', 'STN', 'SVC', 'SYP', 'SZL', 'THB', 'TJS', 'TMT', 'TND', 'TOP',
                'TRY', 'TTD', 'TWD', 'TZS', 'UAH', 'UGX', 'USD', 'USN', 'UYI', 'UYU', 'UZS', 'VEF', 'VND', 'VUV', 'WST', 'XAF', 'XCD', 'XDR', 'XOF', 'XPF', 'XSU', 'XUA', 'YER', 'ZAR', 'ZMW', 'ZWL']

    for j in range(len(CurrencyCode)):
        if extractValueSpecificPattern(temp, CurrencyCode[j], 0, "", True, False) != "":
            temp = extractValueSpecificPattern(temp, CurrencyCode[j], 0, "", True, False)
    return temp

def getCounterparty(bank,description):
    """Extract counterpart of bank from description String

    Args:
        bank (String): Name of the bank currently working on. Bank List: 'MANDIRI','MAYBANK', 'BCA', 'BNI', 'BRI', 'PERMATA', 'DANAMON', 'CIMB', 'PANIN', 'OCBC'
        description (String): Input line description of a transaction to try to extract bank counterpart from

    Returns:
        String: Counterparty
    """
    BankCPTY = '-'
    List_Bank = ['MANDIRI','MAYBANK', 'BCA', 'BNI', 'BRI', 'PERMATA', 'DANAMON', 'CIMB', 'PANIN', 'OCBC']
    List_Bank.remove(bank)
    description = description.split()
    Detected_Bank = list(set(description)&set(List_Bank))
    if(len(Detected_Bank) != 0):
        BankCPTY = Detected_Bank[0]

    return BankCPTY

def checkFooterExistence(x, y, T, result1):
    """Check the existence of the footer

    Args:
        x (List): List of texts extracted from the document
        y (String): The text at the footer based on the specified regular expression
        T (List): Empty list
        result1 (String): Texts extracted from the document

    Returns:
        List: List of texts extracted from the document
        List: List of texts at the footer
    """
    if y in x:
        index = x.index(y)
        for z in range(index,len(x)):
            T.append(result1.splitlines()[z])
            x.remove(result1.splitlines()[z])
    return x, T

def getTesseractCleanedTextMain(text):
    """Clean text with Tesseract_CleanText_1, Tesseract_CleanText_2 abd perform string upper operation

    Args:
        text (String): Dirty String

    Returns:
        String: Cleaned UTF-8 String
    """
    result = getTesseractCleanedText1(text)
    result = getTesseractCleanedText2(result)
    result = result.upper()
    return result

class TVExtract_Common:
    __instance = None
    start_time = None
    IS_DEBUG = None
    IS_IPYNB = None
    IS_DEPLOY = None
    IS_OUTPUT_XLSX = None  # For generating xlsx
    IS_OUTPUT_JSON = None  # For generating json
    FTP_HOST = None
    FTP_USER = None
    FTP_PASSWORD = None

    PROCESS_NAME = None
    FILE_NAME_IN_PROCESS = None

    WORKING_DIRECTORY = None  # By default the targeted source folder
    TVEXTRACT_FOLDER = None  # By default (server) program runs in
    RAW_FOLDER_PATH = None  # For all raw / input files
    PROCESS_FOLDER_PATH = None  # For pre-processed image before pass to Tesseract extraction

    LOG_FOLDER_PATH = None  # For logging every exception
    TIME_FOLDER_PATH = None  # For execution time loggin

    XML_FOLDER_PATH = None  # For AI detection

    OUTPUT_FOLDER_PATH = None  # By default (server) to retrieve process output file at
    OUTPUT_XLS_FOLDER_PATH = None
    OUTPUT_JSN_FOLDER_PATH = None

    #pytesseract.pytesseract.tesseract_cmd   = None  # Tesseract path in Web server ##
    PDF2IMG_POPPLER_PATH = None  # Poppler program path for Windows OS
    #Global IP
    global api_ip
    global URL_Success
    global URL_Error
    global URL_WebApp
    global URL_Receive
    global URL_Json

    #api_ip = '35.186.157.56'
    api_ip='34.124.207.92'  # Check the api during deployment
    URL_Success = 'http://' + api_ip + ':80/index.php/panggilAku'
    URL_Error = 'http://' + api_ip + ':80/index.php/panggilAku/index_error'
    URL_Receive = 'http://' + api_ip + ':80/index.php/panggilAku/receive'
    URL_WebApp = '/var/www/html/maybank-ocr/uploads/static/'
    URL_Json = '/var/www/html/maybank-ocr/uploads/json/'
    def __init__(self, is_debug = False, is_ipynb = False, is_deploy = True):
        """ Virtually private constructor. self is this pointer in c++"""
        print("TVExtract_Common singleton is created.")

        os.environ['OMP_THREAD_LIMIT'] = '1'
        self.start_time     = time.time()
        self.IS_DEBUG       = is_debug
        self.IS_IPYNB       = is_ipynb
        self.IS_DEPLOY      = is_deploy
        self.IS_OUTPUT_XLSX = True
        self.IS_OUTPUT_JSON = True
        self.FTP_HOST       = '10.148.0.3'
        self.FTP_USER       = 'tvx'
        self.FTP_PASSWORD   = 'php@ml123'

        self.PROCESS_NAME           = "DEFAULT_BANKSTATEMENT"
        self.FILE_NAME_IN_PROCESS   = ""

        self.TVEXTRACT_FOLDER    = r"/var/www/html/MayBank_TVEXTRACT/"
        self.RAW_FOLDER_PATH     = self.TVEXTRACT_FOLDER + "raw/"
        self.PROCESS_FOLDER_PATH = self.TVEXTRACT_FOLDER + "process/"

        self.LOG_FOLDER_PATH     = self.TVEXTRACT_FOLDER + "log/"
        self.TIME_FOLDER_PATH    = self.LOG_FOLDER_PATH + "time/"

        self.XML_FOLDER_PATH     = self.TVEXTRACT_FOLDER + "xml/"

        self.OUTPUT_FOLDER_PATH         = "/home/tvx/"
        self.OUTPUT_XLS_FOLDER_PATH     = self.OUTPUT_FOLDER_PATH + "xls/"
        self.OUTPUT_JSN_FOLDER_PATH     = self.OUTPUT_FOLDER_PATH + "json/"

        pytesseract.pytesseract.tesseract_cmd   = r"/opt/tesseract/./tesseract"
        #self.PDF2IMG_POPPLER_PATH = r"C:/Users/Z/poppler-0.68.0/bin"
        self.PDF2IMG_POPPLER_PATH = r"C:\Users\p\poppler-0.68.0\bin"

       # the Singleton creation checking
        if TVExtract_Common.__instance != None:
            raise Exception("Error: TVExtract_Common is being created! use TVExtract_Common.getInstance()")
        else:
            TVExtract_Common.__instance = self
            # TO BE REMOVED !
            print("CHECK: The TVExtract_Common singleton's object pointer ", TVExtract_Common.__instance)

    def setPaths(self, iPROCESS_NAME, iWORKING_DIRECTORY, iTVEXTRACT_FOLDER, iRAW_FOLDER_PATH,
                iPROCESS_FOLDER_PATH, iLOG_FOLDER_PATH, iXML_FOLDER_PATH,
                iOUTPUT_FOLDER_PATH, iOUTPUT_XLS_FOLDER_PATH, iOUTPUT_JSN_FOLDER_PATH,
                iPDF2IMG_POPPLER_PATH=r"/usr/local/bin/pdftoppm"):

        self.setPROCESS_NAME(iPROCESS_NAME)
        self.setWORKING_DIRECTORY(iWORKING_DIRECTORY)
        self.setTVEXTRACT_FOLDER(iTVEXTRACT_FOLDER)
        self.setRAW_FOLDER_PATH(iRAW_FOLDER_PATH)
        self.setPROCESS_FOLDER_PATH(iPROCESS_FOLDER_PATH)
        self.setLOG_FOLDER_PATH(iLOG_FOLDER_PATH)
        self.setTIME_FOLDER_PATH(self.LOG_FOLDER_PATH + "time/")
        self.setXML_FOLDER_PATH(iXML_FOLDER_PATH)
        self.setOUTPUT_FOLDER_PATH(iOUTPUT_FOLDER_PATH)
        self.setOUTPUT_XLS_FOLDER_PATH(iOUTPUT_XLS_FOLDER_PATH)
        self.setOUTPUT_JSN_FOLDER_PATH(iOUTPUT_JSN_FOLDER_PATH)
        self.setPDF2IMG_POPPLER_PATH(iPDF2IMG_POPPLER_PATH)
        self.getPaths()

    def getPaths(self):
        self.printLog("============= TVExtract_Common PATHS =============" + "\n" +
                 "PROCESS_NAME = " + self.PROCESS_NAME + "\n" +
                 "WORKING_DIRECTORY = " + self.WORKING_DIRECTORY + "\n" +
                 "TVEXTRACT_FOLDER = " + self.TVEXTRACT_FOLDER + "\n" +
                 "RAW_FOLDER_PATH = " + self.RAW_FOLDER_PATH + "\n" +
                 "PROCESS_FOLDER_PATH = " + self.PROCESS_FOLDER_PATH + "\n" +
                 "LOG_FOLDER_PATH = " + self.LOG_FOLDER_PATH + "\n" +
                 "TIME_FOLDER_PATH = " + self.TIME_FOLDER_PATH + "\n" +
                 "XML_FOLDER_PATH = " + self.XML_FOLDER_PATH + "\n" +
                 "OUTPUT_FOLDER_PATH = " + self.OUTPUT_FOLDER_PATH + "\n" +
                 "OUTPUT_XLS_FOLDER_PATH = " + self.OUTPUT_XLS_FOLDER_PATH + "\n" +
                 "OUTPUT_JSN_FOLDER_PATH = " + self.OUTPUT_JSN_FOLDER_PATH + "\n" +
                 "PDF2IMG_POPPLER_PATH = " + self.PDF2IMG_POPPLER_PATH)

    def getInstance(is_debug=False, is_ipynb=False, is_deploy=True):
        """Get a TVExtract Common object

        Returns:
            TVExtract_Common Object: An instance of TVExtract_Common
        """
        """ Static access method. """
        if TVExtract_Common.__instance == None:
            TVExtract_Common(is_debug=is_debug, is_ipynb=is_ipynb, is_deploy=is_deploy)
        else:
            TVExtract_Common.IS_DEBUG = is_debug
            TVExtract_Common.IS_IPYNB = is_ipynb
            TVExtract_Common.IS_DEPLOY = is_deploy

        return TVExtract_Common.__instance

    def getPROCESS_NAME(self):              return self.PROCESS_NAME
    def setPROCESS_NAME(self,arg):          self.PROCESS_NAME = arg
    def getWORKING_DIRECTORY(self):         return self.WORKING_DIRECTORY
    def setWORKING_DIRECTORY(self,arg):     self.WORKING_DIRECTORY = arg
    def getTVEXTRACT_FOLDER(self):          return self.TVEXTRACT_FOLDER
    def setTVEXTRACT_FOLDER(self,arg):      self.TVEXTRACT_FOLDER = arg
    def getRAW_FOLDER_PATH(self):           return self.RAW_FOLDER_PATH
    def setRAW_FOLDER_PATH(self,arg):       self.RAW_FOLDER_PATH = arg
    def getPROCESS_FOLDER_PATH(self):       return self.PROCESS_FOLDER_PATH
    def setPROCESS_FOLDER_PATH(self,arg):   self.PROCESS_FOLDER_PATH = arg
    def getLOG_FOLDER_PATH(self):           return self.LOG_FOLDER_PATH
    def setLOG_FOLDER_PATH(self,arg):       self.LOG_FOLDER_PATH = arg
    def getTIME_FOLDER_PATH(self):          return self.TIME_FOLDER_PATH
    def setTIME_FOLDER_PATH(self,arg):      self.TIME_FOLDER_PATH = arg
    def getXML_FOLDER_PATH(self):           return self.XML_FOLDER_PATH
    def setXML_FOLDER_PATH(self,arg):       self.XML_FOLDER_PATH = arg
    def getOUTPUT_FOLDER_PATH(self):        return self.OUTPUT_FOLDER_PATH
    def setOUTPUT_FOLDER_PATH(self,arg):    self.OUTPUT_FOLDER_PATH = arg
    def getOUTPUT_XLS_FOLDER_PATH(self):    return self.OUTPUT_XLS_FOLDER_PATH
    def setOUTPUT_XLS_FOLDER_PATH(self,arg):self.OUTPUT_XLS_FOLDER_PATH = arg
    def getOUTPUT_JSN_FOLDER_PATH(self):    return self.OUTPUT_JSN_FOLDER_PATH
    def setOUTPUT_JSN_FOLDER_PATH(self,arg):self.OUTPUT_JSN_FOLDER_PATH = arg
    def getPDF2IMG_POPPLER_PATH(self):      return self.PDF2IMG_POPPLER_PATH
    def setPDF2IMG_POPPLER_PATH(self,arg):  self.PDF2IMG_POPPLER_PATH = arg

    def printLog(self, text):
        text = str(text)
        print(text)
        now = datetime.datetime.now()
        log_file_name = self.FILE_NAME_IN_PROCESS+"_" + now.strftime("%Y%m%d%H") #self.PROCESS_NAME+"_" + now.strftime("%Y%m%d%H")
        LOG_FILE_PATH = self.LOG_FOLDER_PATH + log_file_name + ".log"
        try:
            if LOG_FILE_PATH != "":
                f = open(LOG_FILE_PATH, "a+")
                f.write(now.strftime("%Y%m%d %H:%M:%S")+"\t"+text+ "\n")
                f.close()
        except:
            f = open(LOG_FILE_PATH, "a+")
            f.write(now.strftime("%Y%m%d %H:%M:%S")+"\t"+"Error when writing Log file:" + LOG_FILE_PATH + "\n")
            print("Error when writing Log file:" + LOG_FILE_PATH)
            f.close()

    def getFILE_NAME_IN_PROCESS(self):           return self.FILE_NAME_IN_PROCESS
    def setFILE_NAME_IN_PROCESS(self, iFILE_NAME_IN_PROCESS):
        base = os.path.basename(iFILE_NAME_IN_PROCESS)
        self.FILE_NAME_IN_PROCESS = os.path.splitext(base)[0]
        self.printLog("============= TVExtract_Common PATHS =============")
        self.printLog("FILE_NAME_IN_PROCESS=" + self.FILE_NAME_IN_PROCESS)

    def directoryCreator(self, dirs):
        try:
            self.printLog("directoryCreator creating..." + dirs)
            os.mkdir(dirs)
            self.printLog("Directory '% s' created successfully." % dirs)
        except:
            self.printLog("Directory " + dirs + " already created!")

    def readFile(self, LocalFileIndex = -1):
        self.printLog("readFile checking...")
        filepath = False
        if LocalFileIndex < 0:
            try:
                filepath = sys.argv[1]          # retrieve the first argument from command
                self.printLog("readFile path : " + filepath)
            except:
                self.printLog("No argument (file path) is defined.")
                return False
        else:
            self.printLog("readFile index : " + str(LocalFileIndex) + " from " + self.WORKING_DIRECTORY)
            onlyfiles = [f for f in listdir(self.WORKING_DIRECTORY) if isfile(join(self.WORKING_DIRECTORY, f))]
            filepath = join(self.WORKING_DIRECTORY, onlyfiles[LocalFileIndex])
        if filepath != False:
            self.setFILE_NAME_IN_PROCESS(filepath)
        else:
            self.printLog("No index " + LocalFileIndex + " is found.")
        self.timestamp_collector(time.time() - self.start_time, "readFile")
        return filepath


    #update: change ftp to API
    def convertFileToImage(self, filepath, start=None, end=None, size=None, pdfdpi=200, tiffmax=250):
        """Convert specified pdf file in a filepath to image
        Args:
            filepath (String): File path of a specified pdf file
            pdfdpi (int, optional): DPI configuration for pdf. Defaults to 200.
            tiffmax (int, optional): TIFF image numbering (count from 0). Defaults to 250.

        Returns:
            list image in OpenCV BGR format: images list
            list of string: images file path list
        """
        self.printLog("convertFileToImage checking...")
        img, list_path = [],[]
        if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.printLog("Reading image file...")
            image = cv2.imread(filepath)#[...,::-1]
            process_filepath = self.PROCESS_FOLDER_PATH + self.FILE_NAME_IN_PROCESS + ".jpg"
            cv2.imwrite(process_filepath, image)
            img.append(image)
            if self.IS_DEPLOY:
                cv2.imwrite(process_filepath,img[0])
                file_name = os.path.basename(filepath)
                myfiles = {'image' : open(process_filepath, 'rb')}
                post_status_to_webapp = requests.post(URL_Receive,{'filename':file_name},files=myfiles)
            list_path.append(process_filepath)
        elif filepath.lower().endswith(('.pdf')):
            self.printLog("Reading pdf file...")

            if platform.system() == 'Linux':
                pages = convert_from_path(filepath, dpi=pdfdpi, first_page=start, last_page=end)
            elif platform.system() == 'Windows':
                pages = convert_from_path(filepath, dpi=pdfdpi, poppler_path=self.PDF2IMG_POPPLER_PATH)

            i = 0
            list_path=[]
            for page in pages:
                process_filepath = self.PROCESS_FOLDER_PATH + self.FILE_NAME_IN_PROCESS + "_" + str(i) + ".jpg"
                page.save(process_filepath, 'JPEG')
                #image = Image.open(process_filepath) # np.float32(image)
                image = cv2.imread(process_filepath)
                img.append(image)
                if self.IS_DEPLOY:
                    send_webapp = URL_WebApp + self.FILE_NAME_IN_PROCESS + "_" + str(i) + '.jpg'
                    page.save(process_filepath, 'JPEG')
                    file_name = os.path.basename(filepath)
                    test = self.PROCESS_FOLDER_PATH + str(self.FILE_NAME_IN_PROCESS) + "_" + str(i) + '.jpg'
                    myfiles = {'image' : open(test, 'rb')}
                    post_status_to_webapp = requests.post(URL_Receive,{'filename':file_name},files=myfiles)
                    list_path.append(send_webapp)
                else:
                    list_path.append(process_filepath)
                i = i + 1
        elif filepath.lower().endswith(('.tif','.tiff',)):
            self.printLog("Reading tif file...")
            img_read = Image.open(filepath) #mandiri.tif[0]dst
            if self.IS_DEPLOY:
                    for i in range(0,tiffmax):
                        try:
                            process_filepath = self.PROCESS_FOLDER_PATH + self.FILE_NAME_IN_PROCESS + "_" + str(i) + ".jpg"
                            img_read.seek(i)
                            img_read.save(process_filepath)
                            image = cv2.imread(process_filepath)
                            img.append(image)
                            send_webapp = URL_WebApp + self.FILE_NAME_IN_PROCESS + "_" + str(i) + '.jpg'
                            try:
                                file_name = os.path.basename(filepath)
                                process_image = self.PROCESS_FOLDER_PATH + self.FILE_NAME_IN_PROCESS + "_" + str(i) + '.jpg'
                                myfiles = {'image' : open(process_image, 'rb')}
                                # if self.IS_DEPLOY:
                                post_status_to_webapp = requests.post(URL_Receive,{'filename':file_name},files=myfiles)
                                list_path.append(send_webapp)
                            except:
                                message = "Error when stor an image"
                                self.printLog(message)
                            i = i + 1
                        except :
                            self.printLog("Tiff file cannot be read")
                            break
        else:
            self.printLog("convertFileToImage : File cannot be read")
        self.timestamp_collector(time.time() - self.start_time, "convertFileToImage")
        return img, list_path

    def isNativePDF(self, extractedText, regEx):
        self.printLog("isNativePDF checking...")
        isNative = False
        if extractedText == ['']*len(extractedText) or extractedText==[]:
            return isNative
        else:
            for text in extractedText:
                if text == extractValueSpecificPattern(text,regEx, 0,'', False, False):
                    isNative=True
                else:
                    isNative=False
                    break

        self.timestamp_collector(time.time() - self.start_time, "isNativePDF")
        return isNative

    def readNativePDF(self,path):
        self.printLog("Reading pdf file...")

        outputPath = self.PROCESS_FOLDER_PATH
        base = os.path.basename(path)
        file_name = os.path.splitext(base)[0]
        fileReader = PdfFileReader(path)

        if fileReader.isEncrypted:
            pdf = pikepdf.open(path)
            path=outputPath + file_name +'_decrypted'+'.pdf'
            pdf.save(path)
            base = os.path.basename(path)
            file_name = os.path.splitext(base)[0]
            fileReader = PdfFileReader(path)

        numPages = fileReader.getNumPages()
        list_page=[]
        if numPages == 1:
            list_page.append(path)
        else:
            for page in range(numPages):
                pdf_writer = PdfFileWriter()
                pdf_writer.addPage(fileReader.getPage(page))
                OutputFilePath = outputPath+'{}_page_{}.pdf'.format(file_name, page)
                with open(OutputFilePath, 'wb') as out:
                    list_page.append(OutputFilePath)
                    pdf_writer.write(out)
        list_filepath=[]
        list_text=[]
        if numPages == 0:
            list_page.append(path)
            OutputFilePath = outputPath+'{}.txt'.format(file_name)
        else:
            for page in range(numPages):
                temp=[]
                OutputFilePath = outputPath+'{}_page_{}.txt'.format(file_name, page)
                # cmd example: "C:\Users\Documents\gs9.54.0\bin\gswin64c.exe -dBATCH -dNOPAUSE -sDEVICE=txtwrite
                #               -sOutputFile=C:\Users\Documents\OCR_MB\file_name_page_XX.txt path"
                args=['gswin64', '-dBATCH', '-dNOPAUSE', '-sDEVICE=txtwrite', '-sOutputFile=%s' %OutputFilePath, list_page[page] ]
                try:
                    subprocess.check_output(args, universal_newlines=True)
                    with open(OutputFilePath) as f:
                        lines = f.readlines()
                    for m in lines:
                        temp.append(' '.join(m.split()))
                    list_text.append('\n'.join(temp))
                    list_filepath.append(OutputFilePath)
                except:
                    self.printLog("readNativePDF : Error on page "+ str(page))
        self.timestamp_collector(time.time() - self.start_time, "readNativePDF")
        return list_filepath,list_text

    #update

    def errorHandling(self, file_name, message):
        #error_message = requests.post('https://dev-box.tvextract.com//panggilAku/index_error',{'nama_file':file_name, 'message':message})
        # error_message = requests.post('http://10.148.0.3:80/index.php/panggilAku/index_error',{'nama_file':file_name, 'message':message})
        error_message = requests.post(URL_Error,{'nama_file':file_name, 'message':message}) #postgre
        # error_message = requests.post('http://34.124.150.81:80/index.php/panggilAku/index_error',{'nama_file':file_name, 'message':message})
        self.timestamp_collector(time.time() - self.start_time, "errorHandling")
        return error_message

    def timestamp_collector(self, ctime, process_name):
        f = open(self.TIME_FOLDER_PATH + self.FILE_NAME_IN_PROCESS + ".txt", "a")
        f.write(process_name + " took " + str(ctime) + " seconds" + '\n')
        f.close()

tve_common = TVExtract_Common.getInstance(iIS_DEBUG, iIS_IPYNB, iIS_DEPLOY)

TVExtract_Extraction_setCommon(tve_common)
TVExtract_PreProcessing_setCommon(tve_common)
TVExtract_PostProcessing_setCommon(tve_common)

date_type=4
# pytesseract.pytesseract.tesseract_cmd   = r"/opt/tesseract/./tesseract"
pytesseract.pytesseract.tesseract_cmd = iTESSERACT_PATH
#iPDF2IMG_POPPLER_PATH = r"C:\Users\Z\poppler-0.68.0\bin"
iPDF2IMG_POPPLER_PATH = iPOPPLER_PATH

def checkDate(TEXT, date_type):
	F=0
	if date_type == 4:
		if extractValueSpecificPattern(TEXT,'^(\d{2}/\d{2}\s|\s\d{2}/\d{2}\s)',0,"",True,False)!='':
			F=1
		elif extractValueSpecificPattern(TEXT,'^(\d{2}[1]\d{2}\s)',0,"",True,False)!='':
			F=1
		elif extractValueSpecificPattern(TEXT,'^(\d{2}[7]\d{2}\s)',0,"",True,False)!='':
			F=1
		# elif extractValueSpecificPattern(TEXT,'^(\d{2}[7][01]\s|\d{2}[7][02]\s|\d{2}[7][03]\s|\d{2}[7][04]\s|\d{2}[7][05]\s|\d{2}[7][06]\s|\d{2}[7][07]\s|\d{2}[7][08]\s|\d{2}[7][09]\s|\d{2}[7][10]\s|\d{2}[7][11]\s|\d{2}[7][12]\s)',0,"",True,False)!='':
		#     F=1
		elif extractValueSpecificPattern(TEXT,'^(\d{2}/\s\d{2}\s)',0,"",True,False)!='':
			F=1
		elif extractValueSpecificPattern(TEXT,'^(\d{2}[1]/\d{2}\s)',0,"",True,False)!='':
			F=1
		elif extractValueSpecificPattern(TEXT,'^(\d{2}/\d{2}[0-9]\s)',0,"",True,False)!='': #02/141 #added
			F=1
		elif extractValueSpecificPattern(TEXT,'^(\d{2}/\d{2}:\s)',0,"",True,False)!='':
			F=1
		elif extractValueSpecificPattern(TEXT,'^(-\s\d{2}/\d{2}\s|-\s\d{2}[1]\d{2}\s|-\s\d{2}[1]\d{2}\s|[1]\s\d{2}[1]\d{2}\s|[1]\s\d{2}[7]\d{2}\s|[1]\s\d{2}/\d{2}\s)',0,"",True,False)!='':
			F=1
	if F == 1:
		return True
	tve_common.timestamp_collector(time.time() - tve_common.start_time, "CheckDate")
	return False

def getDate(TEXT, date_type):
	Date=''
	if date_type == 4:
		if extractValueSpecificPattern(TEXT,'^(\d{2}/\d{2}\s|\s\d{2}/\d{2}\s)',0,"",True,False)!='':
			Date = extractValueSpecificPattern(TEXT,'^(\d{2}/\d{2}\s|\s\d{2}/\d{2}\s)',0,"",True,False)
		elif extractValueSpecificPattern(TEXT,'^(\d{2}[1]\d{2}\s)',0,"",True,False)!='':
			Date = extractValueSpecificPattern(TEXT,'^(\d{2}[1]\d{2}\s)',0,"",True,False)
		elif extractValueSpecificPattern(TEXT,'^(\d{2}[7]\d{2}\s)',0,"",True,False)!='':
			Date = extractValueSpecificPattern(TEXT,'^(\d{2}[7]\d{2}\s)',0,"",True,False)
		elif extractValueSpecificPattern(TEXT,'^(\d{2}/\s\d{2}\s)',0,"",True,False)!='':
			Date = extractValueSpecificPattern(TEXT,'^(\d{2}/\s\d{2}\s)',0,"",True,False)
		elif extractValueSpecificPattern(TEXT,'^(\d{2}/\d{2}[0-9]\s)',0,"",True,False)!='': #02/141 #added
			Date = extractValueSpecificPattern(TEXT,'^(\d{2}/\d{2}[0-9]\s)',0,"",True,False)
		elif extractValueSpecificPattern(TEXT,'^(\d{2}[1]/\d{2}\s)',0,"",True,False)!='':
			Date = extractValueSpecificPattern(TEXT,'^(\d{2}[1]/\d{2}\s)',0,"",True,False)
		elif extractValueSpecificPattern(TEXT,'^(\d{2}\s\d{2}/\d{2}\s)',0,"",True,False)!='': #08 updated
			Date = extractValueSpecificPattern(TEXT,'^(\d{2}\s)',0,"",True,False)
		elif extractValueSpecificPattern(TEXT,'^(\D\d{1}/\d{2}\s)|^([A-Z]\d{1}/\s\d{2}\s)',0,"",True,False)!='': #:6/10 baru
			Date = extractValueSpecificPattern(TEXT,'^(\D\d{1}/\d{2}\s)|^([A-Z]\d{1}/\s\d{2}\s)',0,"",True,False)
		elif extractValueSpecificPattern(TEXT,'^(\d{2}/\d{2}:\s)',0,"",True,False)!='':
			Date=extractValueSpecificPattern(TEXT,'^(\d{2}/\d{2}:\s)',0,"",True,False)
			Date=str(Date)
		elif extractValueSpecificPattern(TEXT,'^(-\s\d{2}/\d{2}\s|-\s\d{2}[1]\d{2}\s|-\s\d{2}[1]\d{2}\s|[1]\s\d{2}[1]\d{2}\s|[1]\s\d{2}[7]\d{2}\s|[1]\s\d{2}/\d{2}\s)',0,"",True,False)!='':
			Date = extractValueSpecificPattern(TEXT,'^(-\s\d{2}/\d{2}\s|-\s\d{2}[1]\d{2}\s|-\s\d{2}[1]\d{2}\s|[1]\s\d{2}[1]\d{2}\s|[1]\s\d{2}[7]\d{2}\s|[1]\s\d{2}/\d{2}\s)',0,"",True,False)
	tve_common.timestamp_collector(time.time() - tve_common.start_time, "getDate")
	return Date

def getDateWF(TEXT, date_type):
    Date=''
    dsc = 0
    if date_type == 4:
        if extractValueSpecificPattern(TEXT,'^(\d{2}/\d{2}\s|\s\d{2}/\d{2}\s)',0,"",True,False)!='':
            Date = extractValueSpecificPattern(TEXT,'^(\d{2}/\d{2}\s|\s\d{2}/\d{2}\s)',0,"",True,False)
            Date = re.sub(' ','',Date)
            dsc = 0
        elif extractValueSpecificPattern(TEXT,'^(\d{2}/\s\d{2}\s)',0,"",True,False)!='':
            Date = extractValueSpecificPattern(TEXT,'^(\d{2}/\s\d{2}\s)',0,"",True,False)
            Date = re.sub(' ','',Date)
            dsc = 0
        elif extractValueSpecificPattern(TEXT,'^(\d{2}/\d{2}[0-9]\s)',0,"",True,False)!='': #02/141 #added
            Date = extractValueSpecificPattern(TEXT,'^(\d{2}/\d{2}[0-9]\s)',0,"",True,False)
            Date = re.sub(' ','',Date)
            dt = list(Date)
            dt[5]=''
            Date = ''.join(dt)
            dsc = 1
        elif extractValueSpecificPattern(TEXT,'^(\d{2}[1]/\d{2}\s)',0,"",True,False)!='':
            Date = extractValueSpecificPattern(TEXT,'^(\d{2}[1]/\d{2}\s)',0,"",True,False)
            Date = re.sub(' ','',Date)
            dt = list(Date)
            dt[2]=''
            Date = ''.join(dt)
            dsc = 1
        elif extractValueSpecificPattern(TEXT,'^(\d{2}[1]\d{2}\s)',0,"",True,False)!='':
            Date = extractValueSpecificPattern(TEXT,'^(\d{2}[1]\d{2}\s)',0,"",True,False)
            Date=re.sub(' ','',Date)
            dt = list(Date)
            dt[2]='/'
            Date=''.join(dt)
            dsc = 1
        elif extractValueSpecificPattern(TEXT,'^(\d{2}[7]\d{2}\s)',0,"",True,False)!='':
            Date = extractValueSpecificPattern(TEXT,'^(\d{2}[7]\d{2}\s)',0,"",True,False)
            Date=re.sub(' ','',Date)
            dt = list(Date)
            dt[2]='/'
            Date=''.join(dt)
            dsc = 1

        elif extractValueSpecificPattern(TEXT,'^(\d{2}\s\d{2}/\d{2}\s)',0,"",True,False)!='': #08 updated
            Date = extractValueSpecificPattern(TEXT,'^(\d{2}\s)',0,"",True,False)
            Date = re.sub(' ','',Date)
            Date = '01/'+Date[0:]
            dsc = 1

        elif extractValueSpecificPattern(TEXT,'^(\D\d{1}/\d{2}\s)|^([A-Z]\d{1}/\s\d{2}\s)',0,"",True,False)!='': #:6/10 baru
            Date = extractValueSpecificPattern(TEXT,'^(\D\d{1}/\d{2}\s)|^([A-Z]\d{1}/\s\d{2}\s)',0,"",True,False)
            Date = re.sub(' ','',Date)
            dt = list(Date)
            dt[0]='0'
            Date = ''.join(dt)
            dsc = 1

        elif extractValueSpecificPattern(TEXT,'^(\d{2}/\d{2}:\s)',0,"",True,False)!='':
            Date=extractValueSpecificPattern(TEXT,'^(\d{2}/\d{2}:\s)',0,"",True,False)
            Date=re.sub(' ','',Date)
            dt = list(Date)
            if dt[2] == '1' or dt[2] == '7':
                dt[2]='/'
            dt[5]=''
            Date = ''.join(dt)
        elif extractValueSpecificPattern(TEXT,'^(-\s\d{2}/\d{2}\s|-\s\d{2}[1]\d{2}\s|-\s\d{2}[1]\d{2}\s|[1]\s\d{2}[1]\d{2}\s|[1]\s\d{2}[7]\d{2}\s|[1]\s\d{2}/\d{2}\s)',0,"",True,False)!='':
            Date = extractValueSpecificPattern(TEXT,'^(-\s\d{2}/\d{2}\s|-\s\d{2}[1]\d{2}\s|-\s\d{2}[1]\d{2}\s|[1]\s\d{2}[1]\d{2}\s|[1]\s\d{2}[7]\d{2}\s|[1]\s\d{2}/\d{2}\s)',0,"",True,False)
            dt = list(Date)
            if dt[4] != '/':
                dt[4]= '/'
            dt[0] = ''
            dt[1] = ''
            Date = ''.join(dt)

    return Date

#17 Jan 2022
def checkFormat(Date):
    # year_default = '2020'
    try:
        date=convertStringToDateTime(Date,'%d/%m').strftime('%d-%m')
        flag=0
        if convertStringToDateTime(date,'%d-%m')>datetime.datetime.now():
            flag=1
    except:
        try : # date = Date+'/'+str(year_default)
            date=convertStringToDateTime(Date,'%d/%m').strftime('%d-%m')
            flag=1
            if convertStringToDateTime(date,'%d-%m')>datetime.datetime.now():
                flag=1
        except:
            date=Date
            flag=1
    return  date,flag

def getValue(date, s):
    if(extractValueSpecificPattern(s, 'SALDO AWAL', 0, "", True, False) == ''):
        desc = extractValueSpecificPattern(s, date + '(.*?)\d{1,3}\.',1,"",True,False)
        mutasi = extractValueSpecificPattern(s, desc + '(.*?)$',1,"",True,False)
        balance = extractValueSpecificPattern(s, desc + '(.*?)$',1,"",True,False)
    else:
        desc = 'SALDO AWAL'
        mutasi = extractValueSpecificPattern(s, desc + ' (.*?)$',1,"",True,False)
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "getValue")
    return desc,mutasi

def getDesc(Description,text) :
    if extractValueSpecificPattern( text,'',0,"",True,False)!='':
        Description+= re.sub(extractValueSpecificPattern( text,'',0,"",True,False),'',text)
    else:
        Description+=text
    Description+=' '
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "getDesc")
    return Description

def fillEmpty(TED,a,b,c,Result):
    n=0
    Empty = '#####'

    s,e = convertStringToDateTime(a, "%d/%m/%Y"), convertStringToDateTime(b, "%d/%m/%Y")
    GENERATEDATE = [s + datetime.timedelta(days=x) for x in range(0, (e-s).days)]
    for date in GENERATEDATE:
        if (date.year < datetime.datetime.now().year) or (date.year == datetime.datetime.now().year) :

            if len(Result)==0:
                Result.append([["Date","Description", "CPTY", "Bank CPTY", "Mutation","Balance","Flag"],[convertStringToDateTime(date.strftime("%d/%m/%Y"),'%d/%m/%Y').strftime('%d-%m-%Y'),'-', '-', '-', Empty,str(TED[-1]['Balance']),2]])
            else:
                Result.append([["Date","Description", "CPTY", "Bank CPTY", "Mutation","Balance","Flag"],[convertStringToDateTime(date.strftime("%d/%m/%Y"),'%d/%m/%Y').strftime('%d-%m-%Y'),'-', '-', '-', Empty,str(Result[-1][1][-1]),2]])
            n=1
        else:
            continue

    if n==1:
        Compare_Date=convertStringToDateTime(GENERATEDATE[-1].strftime("%d/%m/%Y"),'%d/%m/%Y')+datetime.timedelta(days=1)
    else:
        Compare_Date=convertStringToDateTime(c,'%d-%m-%Y')+datetime.timedelta(days=1)
    return Result,Compare_Date.strftime("%d/%m/%Y")

def checkDescBefore(TED,n,result1):
    x=0
    for i in range(n+1,len(result1.splitlines())-1):
        if extractValueSpecificPattern(result1.splitlines()[i],'\d{2}/\d{2}',0,"",True,False)!='':
            break
        else:
            TED[-1]['Description']=getDesc(TED[-1]['Description'],result1.splitlines()[i])
            x+=1
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "checkDesc_Before")
    return TED,x

def checkTransaction(TED,Z,result1,Result,p,f,CM):
    Date=convertStringToDateTime(TED[-1]['Date'],'%d-%m').strftime('%d-%m')
    RDATE=convertStringToDateTime(Date,'%d-%m').strftime('%d/%m')
    m=convertStringToDateTime(Date,'%d-%m')+datetime.timedelta(days=1)
    newdate_c=convertStringToDateTime(m.strftime("%d-%m"),'%d-%m').strftime('%d/%m')
    newdate_c_2=convertStringToDateTime(m.strftime("%d-%m"),'%d-%m').strftime('%d-%m')
    TempDate=getDate1(RDATE,result1.splitlines()[Z],CM)
    try:
        TempDate=convertStringToDateTime(TempDate,'%d/%m').strftime('%d/%m')
    except:
        TempDate=newdate_c
    if  m<convertStringToDateTime(TempDate,'%d/%m')  and  RDATE!=TempDate:
        Result,Compare_Date=fillEmpty(TED,newdate_c,TempDate,newdate_c_2,Result)
        p = 1
    else:
        if RDATE!=TempDate:
            Compare_Date,p=newdate_c,1
        elif RDATE==TempDate:
            Compare_Date,f=convertStringToDateTime(RDATE,'%d/%m').strftime('%d/%m'),1
    compare_date=Compare_Date
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "checkTransaction")
    return Result,compare_date,p,f

def findIndex(text,k):
    S=False
    n = 0
    if k==1:
        for i in range(len(text)-1):
            #HEAD CONTENT
            if 'KETERANGAN' in text[i].split(' ') and 'MUTASI' in text[i].split(' ') and 'SALDO' in text[i].split(' '):
                S=True

            elif 'MUTASI' in text[i].split(' ') and 'REKENING' in text[i].split(' ') and 'INI.' in text[i].split(' '):
                S=True

            elif 'MULASI' in text[i].split(' ') and 'REKENING' in text[i].split(' ') and 'INI.' in text[i].split(' '):
                S=True

            elif 'MUTASI' in text[i].split(' ') and 'REKENING' in text[i].split(' ') and 'INI,' in text[i].split(' '):
                S=True

            elif 'MUTASI' in text[i].split(' ') and 'REKENING' in text[i].split(' ') and 'INI, I' in text[i].split(' '):
                S=True

            elif 'MUTASI' in text[i].split(' ') and 'REKEPING' in text[i].split(' ') and 'INI, I' in text[i].split(' '):
                S=True

            if S:
                if checkDate(text[i+1],date_type):
                    n = i
                    break
    else:
        for i in range(len(text)-1):
            try:
                if checkDate(text[i+1],date_type):
                    n = i
                    S=True
                    break
            except IndexError:
                message='Keyword Cannot Be Found'
                #error_message=tve_common.errorHandling(file_name,message)
                print(message)
    #                 sys.exit()
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "findindex")
    return n,S

def getDate2(text):
    Date=''
    if extractValueSpecificPattern(text,'\d{2}/\d{2}',0,"",True,False)!='':
        Date=extractValueSpecificPattern(text,'\d{2}/\d{2}',0,"",True,False)
    if extractValueSpecificPattern(text,'\d{2}/ \d{2}',0,"",True,False)!='':
        Date=extractValueSpecificPattern(text,'\d{2}/ \d{2}',0,"",True,False)
        Date=re.sub(' ', '',Date)
    elif extractValueSpecificPattern(text,r'(\d+/\d+)',0,"",True,False)!='':
        Date=extractValueSpecificPattern(text,r'(\d+/\d+)',0,"",True,False)
        a=Date
        if len(Date)>10:
            Date=''
        else:
            Date=a
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "getDate2")
    return Date

def firstChecking(text):
    m,n=extractValueSpecificPattern(text,'NO OF CREDIT (\d+)',0,"",True,False),[]
    if m!='':
        y=text.splitlines()
        for i in range(text.splitlines().index(m),text.splitlines().index(m)+5):
            n.append(text.splitlines()[i])
            y.remove(text.splitlines()[i])
        tve_common.timestamp_collector(time.time() - tve_common.start_time, "firstChecking")
        return '\n'.join(y),n
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "firstChecking")
    return text,n

def getMonthYear(result1,L):
    R,Y='',''
    for j in result1.splitlines():
        if extractValueSpecificPattern(j,'PERIODE : FROM ',0,"",True,False)!='':
            R=j
            break
    if R!='':
        Temp=extractValueSpecificPattern(R,'DECEMBER|MARCH|OCTOBER|SEPTEMBER|FEBRUARY|APRIL|MAY|AUGUST|JUNE|JANUARY|JULY|NOVEMBER',0,"",False,False)
        Y=[L[Temp],extractValueSpecificPattern(R,'\d{4}',0,"",False,False)]
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "getMonthYear")
    return Y

def getMutationBalance(B,C,D):
    Mutasi=D if re.sub(' ','',C)=='0.00' else C
    u=list(re.sub(',','.',Mutasi[::-1]))
    u[2]=','
    res=''.join(u)[::-1]
    Mutasi='- '+res if re.sub(' ','',C)=='0.00' else res
    u=list(re.sub(',','.',B[::-1]))
    u[2]=','
    Balance=''.join(u)[::-1]
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "getMutationBalance")
    return Mutasi,Balance

def getName(text,file_name):
    name = ''
    periode = ''
    year=''
    norek=''
    found = False
    for i in range(len(text.splitlines())):
        line = text.splitlines()[i]
        if line.find('KCP') > -1 or line.find('KCU') > -1:
            name = text.splitlines()[i+1]
            name = extractValueSpecificPattern(name, "(.*?)" + "NO. REKENING", 1, "", True, False)
            norek = extractValueSpecificPattern(text, "(?:NO. REKENING : | NO. REKENING I | NO. REKENING 3 )(.*?\d* )", 1, "", True, False)
            periode = extractValueSpecificPattern(text,'(?:PERIODE I |PERIODE H | PERIODE : |PERIODE : |PERIODE :  |HAR 4 PERIODE I |\nPERIODE I |PERIODE .| PERIODE 3 |PERIODE 3 )(.*?JANUARI \d{2,4}|FEBRUARI \d{2,4}|MARET \d{2,4}|APRIL \d{2,4}|MEI \d{2,4}|JUNI \d{2,4}|JULI \d{2,4}|AGUSTUS \d{2,4}|SEPTEMBER \d{2,4}|OKTOBER \d{2,4}|NOVEMBER \d{2,4}|DESEMBER \d{2,4})\s|\n',1,"",True,False)
            year = extractValueSpecificPattern(periode,'(?:JANUARI |FEBRUARI |MARET |APRIL |MEI |JUNI |JULI |AGUSTUS |SEPTEMBER |OKTOBER |NOVEMBER |DESEMBER )(.*?\d{4})',1,"",True,False)
            if periode == '':
                periode = extractValueSpecificPattern(text,'(?:PERIODE I | PERIODE : |PERIODE H |PERIODE : |PERIODE :  |HAR 4 PERIODE I |\nPERIODE I |PERIODE . )(.*?JANUARI \d{2,4}|FEBRUARI \d{2,4}|MARET \d{2,4}|APRIL \d{2,4}|MEI \d{2,4}|JUNI \d{2,4}|JULI \d{2,4}|AGUSTUS \d{2,4}|SEPTEMBER \d{2,4}|OKTOBER \d{2,4}|NOVEMBER \d{2,4}|DESEMBER \d{2,4}|JANUARI\d{2,4}|FEBRUARI\d{2,4}|MARET\d{2,4}|APRIL\d{2,4}|MEI\d{2,4}|JUNI\d{2,4}|JULI\d{2,4}|AGUSTUS\d{2,4}|SEPTEMBER\d{2,4}|OKTOBER\d{2,4}|NOVEMBER\d{2,4}|DESEMBER\d{2,4})',1,"",True,False)
                if periode == extractValueSpecificPattern(text,'(?:PERIODE I | PERIODE : |PERIODE H |PERIODE : |PERIODE :  |HAR 4 PERIODE I |\nPERIODE I |PERIODE . )(.*?JANUARI\d{2,4}|FEBRUARI\d{2,4}|MARET\d{2,4}|APRIL\d{2,4}|MEI\d{2,4}|JUNI\d{2,4}|JULI\d{2,4}|AGUSTUS\d{2,4}|SEPTEMBER\d{2,4}|OKTOBER\d{2,4}|NOVEMBER\d{2,4}|DESEMBER\d{2,4})',1,"",True,False):
                    year = extractValueSpecificPattern(periode,'(?:JANUARI|FEBRUARI|MARET|APRIL|MEI|JUNI|JULI|AGUSTUS|SEPTEMBER|OKTOBER|NOVEMBER|DESEMBER)(.*?\d{4})',1,"",True,False)
                else :
                    year = extractValueSpecificPattern(periode,'(?:JANUARI |FEBRUARI |MARET |APRIL |MEI |JUNI |JULI |AGUSTUS |SEPTEMBER |OKTOBER |NOVEMBER |DESEMBER )(.*?\d{4})',1,"",True,False)
            break
        elif line.find('KCP') > -1 or line.find('KCU') > -1:
            name = text.splitlines()[i+1]
            periode = extractValueSpecificPattern(text,'(?:PERIODE I |PERIODE H |PERIODE : | PERIODE : |PERIODE :  |HAR 4 PERIODE I |\nPERIODE I |PERIODE . )(.*?JANUARI \d{2,4}|FEBRUARI \d{2,4}|MARET \d{2,4}|APRIL \d{2,4}|MEI \d{2,4}|JUNI \d{2,4}|JULI \d{2,4}|AGUSTUS \d{2,4}|SEPTEMBER \d{2,4}|OKTOBER \d{2,4}|NOVEMBER \d{2,4}|DESEMBER \d{2,4})',1,"",True,False)
            year = extractValueSpecificPattern(periode,'(?:JANUARI |FEBRUARI |MARET |APRIL |MEI |JUNI |JULI |AGUSTUS |SEPTEMBER |OKTOBER |NOVEMBER |DESEMBER )(.*?\d{4})',1,"",True,False)

            break
    return name,periode,norek,year,found

def cleanText(text):
    text=re.sub(";EUR'“BWXL%-","",text)
    text=re.sub(";EZ&”&XL i:-g:;%':gg%%%g%","",text)
    text=re.sub(";EZ&%“XL%-%:;%':Q&'","",text)
    text=re.sub(";EZ&%“L%-E:;","",text)
    text=re.sub(";EZ&%“L&:L&ZS&%&'","",text)
    text=re.sub('“','',text)
    text=re.sub(',','',text)
    text=re.sub(';','',text)
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "cleanText")
    return text

def getNextDate(k,result1):
    for c in range(k,len(result1.splitlines())):
        if getDate2(result1.splitlines()[c])!='':
            Date=getDate2(result1.splitlines()[c])
            try:
                Date=convertStringToDateTime(Date,'%d/%m').strftime('%d/%m')
                tve_common.timestamp_collector(time.time() - tve_common.start_time, "getNextDate")
                return Date
            except:
                print("Error occurred when getting a date")

def removeFooter(result1):
    x=result1.splitlines()
    T=[]

    y=re.sub('\n','',extractValueSpecificPattern(result1, 'SALDO AWAL (.*?)\n|SALDO AML (.*?)\n',0,'',False,False))
    x, T = checkFooterExistence(x, y, T, result1)

    y=re.sub('\n','',extractValueSpecificPattern(result1, 'BERSAMBUNG KE HALAMAN BERIKUT',0,'',False,False))
    x, T = checkFooterExistence(x, y, T, result1)

    y=re.sub('\n','',extractValueSpecificPattern(result1, 'MUTASI CR (.*?)\n|MTASI CR (.*?)\n|MUITASI CR (.*?)\n',0,'',False,False))
    x, T = checkFooterExistence(x, y, T, result1)

    y=re.sub('\n','',extractValueSpecificPattern(result1, 'MUTASI DB (.*?)\n|MJITASI DB (.*?)\n|MUITASI DB (.*?)\n',0,'',False,False))
    x, T = checkFooterExistence(x, y, T, result1)

    y=re.sub('\n','',extractValueSpecificPattern(result1, 'SALDO AKHIR (.*?)\n|SALDO AKHI R (.*?)\n',0,'',False,False))
    x, T = checkFooterExistence(x, y, T, result1)

    tve_common.timestamp_collector(time.time() - tve_common.start_time, "removeFooter")
    return '\n'.join(x), '\n'.join(T)

def calculateBalance(Temp_Mutasi, Temp_Balance, Temp_Date):

    j,idxx=[],[]
    for i in Temp_Date:
        j.append(i[3:])

    for g in set(j):
        idxx.append(len(j)-list(reversed(j)).index(g)-1)

    idxx.sort()
    s=1
    for m in range(1,len(idxx)):
        for h in range(s,idxx[m]+1):
            a,b,c = '','',''
            try:
                a,b,c = re.sub('\.','', re.sub('\,','', Temp_Mutasi[h])),re.sub('\.','', re.sub('\,','', Temp_Balance[h-1])),re.sub('\.','', re.sub('\,','', Temp_Balance[h]))
                try:
                    if (int(a)+int(b))!=int(c):
#                         print(a,b,c)
#                         print('NOT Balanced')
                        tve_common.timestamp_collector(time.time() - tve_common.start_time, "calculateBalance")
                        return 'NOT BALANCED'
                except:
                    print("Error when Calculating")
                    print(a,b,c)
                    tve_common.timestamp_collector(time.time() - tve_common.start_time, "calculateBalance")
                    return 'NOT BALANCED'
            except:
                print("Error when Converting")
                print(a,b,c)
                tve_common.timestamp_collector(time.time() - tve_common.start_time, "calculateBalance")
                return 'NOT BALANCED'
        s=idxx[m]+1
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "calculateBalance")
    return 'BALANCE'

def getCPTY(desc, bank,text):
    BankCPTY, CPTY = '-', '-'
    List_Bank = ['MANDIRI','MAYBANK', 'BCA', 'BNI', 'BRI', 'PERMATA', 'DANAMON', 'CIMB', 'CIMB NIAGA', 'PANIN', 'OCBC']
    List_Bank.remove("BCA")
    Detected_Bank = extractValueSpecificPattern(desc, "LLG-(.*?)\d", 1, "", False, False)

    if(Detected_Bank != ''):
        BankCPTY = Detected_Bank
        CPTY = extractValueSpecificPattern(desc, BankCPTY + "\d{4} (.*?) \d", 1, "", False, False)

    else:
        if CPTY != '':
            CPTY = text
        else:
            CPTY = '-'
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "getCPTY")
    return BankCPTY, CPTY

def cleanExtracted(list_x):
    list_y = []
    CUT_MONTH = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    MONTH = ['JANUARI', 'FEBRUARI', 'MARET', 'APRIL', 'MEI', 'JUNI', 'JULI', 'AGUSTUS', 'SEPTEMBER', 'OKTOBER', 'NOVEMBER', 'DESEMBER']

    for i in range(len(list_x)):
        n = 0
        temp = list_x[i]
        temp = re.sub(',', '', temp)
        temp = re.sub('\.', '', temp)
        temp = (' '.join(temp.split()))

        if i == 3:
            temp = getCleanedCurrencyCode(temp)

        list_y.append(temp)

    tve_common.timestamp_collector(time.time() - tve_common.start_time, "cleanExtracted")
    return list_y

def getDate1(date,text,x):
    if extractValueSpecificPattern(text,'\d{2,3}/\d{2,3}',0,"",True,False)!='':
        Temp=extractValueSpecificPattern(text,'\d{2,3}/\d{2,3}',0,"",True,False)
        if int(Temp[0:2])<int(date[0:2]):
            s=extractValueSpecificPattern(text,r'(\d+/\d+)',0,"",True,False)
            Date='' if len(s)>10 else s[0:2]+date[2:]
        else:
            Date=extractValueSpecificPattern(text,'\d{2,3}/\d{2,3}',0,"",True,False)
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "getDate1")
    return Date

def reconCalculation(Result, data):
    j,idx,s=[],[],0
    Balance = []
    for i in range(len(Result)):
        try:
            int(Result[i][1][0][3:5])
            j.append(Result[i][1][0][3:5])
        except:
            j.append(Result[i][1][0][3:6])
    for g in set(j):
        idx.append(len(j)-list(reversed(j)).index(g)-1)
    idx.sort()
    months = list(set(j))
    for m in range(len(months)):
        found = 0
        for mon in range(len(data)):
            if int(data[mon][0]) == int(months[m]) and found == 0:
                found = 1
                balance_check = float(data[mon][1])
                end_balance = float(data[mon][2])
                for h in range(s,idx[m]+1):
                    a, b =re.sub(' ','',re.sub('\,','.',re.sub('\.','',Result[h][1][4]))), re.sub(' ','',re.sub('\,','.',re.sub('\.','',Result[h][1][5])))
                    if( Result[h][1][6] == 2):
                        # Result[h][1][6] = 0
                        continue

                    try:
                        balance_check += float(a)
                        d = float(b)

                    except:
                        print("Error when Calculation in line", h)
                        print(a,b)
                        print(balance_check)
                        Result[h][1][6] = 3
                        break
                    if balance_check != d:
                        Result[h][1][6] = 3
                        break
                if(balance_check != end_balance):
                    Balance.append("MONTH " + str(data[mon][0]) + " NOT BALANCED")
                else:
                    Balance.append("MONTH " + str(data[mon][0]) + " BALANCED")

        if found == 0:
            Result, Balance = calculateBalanceAlt(Result, Balance,str(months[m]),  s+1,idx[m]+1)
        s=idx[m]+1
    tve_common.timestamp_collector(time.time() - tve_common.start_time, "reconCalculation")
    return Balance, Result

def calculateBalanceAlt(Result, Balance, month, start, end):
    j,idxx=[],[]
    for i in range(len(Result)):
        j.append(Result[i][1][0][3:])
    for g in set(j):
        idxx.append(len(j)-list(reversed(j)).index(g)-1)
    idxx.sort()
    s=1
    for h in range(start,end):
        a,b,c = re.sub('\.','', re.sub('\,','', Result[h][1][4])),re.sub('\.','', re.sub('\,','', Result[h-1][1][5])),re.sub('\.','', re.sub('\,','', Result[h][1][5]))
        try:
            if (int(a)+int(b))!=int(c):
                print(a,b,c)
                print('NOT Balanced')
                Result[h][1][6] = 3
                Balance.append("MONTH" + month + "NOT BALANCED")
                break
        except:
            print("Error when Calculating")
            print(a,b,c)
            Result[h][1][6] = 3
            Balance.append("MONTH " + month + " NOT BALANCED")
            break
    Balance.append("MONTH " + month + " BALANCED")

    tve_common.timestamp_collector(time.time() - tve_common.start_time, "calculateBalanceAlt")
    return Result, Balance

#17 Jan 2022
def fillEmptyFirst(a,b,Result, Balance):
    Empty = '#####'
    s = convertStringToDateTime(a, "%d-%m")
    e = convertStringToDateTime(b, "%d-%m")

    GENERATEDATE = [s + datetime.timedelta(days=x) for x in range(0, (e-s).days)]
    a=[]
    for date in GENERATEDATE:
        a.append([["Date","Description", "CPTY", "Bank CPTY", "Mutation","Balance", "Flag"],[convertStringToDateTime(date.strftime("%d/%m"),'%d/%m').strftime('%d-%m'),'-','-','-', Empty,str(Balance), 2]])

    return a

#17 Jan 2022
def FillingEmpty(Result):
    Filled_Result = []
    Empty = '#####'
    for i in range(len(Result)):
        if i==len(Result)-1:
            Filled_Result.append(Result[i])
        else:
            s = convertStringToDateTime(Result[i][1][0], "%d-%m")
            e = convertStringToDateTime(Result[i+1][1][0], "%d-%m")
            Filled_Result.append(Result[i])
            if (e-s).days > 1:
                GENERATEDATE = [s + datetime.timedelta(days=x) for x in range(1, (e-s).days)]
                for date in GENERATEDATE:
                    Filled_Result.append([["Date","Description", "CPTY", "Bank CPTY", "Mutation","Balance", "Flag"],[convertStringToDateTime(date.strftime("%d/%m"),'%d/%m').strftime('%d-%m'),'-','-','-', Empty,Result[i][1][5], 2]])
    return Filled_Result

def checkDateN(line):
    # if extractValueSpecificPattern(line, "\d{2}/\d{2}", 0, "", True, False) != "" and extractValueSpecificPattern(line, "TANGGAL :\d{2}/\d{2}", 0, "", True, False) == "":
    #     return True
    #19 Jan 2022
    if extractValueSpecificPattern(line, "^(\d{2}/\d{2}\s)", 0, "", True, False) != "" and extractValueSpecificPattern(line, "TANGGAL :\d{2}/\d{2}", 0, "", True, False) == "":
        return True

def convertStrMoney(s):
    money = ''
    c1 = 0
    c2 = 0
    s = re.sub('\.|,', '', s)
    s = s[::-1]
    for i in range(len(s)):
        money+=s[i]

        c1+=1
        if c1 == 2:
            money += '.'

        elif c1 > 2:
            c2+=1
            if c2%3 == 0 and c2 != 0 and i != len(s)-1:
                money += ','
    return money[::-1]

def getValueN(n):
    date, desc, mutasi, saldo = '', '', '', ''
    date = extractValueSpecificPattern(n, "^(\d{2}/\d{2}\s)", 0, "", True, False)
    desc = extractValueSpecificPattern(n, date + "(.*?)\d{1,3},", 1,'', True, False)
    mutasi = extractValueSpecificPattern(n, desc + "[\d{1,3},]*\d{1,3}\.\d{2}", 0,'', True, False)
    if extractValueSpecificPattern(desc, "[\d{1,3},]*\d{1,3}\.\d{2}", 0,'', True, False) != "":
        # desc = extractValueSpecificPattern(n, date + "\s(.*?)\d{1,3}\.|,", 1,'', True, False)
        # print("desc2: ", desc)
        mutasi = extractValueSpecificPattern(n, desc + "[\d{1,3},]*\d{1,3}\.\d{2}", 0,'', True, False)
    mutasi = re.sub(desc, "", mutasi)

    if extractValueSpecificPattern(n, "DB", 0,'', True, False):
        mutasi2 = "-"+mutasi
    else:
        mutasi2 = mutasi
    n = re.sub(" DB", "", n)
    saldo = extractValueSpecificPattern(n, mutasi + "(.*?)$", 1,'', True, False)

    return date, desc.lstrip(' '), mutasi2.lstrip(' '), saldo.lstrip(' ')

def getDescN(Description,text) :
    if extractValueSpecificPattern( text,'\d{2}/\d{2}',0,"",True,False)=='':
        Description+=' '
        Description+=text
    else:
        Description+= re.sub(extractValueSpecificPattern(text,'\d{2}/\d{2}/\d{2}',0,"",True,False),'',text)
    return Description

def getCPTYN(desc, bank, name):
    CPTY = '-'
    Detected_CPTY = ''
    Detected_Bank = []

    # BankCPTY = ''

    # List_Bank = ['MANDIRI','MAYBANK', 'BCA', 'BNI', 'BRI', 'PERMATA', 'DANAMON', 'CIMB', 'PANIN', 'OCBC', 'HSBC', 'UOB', 'CITIBANK']
    # List_Bank.remove(bank)
    # desc = desc.split()
    # Detected_Bank = list(set(desc)&set(List_Bank))
    # if(len(Detected_Bank) != 0):
    #     BankCPTY = str(Detected_Bank[0])

    BankCPTY = getCounterparty(bank,desc)
    return BankCPTY, CPTY

def fillEmptyFirstN(a,b,Result, Balance):
    Empty = '#####'
    s = convertStringToDateTime(a, "%d-%m")
    e = convertStringToDateTime(b, "%d-%m")

    GENERATEDATE = [s + datetime.timedelta(days=x) for x in range(0, (e-s).days)]
    Result=[]
    for date in GENERATEDATE:
        Result.append([["Date","Description", "CPTY", "Bank CPTY", "Mutation","Balance", "Flag"],[convertStringToDateTime(date.strftime("%d/%m"),'%d/%m').strftime('%d-%m'),'-','-','-', Empty,str(Balance), 2]])
    return Result

def fillingEmptyN(Result):
    n=0
    Filled_Result = []
    # Empty = '#####'
    Empty = '-'
    for i in range(len(Result)):
        if i==len(Result)-1:
            Filled_Result.append(Result[i])

            day = Result[i][1][0][:2]
            month = Result[i][1][0][3:5]
            daystofill = 0
            try:
                year = Result[i][1][0][6:]
                unused, daystofill = calendar.monthrange(int(year),int(month))

            except:
                today = datetime.datetime.now()
                unused, daystofill = calendar.monthrange(today.year,int(month))
            if int(day) < int(daystofill):
                todate = str(daystofill) + Result[i][1][0][2:]
                s = convertStringToDateTime(Result[i][1][0], "%d-%m")
                e = convertStringToDateTime(todate , "%d-%m")
                GENERATEDATE = [s + datetime.timedelta(days=x) for x in range(1, (e-s).days)]
                for date in GENERATEDATE:
                    Filled_Result.append([["Date","Description","CPTY","Bank CPTY","Mutation","Balance", "Flag"],[convertStringToDateTime(date.strftime("%d/%m"),'%d/%m').strftime('%d-%m'),'-','-','-', Empty,Result[i][1][3], 2]])
                Filled_Result.append([["Date","Description","CPTY","Bank CPTY","Mutation","Balance", "Flag"],[todate,'-','-','-', Empty,Result[i][1][3], 2]])
        else:
            s = convertStringToDateTime(Result[i][1][0], "%d-%m")
            e = convertStringToDateTime(Result[i+1][1][0], "%d-%m")
            Filled_Result.append(Result[i])
            if (e-s).days > 1:
                GENERATEDATE = [s + datetime.timedelta(days=x) for x in range(1, (e-s).days)]
                for date in GENERATEDATE:
                    Filled_Result.append([["Date","Description","CPTY","Bank CPTY","Mutation","Balance", "Flag"],[convertStringToDateTime(date.strftime("%d/%m"),'%d/%m').strftime('%d-%m'),'-','-','-', Empty,Result[i][1][3], 2]])
    return Filled_Result

def PROCESS_ID_BCA_TYPE_1():
    Temp_Mutasi, Temp_Balance, Temp_Date, List_Path, MONTHYEAR, Temp_Debit, Temp_Credit, Temp_Result, Result, JJ, TED, Temp_Img_Data, Temp_Footer, Results, temp_date = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    arrayThreshold=[[300,cv2.THRESH_TRUNC]]
#     arrayThreshold=[[250,cv2.THRESH_TRUNC],[255,cv2.THRESH_TRUNC],[260,cv2.THRESH_TRUNC],[300,cv2.THRESH_TRUNC]]
    r=[]
    TED=[]
    Total_Result_2=[]
    Total_Result2=[]
    Total_Result=[]
    Temp_Result=[]
    Temp_Footer=[]
    Temp_Img_Data=[]
    Temp_Debit=[]
    Temp_Credit=[]
    Temp_Mutasi=[]
    Temp_Balance=[]
    Temp_Date=[]
    Total_Result = []
    Result=[]
    Result2=[]
    Result3=[]
    Result4=[]
    FLAG=[]
    flag_=0
    DATE1,MONTH1=[],[]
    DATE,MONTH=[],[]
    p,f,compare_date=0,0,''
    Empty = '-'
    Date=''
    Description = ''
    Test_CPTY,Month='',''
    Mutasi = 0
    Balance = 0
    first_page = 0
    listpath = []
    start_counting, Check_Balance, found = False, '', False
    DIC={'JANUARY':'01','FEBRUARY':'02', 'MARCH':'03','APRIL':'04', 'MAY':'05','JUNE':'06', 'JULY':'07', 'AUGUST':'08','SEPTEMBER':'09','OCTOBER':'10', 'NOVEMBER':'11',
         'DECEMBER':'12'}

    global iPROCESS_NAME
    global iWORKING_DIRECTORY
    global iTVEXTRACT_FOLDER
    global iRAW_FOLDER_PATH
    global iPROCESS_FOLDER_PATH
    global iLOG_FOLDER_PATH
    global iTIME_FOLDER_PATH
    global iXML_FOLDER_PATH
    global iOUTPUT_FOLDER_PATH
    global iOUTPUT_XLS_FOLDER_PATH
    global iOUTPUT_JSN_FOLDER_PATH
    global iPDF2IMG_POPPLER_PATH

    tve_common.setPaths(iPROCESS_NAME, iWORKING_DIRECTORY, iTVEXTRACT_FOLDER, iRAW_FOLDER_PATH,
                        iPROCESS_FOLDER_PATH, iLOG_FOLDER_PATH, iXML_FOLDER_PATH,
                        iOUTPUT_FOLDER_PATH, iOUTPUT_XLS_FOLDER_PATH, iOUTPUT_JSN_FOLDER_PATH,
                        iPDF2IMG_POPPLER_PATH)
    filepath = tve_common.readFile(0)
    file_name = os.path.basename(filepath)
    check = False
    REGEX = '(REKENING GIRO\n|REKENING TAHAPAN\n)((.|\n)*)(.*?)NO. REKENING : \d+\n(.*?)HALAMAN : (.*)\n(.*?)PERIODE : (.*)\n((.|\n)*)MATA UANG : [A-Z][A-Z][A-Z]\n((.|\n)*)CATATAN:\n((.|\n)*)TANGGAL KETERANGAN CBG MUTASI SALDO\n((.|\n)*)((\d{2}/\d{2} (.*) ((\d{1,3},)*)?\d{1,3}\.\d{2} (DB |)((\d{1,3},)*)?\d{1,3}\.\d{2}\n((.|\n)*))*)((.|\n)*)(SALDO AWAL : (.*)\nMUTASI CR : (.*)\nMUTASI DB : (.*)\nSALDO AKHIR : (.*))?'
    try:
        result, list_text = tve_common.readNativePDF(filepath)
        check = tve_common.isNativePDF(list_text, REGEX)

    except:
        check = False
    img, listpath = tve_common.convertFileToImage(filepath) # convertFileToImageByChris
    Bank = 'BCA'

    if check == False:
        print("This is not native")
        for R in range(len(img)):
            Message = "Currently Page " + str(R+1)
            # tve_common.printLog(Message)
            processedimglvl1, img_data ,parameter, conf = processTesseractMaximumConfidenceLevel1(img[R], arrayThreshold, 1, "ind")
            x = 0
            text = tesseractImageToDataPostprocess(img_data)
            Temp_Img_Data.append(img_data)
            result1 = processTesseractPredefined(processedimglvl1,1)

            # file = open("/var/www/tve1.0/output/json/BCA_"+str(R+1)+' '+str(file_name)+".txt","w")
            # file.write(result1)
            # file.close()

            # file = open(OUTPUT_FOLDER_PATH+str(R+1)+' '+str(file_name)+".txt","w")
            # file.write(result1)
            # file.close()

            Temp_Result.append(result1)
            result, footer = removeFooter(result1)
            idx,S = findIndex(result1.splitlines(), 1)
            if S==False:
                idx,S = findIndex(result1.splitlines(), 0)


            if R==0 :
                if S:
                    LL=['UNIT KERJA','DANAMON','lAPORAN REKENING','BRI','CIMB','CIF','CIF NUMBER']
                    h=result1.splitlines()[:idx]
                    a='\n'.join(h)
                    if len(set(LL) & set(a.split()))!=0 or  extractValueSpecificPattern(a,'OPENING BALANCE|ACCOUNT HUMBER|POSTING DATE|PERIODE LAPORAN|TANGGAL CETAK|ACCOUNT NUMBER|ACCOUNT TYPE|PENCETAKAN|PANIN|DIGITAL CUSTOMER INFORMATION|TRANSACTION INGUIRY|TRANSACTION INQUIRY',0,"",True,False)!='':
                        message='Data Does Not Match'
                        # tve_common.printLog(message)

                        print(message)
                        error_message = tve_common.errorHandling(file_name, message)
                        sys.exit()
                    a='\n'.join(result1.splitlines()[-5:])
                    if extractValueSpecificPattern(a,'OCBC NISP|1500-999|TRANSACTION INGUIRY|TRANSACTION INQUIRY',0,"",True,False)!='':
                        message='Data Does Not Match'
                        # tve_common.printLog(message)
                        print(message)

                        #error_message = errorHandling(file_name, message)
                        sys.exit()
            Message = "Page " + str(R+1) + " index is " + str(idx)
            # tve_common.printLog(Message)
            # print(idx)

            if found == False:
                name,periode,norek,year,found = getName(result1,file_name)
    #         print('result = ', result)
            for i in range(idx+1+x, len(result.splitlines())):
                o=i
                if checkDate(result.splitlines()[i],date_type):
    #                 print('year : ',year)
                    Flag = 0
                    Message = "Checking Date Successfully"
                    # tve_common.printLog(Message)
                    a,Date=result.splitlines()[i],''
                    # if extractValueSpecificPattern(a,'\d{2}/\s\d{2}',0,"",True,0)!='':
                    #    t=[(m.start(0)) for m in re.finditer( '\d{2}/\s\d{2}',a)]
                    #    if len(t)!=0 and t[0]<3:
                    #        Date=extractValueSpecificPattern(a,'\d{2}/\s\d{2}',0,"",True,False)
                    if Date=='':
                        Date=getDate(a,date_type)# Date=extractValueSpecificPattern(a,'(\d{6}|\d{2}7\d{2}|\d{2}1\d{2}|\d{3}\s\d{2}|\d{2}/\d{2})',0,"",True,False)
    #                     Date+=year
    #                     print('Date+year : ',Date)
                    Description=extractValueSpecificPattern(a,Date+'(.*?) \d{1,11}(\.|\,|\:)',1,"",True,0)#\d{1,4}
                    # print('hi:',Date+' '+Description)
                    Temp=Date+Description
                    # print(Temp)
                    Temp=re.sub(Temp,'',a)
                    # print('resub Temp : ', Temp)
                    Temp_=re.sub('DB','',Temp)
                    Temp_=re.sub(':',',',Temp_)
                    Temp_=re.sub(extractValueSpecificPattern(Temp_,'(.*?\d{1,2}.\d{1,4},\D+ )',1,"",True,0),'',Temp_)
                    k=list(Temp_)
                    for s in range(len(k)-2):
                        if (k[s]==',' and k[s+1]==' ' and k[s+2].isdigit()) or (Temp_[s].isdigit() and Temp_[s+1]==' ' and Temp_[s+2]==',') or (Temp_[s]=='.' and Temp_[s+1]==' ' and Temp_[s+2].isdigit()):
                            k[s+1]=''
            #                 if k[s].isdigit() and k[s+1]==' ' and k[s+2]=='-':
            #                     k[s+1]=' '
                    Temp_=" ".join(''.join(k).split())

                    if extractValueSpecificPattern(Temp_,'(.*?)(?: \d{1,4}(\,|\.)| \d{1,11}(\,|\.)| -\d{1,4}(\,|\.))',1,"",True,0) !='':
                        Message = "Checking Mutation Successfully"
                        # tve_common.printLog(Message)
                        Mutasi= extractValueSpecificPattern(Temp_,'(.*?)(?: \d{1,4}(\,|\.)| \d{1,11}(\,|\.)| -\d{1,4}(\,|\.))',1,"",True,0)
                        if '/' in Mutasi :
                            try :
                                Flag=1
                                Mutasi2= extractValueSpecificPattern(Temp_,Mutasi+'(.*? \d{1,4}(\,|\.)+\d{1,4}(\,|\.)\d{1,4}(\,|\.)+\d{1,2})',1,"",True,0)
                                Mutasi=''
                                Mutasi = Mutasi2
                                if extractValueSpecificPattern(Temp_,Mutasi+'(.*? \d{1,4}(\,|\.)+\d{1,4}(\,|\.)\d{1,4}(\,|\.)+\d{1,2})',1,"",True,0) != '':
                                    Balance=re.sub(Mutasi,'',Temp_)
                                    Balance=re.sub('(\.\,)','.',re.sub(',','.',Balance))
                                    Balance=list(Balance)
                                    for i in range(len(Balance)-1,1,-1):
                                        if Balance[i]=='.':
                                            Balance=','
                                            break
                                    Balance=''.join(Balance)
                                else :
                                    Balance = '-'
                            except:
                                Flag=1
                                Mutasi=Mutasi
                        else :
                            Balance=re.sub(Mutasi,'',Temp_)
                            Balance=re.sub('(\.\,)','.',re.sub(',','.',Balance))
                            Balance=list(Balance)
                            for i in range(len(Balance)-1,1,-1):
                                if Balance[i]=='.':
                                    Balance[i]=','
                                    break
                            Balance=''.join(Balance)

                            # fix code : if Balance has 4 digit before . or ,
                            k=list(re.sub(' ','',Balance))
                            for s in range(len(k)-5):
                                s=0
                                if (k[s].isdigit() and k[s+1].isdigit() and k[s+2].isdigit() and k[s+3].isdigit() and k[s+4]=='.'):
                                    k[s+1]=''
                                    Balance="".join(''.join(k).split())
                                if (k[s].isdigit() and k[s+1].isdigit() and k[s+2].isdigit() and k[s+3].isdigit() and k[s+4]==','):
                                    k[s+1]=''
                                    Balance="".join(''.join(k).split())
                    else:
                        Message = "Checking Mutation Successfully"
                        Mutasi=Temp_
                        Balance='-'
                    Mutasi=re.sub('(\,\.)','.',re.sub(',','.',Mutasi))
                    Mutasi=list(Mutasi)
                    for i in range(len(Mutasi)-1,1,-1):
                        if Mutasi[i]=='.':
                            Mutasi[i]=','
                            break
                    Mutasi=''.join(Mutasi)
                    Mutasi='-'+Mutasi if 'DB' in Temp else Mutasi
                    Balance = re.sub("[^0-9.,\-]", "", Balance)
                    Mutasi = re.sub("[^0-9.,\-]", "", Mutasi)
                    if 'SALDO AWAL' in Description:
                        Message = "Checking SALDO AWAL Successfully"
                        Balance = Mutasi
                        Mutasi = '-'
                    #fixing 22159925310,25
                    u = list(re.sub(' ','',Balance))
                    if Balance!='-':
                        try :
                            if Balance !='' and '.' not in u[0:5] and (u[2]=='1' and u[3]=='5'):
                                if '.' not in u[4:9] and u[7]=='5':
                                    u[3]='.'
                                    u[7]='.'
                                    Balance=''.join(u)
                                    Flag=1
                                else:
                                    u[3]='.'
                                    Balance=''.join(u)
                                    Flag=1
                        except:
                            Flag=1
                            Balance=Balance
                else:
                    Message = "Checking Description Successfully"
                    Description = getDesc(Description,result.splitlines()[i])
                    Test_CPTY = extractValueSpecificPattern(result.splitlines()[i], "(.*?)$", 0, "", True, False)
                if Date != '':
                    Date2=getDateWF(Date,date_type)
                    try:
                        sc=int(Mutasi)
                    except:
                        sc=''
                    if Mutasi=='' or Balance=='' or (sc!='' and len(str(sc))<3):
                        Flag=1

                if  o==len(result.splitlines())-1:
                    try:
                        BankCPTY, CPTY = getCPTY(Description, Bank, Test_CPTY)
                        Test_CPTY=''
                        Result2.append([["Date","Description", "CPTY", "Bank CPTY", "Mutation","Balance","Flag"],[Date2,Description,CPTY, BankCPTY,str(Mutasi),str(Balance),Flag]])
                        Result3.append([["Date","Description", "CPTY", "Bank CPTY", "Mutation","Balance","Flag"],[Date2,Description,CPTY, BankCPTY,str(Mutasi),str(Balance),Flag]])
                        Temp_Mutasi.append(Mutasi)
                        Temp_Balance.append(Balance)
                        Temp_Date.append(Date2)
                    except:
                        print("Error on Date")
                        continue
                if  o!=len(result.splitlines())-1 and checkDate(result.splitlines()[o+1],date_type):
                    try:
                        BankCPTY, CPTY = getCPTY(Description, Bank, Test_CPTY)
                        Test_CPTY=''
                        Result2.append([["Date","Description", "CPTY", "Bank CPTY", "Mutation","Balance","Flag"],[Date2,Description,CPTY, BankCPTY,str(Mutasi),str(Balance),Flag]])
                        Result3.append([["Date","Description", "CPTY", "Bank CPTY", "Mutation","Balance","Flag"],[Date2,Description,CPTY, BankCPTY,str(Mutasi),str(Balance),Flag]])
                        Temp_Mutasi.append(Mutasi)
                        Temp_Balance.append(Balance)
                        Temp_Date.append(Date2)
                    except:
                        print("Error on Date")
                        continue

                for i in range(len(Result2)):
                    pairs = zip(Result2[i][0], Result2[i][1])
                    json_values = ('"{}": "{}"'.format(x,y) for x,y in pairs)
                    my_string = '{' + ', '.join(json_values) + '}'
                    TED.append(json.loads(my_string))
                Total_Result += Result2
                Result2=[]

        Result4=[]
        FLAG=[]
        n=0
        DATE1,MONTH1,YEAR1=[],[],[]
        Result2=Total_Result

        for s in range(len(Result2)):
            # if year != '' :
            # 	Result2[s][1][0]=Result2[s][1][0]+'/'+year
            # else :
            # 	year_defaults='2020'
            # 	Result2[s][1][0] = Result2[s][1][0]+'/'+year_defaults
            _,f=checkFormat(Result2[s][1][0])
            if f==0:
                n=s
                break
        DATE,MONTH,YEAR=[],[],[]
        for c in range(len(Result2)):
            # ##
            # if year != '' :
            #     Result2[c][1][0]=Result2[c][1][0]+'/'+year
            # else :
            #     year_defaults='2020'
            #     Result2[c][1][0] = Result2[c][1][0]+'/'+year_defaults
            #     Result2[c][1][6] = 1
            # ##
            Result2[c][1][0],Result2[c][1][6]=checkFormat(Result2[c][1][0])
            if Result2[c][1][6]==1 and c>n:
                Result2[c][1][0]=Result2[c-1][1][0]
            DATE.append(int(Result2[c][1][0][:2]))
            DATE1.append([Result3[c][1][0],int(Result2[c][1][0][:2])])
            MONTH.append(Result2[c][1][0][3:])
            MONTH1.append([Result3[c][1][0],Result2[c][1][0][3:]])
            FLAG.append(Result2[c][1][6])

        for s in range(n):
            MONTH[s]=Result2[n][1][0][3:]
            DATE[s]=int(Result2[n][1][0][:2] )
            FLAG[s]=1

        if len(set(MONTH))!=1:
            t,Z=0,True
            while Z:
                if t==0 and convertStringToDateTime(MONTH[t],'%m')>convertStringToDateTime(MONTH[t+1],'%m'):
                    MONTH[t]=MONTH[t+1]
                    FLAG[t]=1
                elif t>0 and convertStringToDateTime(MONTH[t],'%m')>convertStringToDateTime(MONTH[t+1],'%m')  :
                    MONTH[t]=MONTH[t-1]
                    FLAG[t]=1
                elif t>0 and convertStringToDateTime(MONTH[t],'%m')>convertStringToDateTime(MONTH[t+1],'%m') and FLAG[t-1]==1 and FLAG[t+1]==1:
                    MONTH[t]=MONTH[t-1]
                    FLAG[t]=1
                elif t>0 and convertStringToDateTime(MONTH[t],'%m')>convertStringToDateTime(MONTH[t+1],'%m') and FLAG[t-1]==0 and FLAG[t+1]==0:
                    MONTH[t]=MONTH[t-1]
                    FLAG[t]=1
                elif t>0 and convertStringToDateTime(MONTH[t],'%m')<convertStringToDateTime(MONTH[t+1],'%m') and MONTH[t+1]==MONTH[t-1]:
                    MONTH[t]=MONTH[t-1]
                    FLAG[t]=1
                s,e=t,len(MONTH) - MONTH[::-1].index(MONTH[t]) - 1
                # s,e=t,len(MONTH) - MONTH[::-1].index(MONTH[t]) - 1
                if e-s>20:
                    t+=1
                    continue
                for R in range(s,e+1):
                    if MONTH[R]!=MONTH[t]:
                        MONTH[R]=MONTH[t]
                        FLAG[R]=1
                t=e+1
                if t==len(MONTH)-1 or t==len(MONTH):
                    Z=False

        j=[]
        for n in range(len(MONTH)):
            j.append(MONTH[n])
        kk=[]
        for s in MONTH:
            if s not in kk:
                kk.append(s)
        MY,Date_,FLAG_,MY2,Date_2,FLAG_2=[],[],[],[],[],[]

        for k in range(len(kk)):
            s,e=MONTH.index(kk[k]),len(MONTH) - MONTH[::-1].index(kk[k])
            MY.append(MONTH[s:e])
            MY2.append(MONTH[s:e])
            Date_.append(DATE[s:e])
            Date_2.append(DATE[s:e])
            FLAG_.append(FLAG[s:e])
            FLAG_2.append(FLAG[s:e])
        for u in range(len(Date_)):
            MONTH=Date_[u]
            FLAG=FLAG_[u]
            t,Z=0,True
            while Z and len(MONTH)!=1:
                if t==0 and MONTH[t]>MONTH[t+1]:
                    MONTH[t]=MONTH[t+1]
                    FLAG[t]=1
                elif t>0 and MONTH[t]>MONTH[t+1] :
                    MONTH[t]=MONTH[t-1]
                    FLAG[t]=1
                elif t>0 and  MONTH[t]>MONTH[t+1] and FLAG[t-1]==1 and FLAG[t+1]==1:
                    MONTH[t]=MONTH[t-1]
                    FLAG[t]=1
                elif t>0 and MONTH[t]>MONTH[t+1] and FLAG[t-1]==0 and FLAG[t+1]==0:
                    MONTH[t]=MONTH[t-1]
                    FLAG[t]=1
                elif t>0 and MONTH[t]<MONTH[t+1] and (MONTH[t+1]==MONTH[t-1] or (MONTH[t]<MONTH[t+1] and MONTH[t]<MONTH[t-1])):
                    MONTH[t]=MONTH[t-1]
                    FLAG[t]=1
                elif t>0 and MONTH[t]<MONTH[t-1]:
                    MONTH[t]=MONTH[t-1]
                    FLAG[t]=1
                s,e=t,len(MONTH) - MONTH[::-1].index(MONTH[t])- 1
                if e-s>20:
                    t+=1
                    continue
                for R in range(s,e+1):
                    if MONTH[R]!=MONTH[t]:
                        MONTH[R]=MONTH[t]
                        FLAG[R]=1
                t=e+1
                if t==len(DATE)-1: #added
                    if int(DATE[t])<int(DATE[t-1]):
                        DATE[t]=DATE[t-1]
                        FLAG[t]=1

                if t==len(MONTH)-1 or   t==len(MONTH):
                    Z=False
            Date_[u]=MONTH
            FLAG_[u]=FLAG
        nd=[]
        nf=[]
        nd = [j for i in Date_ for j in i]
        nf = [j for i in FLAG_ for j in i]

        for f in range(len(Result2)):
            try:
                Result2[f][1][0]= convertStringToDateTime(str(nd[f])+'-'+j[f], "%d-%m").strftime("%d-%m")
                Result2[f][1][6]=nf[f]
            except:
                try:
                    if '/' in Result2[f][1][0]:
                        Result2[f][1][0]=re.sub('/','',Result2[f][1][0])
                        Result2[f][1][0]= convertStringToDateTime(str(nd[f])+'-'+j[f], "%d-%m").strftime("%d-%m")
                        Result2[f][1][6]=nf[f]
                except:
                    message='Machine cannot read the date'
                    print(message)
                    #error_message=errorHandling(file_name,message)
                    sys.exit()

        for g in range(len(Result3)):
            if Result3[g][1][6]==1 and Result2[f][1][6]==0:
                Result2[g][1][6]=1
        for g in range(len(Result2)):
            if Result2[g][1][5]!='' or Result2[g][1][5]!='-' :
                Bal=Result2[g][1][5]

                k = list(Bal)
                for s in range(len(k)-2):
                    try :
                        if (k[s].isdigit() and k[s+1].isdigit() and k[s+2].isdigit() and k[s+3].isdigit() and k[s+4]=='.'):
                            k[1]=''
                            Result2[g][1][6]=1
                            Result2[g][1][5]=" ".join(''.join(k).split())
                        elif (k[s].isdigit() and k[s+1].isdigit() and k[s+2].isdigit() and k[s+3].isdigit() and k[s+4].isdigit() and k[s+5]=='.'):
                            k[s+1]=''
                            k[s+3]=''
                            Result2[g][1][6]=1
                            Result2[g][1][5]=" ".join(''.join(k).split())
                        elif (k[s].isdigit() and k[s+1]=='.' and k[s+2]=='.' and k[s+3].isdigit()):
                            k[s+1]=''
                            Result2[g][1][6]=1
                            Result2[g][1][5]=" ".join(''.join(k).split())
                        elif (k[s].isdigit() and k[s+1]=='.' and k[s+2]==',' and k[s+3]=='.' and k[s+4].isdigit()):
                            k[s+1]='.'
                            k[s+2]=''
                            k[s+3]=''
                            Result2[g][1][6]=1
                            Result2[g][1][5]=" ".join(''.join(k).split())
                        elif (k[s].isdigit() and k[s+1].isdigit() and k[s+2].isdigit() and k[s+3].isdigit() and k[s+4]=='.'):
                            k[s+1]=''
                            Result2[g][1][6]=1
                            Result2[g][1][5]=" ".join(''.join(k).split())
                    except:
                        Result2[g][1][5]=Result2[g][1][5]
            if Result2[g][1][4]!='' or Result2[g][1][4]!='-' :
                Bal=Result2[g][1][4]
                k = list(Bal)
                for s in range(len(k)-2):
                    try :
                        if (k[s].isdigit() and k[s+1]=='.' and k[s+2]=='.' and k[s+3].isdigit()):
                            k[s+1]=''
                            Result2[g][1][6]=1
                            Result2[g][1][4]=" ".join(''.join(k).split())
                        elif (k[s].isdigit() and k[s+1]=='.' and k[s+2]==',' and k[s+3]=='.' and k[s+4].isdigit()):
                            k[s+2]=''
                            k[s+3]=''
                            Result2[g][1][6]=1
                            Result2[g][1][4]=" ".join(''.join(k).split())
                        elif (k[s].isdigit() and k[s+1]=='.' and k[s+2]=='.' and k[s+3]=='.' and k[s+4].isdigit()):
                            k[s+2]=''
                            k[s+3]=''
                            Result2[g][1][6]=1
                            Result2[g][1][4]=" ".join(''.join(k).split())
                        elif (k[s].isdigit() and k[s+1].isdigit() and k[s+2].isdigit() and k[s+3].isdigit() and k[s+4]=='.'):
                            k[s+1]='.'
                            Result2[g][1][6]=1
                            Result2[g][1][4]=" ".join(''.join(k).split())
                        elif (k[0].isdigit() and k[1].isdigit() and k[2].isdigit() and k[3].isdigit() and k[4].isdigit()and k[4]==','):
                            Result2[g][1][6]=1
                    except:
                        Result2[g][1][4]=Result2[g][1][4]
            # if Result2[-1][1][1]!='' or Result2[-1][1][1]!='-':
            # 	if 'KCP' in Result2[-1][1][1] or 'KCU' in Result2[-1][1][1]:
            # 		Result2[-1][1][6]=1


        try:
            recon_file_name = iOUTPUT_RECON_FOLDER_PATH + file_name[:-4] + '.json'
            f = open(recon_file_name, "r")
            data_recon = json.load(f)
            # if found_opbalance == False:
            #     OB,found_opbalance = getInBalance(data_recon)
        except:
            try:
                thousands_separator = "."
                decimal_separator =","
                RB=Result2[0][1][5]
                RB=re.sub('[,\.]','',RB)
                RM=Result2[0][1][4]
                RM=re.sub('[,\.]','',RM)
                RB=int(RB)/100
                RM=int(RM)/100
                OB=round(RB-RM,2)
                OB="{0:,.2f}".format(float(OB))
                if thousands_separator==".":
                    main_curr, decimal_curr = OB.split(".")[0], OB.split(".")[1]
                    new_main_curr = main_curr.replace(",",".")
                    OB = new_main_curr + decimal_separator + decimal_curr
            except:
                print('Error on Calculation')
        #added
        Temp_Date=[]
        for p in range(len(Result2)):
            Temp_Date.append(Result2[p][1][0])

        if first_page == 0:
            first_day = 1
            try:
                first_day = int(Result2[0][1][0][:2])
            except:
                first_page+=1
            if first_day != 1:
                try:
                    Total_Result_2 = fillEmptyFirst('1' + Result2[0][1][0][2:],Result2[0][1][0], Result2, OB)
                except:
                    print("Filling Early Fail")
        Total_Result2 = FillingEmpty(Total_Result_2+Result2)
        try:
            Message = "Check Point Data Recon"
            recon_file_name = iOUTPUT_RECON_FOLDER_PATH + file_name[:-4] + '.json'
            f = open(recon_file_name, "r")
            data_recon = json.load(f)
            if len(data_recon) > 0:
                BalanceChecking, Total_Result2 = reconCalculation(Total_Result2, data_recon)
                Message = "Check Point reconCalculation"

            else:
                BalanceChecking = calculateBalance(Temp_Mutasi, Temp_Balance, Temp_Date)
                Message = "Check Point calculateBalance"
        except:
            BalanceChecking = calculateBalance(Temp_Mutasi, Temp_Balance, Temp_Date)
            Message = "Check Point calculateBalance exception"
            tve_common.printLog(Message)

        for u in range(len(Total_Result2)):
            if Total_Result2[-1][1][1]!='' or Total_Result2[-1][1][1]!='-':
                if 'KCP' in Total_Result2[-1][1][1] or 'KCU' in Total_Result2[-1][1][1]:
                    Total_Result2 = np.delete(Total_Result2, -1, axis=0)

        #17 Jan 2022
        for yr in range(len(Total_Result2)):
            if year != '' :
                Total_Result2[yr][1][0]=Total_Result2[yr][1][0]+'-'+year
            else:
                year_default = '2020'
                Total_Result2[yr][1][0]=Total_Result2[yr][1][0]+'-'+year_default
                Total_Result2[yr][1][6] = 1

        Rev_TED = []
        for i in range(len(Total_Result2)):
            pairs = zip(Total_Result2[i][0], Total_Result2[i][1])
            json_values = ('"{}": "{}"'.format(x,y) for x,y in pairs)
            my_string = '{' + ', '.join(json_values) + '}'
            Rev_TED.append(json.loads(my_string))

        user_defined_list = ["PERIODE","NAME", "BANK", "MATA UANG","EXCHANGE", "NO. REKENING"]
        extractedResult = extractValuePredefinedForeList(Temp_Result[0], user_defined_list)
        extractedResult = cleanExtracted(extractedResult)
        if periode == '':
            periode = '*****'
        extractedResult[0], extractedResult[1], extractedResult[2], extractedResult[5]=periode, name, "BCA", norek
        extractedResult[3] = re.sub(" ", "",re.sub("\d+", "", extractedResult[3]))

        ConfLv = round(getConfidenceLevel(Temp_Img_Data,check),2)
        Conf = str(ConfLv) + ' %'
        user_defined_list = ["PERIODE","NAME", "BANK", "CURRENCY","EXCHANGE","ACCOUNT NO"]
        my_array = [user_defined_list, extractedResult]
        pairs = zip(my_array[0], my_array[1])
        json_values = ('"{}": "{}"'.format(label, value) for label, value in pairs)
        field_string = '{' + ', '.join(json_values) + '}'
        field_turn = json.loads(field_string)

    elif check == True:
        print("This is native")
        tve_common.printLog("This is native")
        print("Extraction has started!")
        new_line = ''
        sub = '\n'
        TED = []
        Total_Result = []
        List_Path = ""

        header_found = False
        name, periode, currency, norek, exchange = '','','','','#####'

        year = ''
        for i in range(len(list_text)):
            Result2 = []
            result1 = []
            saldo_awal = ''
            for j in range(len(list_text[i])):
                if (list_text[i][j] == '\n' and new_line != '' ):
                    new_line = new_line.replace('\n','')
                    if new_line == '':
                        continue
                    result1.append(new_line)
                    new_line = ''
                elif j == len(list_text[i])-1:
                    new_line += list_text[i][j]
                    result1.append(new_line)
                new_line += list_text[i][j]
            cek_trx = False

            for n in range(len(result1)):
                line = result1[n]
                if extractValueSpecificPattern(line, "Bersambung ke Halaman berikut|SALDO AWAL :", 0, "", True, False) != "":
                    cek_trx = False
                if cek_trx:
                    if checkDateN(line) == True:
                        if extractValueSpecificPattern(line, "SALDO AWAL", 0, "", True, False):
                            saldo_awal = extractValueSpecificPattern(line, "[\d{1,3},]*\.\d{2}$", 0,'', True, False)
                            continue
                        date, desc, mutasi, saldo = '', '', '', ''
                        date, desc, mutasi, saldo = getValueN(line)
                        date = re.sub('\s','',date) #19 Jan 2022
                    else:
                        desc = getDescN(desc, line)
                        if extractValueSpecificPattern(result1[n-1], "TANGGAL KETERANGAN CBG MUTASI SALDO", 0, "", True, False) != "":
                            Total_Result[-1][1][1] = desc
                            continue

                    CPTY, BankCPTY = getCPTYN(desc, "BCA", name)
                    if n!=len(result1)-1 and checkDateN(result1[n+1]):
                        Result2.append([["Date","Description", "CPTY", "Bank CPTY", "Mutation","Balance","Flag"],[convertStringToDateTime(date,'%d/%m').strftime('%d-%m'),desc,CPTY, BankCPTY,mutasi,saldo,0]])

                    if n==len(result1)-2 or extractValueSpecificPattern(result1[n+1], "Bersambung ke Halaman berikut|SALDO AWAL :", 0, "", True, False) != "":
                            try:
                                Result2.append([["Date","Description", "CPTY", "Bank CPTY", "Mutation","Balance","Flag"],[convertStringToDateTime(date,'%d/%m').strftime('%d-%m'),desc,CPTY, BankCPTY,mutasi,saldo,0]])
                            except:
                                print("Error occurred onTime Data at line : "+str(0))
                if extractValueSpecificPattern(line, "TANGGAL KETERANGAN CBG MUTASI SALDO", 0, "", True, False) != "":
                    cek_trx = True
                if header_found != True:
                    if extractValueSpecificPattern(line, "NO. REKENING", 0, "", True, False):
                        name = extractValueSpecificPattern(line, "(.*?) NO. REKENING",1, "", True, False)
                        norek = extractValueSpecificPattern(line, "\d{1,14}",0, "", True, False)
                    if extractValueSpecificPattern(line, "PERIODE",0, "", True, False):
                        periode = extractValueSpecificPattern(line, "PERIODE : (.*?)$",1, "", True, False)
                        year = extractValueSpecificPattern(periode, "\d{4}$",0, "", True, False)
                    if extractValueSpecificPattern(line, "MATA UANG",0, "", True, False):
                        currency = extractValueSpecificPattern(line, "MATA UANG : (.*?)$",1, "", True, False)
                        header_found= True

            Total_Result += Result2

        Total_Result_2 = []
        if int(Total_Result[0][1][0][0:2]) != 1 :
            Total_Result_2 = fillEmptyFirstN('01' + Total_Result[0][1][0][2:],Total_Result[0][1][0], Total_Result, Total_Result[0][1][5])
        Total_Result2 = fillingEmptyN(Total_Result_2+Total_Result)

        #17 Jan 2020
        for yr in range(len(Total_Result2)):
            if year != '' :
                Total_Result2[yr][1][0]=Total_Result2[yr][1][0]+'-'+year
            else:
                year_default = '2020'
                Total_Result2[yr][1][0]=Total_Result2[yr][1][0]+'-'+year_default
                try:
                    Total_Result2[yr][1][6] = 1
                except:
                    tve_common.printLog("Flagging Failed")

        Rev_TED = []
        for i in range(len(Total_Result2)):
            pairs = zip(Total_Result2[i][0], Total_Result2[i][1])
            json_values = ('"{}": "{}"'.format(x,y) for x,y in pairs)
            my_string = '{' + ', '.join(json_values) + '}'
            Rev_TED.append(json.loads(my_string))
        user_defined_list = ["PERIODE","NAME", "BANK", "MATA UANG","EXCHANGE", "NO. REKENING"]
        extractedResult = [periode, name, "BCA", currency,exchange,norek]
        Conf = str(100) + ' %'
        user_defined_list = ["PERIODE","NAME", "BANK", "CURRENCY","EXCHANGE","ACCOUNT NO"]
        my_array = [user_defined_list, extractedResult]
        pairs = zip(my_array[0], my_array[1])
        json_values = ('"{}": "{}"'.format(label, value) for label, value in pairs)
        field_string = '{' + ', '.join(json_values) + '}'
        field_turn = json.loads(field_string)
        Check_Balance = "BALANCE"

    Top_Head = ["CONF_LV", "CALCULATION", "IMG_RAW", "FIELD_LIST","TRANSACTION_LIST"]
    Top_Head_Value = [Conf, Check_Balance, listpath, field_turn, Rev_TED]

    mystring = {}
    mystring[Top_Head[0]] = Top_Head_Value[0]
    mystring[Top_Head[1]] = Top_Head_Value[1]
    mystring[Top_Head[2]] = Top_Head_Value[2]
    mystring[Top_Head[3]] = Top_Head_Value[3]
    mystring[Top_Head[4]] = Top_Head_Value[4]

    json_formatted_str = json.dumps(mystring, indent=2)

    print(json_formatted_str)
    print("--- %s seconds ---" % (time.time() - tve_common.start_time))
    with open(iOUTPUT_JSN_FOLDER_PATH+file_name[:-4]+'.txt', "w") as d:
        json.dump(mystring, d,indent=3)

    print("Extraction has completed!")

    return (json_formatted_str)
'''Code'''
#
# @app.route('/')
# def bankStatement():
#     return render_template('bankStatement.html', result_1="", result_2="", result_3="", result_4="", result_5="")
#
# @app.route('/upload-image', methods=['GET', 'POST'])
# def upload_image():
#     if request.method == "POST":
#         print(f"Route.request.files:{request.files}")
#     if request.files:
#         image = request.files["img"]
#
#         if image.filename != '':
#             image.save(os.path.join(iWORKING_DIRECTORY, image.filename))
#
#             #Call Fcuntion
#             result_json = PROCESS_ID_BCA_TYPE_1()
#             result_json = json.loads(result_json)
#             result_1 = result_json["CONF_LV"]
#             result_2 = result_json["CALCULATION"]
#             result_3 = result_json["IMG_RAW"]
#
#             for x in result_json["IMG_RAW"]:
#                 print(x)
#
#             result_4 = result_json["FIELD_LIST"]
#             result_5 = result_json["TRANSACTION_LIST"]
#             #Printed Final Results
#
#             return render_template("main.html", result_1=result_1, result_2=result_2, result_3=result_3, len=len(result_3), result_4=result_4, result_5=result_5)
#
#         else:
#             error = "upload again"
#             return render_template("main.html", result_1="", result_2="", result_3="", result_4="", result_5="")
#
#     return render_template("main.html", result_1="", result_2="", result_3="", result_4="", result_5="")
#
# if __name__ == "__main__":
#     app.run()
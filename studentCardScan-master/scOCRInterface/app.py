import codecs
from imutils import contours
from flask import Flask, request, jsonify
from os import listdir
from os.path import isfile, join
from pdf2image import convert_from_path
# from skimage.filters import threshold_local
# from operator import itemgetter, attrgetter
from PIL import Image
from tf_model_helper import TFModel
# from skimage import color
# import datefinder
# import textdistance
import pytesseract
import csv
import cv2
import ftfy
import imutils
import json
import math
import matplotlib.pyplot as plt
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
from io import BytesIO
import base64
from pathlib import Path
from studentCard import Process_StudentID
from bpjs import ProcessID_BPJS
from bankStatement import PROCESS_ID_BCA_TYPE_1

'''Folder Location'''
path = pathlib.Path().resolve()

# server path
# path = r"/home/hostinger/ftp/myfyp1/scOCRInterface"

pytesseract.pytesseract.tesseract_cmd = r"C:/Users/Z/Tesseract-OCR/tesseract.exe"  # version 5.0.0
# pytesseract.pytesseract.tesseract_cmd = r"C:/Users/p/Tesseract-OCR/tesseract.exe" #version 5.0.0
# pytesseract.pytesseract.tesseract_cmd = r"D:/tesseract/tesseract.exe" #version 5.0.0

# Student Card Folder Path
STUDENTCARD_FOLDER_PATH = str(path).replace('\\', '/') + "/StudentCardDB"
STUDENTCARD_OUTPUT = STUDENTCARD_FOLDER_PATH + "/Output"
STUDENTCARD_PROCESS = STUDENTCARD_FOLDER_PATH + "/Process"
STUDENTCARD_UPLOAD = STUDENTCARD_FOLDER_PATH + "/upload"

# BPJS Folder Path
BPJS_FOLDER_PATH = str(path).replace('\\', '/') + "/BPJS"
BPJS_OUTPUT = BPJS_FOLDER_PATH + "/Output"
BPJS_PROCESS = BPJS_FOLDER_PATH + "/Process"
BPJS_UPLOAD = BPJS_FOLDER_PATH + "/upload"

# Bank Statement
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

isDebug = True
app = Flask(__name__)

'''Student Card'''


@app.route('/')
def main():
    return render_template('main.html')


def clear_studentCardDB():
    # Clear content of Folders
    for i in range(3):
        if i == 0:
            folder = STUDENTCARD_UPLOAD
        elif i == 1:
            folder = STUDENTCARD_PROCESS
        else:
            folder = STUDENTCARD_OUTPUT

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


@app.route('/scanStudentCard', methods=['GET', 'POST'])
def scan_studentCard():
    filename = request.args['filename']

    if filename != '':
        json_result = Process_StudentID(filename)

        decoded_json_result = json.loads(json_result)
        print(decoded_json_result)

        # error = " "
        return render_template("main.html", raw_image=decoded_json_result["raw_img_filepath"],
                               threshold_image=decoded_json_result["threshold_filepath"],
                               boxes_text_image=decoded_json_result["boxes_text_filepath"],
                               barcode_image=decoded_json_result["detectedBarcode_path"],
                               face_image=decoded_json_result["detectedFace_path"],
                               logo_image=decoded_json_result["detectedLogo_path"],
                               final_results=decoded_json_result)
    else:
        error = "upload again"
        return render_template("main.html", raw_image="",
                               threshold_image="",
                               boxes_text_image="",
                               barcode_image="",
                               face_image="",
                               logo_image="",
                               final_results="")


@app.route('/studentCard/camera-image', methods=['GET', 'POST'])
def studentCard_base64_image():
    if request.method == "POST":
        clear_studentCardDB()

        # print(f"Route.request.files:{request.files}")
        if request.json:
            data = request.json[22:]
            # print(f"image from json:{data}")
            im = Image.open(BytesIO(base64.b64decode(data)))

            file_name = "scanned_img.png"
            im.save(os.path.join(STUDENTCARD_UPLOAD, file_name))  # "scan_folder"
            return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}
        else:
            print("json is empty")

@app.route('/studentCard/scan-camera-image', methods=['GET', 'POST'])
def studentCard_camera_image():
    # retrieve image from upload_folder
    file_name = "scanned_img.png"
    return redirect(url_for('.scan_studentCard', filename=os.path.join(STUDENTCARD_UPLOAD, file_name)))


@app.route('/studentCard/upload-image', methods=['GET', 'POST'])
def studentCard_upload_image():
    if request.method == "POST":
        print(f"Route.request.files:{request.files}")

    if request.files:
        image = request.files["img"]
        print(f"image from upload:{image}")

        if image.filename != '':
            clear_studentCardDB()
            image.save(os.path.join(STUDENTCARD_UPLOAD, image.filename))
            # Call StudentCard Function
            return redirect(url_for('.scan_studentCard', filename=os.path.join(STUDENTCARD_UPLOAD, image.filename)))
        else:
            # error = "upload again"
            return render_template("main.html", raw_image="",
                                   threshold_image="",
                                   boxes_text_image="",
                                   barcode_image="",
                                   face_image="")

    return render_template("main.html", raw_image="",
                           threshold_image="",
                           boxes_text_image="",
                           barcode_image="",
                           face_image="")


@app.route('/studentCard/demo-image')
def studentCard_demo_image():
    clear_studentCardDB()

    # Call StudentCard Function
    # file_name = "caydenSC.jpeg"
    # file_name = "nicholasSC.png"
    # file_name = "JamesSC.jpeg"
    file_name = "peterSC.jpeg"
    return redirect(url_for('.scan_studentCard', filename=STUDENTCARD_FOLDER_PATH + "/defaultImg/" + file_name))


@app.route('/studentCard/upload/<filename>')
def studentCard_img_process(filename=''):
    from flask import send_from_directory
    return send_from_directory(STUDENTCARD_PROCESS, filename)


'''Student Card'''

'''BPJS'''


@app.route('/bpjs')
def bpjs():
    return render_template('bpjs.html')


@app.route('/bpjs/upload-image', methods=['GET', 'POST'])
def bpjs_upload_image():
    if request.method == "POST":
        print(f"Route.request.files:{request.files}")
    if request.files:
        image = request.files["img"]

        # if image.filename != '':
        # Clear content of Folders
        for i in range(3):
            if i == 0:
                folder = BPJS_UPLOAD
            elif i == 1:
                folder = BPJS_PROCESS
            else:
                folder = BPJS_OUTPUT

            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

            # image.save(os.path.join(BPJS_UPLOAD, image.filename))

            # Call BPJS Fcuntion
            # rawimg_filepath, detectedQRCode_filepath, resizedImg_filepath, threshold_filepath, boxes_text_filepath, bpjs_json = ProcessID_BPJS(
            #     os.path.join(pathlib.Path(image.filename).parent.resolve(), "BPJS/upload", image.filename))
            rawimg_filepath, detectedQRCode_filepath, resizedImg_filepath, threshold_filepath, boxes_text_filepath, bpjs_json = ProcessID_BPJS(
                str(path) + "/BPJS/defaultImg/BPJS_TK-GERRY A.png")

            # Decode Final Results
            decoded_bpjs_json = json.loads(bpjs_json)
            print(decoded_bpjs_json)

            return render_template("bpjs.html", raw_image=rawimg_filepath, qrCode_image=detectedQRCode_filepath,
                                   resized_image=resizedImg_filepath, threshold_image=threshold_filepath,
                                   boxes_text_image=boxes_text_filepath,
                                   final_result=decoded_bpjs_json)

        else:
            error = "upload again"
            return render_template("bpjs.html", raw_image="", qrCode_image="", resized_image="", threshold_image="",
                                   boxes_text_image="", )

    return render_template("bpjs.html", raw_image="", qrCode_image="", resized_image="", threshold_image="",
                           boxes_text_image="", )


@app.route('/bpjs/upload/<filename>')
def bpjs_img_process(filename=''):
    from flask import send_from_directory
    return send_from_directory(BPJS_PROCESS, filename)


'''BPJS'''

'''Bank Statement'''


@app.route('/bankStatement')
def bankStatement():
    return render_template('bankStatement.html')

@app.route('/bankStatement/upload-image', methods=['GET', 'POST'])
def bankStatement_upload():
    if request.method == "POST":
        print(f"Route.request.files:{request.files}")

        # Call Function
        result_json = PROCESS_ID_BCA_TYPE_1()
        result_json = json.loads(result_json)
        result_1 = result_json["CONF_LV"]
        result_2 = result_json["CALCULATION"]
        result_3 = result_json["IMG_RAW"]

        # for x in result_json["IMG_RAW"]:
        #     print(x)

        result_4 = result_json["FIELD_LIST"]
        result_5 = result_json["TRANSACTION_LIST"]

        return render_template("bankStatement.html", result_1=result_1, result_2=result_2, result_3=result_3,
                               len=len(result_3), result_4=result_4, result_5=result_5)

    return render_template("bankStatement.html", result_1="", result_2="", result_3="", result_4="", result_5="")


'''Bank Statement'''


# Path to signature.json and model file
# ASSETS_PATH = os.path.join(".", "./model")
# TF_MODEL = TFModel(ASSETS_PATH)
# @app.post('/predict')
# def predict_image():
#     req = request.get_json(force=True)
#     image = _process_base64(req)
#     return TF_MODEL.predict(image)

# def _process_base64(json_data):
#     image_data = json_data.get("image")
#     image_data = re.sub(r"^data:image/.+;base64,", "", image_data)
#     image_base64 = bytearray(image_data, "utf8")
#     image = base64.decodebytes(image_base64)
#     return Image.open(io.BytesIO(image))
#

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)

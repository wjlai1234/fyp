# import cv2
#
# cv2.namedWindow("preview")
# vc = cv2.VideoCapture(0)
#
# if vc.isOpened(): # try to get the first frame
#     rval, frame = vc.read()
# else:
#     rval = False
#
# while rval:
#     cv2.imshow("preview", frame)
#     rval, frame = vc.read()
#     key = cv2.waitKey(20)
#     if key == 27: # exit on ESC
#
#         break
#
# vc.release()
# cv2.destroyWindow("preview")

# Python program to capture a single image
# using pygame library
import shutil
import cv2
import pathlib
import os

path = pathlib.Path().resolve()
STUDENTCARD_FOLDER_PATH = str(path) + "/StudentCardDB"
FACE_XML_PATH = str(path) + "/haarcascade_frontalface_alt.xml"
OUTPUT_FOLDER_PATH = STUDENTCARD_FOLDER_PATH + "/Output"
PROCESS_FOLDER_PATH = STUDENTCARD_FOLDER_PATH + "/Process"
UPLOAD_FOLDER_PATH = STUDENTCARD_FOLDER_PATH + "/upload"

cam = cv2.VideoCapture(0)
cv2.namedWindow("scanner")

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("scanner", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        folder = OUTPUT_FOLDER_PATH
        totalImg = 0
        for base, dirs, files in os.walk(folder):
            print('Searching in : ', base)
            for Files in files:
                totalImg += 1

        scanned_img = "scanned_img" + str(totalImg) + ".jpeg"
        process_filepath = folder + "/" + scanned_img
        cv2.imwrite(process_filepath, frame)
        captured_img_filepath = os.path.basename(process_filepath)

        print("{} written!".format(scanned_img))

cam.release()

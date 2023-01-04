import shutil
import pathlib
import os
import glob
from IPython.display import Image, display

path = pathlib.Path().resolve()
path = str(path).replace('\\', '/') + "/yolov7"
detectPath = path + "/detectLogo.py"
bestPath = path + "/bestLogo.pt"
img = path + "/testSC1.jpeg"
imgFolder = path + "/runs/detect/exp"


def clear_studentCardDB():
    # Clear content of Folders
    for filename in os.listdir(imgFolder):
        file_path = os.path.join(imgFolder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


clear_studentCardDB()

# !python {detectPath} --weights {bestPath} --source {img} --save-txt --save-conf --view-img --exist-ok
os.system("python " + detectPath + " --weights " + bestPath + " --source " + img + " --save-txt --save-conf --view-img --exist-ok")

i = 0
limit = 10000  # max images to print
for imageName in glob.glob(imgFolder + '/*.jpeg'):  # assuming JPG
    if i < limit:
        display(Image(filename=imageName))
        print("\n")
    i = i + 1

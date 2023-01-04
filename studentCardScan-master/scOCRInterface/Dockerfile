# FROM fyp-ocr-flask-app
FROM ubuntu:20.04
RUN apt-get update -y
RUN apt-get install python -y
RUN apt-get install python3-pip -y
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Malaysia
RUN apt-get install -y tzdata
RUN apt install libgl1-mesa-glx -y
RUN apt install tesseract-ocr -y
RUN apt install libtesseract-dev -y
RUN apt-get install nano -y
RUN apt install  libglib2.0-0 libsm6 libxrender1 libxext6 -y
RUN pip install tensorflow
RUN pip install flask
RUN pip install pdf2image
RUN pip install pytesseract
RUN pip install pillow
RUN pip install opencv-python
RUN pip install imutils
RUN pip install ftfy
RUN pip install matplotlib
RUN pip install pandas
RUN pip install numpy
RUN pip install pyzbar
RUN apt-get install zbar-tools -y
WORKDIR /home
# ADD ./requirements.txt /home/requirements.txt
# RUN  pip ins tall -r requirements.txt
ADD . /home
ENTRYPOINT FLASK_APP=/home/app.py flask run --host=0.0.0.0
# CMD [ "python", "./studentCard.py" ]
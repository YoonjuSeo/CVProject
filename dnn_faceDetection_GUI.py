import cv2
import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

model_name='res10_300x300_ssd_iter_140000.caffemodel' #실제 parameter값이 들어있는 파일
prototxt_name='deploy.prototxt.txt' #caffe 모델이 어떤 구성으로 이루어져있다는 것을 보여주는 것
min_confidence = 0.5 # 이 값 이상의 확률값이 나오는 것만 detection 할 것임
file_name = "image/marathon_01.jpg"
title_name = "dnn Deep Learning object detection"
frame_width = 300
frame_height = 300


def selectFile():
    file_name = filedialog.askopenfilename(initialdir="./image", title = "Select file", filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print("File name: ", file_name)
    read_image = cv2.imread(file_name)
    image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    (height, width) = read_image.shape[:2]
    detectAndDisplay(read_image,width,height)

def detectAndDisplay(frame,w,h):
    #Pass the blob through the model and obstain the detections
    #입력된 이미지를 바로 사용할 수 있는게 아니라 blob 형태로 변경해줘야 함
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)

    #Resizing to a fixed 300x300 pixels and then normalizing it
    #300x300 사이즈로 변경
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()
    min_confidence = float(sizeSpin.get())

    #loop over the detections
    #channel 갯수 만큼 for loop수행
    for i in range(0,detections.shape[2]):
        #extract the confidence(i.e., probability) associated with the prediction
        #확률(confidence)값
        confidence = detections[0,0,i,2]

        #filter out weak detections by ensuring the 'confidence' is greater than the minimum confidence
        if confidence>min_confidence:
            #compute the (x,y)-coordinates of the bounding box for the object
            #dnn모델로 face detection 한 이후 바운딩박스를 계산하는 단계
            #Caffe 모델의 출력 결과인 detections 배열에서 현재 i번째 감지 결과를 추출한다.
            #인덱스 3:7은 감지된 객체의 바운딩 박스 좌표를 나타내며, 일반적으로 순서는 다음과 같다: [xmin, ymin, xmax, ymax]
            box = detections[0,0,i,3:7] * np.array([w,h, w,h])
            (startX, startY, endX, endY) = box.astype("int")
            print(confidence, startX, startY, endX, endY)

            #draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence*100)
            #text가 표시될 위치를 지정
            y = startY - 10 if startY-10 >10 else startY+10

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0),2)
            cv2.putText(frame, text,(startX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
        
        #show the output image
        #cv2.imshow("Face Detection by dnn", frame)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image = image)
        detection.config(image=imgtk)
        detection.image = imgtk

"""
🖼️ ImageTk의 역할
ImageTk 모듈은 Python의 이미지 처리 라이브러리인 **Pillow (PIL)**와 GUI 라이브러리인 Tkinter 사이의 다리(Bridge) 역할을 수행합니다.

OpenCV 이미지 형식: OpenCV는 이미지를 NumPy 배열 형태로 다룹니다.

Tkinter 이미지 형식: Tkinter 캔버스나 위젯은 NumPy 배열을 직접 표시할 수 없고, **PhotoImage**와 같은 특정 Tkinter 호환 이미지 객체만 인식합니다.

ImageTk의 변환: ImageTk는 Pillow의 이미지 객체(Image.fromarray로 생성된 객체)를 Tkinter의 PhotoImage 객체로 효율적으로 변환하여 GUI에 사용할 수 있게 만듭니다.
"""

#main (components 구성하기)
main = Tk()
main.title(title_name)
main.geometry()

read_image = cv2.imread(file_name) # 1. 초기 이미지 로드 (이후 detectAndDisplay로 전달되지 않음)
image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB) # 2. GUI 표시용 이미지 변환 (RGB)
image = Image.fromarray(image) # 3. GUI 표시용 이미지 변환 (Pillow 객체)
imgtk = ImageTk.PhotoImage(image = image)# 4. GUI 표시용 이미지 변환 (Tkinter 객체)
(height, width) = read_image.shape[:2] # 5. 전역 변수 width, height 설정

#GUI 화면구성
label = Label(main, text=title_name)
label.config(font=("Courier",18))
label.grid(row=0, column=0,columnspan=4)
sizeLabel=Label(main, text='Min Confidence: ')
sizeLabel.grid(row=1, column=0)
sizeVal = IntVar(value=min_confidence)
sizeSpin= Spinbox(main, textvariable=sizeVal, from_=0, to=1, increment=0.05, justify=RIGHT)
sizeSpin.grid(row=1, column=1)
#W (West/서쪽): 위젯을 왼쪽(시작점)에 고정합니다.
#E (East/동쪽): 위젯을 오른쪽(끝점)에 고정합니다.
#W,E : 가로확장 (좌우로 꽉채움)
Button(main, text="File Select", height=2, command=lambda:selectFile()).grid(row=1, column=2, columnspan=2, sticky=(W, E))
detection = Label(main, image=imgtk)
detection.grid(row=2, column=0, columnspan=4)
detectAndDisplay(read_image, width, height)

main.mainloop()

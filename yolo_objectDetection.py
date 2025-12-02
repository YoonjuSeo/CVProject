import cv2
import numpy as np

min_confidence = 0.5

#Load Yolo
net = cv2.dnn.readNet("yolo/yolov3.weights","yolo/yolov3.cfg")
classes =[]

#coco.names 에 기록되어있는 object를 detect 하게 됨 (detection 대상 class로 취급)
with open("yolo/coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

#로드된 YOLO 모델(net)에 존재하는 모든 레이어(층)의 이름을 가져와 layer_names 변수에 리스트 형태로 저장합니다.
#각 레이어의 이름을 순서대로 담고 있으며, 다음 단계에서 출력 레이어를 식별하는 데 사용됩니다
layer_names = net.getLayerNames()
#output_layers 변수는 이후 이미지 처리 시 모델의 최종 출력을 얻기 위해 net.forward() 함수에 전달될 레이어 이름들의 리스트를 담게 됩니다
output_layers = [layer_names[i -1]for i in net.getUnconnectedOutLayers()]
#탐지된 각 객체 클래스(예: 사람, 자동차, 의자)에 고유한 색상을 무작위로 할당하여, 나중에 바운딩 박스(Bounding Box)를 그릴 때 사용하기 위함
colors = np.random.uniform(0,255,size=(len(classes),3))

#loading image
img = cv2.imread("image/cat_and_person.jpg")
img = cv2.resize(img, None, fx=0.2, fy=0.2) # 사이즈를 0.4 만큼 줄인다
height, width, channels = img.shape 
cv2.imshow("Original Image",img)

#Detecting objects
#입력 이미지를 신경망이 요구하는 형식인 블롭(Blob) 객체로 변환하는 역할
blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416),(0,0,0),True, crop = False)
"""
img (원본이미지) : OpenCV로 로드된 이미지(NumPy 배열)
0.00392 (스케일 팩터) : 이미지 픽셀 값(0~255)을 신경망이 사용하는 0~1 사이의 값으로 정규화하기 위한 값. ($1/255 \approx 0.00392$)
(416, 416) (크기) : 입력 이미지를 YOLO 모델이 요구하는 416x416 픽셀로 강제 크기 조정(Resize)합니다.
(0,0,0) (평균 값) : 이미지의 각 채널에서 뺄 평균 값입니다. 여기서는 (0, 0, 0)이므로 평균을 빼는 작업을 건너뜁니다.
True (채널순서) : 채널 순서를 BGR에서 RGB로 변환할지 여부를 지정합니다.
crop = False (자르기) : 크기 조정 시 이미지를 자르지 않고 전체를 유지하도록 지정합니다.
"""

net.setInput(blob) # 변환된 블롭(blob)을 로드된 신경망(net)의 입력 계층에 설정
outs = net.forward(output_layers) # 신경망 실행

#Showing information on the screen
class_ids =[]
confidences = []
boxes = []

for out in outs :
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > min_confidence:
            #Object detected
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            #Rectangle coordinates
            x = int(center_x - w / 2) # 사각형의 왼쪽 꼭지점
            y = int(center_y - h / 2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

#NMSBox : 노이즈 상자를 없애주는 알고리즘
indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN

#위에서 구한 box 갯수 만큼 for each 수행하며 네모 및 텍스트 기입
for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        print(i, label)
        color = colors[i]
        cv2.rectangle(img,(x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x,y-10), font, 3,(0,0,255),3)

cv2.imshow("YOLO Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
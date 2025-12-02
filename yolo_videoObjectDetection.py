import cv2
import numpy as np
import time

file_name = 'video/skating.mp4'
min_confidence = 0.5

def detectAndDisplay(frame):
    start_time = time.time()
    img = cv2.resize(frame, None, fx=0.4, fy=0.4) # 사이즈를 0.4 만큼 줄인다
    height, width, channels = img.shape 
    cv2.imshow("Original Image",img)

    #Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416),(0,0,0),True, crop = False)

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
            label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i]*100)
            print(i, label)
            color = colors[i]
            cv2.rectangle(img,(x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x,y-5), font, 1,color,1)

    end_time = time.time()
    process_time = end_time-start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    cv2.imshow("YOLO Video", img)


#Load Yolo
net = cv2.dnn.readNet("yolo/yolov3.weights","yolo/yolov3.cfg")
classes =[]

#coco.names 에 기록되어있는 object를 detect 하게 됨 (detection 대상 class로 취급)
with open("yolo/coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i -1]for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes),3))

#Read the video stream
cap = cv2.VideoCapture(file_name)
if not cap.isOpened:
    print('---(!) Error opening video capture')
    exit(0)
while True:
    ret,frame = cap.read()
    if frame is None:
        print('---(!) No Captured frame -- Break!')
        break
    detectAndDisplay(frame)
    #Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF  == ord('q'):
        break
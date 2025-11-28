import cv2
import numpy as np

model_name='res10_300x300_ssd_iter_140000.caffemodel' #실제 parameter값이 들어있는 파일
prototxt_name='deploy.prototxt.txt' #caffe 모델이 어떤 구성으로 이루어져있다는 것을 보여주는 것
min_confidence = 0.5 # 이 값 이상의 확률값이 나오는 것만 detection 할 것임
file_name = "image/character.png"

def detectAndDisplay(frame):
    #Pass the blob through the model and obstain the detections
    #입력된 이미지를 바로 사용할 수 있는게 아니라 blob 형태로 변경해줘야 함
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)

    #Resizing to a fixed 300x300 pixels and then normalizing it
    #300x300 사이즈로 변경
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()

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
            box = detections[0,0,i,3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            print(confidence, startX, startY, endX, endY)

            #draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence*100)
            #text가 표시될 위치를 지정
            y = startY - 10 if startY-10 >10 else startY+10

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0),2)
            cv2.putText(frame, text,(startX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
        
        #show the output image
        cv2.imshow("Face Detection by dnn", frame)


print("OpenCV version:")
print(cv2.__version__)

img = cv2.imread(file_name)
print("width: {} pixels".format(img.shape[1]))
print("Height: {} pixels".format(img.shape[0]))
print("Channels: {} pixels".format(img.shape[2]))

(height, width) = img.shape[:2]
cv2.imshow("Original Image", img)
detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # -- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces :
        center = (x+w//2, y+h//2)
        frame = cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0),4)
        faceROI = frame_gray[y:y+h, x:x+w]
        # -- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eyes_center = (x+x2+w2//2, y+y2+h2//2)
            radius = int(round((w2+h2)*0.25))
            frame = cv2.circle(frame, eyes_center, radius, (255,0,0),4)
        cv2.imshow('Capture - Face detection', frame)

img = cv2.imread("image/marathon_02.jpg")

(height, width) = img.shape[:2]
cv2.imshow("Original Image", img)

face_cascade_name = 'haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name = 'haarcascades/haarcascade_eye_tree_eyeglasses.xml'

face_cascade = cv2.CascadeClassifier()
eyes_cascade  = cv2.CascadeClassifier()

# --1. Load the cascades
# -- 1. 가스케이드 모델 로드 (오타 수정: samples.findFile)
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print(f'--(!)Error loading face cascade: {face_cascade_name}')
    exit(0)

if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print(f'--(!)Error loading eyes cascade: {eyes_cascade_name}')
    exit(0)

# -- 2. 검출 함수 호출
detectAndDisplay(img)

# 키 입력 대기 및 윈도우 종료
cv2.waitKey(0)
cv2.destroyAllWindows()
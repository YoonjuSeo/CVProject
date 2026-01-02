import numpy as np
import dlib
import cv2

RIGHT_EYE = list(range(36,42))
LEFT_EYE = list(range(42,48))
MOUTH = list(range(48,68))
NOSE = list(range(27,36))
EYEBROWS = list(range(17,27))
JAWLINE = list(range(0,17))
ALL = list(range(0,68))

predictor_file = 'model/shape_predictor_68_face_landmarks.dat'
image_file ='image/faceDetection_sample6.jpg'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

image = cv2.imread(image_file)
max_width = 800
if image.shape[1] > max_width:
    ratio = max_width / image.shape[1]
    dim = (max_width, int(image.shape[0] * ratio))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray,1)

for (i, rect) in enumerate(rects):
    points = np.matrix([[p.x, p.y] for p in predictor(gray,rect).parts()])
    show_parts = points[ALL]

    for (i, point) in enumerate(show_parts):
        x = point[0,0]
        y = point[0,1]
        cv2.circle(image, (x,y), 1, (0,255,255),-1)
        cv2.putText(image, "{}".format(i+1),(x,y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0),1)

cv2.namedWindow("Face Landmark", cv2.WINDOW_NORMAL) # 창 크기 조절 가능하게 설정
cv2.resizeWindow("Face Landmark", 800, 600)        # 출력될 창의 크기 지정
cv2.imshow("Face Landmark", image)
cv2.waitKey(0)
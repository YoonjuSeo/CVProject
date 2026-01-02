import numpy as np
import dlib
import cv2

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))

predictor_file = 'model/shape_predictor_68_face_landmarks.dat'
image_file = 'dataset/tedy/8.jpg'
MARGIN_RATIO = 1.5
OUTPUT_SIZE = (300, 300)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

image = cv2.imread(image_file)
if image is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
    exit()

image_origin = image.copy()
(image_height, image_width) = image.shape[:2]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

def getFaceDimension(rect):
    return (rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top())

def getCropDimension(rect, center):
    width = (rect.right() - rect.left())
    half_width = width // 2
    (centerX, centerY) = center
    # 좌표가 정수여야 슬라이싱이 가능하므로 int 변환
    startX = int(centerX - half_width)
    endX = int(centerX + half_width)
    startY = int(rect.top())
    endY = int(rect.bottom()) 
    return (startX, endX, startY, endY)    

for (i, rect) in enumerate(rects):
    (x, y, w, h) = getFaceDimension(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
    
    # 각 눈의 중앙점 계산
    right_eye_center = np.mean(points[RIGHT_EYE], axis=0).astype("int")
    left_eye_center = np.mean(points[LEFT_EYE], axis=0).astype("int")

    # 1. 시각화 (원본 이미지 위)
    cv2.circle(image, (right_eye_center[0,0], right_eye_center[0,1]), 5, (0, 0, 255), -1)
    cv2.circle(image, (left_eye_center[0,0], left_eye_center[0,1]), 5, (0, 0, 255), -1)

    # 2. 각도 및 스케일 계산
    eye_delta_x = right_eye_center[0,0] - left_eye_center[0,0]
    eye_delta_y = right_eye_center[0,1] - left_eye_center[0,1]
    degree = np.degrees(np.arctan2(eye_delta_y, eye_delta_x)) - 180

    eye_distance = np.sqrt((eye_delta_x ** 2) + (eye_delta_y ** 2))
    aligned_eye_distance = left_eye_center[0,0] - right_eye_center[0,0]
    scale = aligned_eye_distance / eye_distance

    # [수정 포인트] eyes_center를 float 튜플로 변환하여 OpenCV TypeError 방지
    eyes_center = (float((left_eye_center[0,0] + right_eye_center[0,0]) / 2),
                   float((left_eye_center[0,1] + right_eye_center[0,1]) / 2))
            
    # 회전 행렬 생성
    metrix = cv2.getRotationMatrix2D(eyes_center, degree, scale)

    # 3. 아핀 변환 (회전 및 스케일 조정)
    warped = cv2.warpAffine(image_origin, metrix, (image_width, image_height),
                            flags=cv2.INTER_CUBIC)
    
    cv2.imshow("warpAffine", warped)

    # 4. 크롭 및 리사이즈
    (startX, endX, startY, endY) = getCropDimension(rect, eyes_center)
    
    # 이미지 경계 밖으로 나가는 경우를 대비한 안전 조치
    startX, startY = max(0, startX), max(0, startY)
    endX, endY = min(image_width, endX), min(image_height, endY)
    
    croped = warped[startY:endY, startX:endX]
    
    if croped.size != 0:
        output = cv2.resize(croped, OUTPUT_SIZE)
        cv2.imshow("output", output)

    # 눈 랜드마크 표시
    for point in points[EYES]:
        cv2.circle(image, (point[0,0], point[0,1]), 1, (0, 255, 255), -1)

cv2.imshow("Face Alignment", image)
cv2.waitKey(0)   
cv2.destroyAllWindows()
import cv2
import face_recognition
import pickle
import time

image_file = 'image/faceDetection_sample7.jpg'
encoding_file='encodings.pickle'
unknown_name = 'Unknown'
model_method = 'cnn'


def detectAndDisplay(image):
    start_time = time.time()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model = model_method)
    encodings = face_recognition.face_encodings(rgb, boxes)

    #initialize the list of names for each face detected
    names =[]

    #loop over the facial embeddings
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = unknown_name
        counts = {}

        if True in matches:
            matchedIdxs = [i for (i,b) in enumerate(matches) if b]
            for i in matchedIdxs:
                name_found = data["names"][i]
                counts[name_found] = counts.get(name_found, 0) + 1

            name = max(counts, key=counts.get)
        names.append(name)
    
    for ((top,right,bottom,left), name) in zip(boxes, names):
        y = top -15 if top-15 >15 else top+15
        color = (0,255,0) # 찾은 얼굴은 초록색으로 네모 그림 
        line = 2
        if (name == unknown_name):
            color = (0,0,255) # unknown 얼굴은 빨간색으로 네모 그림
            line = 1
            name = ''
        
        cv2.rectangle(image, (left, top), (right,bottom), color, line)
        y = top-15 if top-15>15 else top+15
        cv2.putText(image, name, (left,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, line)
    
    end_time = time.time()
    process_time = end_time-start_time
    print("=== A frame took {:.3f} seconds".format(process_time))

    cv2.imshow("Recognition", image)


#load the known faces and embeddings
data= pickle.loads(open(encoding_file, "rb").read())

#load the input image
image = cv2.imread(image_file)
image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2) #이미지 사이즈를 0.2로 축소
detectAndDisplay(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
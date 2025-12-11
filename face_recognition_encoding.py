import cv2
import face_recognition
import pickle

dataset_paths=['dataset/Soma/']
names = ['Soma']
number_images=20
image_type = '.jpg'
encoding_file = 'encodings.pickle'
#Either cnn or hog. The CNN method is more accurate but slower. HOG is faster but less accurate
model_method = 'cnn'

#initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

#loop over the image paths
for (i, dataset_path) in enumerate(dataset_paths):
    #extract the person name from names
    name = names[i]

    #폴더 안에 있는 이미지 파일 하나씩 읽어서 처리
    for idx in range(number_images):
        file_name = dataset_path+str(idx+1)+image_type
        
        #load the input image and convert it from BGR to RGB
        image = cv2.imread(file_name)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #detect the (x,y) - coordinates of thge bounding boxes
        #corresponding to each face in the input image
        #이미지 마다 얼굴 bounding box 찾기. CNN 방식으로
        boxes = face_recognition.face_locations(rgb, model = model_method)

        #compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        #loop over the encodings
        for encoding in encodings:
            #add each encoding + name to our set of known names and encodings
            print(file_name, name, encoding)
            knownEncodings.append(encoding)
            knownNames.append(name)

#Save the facial encodings + names to disk
data = {"encodings": knownEncodings, "names": knownNames}
f = open(encoding_file, "wb")
f.write(pickle.dumps(data))
f.close()
import os

import cv2
import numpy as np
import face_recognition

path = "res"
images = []
names = []
files = os.listdir(path)
for file in files:
    thisImg = cv2.imread(f'{path}/{file}')
    images.append(thisImg)
    names.append(os.path.splitext(file)[0])
print(names)


def find_encodings(images_list):
    encode_list = []
    for img in images_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)

    return encode_list


encodeListKnownFaces = find_encodings(images)
print("Encoding done.")

capt = cv2.VideoCapture(0)

while True:
    success, camImg = capt.read()
    smallImg = cv2.resize(camImg, (0, 0), None, 0.25, 0.25)
    smallImg = cv2.cvtColor(smallImg, cv2.COLOR_BGR2RGB)

    facesInFrame = face_recognition.face_locations(smallImg)
    encodingsInFrame = face_recognition.face_encodings(smallImg, facesInFrame)

    for encoding, location in zip(encodingsInFrame, facesInFrame):
        matches = face_recognition.compare_faces(encodeListKnownFaces, encoding)
        distances = face_recognition.face_distance(encodeListKnownFaces, encoding)
        matchIndex = np.argmin(distances)

        if matches[matchIndex]:
            name = names[matchIndex]
            y1, x2, y2, x1 = location
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(camImg, (x1, y1), (x2, y2), (150, 77, 100), 3)
            cv2.rectangle(camImg, (x1, y2 - 35), (x2, y2), (150, 77, 100), cv2.FILLED)
            cv2.putText(camImg, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (255, 255, 255), 2)

    cv2.imshow("Camera", camImg)
    cv2.waitKey(1)

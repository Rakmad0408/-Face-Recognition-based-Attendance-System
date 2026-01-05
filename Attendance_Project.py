import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = 'Images'
images = []
classNames = []

myList = os.listdir(path)
# print(myList)

for cls in myList:
  curImg = cv2.imread(f'{path}/{cls}')
  images.append(curImg)
  classNames.append(cls.split('.')[0])
# print(classNames)

def findEncodings(images):
  encode_list = []
  for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    encode_list.append(encode)
  
  return encode_list


def markAttendance(name):
  with open('Attendance.csv', 'r+') as file:
    myDataList = file.readlines()
    nameList = []
    # print(myDataList)
    for line in myDataList:
      entry = line.split(',')
      nameList.append(entry[0])
    if name not in nameList:
      now = datetime.now()
      dtString = now.strftime('%H:%M:%S')
      file.writelines(f'\n{name}, {dtString}')


encode_list_known = findEncodings(images)
# print(len(encode_list_known))
print('Encoding Complete')

webcam = cv2.VideoCapture(0)

while webcam.isOpened():
  success, img = webcam.read()
  img = cv2.flip(img, 1)
  imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
  imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

  facesCurFrame = face_recognition.face_locations(imgS)
  encodingCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

  for encodeFace, faceLoc in zip(encodingCurFrame, facesCurFrame):
    matches = face_recognition.compare_faces(encode_list_known, encodeFace)
    faceDist = face_recognition.face_distance(encode_list_known, encodeFace)
    # print(faceDist)
    matchIndex = np.argmin(faceDist)

    if matches[matchIndex]:
      name = classNames[matchIndex].upper()
      # print(name)
      top, right, bottom, left = faceLoc
      top, right, bottom, left = top*4, right*4, bottom*4, left*4
      cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
      cv2.rectangle(img, (left, bottom-35), (right, bottom), (0, 255, 0), cv2.FILLED)
      cv2.putText(img, name, (left+6, bottom-6), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)

      markAttendance(name)

    # if faceDist[matchIndex]<0.50:
    #   name = classNames[matchIndex].upper()
    #   markAttendance(name)
    else:
      name = 'Unknown'
      top, right, bottom, left = faceLoc
      top, right, bottom, left = top*4, right*4, bottom*4, left*4
      cv2.rectangle(img, (left, top), (right, bottom), (7, 28, 224), 2)
      cv2.rectangle(img, (left, bottom-35), (right, bottom), (7, 28, 224), cv2.FILLED)
      cv2.putText(img, name, (left+6, bottom-6), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)



  cv2.imshow('webcam', img)
  if cv2.waitKey(1) & 255 == ord('q'):
    break


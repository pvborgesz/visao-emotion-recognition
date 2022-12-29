import cv2 

import mediapipe as mp
import keras.utils as image
import numpy as np
from keras.models import model_from_json

solution = mp.solutions.face_detection
face_detect = solution.FaceDetection()
draw = mp.solutions.drawing_utils
webcam = cv2.VideoCapture(0)

json_file = open('model8267yolo.json', 'r') 
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights('model8267yolo.h5')
while True:
    verificador, frame = webcam.read()
    if not verificador:
      print(verificador, "err")
      break

    faces_list = face_detect.process(frame)
    
    if (faces_list.detections):
      for face in faces_list.detections:
        draw.draw_detection(frame,face)
    
    img_copy = frame.copy()

    # faces = faces_list.detections[0].location_data.relative_bounding_box.xmin

    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    # x,y,w,h = faces[0]


    # cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    # roi_gray = gray_img[y:y + w, x:x + h]
    # i wanna convert the img to gray

    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    gray_img = cv2.filter2D(src=gray_img, ddepth=-1, kernel=kernel)

    roi_gray = cv2.resize(gray_img, (48, 48))
    # roi_gray = cv2.resize(roi_gray, (48, 48))
    img_pixels = image.img_to_array( cv2.resize(roi_gray, (48,48)))
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255.0
    predictions = model.predict(img_pixels)
    # predictions = ssd_detect(img_pixels, 0.5, False, model,x, y, w, h)
    print(predictions, 'predictions')
    
    max_index = int(np.argmax(predictions))
    emotions = ['Neutro', 'Feliz', 'Surpreso', 'Triste', 'Raiva', 'Nojo', 'Medo']
    predicted_emotion = emotions[max_index]
    # cv2.putText(frame, predicted_emotion,( (int(faces_list.detections[0].location_data.relative_bounding_box.xmin), int(faces_list.detections[0].location_data.relative_bounding_box.ymin))), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255), 2)
    # i wanna put the text in the face
    cv2.putText(frame, predicted_emotion,(10,20), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255), 2) #thats not working
    # cv2.putText(img_copy, predicted_emotion,)

    resized_img = cv2.resize(frame, (1000, 700))
    cv2.imshow('Facial Emotion Recognition', resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

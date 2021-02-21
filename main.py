

import cv2
import numpy as np
from tensorflow.python.keras.models import load_model

#Step 1:- detect the face of the person

model_file="opencv_face_detector_uint8.pb"   #path for model file
config_file="opencv_face_detector.pbtxt"      #path for the config file
net=cv2.dnn.readNetFromTensorflow(model_file,config_file)

#Step 2:- Start capturing the footage using opencv

cap=cv2.VideoCapture(0)             #start capturing through webcam
model=load_model("model.h5")        #loading the model
while cap:
    ret,frame=cap.read()
    frame = cv2.flip(frame, 1)      #to flip the camera or prevent lateral inversion
    (frameHeight, frameWidth) = frame.shape[:2]    #get the
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1, (300, 300),
                                 (78.4263377603, 87.7689143744, 114.895847746), swapRB=False, crop=False)

    net.setInput(blob)
    detections = net.forward()       #detecting the face of the person
    bboxes = []

    conf_threshold = 0.5
# Step 3:- Find your ROI
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]                   #extracting the confidence that is associated with the prediction
        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])  #creating a bounding box around the face
        (x1, y1, x2, y2) = box.astype("int")
        text = "{:.2f}%".format(confidence * 100)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)                #displaying a rectangle over the bounding box
        cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        #alpha = 1.0            # Contrast control (1.0-3.0) #optional
        #beta = 20              # Brightness control (0-100)
        #frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        roi = frame
        roi = cv2.resize(roi, (224, 224))           # resizing ROI
        roi = np.expand_dims(roi, axis=0)
        print(roi.shape)
#Step 4:- making prediction on the ROI
        preds = model.predict(roi)          #making prediction on the ROI
        preds = np.argmax(preds)            #returning maximum value
        print(preds)
        if preds == 0:                      #if 0 then person is wearing a mask
            text = "With_mask"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,255, 0), 4)
        elif preds == 1:
            text = "Without_mask"           #if 1 then person is not wearing a mask
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 4)
    cv2.imshow("Video", frame)                #display the ongoing video
    if cv2.waitKey((1)) % 0XFF == ord('q'):
        breakq
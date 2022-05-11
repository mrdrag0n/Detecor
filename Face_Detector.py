import cv2


#Import Trained Data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Capture Video from Webcam
webcam = cv2.VideoCapture(0)

#Iterate Each Frame
while True:
    #Read Curent Frame
    successful_frame_read, frame = webcam.read()
    #convert Image into Black And White
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Determine Face Coordinate
    face_coordinates = trained_face_data.detectMultiScale(gray_img)
    #Draw The Rectangle Around The Face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("This is a Test", frame)
    cv2.waitKey(1)
webcam.release()
import cv2,time

# type 0 for built-in camera
# type 2 for the external camera
video = cv2.VideoCapture(1)

# frame is just the first frame
a=1

# adding face detection
face_cascade = cv2.CascadeClassifier('C:\\Users\\gdsyz\Miniconda3\\envs\\smartReminder\\Lib\\site-packages\\opencv_python-4.1.2.30.dist-info\\opencv_data\\opencv\\data\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\gdsyz\\Miniconda3\\envs\\smartReminder\\Lib\\site-packages\\opencv_python-4.1.2.30.dist-info\\opencv_data\\opencv\\data\\haarcascades\\haarcascade_eye.xml')



# loop the sequence of frames
while True:
    a=a+1
    check, frame = video.read()
    # print(check)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print(gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('Capture',frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# print the number of frames
print(a)

video.release()

cv2.destroyAllWindows()

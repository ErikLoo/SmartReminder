import cv2

# colored image
# Images are read as numpy array
# note use \\ instead of
# you need to specify the locations of
face_cascade = cv2.CascadeClassifier('C:\\Users\\gdsyz\Miniconda3\\envs\\smartReminder\\Lib\\site-packages\\opencv_python-4.1.2.30.dist-info\\opencv_data\\opencv\\data\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\gdsyz\\Miniconda3\\envs\\smartReminder\\Lib\\site-packages\\opencv_python-4.1.2.30.dist-info\\opencv_data\\opencv\\data\\haarcascades\\haarcascade_eye.xml')

img = cv2.imread("test_images/girl.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow("girl",img2)
# cv2.waitKey(0)
#
# print(img.shape)

#Face detection


faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
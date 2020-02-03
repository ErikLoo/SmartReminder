
import cv2,pandas,time
from datetime import datetime

# type 0 for built-in camera
# type 2 for the external camera
video = cv2.VideoCapture(2)

# frame is just the first frame
first_frame = None

status_list= [None,None]

times = []

df = pandas.DataFrame(columns = ['Start','End'])

# loop the sequence of frames
while True:

    check, frame = video.read()

    status = 0
    # print(check)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame,gray)
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta,None,iterations=0)

    (cnts, _) = cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        # won't detect object less than 1000 in area
        if cv2.contourArea(contour)<1000:
            continue

        # change status 1 when an object is being detected

        status = 1
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    status_list.append(status)
    status_list = status_list[-2:]

    # store the time when the object appears and exits the view
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    cv2.imshow('frame',frame)
    cv2.imshow('capturing',gray)
    cv2.imshow('delta',delta_frame)
    cv2.imshow('thresh',thresh_delta)

    key = cv2.waitKey(1)

    if key== ord('q'):
        break

print(status_list)
print(times)

# store the time and save the data as csv file
for i in range(0,len(times),2):
    df = df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("Time.csv")
video.release()

cv2.destroyAllWindows()


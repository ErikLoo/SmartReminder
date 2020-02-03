import numpy as np
import cv2
import cv2.aruco as aruco
'''
    drawMarker(...)
        drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
'''

def generate_tag():
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    print(aruco_dict)
    # second parameter is id number
    # last parameter is total image size
    img = aruco.drawMarker(aruco_dict, 2, 700)
    cv2.imwrite("test_marker.jpg", img)

    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_tag():
    cap = cv2.VideoCapture(2)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # print(frame.shape) #480x640
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = frame

        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()

        # print(parameters)

        '''    detectMarkers(...)
            detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
            mgPoints]]]]) -> corners, ids, rejectedImgPoints
            '''
        # lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        print(corners)

        # It's working.
        # my problem was that the cellphone put black all around it. The alrogithm
        # depends very much upon finding rectangular black blobs

        gray = aruco.drawDetectedMarkers(gray, corners)

        # print(rejectedImgPoints)
        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # generate_tag()
    detect_tag()
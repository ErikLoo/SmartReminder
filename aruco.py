import numpy as np
import cv2
import cv2.aruco as aruco
import glob

'''
    drawMarker(...)
        drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
'''

def video_to_images():
    # Opens the Video file
    cap = cv2.VideoCapture('calib_v.mp4')
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break

        # only write out when there are multipliers of 10
        if(i%10==0):
            cv2.imwrite('camera_calib/test_img' + str(i) + '.jpg', frame)
            print("converting frame " + str(i))

        i += 1

    cap.release()
    cv2.destroyAllWindows()

def calib_camera():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # find all files with extension .jpg
    images = glob.glob('camera_calib/*.jpg')
    total_count = len(images)
    i=0

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        i+=1

        print("Calibrating based on image : " + str(i) + "/" + str(total_count))
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    print(imgpoints)

    # get the calibrated parameters
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # save the calibration parameters
    np.savez("calib_parameters",mtx = mtx,dist=dist,rvecs = rvecs,tvecs = tvecs)

    cv2.destroyAllWindows()


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

    # loaded saved calibration parameters
    with np.load('calib_parameters.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # print(frame.shape) #480x640
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # gray = frame

        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()

        # print(parameters)

        '''    detectMarkers(...)
            detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
            mgPoints]]]]) -> corners, ids, rejectedImgPoints
            '''
        # lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=mtx, distCoeff=dist)


        # print(corners)



        # draw borders on the original image
        gray = aruco.drawDetectedMarkers(gray, corners)

        # the pose estimate is not accurate. I suggest we not use it.
        # # draw coordinate system on the original image
        rvecs, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.0229, mtx, dist)
        # #
        if ids is not None:

            # plot for all markers if there are multiple
            for j in range(rvecs.shape[0]):
                gray = cv2.aruco.drawAxis(gray, mtx, dist, rvecs[j], tvec[j], 0.02)
            # print(rvecs.shape)
                print(tvec)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # video_to_images()
    # calib_camera()
    # generate_tag()
    detect_tag()
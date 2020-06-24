# import open-cv2 library for functions
import cv2

# Haar-cascadeClassifier for Face Modules loading here. Which loads the
#face Classifier for further fucntions.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Haar-cascadeClassifier for Eye Modules loading here. Which loads the eye Classifier.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Capturing the video from live webcam for detecting the face.
cap = cv2.VideoCapture(0)

# To Also use a video file as input from that video the program will detect #the face and eyes.
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # At this stage the program will read the Video frames from video.
    _, myImage = cap.read()

    # At this stage the video which is given by user is Convert into grayscale image.
    grayScale = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)

    # Here the program Detects the faces from video or webcam. With the given scale factor aurguments.
    facesDetection = face_cascade.detectMultiScale(grayScale, 1.1, 4)

    # print the number of faces
    print('Faces Found: ', len(facesDetection))
    print('The image height, width, and channel: ', facesDetection)
    print('The coordinates of each face detected: ', facesDetection)

    # This is the stage where the program Draw rectangle around the each face for detection indication to user.
    for (x, y, w, h) in facesDetection:
        cv2.rectangle(myImage, (x, y), (x+w, y+h), (255, 0, 0), 2)
        rec_face = myImage[y:y + h, x:x + w]

        # This is the stage where the program Draw rectangle around the Eyes for detection indication to user.
        eyes_corners = eye_cascade.detectMultiScale(rec_face, 1.3, 5)
        for (ex, ey, ew, eh) in eyes_corners:
            cv2.rectangle(rec_face, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    # This is the Code to Display the outputs from the given detected faces.
    cv2.imshow('img', myImage)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
        
# Release the VideoCapture object
cap.release()
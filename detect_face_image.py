# import open-cv2 library for functions
import cv2

# Haar-cascadeClassifier for Face Modules loading here. Which loads the
#face Classifier for further fucntions.
for_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')                                   


# Haar-cascadeClassifier for Eye Modules loading here. Which loads the eye Classifier.
for_eye = cv2.CascadeClassifier('haarcascade_eye.xml')

# From here the program read the input which giver by the user.from the given path to futher process.
myImage = cv2.imread('DataSet/group3.jpg')

# At this stage the image which is given by user is Convert into grayscale image.for detection.
grayScale = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)

# Here the program Detects the faces. With given scale factor aurguments.
facesDetection = for_face.detectMultiScale(grayScale, 1.1, 4)

# print the number of faces.
print('Faces Found: ', len(facesDetection))
print('The image height, width, and channel: ', facesDetection)
print('The coordinates of each face detected: ', facesDetection)

# This is the stage where the program Draw rectangle shape Indicator around the each face for
# detection indication to user.
for (x, y, w, h) in facesDetection:
    cv2.rectangle(myImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
    rec_face = myImage[y:y+h, x:x+w]

#This is the stage where the program Draw rectangle shape around the Eyes for detection indication to user.
    eyes_corners = for_eye.detectMultiScale(rec_face, 1.3, 5)
    for (ex, ey, ew, eh) in eyes_corners:
        cv2.rectangle(rec_face, (ex, ey), (ex + ew, ey+eh), (255, 0, 0), 2)

#This is the stage where the text write on the detected face.
font_style = cv2.FONT_HERSHEY_SIMPLEX
text_On_image = cv2.putText(myImage, 'Face Detected', (0, myImage.shape[0]), font_style, 2, (255, 255, 255), 2)

# This is the Code to Display the outputs from the given detected faces.
cv2.imshow('myImage', myImage)
cv2.waitKey(0)

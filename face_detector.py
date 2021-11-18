import cv2

# Load some pre-trained data on face frontasls from opencv (haar cascade algorythm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
img = cv2.imread('gal-gadot.jpg')
# img = cv2.imread('palka.jpg')

# Convert img into grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# drow rectangle around face
(x, y, w, h) = face_coordinates[-1]

cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# print(face_coordinates)

#
cv2.imshow('Face detector', img)
cv2.waitKey()

print("Code completed")

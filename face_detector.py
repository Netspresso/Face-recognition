import cv2

# Load some pre-trained data on face frontasls from opencv (haar cascade algorythm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
webcam = cv2.VideoCapture(0)

# Iterate forere over frames
while True:

    # Read the current frame
    succesful_frame_read, frame = webcam.read()

    # Convert img into grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # drow rectangle around face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the img
    cv2.imshow('Face detector', frame)
    key = cv2.waitKey(1)

    # Stop programm when Q key is pressed
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
webcam.release()

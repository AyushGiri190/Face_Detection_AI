#importing module Cv2 Computer Vision
import cv2

# if we write 0 it will make the webcam the default source for the video capture
# pasting the name of the name of the window will let the video play on the screen
webcam = cv2.VideoCapture(0)

#this is a cascade classifier used for detecting faces of a person 
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    count = 0
    
    # reading each frame recorded by the webcam 
    sucessfull_frame_read, frame = webcam.read()  # successfull_frame_read might be true or false
    
    # conerting the coloured image into black and white as it is easy to detect the image in black and white then in colour
    grayscaled_img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Getting the face coordinates by the cascade classifier we previously trained
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img1)
    for (x, y, w, h) in face_coordinates:
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Drawing a rectangle over the face coordinates
        
        count += 1 # getting the count of the total number of faces
        
        cv2.putText(frame, 'Face Found', (h, h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) # putting the text on the faces found
        
    c = str(count)
    
    cv2.putText(frame,'Total faces found'+c, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) # putiing the total number of faces found on the frame

   # Showing the processed image to the monitor
    cv2.imshow('Face Detector', frame)

     # wait key is used to stop the screen from quickly getting away
    key = cv2.waitKey(1)
    if key == 13 or key == 27:  # Escapes the loops and terminates cv window when enter or escape is entered.
        break
    
# destroying and releasing all the previously created windows
webcam.release()
cv2.destroyAllWindows()
# program completed to find a face in a image

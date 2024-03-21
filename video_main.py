import os 
import face_recognition
import cv2
import numpy as np

name_file = input("tapez le nom de votre video a traiter !")
name_file = "video_test/" + name_file + ".MOV"

unknown_image = face_recognition.load_image_file(name_file)
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]


mike_tyson_img = face_recognition.load_image_file("image_test/mike tyson.jpg")
mike_tyson_face_encoding = face_recognition.face_encodings(mike_tyson_img)[0]

brad_pitt_img = face_recognition.load_image_file("image_test/brad pitt.jpg")
brad_pitt_face_encoding = face_recognition.face_encodings(brad_pitt_img)[0]

lewis_hamilton_img = face_recognition.load_image_file("image_test/lewis hamilton.jpg")
lewis_hamilton_face_encoding = face_recognition.face_encodings(lewis_hamilton_img)[0]

squeezie_img = face_recognition.load_image_file("image_test/squeezie.jpg")
squeezie_face_encoding = face_recognition.face_encodings(squeezie_img)[0]

baki_img = face_recognition.load_image_file("image_test/baki.jpg")
baki_face_encoding = face_recognition.face_encodings(baki_img)[0]

adriana_lima_img = face_recognition.load_image_file("image_test/adriana lima.jpg")
adriana_lima_face_encoding = face_recognition.face_encodings(adriana_lima_img)[0]

known_face_encodings = [
    mike_tyson_face_encoding,
    brad_pitt_face_encoding,
    lewis_hamilton_face_encoding,
    squeezie_face_encoding,
    baki_face_encoding,
    adriana_lima_face_encoding
    
]

known_face_names = [
    "Mike Tyson",
    "Brad Pitt",
    "Lewis Hamilton",
    "Squeezie",
    "Baki",
    "Adriana Lima"
]

face_locations = []
face_encodings = []
face_names = []
frame_number = 0


input_movie = cv2.VideoCapture(name_file)
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (640, 360))

ret, frame = input_movie.read()
frame_number += 1

rgb_frame = frame[:, :, ::-1]

# Find all the faces and face encodings in the current frame of video
face_locations = face_recognition.face_locations(rgb_frame)
face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

face_names = []

for face_encoding in face_encodings:
    results = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.50)

for (top, right, bottom, left), name in zip(face_locations, face_names):
    if not name:
        continue

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


print("Writing frame {} / {}".format(frame_number, length))
output_movie.write(frame)

print(results)






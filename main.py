import os 
import face_recognition
import cv2
import numpy as np


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

name_file = input("tapez le nom de votre image a traiter !")
name_file_test = name_file + "Vtest" + ".jpg"
name_file_img = name_file
name_file = "image_test/" + name_file + ".jpg"

unknown_image = face_recognition.load_image_file(name_file)
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]


# Get all images in a directory
def get_images(path):
    images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            images.append(os.path.join(root, file))
    return images

def get_face(image):
    image = face_recognition.load_image_file(image)
    face_locations = face_recognition.face_locations(image)
    return face_locations[0]


def draw_rectangle(image, output, coordinates):
    image =  cv2.imread(image)
    x, y, w, h = coordinates
    cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 10)
    cv2.rectangle(image, (x, y - 100), (w, y), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name_file_img, (x + 6, y - 6), font, 3.5, (255, 255, 255), 3)
    cv2.imwrite(os.path.join(os.getcwd(), "output", output) ,image)

results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)

print("traitement en cours...")

print((name_file))

print(get_face(name_file))

draw_rectangle(name_file, name_file_test, get_face(name_file))

print(results)

print("traitement termine !")




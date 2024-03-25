import os 
import face_recognition
import cv2
import numpy as np


mael_llado_img = face_recognition.load_image_file("now_people_face/mael.jpeg")
mael_llado_face_encoding = face_recognition.face_encodings(mael_llado_img)[0]
#(869, 1380, 2019, 230)

sawyer_soule_img = face_recognition.load_image_file("now_people_face/sawyer.jpeg")
sawyer_soule_face_encoding = face_recognition.face_encodings(sawyer_soule_img)[0]
#(724, 1149, 1682, 191)

sulyvan_rouzeau_img = face_recognition.load_image_file("now_people_face/sulyvan.jpeg")
sulyvan_rouzeau_face_encoding = face_recognition.face_encodings(sulyvan_rouzeau_img)[0]
#(583, 1503, 1963, 123)

azdine_bachiri_img = face_recognition.load_image_file("now_people_face/azdine bachiri.jpeg")
azdine_bachiri_face_encoding = face_recognition.face_encodings(azdine_bachiri_img)[0]
#(997, 1508, 2147, 357)

mathis_dumas_img = face_recognition.load_image_file("now_people_face/mathis dumas.jpeg")
mathis_dumas_face_encoding = face_recognition.face_encodings(mathis_dumas_img)[0]
#(613, 1380, 1764, 230)

known_face_encodings = [
    mael_llado_face_encoding,
    sawyer_soule_face_encoding,
    sulyvan_rouzeau_face_encoding,
    azdine_bachiri_face_encoding,
    mathis_dumas_face_encoding, 
]

known_face_names = [
    "Mael llado",
    "Sawyer soule",
    "Sulyvan rouzeau",
    "Azdine bachiri",
    "mathis dumas"
]

face_locations = []
face_encodings = []
face_names = []

name_file = input("tapez le nom de votre image a traiter !")
name_file_test = name_file + "Vtest" + ".jpeg"
name_file_img = name_file
name_file = os.path.join(os.getcwd(), "now_people_face", f"{name_file}.jpeg")

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




import os 
import face_recognition
import cv2
import numpy as np

face_locations = []
face_encodings = []
face_names = []

name_file = input("tapez le nom de votre image a traiter !")
name_file_test = name_file + "Vtest" + ".jpeg"
name_file_img = name_file
name_file = os.path.join(os.getcwd(), "image_test", f"{name_file}.jpeg")

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

known_people = get_images(os.path.join(os.getcwd(),"now_people_face"))
known_face_encodings = []
known_face_names = []

for people in known_people:
    image = face_recognition.load_image_file(people)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(str(os.path.basename(people)).replace(".jpeg", ""))
    print(f"chargement du visage de:{people}")


results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.55)

print("traitement en cours...")

print((name_file))

print(get_face(name_file))

draw_rectangle(name_file, name_file_test, get_face(name_file))

print(results)

for i, result in enumerate(results):
    if result:
        print("Personne reconnue :", known_face_names[i])

print("traitement termine !")
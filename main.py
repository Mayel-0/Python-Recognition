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

sulyvan_rouzeaud_img = face_recognition.load_image_file("now_people_face/sulyvan.jpeg")
sulyvan_rouzeaud_face_encoding = face_recognition.face_encodings(sulyvan_rouzeaud_img)[0]
#(583, 1503, 1963, 123)

fabio_barbot_img = face_recognition.load_image_file("now_people_face/fabio.jpeg")
fabio_barbot_face_encoding = face_recognition.face_encodings(fabio_barbot_img)[0]

#alexis_benoit_img = face_recognition.load_image_file("now_people_face/alexis.jpeg")
#alexis_benoit_face_encoding = face_recognition.face_encodings(alexis_benoit_img)[0]

#eugene_blin_img = face_recognition.load_image_file("now_people_face/eugene.jpeg")
#eugene_blin_face_encoding = face_recognition.face_encodings(eugene_blin_img)[0]

noa_caubet_img = face_recognition.load_image_file("now_people_face/noa.jpeg")
noa_caubet_face_encoding = face_recognition.face_encodings(noa_caubet_img)[0]

gabriel_chinarro_img = face_recognition.load_image_file("now_people_face/gabriel.jpeg")
gabriel_chinarro_face_encoding = face_recognition.face_encodings(gabriel_chinarro_img)[0]

#nathan_corre_img = face_recognition.load_image_file("now_people_face/nathan-corre.jpeg")
#nathan_corre_face_encoding = face_recognition.face_encodings(nathan_corre_img)[0]

nolan_delmont_img = face_recognition.load_image_file("now_people_face/nolan.jpeg")
nolan_delmont_face_encoding = face_recognition.face_encodings(nolan_delmont_img)[0]

fatih_emiral_img = face_recognition.load_image_file("now_people_face/fatih.jpeg")
fatih_emiral_face_encoding = face_recognition.face_encodings(fatih_emiral_img)[0]

leandro_fargeas_img = face_recognition.load_image_file("now_people_face/leandro.jpeg")
leandro_fargeas_face_encoding = face_recognition.face_encodings(leandro_fargeas_img)[0]

nathan_gaudin_img = face_recognition.load_image_file("now_people_face/nathan-gaudin.jpeg")
nathan_gaudin_face_encoding = face_recognition.face_encodings(nathan_gaudin_img)[0]

jean_baptiste_giral_img = face_recognition.load_image_file("now_people_face/jb.jpeg")
jean_baptiste_giral_face_encoding = face_recognition.face_encodings(jean_baptiste_giral_img)[0]

alexy_leroy_img = face_recognition.load_image_file("now_people_face/alexy-leroy.jpeg")
alexy_leroy_face_encoding = face_recognition.face_encodings(alexy_leroy_img)[0]

steeven_magloire_img = face_recognition.load_image_file("now_people_face/steeven.jpeg")
steeven_magloire_face_encoding = face_recognition.face_encodings(steeven_magloire_img)[0]

#quentin_moinard_img = face_recognition.load_image_file("now_people_face/quentin.jpeg")
#quentin_moinard_face_encoding = face_recognition.face_encodings(quentin_moinard_img)[0]

mathis_perey_img = face_recognition.load_image_file("now_people_face/mathys.jpeg")
mathis_perey_face_encoding = face_recognition.face_encodings(mathis_perey_img)[0]

#youssef_post_img = face_recognition.load_image_file("now_people_face/azdine bachiri.jpeg")
#youssef_post_face_encoding = face_recognition.face_encodings(youssef_post_img)[0]

fabrice_rouchard_img = face_recognition.load_image_file("now_people_face/fabrice.jpeg")
fabrice_rouchard_face_encoding = face_recognition.face_encodings(fabrice_rouchard_img)[0]

adam_serra_img = face_recognition.load_image_file("now_people_face/adam.jpeg")
adam_serra_face_encoding = face_recognition.face_encodings(adam_serra_img)[0]

sevan_vazquez_img = face_recognition.load_image_file("now_people_face/sevan.jpeg")
sevan_vazquez_face_encoding = face_recognition.face_encodings(sevan_vazquez_img)[0]

azdine_bachiri_img = face_recognition.load_image_file("now_people_face/azdine bachiri.jpeg")
azdine_bachiri_face_encoding = face_recognition.face_encodings(azdine_bachiri_img)[0]
#(997, 1508, 2147, 357)

mathis_dumas_img = face_recognition.load_image_file("now_people_face/mathis dumas.jpeg")
mathis_dumas_face_encoding = face_recognition.face_encodings(mathis_dumas_img)[0]
#(613, 1380, 1764, 230)

known_face_encodings = [
    azdine_bachiri_face_encoding,
    fabio_barbot_face_encoding,
    #alexis_benoit_face_encoding,
    #eugene_blin_face_encoding,
    noa_caubet_face_encoding,
    gabriel_chinarro_face_encoding,
    #nathan_corre_face_encoding,
    nolan_delmont_face_encoding,
    mathis_dumas_face_encoding,
    fatih_emiral_face_encoding,
    leandro_fargeas_face_encoding,
    nathan_gaudin_face_encoding,
    jean_baptiste_giral_face_encoding,
    alexy_leroy_face_encoding,
    mael_llado_face_encoding,
    steeven_magloire_face_encoding,
    #quentin_moinard_face_encoding,
    mathis_perey_face_encoding,
    #youssef_post_face_encoding,
    fabrice_rouchard_face_encoding,
    sulyvan_rouzeaud_face_encoding,
    adam_serra_face_encoding,
    sawyer_soule_face_encoding,
    sevan_vazquez_face_encoding
]

known_face_names = [
    "Azdine Bachiri",
    "Fabio Barbot-krisa",
    #"Alexis Benoit",
    #"Eugene Blin",
    "Noa Caubet",
    "Gabriel Chinarro",
    #"Nathan Corre",
    "Nolan Delmont",
    "Mathis Dumas",
    "Fatih Emiral",
    "Leandro Fargeas",
    "Nathan Gaudin",
    "Jean-Baptiste Giral",
    "Alexy Leroy",
    "Mael Llado",
    "Steeven Magloire",
    #"Quentin Moinard",
    "Mathis Perey",
    #"Youssef Post",
    "Fabrice Rouchard",
    "Sulyvan Rouzeaud",
    "Adam Serra",
    "Sawyer Soule",
    "Sevan Vazquez"
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




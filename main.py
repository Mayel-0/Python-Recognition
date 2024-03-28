import os 
import face_recognition
import cv2
import numpy as np

#Live webcam recognition
def webcam_recognition(known_face_encodings, known_face_names):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

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


def draw_rectangle(image, output, coordinates, name_file_img):
    image =  cv2.imread(image)
    x, y, w, h = coordinates
    cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 10)
    cv2.rectangle(image, (x, y - 100), (w, y), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name_file_img, (x + 6, y - 6), font, 3.5, (255, 255, 255), 3)
    cv2.imwrite(os.path.join(os.getcwd(), "output", output) ,image)


def face_encoding():
    known_people = get_images(os.path.join(os.getcwd(),"now_people_face"))

    known_face_encodings = []
    known_face_names = []

    for people in known_people:
        image = face_recognition.load_image_file(people)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(str(os.path.basename(people)).replace(".jpeg", ""))
        print(f"chargement du visage de:{people}")

    return known_face_encodings, known_face_names

def start_image(known_face_encodings, known_face_names):
    face_locations = []
    face_encodings = []
    face_names = []

    name_file = input("tapez le nom de votre image a traiter !")
    name_file_test = name_file + "Vtest" + ".jpeg"
    name_file_img = name_file
    name_file = os.path.join(os.getcwd(), "image_test", f"{name_file}.jpeg")

    unknown_image = face_recognition.load_image_file(name_file)
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.55)

    print("traitement en cours...")

    print((name_file))

    print(get_face(name_file))

    draw_rectangle(name_file, name_file_test, get_face(name_file), name_file_img)

    print(results)

    for i, result in enumerate(results):
        if result:
            print("Personne reconnue :", known_face_names[i])

    print("traitement termine !")


known_face_encodings, known_face_names = face_encoding()

start_image(known_face_encodings, known_face_names)

webcam_recognition(known_face_encodings, known_face_names)


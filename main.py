import os 
import face_recognition
import cv2
import numpy as np

def get_difference_images(image1, image2):
    image1 = face_recognition.load_image_file(image1)
    image2 = face_recognition.load_image_file(image2)
    image1 = np.array(image1)
    image2 = np.array(image2)
    difference_images = np.subtract(image1, image2)
    return difference_images


name_file = input("tapez le nom de votre image a traiter !")
name_file_test = name_file + "Vtest" + ".jpg"
name_file_img = name_file
name_file = "image_test/" + name_file + ".jpg"

user_input = int(input("voulez vous faire une difference image ? si oui tapez 1 sinon 0 :"))
if user_input == 1:
    name_file_diff = input("avec quel image souhaiter vous faire une difference ?")
    name_file_diff = "image_test/" + name_file_diff + ".jpg"

elif user_input != 0 and user_input != 1:
    pass

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
    cv2.putText(image, os.path.basename(name_file_img), (int((x+h)/2),y-30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 10)
    cv2.imwrite(os.path.join(os.getcwd(), "output", output) ,image)



print("traitement en cours...")

print((name_file))
print(get_face(name_file))

if user_input == 1: print((name_file_diff) + get_face(name_file_diff))
elif user_input != 0 and user_input != 1:
    pass

draw_rectangle(name_file, name_file_test, get_face(name_file))

if user_input == 1:
    difference_images = get_difference_images(name_file, name_file_diff)
    cv2.imwrite(os.path.join(os.getcwd(), "image_difference", "difference_images.jpg"), difference_images)
elif user_input != 0 and user_input != 1:
    pass

print("traitement termine !")




import os 
import face_recognition
import cv2

IMAGES_DIR = os.path.join(os.getcwd(), 'image_reconnaissable')

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
    return face_locations

def draw_rectangle(image, output, coordinates):
    image =  cv2.imread(image)
    x, y, w, h = coordinates
    cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 10)
    cv2.imwrite(output, image)


print(get_face("test.png"))

draw_rectangle("test.png","v.jpg", get_face("test.png")[0])


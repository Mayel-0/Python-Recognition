import os 
import face_recognition

IMAGES_DIR = os.path.join(os.getcwd(), 'image_reconnaissable')

# Get all images in a directory
def get_images(path):
    images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            images.append(os.path.join(root, file))
    return images

print(get_images(IMAGES_DIR))


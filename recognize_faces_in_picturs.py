import face_recognition

# Load the jpg files into numpy arrays
mike_image = face_recognition.load_image_file("image_test/mike tyson.jpg")
brad_image = face_recognition.load_image_file("image_test/brad pitt.jpg")
baki_image = face_recognition.load_image_file("image_test/baki.jpg")
unknown_image = face_recognition.load_image_file("image_test/bite.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    mike_face_encoding = face_recognition.face_encodings(mike_image)[0]
    brad_face_encoding = face_recognition.face_encodings(brad_image)[0]
    baki_face_encoding = face_recognition.face_encodings(baki_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    mike_face_encoding,
    brad_face_encoding,
    baki_face_encoding
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

print("Le visage inconnu est Brad pitt si {}".format(results[0]))
print("Le visage inconnu est Mike Tyson si {}".format(results[1]))
print("Le visage inconnu est Baki si {}".format(results[2]))
print("Le visage inconnu correspont a la photo {}".format(not [] in results))
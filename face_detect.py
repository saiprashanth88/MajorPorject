import face_recognition
import os

# Path to the folder containing images
image_folder = "image"

# Load known face encodings and their names
known_face_encodings = []
known_face_names = []

# Iterate through each image file in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        # Load the image and extract face encodings
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        for face_encoding in face_encodings:
            known_face_encodings.append(face_encoding)
            # Extract the name from the file name
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)

# Load the sample image
sample_image_path = "mypic.jpg"
sample_image = face_recognition.load_image_file(sample_image_path)

# Find face locations and encodings in the sample image
face_locations = face_recognition.face_locations(sample_image)
face_encodings = face_recognition.face_encodings(sample_image, face_locations)

# Initialize an array for storing the names of recognized faces
recognized_names = []

# Compare each face encoding found in the sample image with known face encodings
for face_encoding in face_encodings:
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    names = [known_face_names[i] for i, match in enumerate(matches) if match]
    recognized_names.extend(names)

# Output the recognized names or "Person" if no recognized names are found
if recognized_names:
    print("Recognized names:", recognized_names)
else:
    print("Person")

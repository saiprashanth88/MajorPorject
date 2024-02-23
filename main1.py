import os
import cv2
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import face_recognition
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import time



model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
genai.configure(api_key="Your API KEY")

def image():
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        videoCaptureObject = cv2.VideoCapture(0)
        ret, frame = videoCaptureObject.read()
        cv2.imwrite("mypic.jpg", frame)
        videoCaptureObject.release()
    except Exception as e:
        print(e+"Cannot take image!!!")

def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds



def recog(sample):
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
    sample_image_path = sample
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
        return recognized_names
    else:
        return "Person"

def palm(string, names):
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])
    # print(chat)
    sentence = string
    word = names
    response = chat.send_message(f"I will give you Sentence and some words, Example: A Man is sitting and the other words like Prashanth. Now replace the man with Prashanth and give me putput as Prashanth is sitting. So give me output in that manner and my sentence is {sentence} and word is {word}, Note: You have to output only The modified sentence. Optimize the sentence to more accurate if needed, Note:If there are No persons in the list then do not do anything, Just return the sentence, Note: Replace with correct prepositions in sentece only, If girl name is in words then replace with girl prepostion only" )

    # for chunk in response:
    #     print(chunk.text)
    return response
def main():

   image()
   string = predict_step(['mypic.jpg'])
   print(string)
   name = recog('mypic.jpg')
#    print(name)
   prompt = palm(string, name)
   for chunk in prompt:
      return chunk.text
print(main())

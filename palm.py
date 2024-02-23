import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))



genai.configure(api_key="AIzaSyCzLk7mX5JdNa9QaPMoMV64lwLLFdV0TMY")

# for m in genai.list_models():
#   if 'generateContent' in m.supported_generation_methods:
#     print(m.name)
# model = genai.GenerativeModel('gemini-pro-vision')
# response = model.generate_content('mypic.jpg')

# to_markdown(response.text)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])
# print(chat)
sentence = "a person sitting in a room with a computer"
word = "Vandana"
response = chat.send_message(f"I will give you Sentence and some words, Example: A Man is sitting and the other words like Prashanth. Now replace the man with Prashanth and give me putput as Prashanth is sitting. So give me output in that manner and my sentence is {sentence} and word is {word}, Note: You have to output only The modified sentence. Optimize the sentence to more accurate if needed" )

for chunk in response:
  print(chunk.text)


# print(chat.history)


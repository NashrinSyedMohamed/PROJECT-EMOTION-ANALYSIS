# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages

import streamlit as st
from streamlit_player import st_player

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
#outputFrame = None
#lock = threading.Lock()
# initialize a flask object
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
#vs = VideoStream(src="http://81.83.10.9:8001/mjpg/video.mjpg").start()
#vs = VideoStream(src="http://81.149.56.38:8084/mjpg/video.mjpg").start()
#vs = VideoStream(src="http://81.149.56.38:8081/mjpg/video.mjpg").start()
#vs = st_player("http://81.149.56.38:8081/mjpg/video.mjpg")

# Play a youtube video with streamlit
vs = st_player("https://youtu.be/KMJS66jBtVQ")
#vs = st_player("http://81.149.56.38:8081/mjpg/video.mjpg")

# Add a title
st.title("Form for the Users")

# Write a text with streamlit
st.write("It's streamlit")

# display an image with streamlit
#st.image("photo.jpg", width=None)

# Textbox for the model => When we want to add a description sentence =>if we already have 10 sentences => the code code will erase 
# the last sentence and add the new one
# You just have to do new_sentence = the sentence you want to add and it will work

# At the begining of the program we need to intialize test, 
# you just have to replace text = 'Ligne1\nLigne2\nLigne3\nLigne4\nLigne5\nLigne6\nLigne7\nLigne8\nLigne9\nLigne10' by test = ""
# test = ""
text = 'Ligne1\nLigne2\nLigne3\nLigne4\nLigne5\nLigne6\nLigne7\nLigne8\nLigne9\nLigne10'


new_sentence ='new description by the model\n'
text = new_sentence + text 
if text.count('\n') >= 10:
     find_last_sentence = "\n"
     number = text.rfind(find_last_sentence)
     number = len(text) - number
     text = text[:-number]

st.text_area("Description model", value=text, height=275, max_chars=0, key=10)

from pickle import load
from numpy import argmax
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import numpy as np
gpus = tensorflow.config.experimental.list_physical_devices('GPU') 
tensorflow.config.experimental.set_memory_growth(gpus[0], True)
class ImageCaptioning:
    def __init__(self,tokenizer, model):
        self.tokenizer = load(open(tokenizer, 'rb'))
        self.model = load_model(model)
        self.vgg16 = VGG16()
        self.vgg16 = Model(inputs=self.vgg16.inputs, outputs=self.vgg16.layers[-2].output)

    # extract features from each photo in the directory
    def extract_features(self,frame):
        # load the model
        #model = VGG16()
        # re-structure the model
        #model = Model(inputs=self.vgg16.inputs, outputs=self.vgg16.layers[-2].output)
        # load the photo
        #image = load_img(filename, target_size=(224, 224))
        image = Image.fromarray(frame)
        image = image.convert("RGB")
        image = image.resize((224,224),Image.NEAREST)

        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = self.vgg16.predict(image, verbose=0)
        return feature

    # map an integer to a word
    def word_for_id(self,integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    # generate a description for an image
    def generate_desc(self, photo, max_length):
        # seed the generation process
        in_text = 'startseq'
        # iterate over the whole length of the sequence
        for i in range(max_length):
            # integer encode input sequence
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            # pad input
            sequence = pad_sequences([sequence], maxlen=max_length)
            # predict next word
            yhat = self.model.predict([photo,sequence], verbose=0)
            # convert probability to integer
            yhat = argmax(yhat)
            # map integer to word
            word = self.word_for_id(yhat, self.tokenizer)
            # stop if we cannot map the word
            if word is None:
                break
            # append as input for generating the next word
            in_text += ' ' + word
            # stop if we predict the end of the sequence
            if word == 'endseq':
                break
        return in_text
    def run(self,frame):
        with tensorflow.device('device:cpu:0'):
            # pre-define the max sequence length (from training)
            max_length = 42
            # load the model
            #model = load_model('model-ep005-loss3.391-val_loss3.705.h5')
            # load and prepare the photograph
            photo = self.extract_features(frame)
            # generate description
            description = self.generate_desc(photo, max_length)
            return description

if __name__ == "__main__":
    photo = cv2.imread("photo.jpg")
    tokenizer = load(open('New_Tok.pkl', 'rb'))
    model = load_model('New_Model.h5')
    cap = ImageCaptioning(model=model,tokenizer=tokenizer)
    print(cap.run(photo))
    




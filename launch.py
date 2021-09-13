import gradio as gr
import numpy as np
import cv2
from tensorflow import keras

model = keras.models.load_model('CounterModel.h5')

def Counter(img):
  IMG_WIDTH = 224*3
  IMG_HEIGHT = 224*3

  img_scaled = img.astype(np.float32)/255.
  img_resized =  cv2.resize(img_scaled,(IMG_WIDTH,IMG_HEIGHT))
  prdct = np.round(model.predict(img_resized.reshape(1,IMG_WIDTH,IMG_HEIGHT,3)))
  return '{} people, {} cat/s, {} dog/s'.format(int(prdct[0][0]),int(prdct[0][1]),int(prdct[0][2]))
    
inputs = gr.inputs.Image()
outputs = gr.outputs.Textbox()
sample_images = [['examples/pic6_1p1c0d.jpeg'],['examples/pic9_1p1c0d.jpeg'],['examples/pic63_2p0c1d.jpeg']
                 ,['examples/pic394_0p3c2d.jpeg'],['examples/pic390_0p1c1d.jpeg']]
title = 'How many cats / dogs / people?'
description = 'Upload a picture or use one of the examples below. The model will count how many cats, dogs and people are in the picture'

gr.Interface(fn=Counter, inputs=inputs, outputs=outputs,examples = sample_images,title=title,description=description).launch()

import gradio as gr
from fastai.vision.all import *
import torch

learn = load_learner('model.pkl')
categories = learn.dls.vocab

def greet(name):
    return "Hello " + name + "!!"

def bear_classifier(image):
    pred, idx, probs = learn.predict(image)
    return dict(zip(categories, map(float, probs)))

image = gr.Image(shape=(224, 224))
label = gr.outputs.Label()
examples = ['panda.jpeg', 'poler.jpeg']

iface = gr.Interface(bear_classifier, inputs=image, outputs=label, examples=examples)
iface.launch()

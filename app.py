from flask import Flask, render_template, request, url_for, redirect,session
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import json

app = Flask(__name__)
app = Flask(__name__,template_folder='template')

model = tf.keras.models.load_model("model/model hsn.h5")

@app.route('/')
def index():
  return render_template("index.html")

@app.route('/fer_smile',methods=['POST'])
def fer_smile():
  base64_image_string = request.form['image'].split(',')[1]
  image = Image.open( BytesIO( base64.b64decode( base64_image_string ))).resize((48, 48))
  img = np.array(image)

  gray = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype('float32')
  grayre = np.reshape(gray,(48, 48, 1))

  print(grayre)
  pre = model.predict(np.array([grayre]))
  print(pre)
  
  hasil = {
    'happy':pre[0][0]*100,
    'sad':pre[0][1]*100,
    'neutral':pre[0][2]*100
  }
  return str(json.dumps(hasil))

if __name__=='__main__':
    app.run()
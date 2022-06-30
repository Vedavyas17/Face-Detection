"""Utilities
"""

# Imports
import re
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from io import StringIO
import cv2
from flask import jsonify

def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")


def readb64(base64_string):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    print(type(encoded_data), encoded_data[0:60])
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def predFacesUsingCV2(recImg):
    # Convert Image to CV@ format
    img = data_uri_to_cv2_img(recImg)

    # Convert to grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Init Classifier
    face_cascade = cv2.CascadeClassifier("./classifier/haarcascade_frontalface_default.xml")

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Drawing reactangle for the detected faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    # Encode as base64
    processed_string = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
    
    # Prepare img src
    img_src = "data:image/jpg;base64," + processed_string

    # Prepare output to return
    out_json = jsonify(result=img_src, probability="")

    # Return
    return out_json
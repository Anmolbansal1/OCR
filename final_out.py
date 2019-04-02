#!/usr/bin/env python3
# -*- coding: cp1252 -*-

import cv2
import io
from PIL import Image
import numpy as np
import base64
from base64 import decodestring

from flask import Flask, request, Response
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)

@app.route('/', methods=['POST'])
@cross_origin()
def index():
    print(request)
    # data = request.data
    content = request.get_json()

    print(content['text'])
    image = base64.b64decode(content['img'])
    # image = Image.frombytes('RGB',(350,350),decodestring(data))
    # image.save("foo.png")
    with open('foo.png', 'wb') as f_output:
        f_output.write(image)
    print('done')
    final_out(content)
    # data = np.fromstring(data, np.uint8)
    # print(data)
    # print(img)
    # print(type(img))
    # cv2.imshow("s",img)
    # cv2.waitKey(0)
    # img = base64.b64decode(data)
    # print(type(img))
    # image = io.BytesIO(img)
    # image = Image.open(image)
    # print(type(image))
    # ph = open("img.png", "wb")
    # fh.write(data.decode('base64'))
    # fh.close()
    # cv2.imshow("a", img)
    # cv2.waitKey(0)
    # print
    # print(data)
    # final_out(data)
    return Response({'message': 'done!!'})

def final_out(data):

    image = cv2.imread('foo.png')
    image1 = image.copy()
    num_boxes = len(data["cordinates"])
    
    text = data["text"]
   
    font= cv2.FONT_HERSHEY_SIMPLEX
    
    fontScale = 1
    fontColor = (255,255,255)
    lineType = 2
    
    for x in range(num_boxes):
        pt1 = np.zeros(shape=(4,1), dtype=int)
        pt1[0] = data["cordinates"][x]["startX"]
        pt1[1] = data["cordinates"][x]["startY"]
        pt1[2] = data["cordinates"][x]["endX"]
        pt1[3] = data["cordinates"][x]["endY"]
        print('Points are - ', pt1)

        bottomLeftCornerOfText = (pt1[0],pt1[3])
        
        pnt1 = [pt1[0], pt1[1]]
        pnt2 = [pt1[2], pt1[3]]

        cv2.rectangle(image, (pnt1[0], pnt1[1]), (pnt2[0],pnt2[1]),0, thickness=cv2.FILLED)
        cv2.rectangle(image1, (pnt1[0], pnt1[1]), (pnt2[0],pnt2[1]),0, thickness=1)
        cv2.putText(image,str(text[x][0]), 
	      bottomLeftCornerOfText, 
	      font, 
	      fontScale,
	      fontColor,
	      lineType)
    print('Now see the magic')
    image_final = np.concatenate((image,image1) , axis=1)
    cv2.imshow("ocr" , image_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




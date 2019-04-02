import text_detection

import numpy as np


def ocr(image):
        retVals = []

        test_image = image


        ret_vals = predict_interface.pred_from_img(test_image)

        print(" ")
        print("----------")

        print("A modified image with the predictions: pro-img/IMAGE_NAME_digitized_image.png")



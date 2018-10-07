import hashlib
import os

import cv2

from PIL import Image
import numpy as np
from flask import request
from skimage.transform import resize
from keras import backend as kb

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


# Local Image/Prediction Commands

def local_load_image(image_file):
    image = Image.open(image_file)
    image_array = np.asarray(image.convert('RGB'))
    image_array = image_array / 255
    image_array = resize(image_array, (224, 224))
    return image_array


def local_predict_overlay(image_path, model):
    img_ori = cv2.imread(filename=image_path)
    img_array = local_load_image(image_path)

    prediction = model.predict(np.expand_dims(img_array, axis=0))
    index = np.argmax(prediction[0])

    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = get_output_layer(model, "bn")
    get_output = kb.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([np.array([img_array])])
    conv_outputs = conv_outputs[0, :, :, :]

    cam = np.zeros(dtype=np.float32, shape=(conv_outputs.shape[:2]))
    for i, w in enumerate(class_weights[index]):
        cam += w * conv_outputs[:, :, i]
    cam /= np.max(cam)
    cam = cv2.resize(cam, img_ori.shape[:2])
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap * 0.5 + img_ori
    cv2.putText(img, text=CLASS_NAMES[index], org=(5, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8, color=(0, 0, 255), thickness=1)
    cv2.imwrite('prediction/predict-' + image_path.split('/')[-1], img)
    return index


# Here lies POST specific operations - generally mirrors local commands

def post_image_prepare(image):
    image_array = np.asarray(image.convert('RGB'))
    image_array = image_array / 255
    image_array = resize(image_array, (224, 224))
    return image_array


def decode_prediction(prediction):
    output_dict = {}
    for i in range(len(prediction[0])):
        output_dict[CLASS_NAMES[i]] = prediction[0][i]
    return output_dict


def post_image_overlay(img_in, prediction_array, model, upload_folder):
    img_ori = np.asarray(img_in.convert('RGB'))
    img_array = post_image_prepare(img_in)
    out_image_name = 'predict-' + hashlib.sha224(
        (str(request.remote_addr) + str(np.random.random_sample())).encode('utf-8')).hexdigest() + '.png'

    index = np.argmax(prediction_array[0])

    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = get_output_layer(model, "bn")
    get_output = kb.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([np.array([img_array])])
    conv_outputs = conv_outputs[0, :, :, :]

    cam = np.zeros(dtype=np.float32, shape=(conv_outputs.shape[:2]))
    for i, w in enumerate(class_weights[index]):
        cam += w * conv_outputs[:, :, i]
    cam /= np.max(cam)
    cam = cv2.resize(cam, img_ori.shape[:2])
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap * 0.5 + img_ori
    # cv2.putText(img, text=CLASS_NAMES[index], org=(5, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #             fontScale=0.8, color=(0, 0, 255), thickness=1)
    cv2.imwrite(os.path.join(upload_folder, out_image_name), img)
    return out_image_name


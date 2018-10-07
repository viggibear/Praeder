import io
import json
import os

import flask
from PIL import Image
from flask import send_file, render_template, Flask
from keras import Input, Model
from keras.applications import DenseNet121
from keras.layers import Dense
import image_handler
import numpy as np

app: Flask = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'predictions')

input_shape = (224, 224, 3)
model = None
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def load_model():
    global model
    img_input = Input(shape=input_shape)
    base_model = DenseNet121(include_top=False, input_tensor=img_input, input_shape=input_shape, pooling='avg')
    prediction_layer = Dense(len(CLASS_NAMES), activation='sigmoid', name='predictions')(base_model.output)
    model = Model(inputs=img_input, outputs=prediction_layer)
    model.load_weights('brucechou1983_CheXNet_Keras_0.3.0_weights.h5')


@app.route("/predict_chest", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image_ori = flask.request.files["image"].read()
            try:
                image_ori = Image.open(io.BytesIO(image_ori))
            except OSError:
                data["error"] = "Please check you have uploaded an Image File (.jpg, .tiff, .bmp, .png)"
                return render_template("results_page_chest.html", data=data)

            image = image_handler.post_image_prepare(image_ori)

            model._make_predict_function()  # have to initialize before threading
            prediction_ndarray = model.predict(np.expand_dims(image, axis=0))
            preds = image_handler.decode_prediction(prediction_ndarray)
            print(preds)
            diagnosis_list = []
            for diagnosis, prob in preds.items():
                r = {"label": diagnosis, "probability": float(prob)}
                diagnosis_list.append(r)
            data["predictions"] = sorted(diagnosis_list, key=lambda k: k['probability'], reverse=True)

            data["image_dir"] = image_handler.post_image_overlay(image_ori, prediction_ndarray, model,
                                                                 app.config['UPLOAD_FOLDER'])

            data["success"] = True

        return render_template("results_page_chest.html", data=data)


@app.route('/images/<string:pid>')
def get_image(pid):
    return send_file(
        os.path.join(app.config['UPLOAD_FOLDER'], pid),
        mimetype='image/png',
        attachment_filename='%s' % pid)


@app.route('/')
def hello_world():
    return render_template('landing_page.html')


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(threaded=False, debug=False)

# image_handler.local_predict_overlay('images/00009242_003.png', model)

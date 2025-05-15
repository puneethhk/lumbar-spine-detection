import flask , request, render_template
import tensorflow as tf
import numpy as np
import cv2
import pydicom
import os
app = Flask(__name__)
model = tf.keras.models.load_model('model/lumbar_spine_model.h5')
def preprocess_dicom_image(dicom_path):
    dicom = pydicom.dcmread(dicom_path)
    image = dicom.pixel_array
    image = cv2.resize(image, (224,224))
    image = image / np.max(image)
    return np.expand_dims(image,axis=[0,-1])
    @app.route('/',methods=['GET','POST'])
    def index():
        if request.method == 'POST':
            file = request.files['file']
            file_path = os.path.join('uploads',file.filename)
            file.save(file_path)
            image = preprocess_dicom_image(file_path)
            prediction = model.predict(image)
            return render_template('result.html',prediction=prediction)
            return render_template('index.html')
            if_name_=='_main_'
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
                app.run(debug=True)


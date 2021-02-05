# import the necessary packages
from imutils import face_utils
import numpy as np
import os
import imutils
import dlib
import cv2
import glob
from pathlib import Path
from PIL import Image, ImageOps
import streamlit as st
import boto3
import uuid
from secretsss import access_key, secret_access_key

client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_access_key)
upload_bucket = 'test-ml-facedistance'

uniqueId = uuid.uuid1()

# st.set_option('depracation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def load_model():
    model = "protos/shape_predictor_68_face_landmarks.dat"
    return model


def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth


def import_and_predict(image_data):
    distances = []
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model)

    image = cv2.imread(image_data)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        focalLength = 472  # my camera Focal length
        avg_width = 6.8
        color = None
        color0 = (255, 0, 0)
        color1 = (0, 50, 255)

        dist = distance_to_camera(avg_width, focalLength, int(int(w + x) - int(x)))
        distance = str(round(dist, 2))
        distances.append(distance)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(image, "distance:" + distance + ' inches', (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color0, 2)

        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (255, 255, 0), -1)

    return image, distances


model = load_model()

st.title("Distance Calculation :camera_with_flash::wink:")

file = st.file_uploader("Upload here", type=["jpg", "png", "jpeg"])

if file is None:
    st.write("## Please upload your selfie w/o zoom")
else:
    image = Image.open(file)
    st.image(image, caption='Selfie', use_column_width=True)
    img_array = np.array(image)
    cv2.imwrite('input.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2RGBA))

    if st.button("Process"):
        jpgCounter = len(glob.glob1('./runs', "*.jpg"))
        path = os.getcwd() + '/runs'
        result_img, distances = import_and_predict('input.jpg')
        cv2.imwrite(os.path.join(path, str(uniqueId) + '.jpg'), cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        st.image(result_img, use_column_width=True)

        # S3 config
        upload_key = str(uniqueId) + '.jpg'
        img = Image.fromarray(result_img, 'RGB')
        client.upload_file(os.getcwd() + '\\runs\\' + str(uniqueId) + '.jpg', upload_bucket, upload_key)

        st.success("The distance is: " + " ".join(str(x) for x in distances) + " inches")

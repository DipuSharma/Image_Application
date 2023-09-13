import os
import base64
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
from modal import UploadImage
from db.db_config import get_db
from sqlalchemy.orm import Session
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, File, UploadFile, Depends

app = FastAPI(docs_url="/dts")

origins = ['0.0.0.0']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/upload-original-image")
async def upload_original_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File not found")
    File_DIR = './static/upload/'
    if not os.path.exists(File_DIR):
        os.makedirs(File_DIR)
    file_path = File_DIR + file.filename

    existing_image = db.query(UploadImage).filter(
        UploadImage.image_path == file_path).first()
    if existing_image:
        return {"status": "failed", "message": 'this image already existed'}

    try:
        contents = file.file.read()
        with open(file_path, 'wb+') as f:
            f.write(contents)
            f.close()

        image = UploadImage(image_path=file_path)
        db.add(image)
        db.commit()

        data = {"status": "success", "image": file_path}
        return FileResponse(f'{file_path}', status_code=200)
    except:
        return False


@app.post('/check_similarity')
def find_similar_images(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # features, database_features,

    if not file.filename:
        raise HTTPException(status_code=400, detail="File not found")
    File_DIR = './static/verified/'
    if not os.path.exists(File_DIR):
        os.makedirs(File_DIR)
    file_path2 = File_DIR + file.filename

    db_file_path = db.query(UploadImage).first()
    contents = file.file.read()
    with open(file_path2, 'wb+') as f:
        f.write(contents)
        f.close()

    # Extracting code
    model = tf.keras.applications.VGG16(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # Preprocess the image and extract features
    img1 = tf.keras.preprocessing.image.load_img(
        file_path2, target_size=(224, 224))
    x1 = tf.keras.preprocessing.image.img_to_array(img1)
    x1 = np.expand_dims(x1, axis=0)
    x1 = tf.keras.applications.vgg16.preprocess_input(x1)

    features1 = model.predict(x1)
    # End Extracting code

    # Preprocess the image and extract features
    img2 = tf.keras.preprocessing.image.load_img(
        db_file_path.image_path, target_size=(224, 224))
    x2 = tf.keras.preprocessing.image.img_to_array(img2)
    x2 = np.expand_dims(x2, axis=0)
    x2 = tf.keras.applications.vgg16.preprocess_input(x2)

    features2 = model.predict(x2)
    print("feature first_______________",
          features1[0], "feature second_______________", features2[0])

    # # Calculate similarity scores
    similarity_scores = cosine_similarity(features1[0][0], features2[0][0])

    # # Find the most similar image
    most_similar_image_idx = np.argmax(similarity_scores)
    print("Similarity____________", most_similar_image_idx)
    return True

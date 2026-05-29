from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

app   = FastAPI()
model = YOLO("best.pt")  # ton modèle entraîné

@app.post("/compter-photo")
async def compter_photo(fichier: UploadFile = File(...)):
    # Lire l'image
    contenu = await fichier.read()
    img_array = np.frombuffer(contenu, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Prédiction
    resultats = model.predict(img, conf=0.5)
    nb_sujets = len(resultats[0].boxes)
    confiance = float(resultats[0].boxes.conf.mean()) if nb_sujets > 0 else 0
    
    return {
        "nb_sujets": nb_sujets,
        "confiance": round(confiance, 2)
    }

@app.post("/compter-video")
async def compter_video(fichier: UploadFile = File(...)):
    # Sauvegarder la vidéo temporairement
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await fichier.read())
        tmp_path = tmp.name
    
    # Extraire les frames et compter
    cap = cv2.VideoCapture(tmp_path)
    max_sujets = 0
    confiance_total = 0
    nb_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Analyser 1 frame sur 5 pour aller plus vite
        if nb_frames % 5 == 0:
            resultats = model.predict(frame, conf=0.5, verbose=False)
            nb = len(resultats[0].boxes)
            if nb > max_sujets:
                max_sujets = nb
                if nb > 0:
                    confiance_total = float(resultats[0].boxes.conf.mean())
        nb_frames += 1
    
    cap.release()
    os.unlink(tmp_path)
    
    return {
        "nb_sujets": max_sujets,
        "confiance": round(confiance_total, 2)
    }
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

app   = FastAPI()
model = YOLO("best.pt")

@app.post("/compter-photo")
async def compter_photo(fichier: UploadFile = File(...)):
    contenu = await fichier.read()
    img_array = np.frombuffer(contenu, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"erreur": "Image invalide", "nb_sujets": 0, "confiance": 0}
    
    print(f"Image reçue : {img.shape}")
    
    # Seuil plus bas pour détecter plus
    resultats = model.predict(img, conf=0.25, imgsz = 1280, verbose=True)
    nb_sujets = len(resultats[0].boxes)
    confiance = float(resultats[0].boxes.conf.mean()) if nb_sujets > 0 else 0
    
    print(f"Résultat : {nb_sujets} sujets, confiance {confiance}")
    
    return {
        "nb_sujets": nb_sujets,
        "confiance": round(confiance, 2)
    }

@app.post("/compter-video")
async def compter_video(fichier: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await fichier.read())
        tmp_path = tmp.name
    
    print(f"Vidéo reçue : {tmp_path}")
    
    cap = cv2.VideoCapture(tmp_path)
    max_sujets = 0
    confiance_total = 0
    nb_frames = 0
    resultats_par_frame = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if nb_frames % 5 == 0:
            resultats = model.predict(frame, conf=0.25, imgsz=1280, verbose=False)
            nb = len(resultats[0].boxes)
            resultats_par_frame.append(nb)
            if nb > max_sujets:
                max_sujets = nb
                if nb > 0:
                    confiance_total = float(resultats[0].boxes.conf.mean())
        nb_frames += 1
    
    cap.release()
    os.unlink(tmp_path)
    
    print(f"Frames analysées : {nb_frames}, max détecté : {max_sujets}")
    print(f"Résultats par frame : {resultats_par_frame}")
    
    return {
        "nb_sujets": max_sujets,
        "confiance": round(confiance_total, 2)
    }
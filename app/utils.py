import cv2
import json
import numpy as np
from deepface import DeepFace

# Haar Cascade para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detectar_rostro(frame_bgr):
    """
    Detecta el primer rostro en un frame y devuelve el recorte (BGR).
    Si no encuentra rostro, devuelve None.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) > 0:
        x, y, w, h = faces[0]
        rostro = frame_bgr[y:y+h, x:x+w]
        return rostro
    return None

def obtener_embedding(imagen_bgr, modelo="Facenet"):
    imagen_rgb = imagen_bgr[:, :, ::-1]
    embedding_vec = DeepFace.represent(img_path=imagen_rgb, model_name=modelo, enforce_detection=False)
    if isinstance(embedding_vec, list) and len(embedding_vec) > 0 and "embedding" in embedding_vec[0]:
        vec = np.array(embedding_vec[0]["embedding"], dtype=np.float32)
    elif isinstance(embedding_vec, dict) and "embedding" in embedding_vec:
        vec = np.array(embedding_vec["embedding"], dtype=np.float32)
    else:
        vec = np.array(embedding_vec, dtype=np.float32)
    return vec

def embedding_a_json(vec: np.ndarray):
    return json.dumps(vec.tolist())

def distancia_coseno(vec1: np.ndarray, vec2: np.ndarray):
    num = np.dot(vec1, vec2)
    den = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-8
    return 1.0 - (num / den)

def es_duplicado(embedding_nuevo, embeddings_existentes, umbral=0.35):
    for emb_json in embeddings_existentes:
        vec = np.array(json.loads(emb_json), dtype=np.float32)
        if distancia_coseno(embedding_nuevo, vec) < umbral:
            return True
    return False
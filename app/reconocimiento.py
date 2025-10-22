import cv2
import json
import numpy as np
from deepface import DeepFace
from db import init_db, get_all_personas, insert_deteccion
from utils import detectar_rostro, obtener_embedding, distancia_coseno

def reconocer_en_tiempo_real(umbral=0.35):
    init_db()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se puede acceder a la cámara")

    print("Presiona 'q' para salir.")

    # Cargar embeddings de la DB
    personas = get_all_personas()
    ids = [p[0] for p in personas]
    nombres = [f"{p[1]} {p[2]}" for p in personas]
    embeddings = [np.array(json.loads(p[4]), dtype=np.float32) for p in personas]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar rostro
        rostro = detectar_rostro(frame)
        emb_vec = None
        if rostro is not None:
            try:
                emb_vec = obtener_embedding(rostro)
                # Dibujar rectángulo en el rostro detectado
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            except Exception as e:
                print("Error generando embedding:", e)

        nombre = "No registrado"
        persona_id = None

        # Comparar con embeddings existentes
        if emb_vec is not None and len(embeddings) > 0:
            mejor_dist = 1.0
            mejor_idx = -1
            for i, emb in enumerate(embeddings):
                dist = distancia_coseno(emb_vec, emb)
                if dist < mejor_dist:
                    mejor_dist = dist
                    mejor_idx = i
            if mejor_idx != -1 and mejor_dist < umbral:
                persona_id = ids[mejor_idx]
                nombre = nombres[mejor_idx]

        # Analizar emoción
        try:
            analisis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emocion = analisis[0]['dominant_emotion']
            confianza = analisis[0]['emotion'][emocion]
        except Exception:
            emocion = "Desconocida"
            confianza = 0.0

        # Guardar detección si es persona conocida
        if persona_id:
            insert_deteccion(persona_id, emocion, float(confianza))

        # Mostrar overlay en pantalla
        texto = f"{nombre} | {emocion} ({confianza:.1f}%)"
        cv2.putText(frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Reconocimiento", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    reconocer_en_tiempo_real()
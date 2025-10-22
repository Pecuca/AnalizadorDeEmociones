import cv2
import sqlite3
import st
from db import   init_db, insert_persona, get_all_personas
from utils import detectar_rostro, obtener_embedding, embedding_a_json, es_duplicado

try:
    persona_id = insert_persona(nombre, apellido, email, emb_json)
    st.success(f"🎉 Persona registrada con ID {persona_id}")
except sqlite3.IntegrityError:
    st.error("⚠️ Ya existe una persona registrada con ese email.")

def capturar_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se puede acceder a la cámara")

    print("Presiona 'c' para capturar, 'q' para salir.")
    frame_capturado = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturando frame")
            break

        # Dibujar rectángulo si se detecta rostro
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Registro - Vista previa", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            frame_capturado = frame.copy()
            print("Imagen capturada.")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame_capturado

def registrar_persona(nombre, apellido, email):
    init_db()
    frame = capturar_frame()
    if frame is None:
        print("No se capturó imagen. Registro cancelado.")
        return

    rostro = detectar_rostro(frame)
    if rostro is None:
        print("⚠️ No se detectó rostro en la imagen capturada.")
        return
    else:
        print("✅ Rostro detectado, generando embedding...")

    embedding_vec = obtener_embedding(rostro)

    personas = get_all_personas()
    embeddings_existentes = [p[4] for p in personas]
    if es_duplicado(embedding_vec, embeddings_existentes):
        print("⚠️ Registro rechazado: el rostro ya existe en la base de datos.")
        return

    emb_json = embedding_a_json(embedding_vec)
    persona_id = insert_persona(nombre, apellido, email, emb_json)
    print(f"🎉 Registro exitoso. ID persona: {persona_id}")
    print(f"Guardado: {nombre} {apellido} ({email})")

if __name__ == "__main__":
    nombre = input("Nombre: ").strip()
    apellido = input("Apellido: ").strip()
    email = input("Email: ").strip()
    registrar_persona(nombre, apellido, email)
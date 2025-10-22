import cv2
import streamlit as st
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from deepface import DeepFace

# Importar utilidades y DB
from app.db import init_db, insert_persona, get_all_personas, insert_deteccion
from app.utils import detectar_rostro, obtener_embedding, embedding_a_json, distancia_coseno

st.set_page_config(page_title="Sistema de Reconocimiento Facial", layout="wide")
st.title("🧠 Sistema de Reconocimiento Facial con Emociones")

# Inicializar DB
init_db()

# Sidebar de navegación
opcion = st.sidebar.radio("Menú", ["Registro", "Reconocimiento", "Reportes"])

# ---------------------------
# 1. REGISTRO
# ---------------------------
if opcion == "Registro":
    st.header("📌 Registro de Persona")

    nombre = st.text_input("Nombre")
    apellido = st.text_input("Apellido")
    email = st.text_input("Email")

    # Captura con cámara integrada de Streamlit
    img_file = st.camera_input("📸 Toma tu foto para el registro")

    if img_file is not None and st.button("✅ Confirmar Registro", key="confirmar_registro"):
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # Detectar rostro y dibujar rectángulo
        rostro = detectar_rostro(frame)
        if rostro is None:
            st.error("⚠️ No se detectó rostro en la imagen capturada.")
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            st.image(frame, channels="BGR", caption="Foto con detección de rostro")

            try:
                emb_vec = obtener_embedding(rostro)
                emb_json = embedding_a_json(emb_vec)
                persona_id = insert_persona(nombre, apellido, email, emb_json)
                st.success(f"🎉 Persona registrada con ID {persona_id}")
            except sqlite3.IntegrityError:
                st.error("⚠️ Ya existe una persona registrada con ese email.")

# ---------------------------
# 2. RECONOCIMIENTO (streaming simulado)
# ---------------------------
elif opcion == "Reconocimiento":
    st.header("🎥 Reconocimiento en Tiempo Real")

    frame_placeholder = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("No se pudo acceder a la cámara.")
    else:
        personas = get_all_personas()
        ids = [p[0] for p in personas]
        nombres = [f"{p[1]} {p[2]}" for p in personas]
        embeddings = [np.array(json.loads(p[4]), dtype=np.float32) for p in personas]

        # Loop de streaming simulado
        for _ in range(200):  # muestra ~20 segundos (200 * 0.1s)
            ret, frame = cap.read()
            if not ret:
                break

            rostro = detectar_rostro(frame)
            emb_vec = None
            if rostro is not None:
                try:
                    emb_vec = obtener_embedding(rostro)
                    # Dibujar rectángulo
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                except Exception as e:
                    st.warning(f"Error embedding: {e}")

            nombre = "No registrado"
            persona_id = None

            if emb_vec is not None and len(embeddings) > 0:
                mejor_dist = 1.0
                mejor_idx = -1
                for i, emb in enumerate(embeddings):
                    dist = distancia_coseno(emb_vec, emb)
                    if dist < mejor_dist:
                        mejor_dist = dist
                        mejor_idx = i
                if mejor_idx != -1 and mejor_dist < 0.35:
                    persona_id = ids[mejor_idx]
                    nombre = nombres[mejor_idx]

            try:
                analisis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emocion = analisis[0]['dominant_emotion']
                confianza = analisis[0]['emotion'][emocion]
            except Exception:
                emocion = "Desconocida"
                confianza = 0.0

            if persona_id:
                insert_deteccion(persona_id, emocion, float(confianza))

            texto = f"{nombre} | {emocion} ({confianza:.1f}%)"
            cv2.putText(frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)

            frame_placeholder.image(frame, channels="BGR")
            time.sleep(0.1)

        cap.release()

# ---------------------------
# 3. REPORTES
# ---------------------------
elif opcion == "Reportes":
    st.header("📊 Reportes de Detecciones")

    conn = sqlite3.connect("app_data.db", check_same_thread=False)
    df_personas = pd.read_sql_query("SELECT * FROM personas;", conn)
    df_det = pd.read_sql_query("SELECT * FROM detecciones;", conn)

    if df_personas.empty:
        st.warning("No hay personas registradas.")
    else:
        persona_sel = st.selectbox(
            "Selecciona persona:",
            df_personas["nombre"] + " " + df_personas["apellido"]
        )
        persona_id = df_personas.loc[
            (df_personas["nombre"] + " " + df_personas["apellido"]) == persona_sel,
            "id"
        ].values[0]

        df_p = df_det[df_det["persona_id"] == persona_id]

        if df_p.empty:
            st.info("No hay detecciones para esta persona.")
        else:
            st.subheader("📋 Tabla de detecciones")
            st.dataframe(df_p)

            st.subheader("📈 Distribución de emociones")
            fig, ax = plt.subplots()
            df_p["emocion"].value_counts().plot(kind="bar", ax=ax, color="skyblue")
            ax.set_ylabel("Cantidad")
            ax.set_xlabel("Emoción")
            st.pyplot(fig)

    conn.close()
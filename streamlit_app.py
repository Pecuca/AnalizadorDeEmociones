import streamlit as st
import cv2
import numpy as np
import json
import time
from pathlib import Path
from app.db import init_db, insert_persona, get_all_personas, insert_deteccion
from app.utils import obtener_embedding, embedding_a_json, es_duplicado, distancia_coseno
from deepface import DeepFace

st.set_page_config(page_title="Reconocimiento facial y emociones", layout="wide")
init_db()

def abrir_camara():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("No se puede acceder a la cámara")
        return None
    return cap

def frame_a_stimage(frame):
    # BGR -> RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# --------------------- Pantalla Registro --------------------- #
def pantalla_registro():
    st.header("Registro de personas")
    with st.form("form_registro"):
        nombre = st.text_input("Nombre")
        apellido = st.text_input("Apellido")
        email = st.text_input("Email")
        capturar = st.form_submit_button("Capturar y registrar")

    vista = st.empty()
    if capturar:
        cap = abrir_camara()
        if cap is None:
            return
        st.info("Mostrando cámara. Presiona el botón 'Capturar' debajo del video.")
        boton_captura = st.button("Capturar imagen")
        frame_guardado = None

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error en la cámara.")
                break
            vista.image(frame_a_stimage(frame), channels="RGB", use_column_width=True)
            if boton_captura:
                frame_guardado = frame.copy()
                break
            # salir si el usuario cambia de página
            if st.session_state.get("salir_registro", False):
                break

        cap.release()

        if frame_guardado is not None:
            st.success("Imagen capturada. Generando embedding...")
            try:
                emb_vec = obtener_embedding(frame_guardado)
            except Exception as e:
                st.error(f"No se pudo generar el embedding: {e}")
                return

            personas = get_all_personas()
            embeddings_existentes = [p[4] for p in personas]
            if es_duplicado(emb_vec, embeddings_existentes):
                st.warning("Registro rechazado: rostro ya existe en la base de datos.")
                return

            persona_id = insert_persona(nombre, apellido, email, embedding_a_json(emb_vec))
            st.success(f"Registro exitoso. ID persona: {persona_id}")
            st.image(frame_a_stimage(frame_guardado), caption="Rostro registrado", channels="RGB")

# --------------------- Pantalla Detección --------------------- #
def pantalla_deteccion():
    st.header("Detección en tiempo real")
    umbral = st.slider("Umbral de coincidencia (menor es más estricto)", 0.20, 0.60, 0.35, 0.01)
    iniciar = st.button("Iniciar detección")
    detener = st.button("Detener")

    personas = get_all_personas()
    ids = [p[0] for p in personas]
    nombres = [f"{p[1]} {p[2]}" for p in personas]
    embeddings = [np.array(json.loads(p[4]), dtype=np.float32) for p in personas]

    vista = st.empty()
    info = st.empty()

    if iniciar:
        cap = abrir_camara()
        if cap is None:
            return

        while True:
            if detener:
                break
            ret, frame = cap.read()
            if not ret:
                st.error("Error en la cámara.")
                break

            # Emoción
            try:
                analisis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emocion = analisis[0]['dominant_emotion']
                confianza_emocion = float(analisis[0]['emotion'][emocion])
            except Exception:
                emocion = "Desconocida"
                confianza_emocion = 0.0

            # Identidad
            try:
                emb_vec = obtener_embedding(frame)
            except Exception:
                emb_vec = None

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
                if mejor_idx != -1 and mejor_dist < umbral:
                    persona_id = ids[mejor_idx]
                    nombre = nombres[mejor_idx]

            # Overlay texto
            texto = f"{nombre} | {emocion} ({confianza_emocion:.1f}%)"
            cv2.putText(frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            vista.image(frame_a_stimage(frame), channels="RGB", use_column_width=True)
            info.markdown(f"**Nombre:** {nombre}  |  **Emoción:** {emocion}  |  **Confianza:** {confianza_emocion:.1f}%  |  **Hora:** {time.strftime('%H:%M:%S')}")

            # Guardar detección si es conocida
            if persona_id:
                insert_deteccion(persona_id, emocion, confianza_emocion)

        cap.release()
        st.info("Detección detenida.")

# --------------------- Pantalla Reportes --------------------- #
def pantalla_reportes():
    import pandas as pd
    import sqlite3
    import seaborn as sns
    import matplotlib.pyplot as plt

    st.header("Reportes y estadísticas")

    db_path = Path("app_data.db")
    if not db_path.exists():
        st.warning("No hay base de datos aún.")
        return

    conn = sqlite3.connect(db_path)
    df_personas = pd.read_sql_query("SELECT id, nombre, apellido, email FROM personas", conn)
    df_det = pd.read_sql_query("SELECT persona_id, emocion, confianza, timestamp FROM detecciones", conn)
    conn.close()

    if df_det.empty:
        st.info("Aún no hay detecciones registradas.")
        return

    # Selector de persona
    opciones = {f"{r['nombre']} {r['apellido']} ({r['email']})": r['id'] for _, r in df_personas.iterrows()}
    seleccion = st.selectbox("Persona", options=list(opciones.keys()))
    persona_id = opciones[seleccion]

    df_p = df_det[df_det['persona_id'] == persona_id]
    st.subheader("Historial de detecciones")
    st.dataframe(df_p.sort_values("timestamp", ascending=False), use_container_width=True)

    # Conteo por emoción
    st.subheader("Emociones por persona")
    conteo = df_p['emocion'].value_counts().rename_axis('emocion').reset_index(name='conteo')
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5,3))
        sns.barplot(data=conteo, x='emocion', y='conteo', ax=ax)
        ax.set_xlabel("Emoción")
        ax.set_ylabel("Conteo")
        ax.set_title("Conteo de emociones")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        prom = df_p.groupby('emocion')['confianza'].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(5,3))
        sns.barplot(data=prom, x='emocion', y='confianza', ax=ax2)
        ax2.set_xlabel("Emoción")
        ax2.set_ylabel("Confianza promedio")
        ax2.set_title("Confianza promedio por emoción")
        plt.tight_layout()
        st.pyplot(fig2)

    # Exportación
    st.subheader("Exportación")
    csv = df_p.to_csv(index=False).encode('utf-8')
    st.download_button("Exportar CSV del historial", csv, file_name="historial_detecciones.csv", mime="text/csv")

# --------------------- Navegación --------------------- #
def main():
    st.sidebar.title("Navegación")
    pagina = st.sidebar.selectbox("Ir a", ["Registro", "Detección", "Reportes"])

    if pagina == "Registro":
        pantalla_registro()
    elif pagina == "Detección":
        pantalla_deteccion()
    else:
        pantalla_reportes()

if __name__ == "__main__":
    main()
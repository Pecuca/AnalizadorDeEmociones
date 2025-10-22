import cv2
from db import init_db, insert_persona, get_all_personas
from utils import detectar_rostro, obtener_embedding, embedding_a_json, es_duplicado

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
        print("No se detectó rostro. Intenta de nuevo.")
        return

    print("Generando embedding facial...")
    embedding_vec = obtener_embedding(rostro)

    personas = get_all_personas()
    embeddings_existentes = [p[4] for p in personas]
    if es_duplicado(embedding_vec, embeddings_existentes):
        print("Registro rechazado: el rostro ya existe en la base de datos.")
        return

    emb_json = embedding_a_json(embedding_vec)
    persona_id = insert_persona(nombre, apellido, email, emb_json)
    print(f"Registro exitoso. ID persona: {persona_id}")

if __name__ == "__main__":
    nombre = input("Nombre: ").strip()
    apellido = input("Apellido: ").strip()
    email = input("Email: ").strip()
    registrar_persona(nombre, apellido, email)
## 📄 README.md

```markdown
# Proyecto 2 – Sistema de Reconocimiento Facial con Análisis de Emociones

Este proyecto implementa un sistema capaz de **registrar personas mediante cámara web**, **reconocerlas en tiempo real** y **analizar sus emociones** utilizando técnicas de visión por computadora y aprendizaje profundo.

---

## Funcionalidades

### 🔹 Módulo de Registro
- Captura de rostro mediante cámara web.
- Extracción de embeddings faciales con **DeepFace**.
- Almacenamiento en base de datos SQLite junto con nombre, apellido y email.
- Validación de duplicados para evitar registros repetidos.

### 🔹 Módulo de Reconocimiento
- Identificación en tiempo real de personas previamente registradas.
- Manejo de casos de “persona no registrada”.
- Detección de emociones básicas: **felicidad, tristeza, enojo, sorpresa, neutral, miedo, disgusto**.
- Visualización en pantalla de:
  - Nombre de la persona
  - Emoción detectada
  - Nivel de confianza
  - Hora de detección

### 🔹 Módulo de Reportes
- Historial de detecciones por persona.
- Gráficas de distribución de emociones y confianza promedio.
- Exportación de reportes a CSV.

---

## Tecnologías utilizadas
- **Python 3.13**
- [OpenCV](https://opencv.org/) – captura de video y procesamiento de imágenes
- [DeepFace](https://github.com/serengil/deepface) – embeddings faciales y análisis de emociones
- [SQLite](https://www.sqlite.org/) – base de datos ligera
- [Streamlit](https://streamlit.io/) – interfaz gráfica
- [Matplotlib / Seaborn](https://seaborn.pydata.org/) – visualización de datos

---

## Estructura del proyecto

```
app/
 ├── db.py              # Gestión de base de datos
 ├── utils.py           # Funciones auxiliares (detección, embeddings, etc.)
 ├── registro.py        # Registro de personas
 ├── reconocimiento.py  # Reconocimiento en tiempo real
streamlit_app.py        # Interfaz gráfica con Streamlit
requirements.txt        # Dependencias
```

---

## Instalación y ejecución

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/TU_USUARIO/DetectorDeBasura.git
   cd DetectorDeBasura
   ```

2. **Crear entorno virtual (opcional pero recomendado)**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # En Windows
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar la aplicación con Streamlit**
   ```bash
   python -m streamlit run streamlit_app.py
   ```

---

## Uso del sistema

- **Registro**: completa el formulario y captura el rostro → se guarda en la base de datos.
- **Detección**: inicia la cámara → muestra nombre, emoción y confianza en tiempo real.
- **Reportes**: selecciona una persona → verás estadísticas y podrás exportar el historial.

---

## Notas importantes
- Los embeddings se almacenan en formato JSON dentro de SQLite.
- El umbral de coincidencia puede ajustarse (por defecto 0.35).
- Para pruebas rápidas, se recomienda buena iluminación y cámara frontal.

---

## Autores
- **Alex Hernandez** – Universidad Rafael Urdaneta
- Proyecto académico para la materia de **Inteligencia Artificial**
```




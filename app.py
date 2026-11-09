
import streamlit as st
from streamlit_cropper import st_cropper
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

BASE = Path(__file__).parent
MODEL = BASE / "assets" / "model.tflite"
LABELS = BASE / "assets" / "labels.txt"

st.set_page_config(page_title="LiveDetection", page_icon="🔬")

st.title("🔬 LiveDetection")
st.write("Desmiden-Erkennung – Top 3 Arten")

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=str(MODEL))
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_labels():
    return [
        line.strip()
        for line in LABELS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def recognize(image):
    interpreter = load_model()
    labels = load_labels()

    image = image.convert("RGB")
    image = image.resize((224, 224), Image.Resampling.BOX)

    # MobileNetV3-Vorverarbeitung ist im Modell enthalten.
    input_data = np.asarray(image, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    scores = interpreter.get_tensor(output_details[0]["index"])[0]

    if len(labels) != len(scores):
        raise ValueError("Klassenliste und Modell passen nicht zusammen.")

    top_indices = np.argsort(scores)[::-1][:3]

    return [(labels[i], float(scores[i])) for i in top_indices]

aufnahme = st.camera_input("Mikroskopfoto aufnehmen")

galerie = st.file_uploader(
    "Oder Foto aus der Galerie auswählen",
    type=["jpg", "jpeg", "png"]
)

photo = aufnahme if aufnahme is not None else galerie

if photo is not None:
    image = Image.open(photo).convert("RGB")

    st.subheader("Bildausschnitt auswählen")
    st.write(
        "Verschiebe den quadratischen Rahmen über die Alge. "
        "Du kannst den Rahmen auch vergrößern oder verkleinern."
    )

    cropped_image = st_cropper(
        image,
        realtime_update=True,
        box_color="#00FF00",
        aspect_ratio=(1, 1),
        return_type="image",
        key="desmiden_cropper",
    )

    st.image(
        cropped_image,
        caption="Ausgewählter Bildausschnitt",
        width=250,
    )

    if st.button("Desmide erkennen", type="primary"):
        try:
            results = recognize(cropped_image)

            st.subheader("Top 3 Erkennung")

            for name, score in results:
                st.write(f"**{name}** – {score * 100:.1f} %")

            st.caption(
                "KI-Modellwerte sind keine gesicherte Artbestimmung."
            )
        except Exception as error:
            st.error(f"Erkennung fehlgeschlagen: {error}")

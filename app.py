import streamlit as st
from streamlit_cropper import st_cropper
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from io import BytesIO
from datetime import datetime
import hashlib
import base64

BASE = Path(__file__).parent
MODEL = BASE / "assets" / "model.tflite"
LABELS = BASE / "assets" / "labels.txt"

st.set_page_config(
    page_title="LiveDetection",
    page_icon=str(BASE / "livedetection_icon.png"),
)

# Versionsnummer für die Aufnahme-Widgets.
if "input_epoch" not in st.session_state:
    st.session_state["input_epoch"] = 0

epoch = st.session_state["input_epoch"]

st.image(str(BASE / "livedetection_icon.png"), width=180)
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


def image_to_jpeg(image):
    output = BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=95)
    return output.getvalue()


def create_result_image(cropped_image, results):
    cropped_image = cropped_image.convert("RGB")

    width = 1400
    margin = 65
    photo = cropped_image.copy()
    photo.thumbnail((width - 2 * margin, 950))

    header_height = 180
    footer_height = 390
    height = header_height + photo.height + footer_height

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font_bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48
        )
        font_regular = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 35
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25
        )
    except OSError:
        font_bold = ImageFont.load_default()
        font_regular = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.rectangle((0, 0, width, header_height), fill="#174D35")
    draw.text(
        (margin, 45),
        "LiveDetection",
        fill="white",
        font=font_bold,
    )
    draw.text(
        (margin, 110),
        "Erkennungsergebnis – Top 3",
        fill="white",
        font=font_regular,
    )

    photo_x = (width - photo.width) // 2
    photo_y = header_height
    canvas.paste(photo, (photo_x, photo_y))

    y = photo_y + photo.height + 35
    draw.text(
        (margin, y),
        "Erkannte Arten",
        fill="#174D35",
        font=font_bold,
    )

    y += 85
    for index, (name, score) in enumerate(results, start=1):
        draw.text(
            (margin, y),
            f"{index}.  {name}",
            fill="#222222",
            font=font_regular,
        )
        draw.text(
            (width - margin - 230, y),
            f"{score * 100:.1f} %",
            fill="#174D35",
            font=font_regular,
        )
        y += 75

    y += 15
    timestamp = datetime.now().strftime("%d.%m.%Y – %H:%M Uhr")

    draw.text(
        (margin, y),
        f"Aufnahme: {timestamp}",
        fill="#666666",
        font=font_small,
    )
    draw.text(
        (margin, y + 45),
        "KI-Modellwerte sind keine gesicherte Artbestimmung.",
        fill="#666666",
        font=font_small,
    )

    output = BytesIO()
    canvas.save(output, format="JPEG", quality=95)
    return output.getvalue()


aufnahme = st.camera_input(
    "Mikroskopfoto aufnehmen",
    key=f"camera_{epoch}",
)

galerie = st.file_uploader(
    "Oder Foto aus der Galerie auswählen",
    type=["jpg", "jpeg", "png"],
    key=f"upload_{epoch}",
)

photo = aufnahme if aufnahme is not None else galerie

if photo is not None:
    original_bytes = photo.getvalue()
    photo_id = hashlib.sha256(original_bytes).hexdigest()

    if st.session_state.get("photo_id") != photo_id:
        st.session_state["photo_id"] = photo_id
        st.session_state.pop("results", None)
        st.session_state.pop("result_image", None)
        st.session_state.pop("result_crop_id", None)

    image = Image.open(BytesIO(original_bytes)).convert("RGB")

    st.subheader("Originalfoto")
    st.image(image, width=350)

    suffix = Path(
        getattr(photo, "name", "Mikroskopfoto.jpg")
    ).suffix.lower()

    if suffix not in [".jpg", ".jpeg", ".png"]:
        suffix = ".jpg"

    # Teilen über das native iPhone-Teilen-Menü
image_base64 = base64.b64encode(cropped_bytes).decode("ascii")

share_html = """
<button id="share-crop"
style="
    background:#174D35;
    color:white;
    border:0;
    border-radius:8px;
    padding:12px 18px;
    font-size:16px;
    width:100%;
    cursor:pointer;">
    📲 Zuschnitt teilen / in Fotos sichern
</button>
<p id="share-status" style="font-size:13px;"></p>

<script>
const imageData = "__IMAGE_DATA__";

document.getElementById("share-crop").addEventListener("click", async () => {
    const status = document.getElementById("share-status");

    try {
        const binary = atob(imageData);
        const bytes = new Uint8Array(binary.length);

        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }

        const file = new File(
            [bytes],
            "LiveDetection_Zuschnitt.jpg",
            {type: "image/jpeg"}
        );

        if (navigator.canShare && navigator.canShare({files: [file]})) {
            await navigator.share({
                files: [file],
                title: "LiveDetection Bildausschnitt"
            });
            status.textContent = "Teilen-Menü geöffnet.";
        } else {
            status.textContent =
                "Teilen nicht unterstützt. Bitte den Download darunter verwenden.";
        }
    } catch (error) {
        if (error.name !== "AbortError") {
            status.textContent = "Teilen nicht möglich. Bitte Download verwenden.";
        }
    }
});
</script>
"""

share_html = share_html.replace("__IMAGE_DATA__", image_base64)

st.html(share_html, unsafe_allow_javascript=True)

# Bisheriger Download bleibt als Rückfallebene
st.download_button(
    "⬇️ Zuschnitt herunterladen",
    data=cropped_bytes,
    file_name="LiveDetection_Zuschnitt.jpg",
    mime="image/jpeg",
    key=f"download_crop_{epoch}",
)

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
        key=f"desmiden_cropper_{epoch}",
    )

    st.image(
        cropped_image,
        caption="Ausgewählter Bildausschnitt",
        width=250,
    )

    cropped_bytes = image_to_jpeg(cropped_image)
    crop_id = hashlib.sha256(cropped_bytes).hexdigest()

    st.download_button(
        "✂️ Zugeschnittenes Bild auf iPhone speichern",
        data=cropped_bytes,
        file_name="LiveDetection_Zuschnitt.jpg",
        mime="image/jpeg",
        key=f"download_crop_{epoch}",
    )

    if st.button(
        "Desmide erkennen",
        type="primary",
        key=f"recognize_{epoch}",
    ):
        try:
            results = recognize(cropped_image)

            st.session_state["results"] = results
            st.session_state["result_image"] = create_result_image(
                cropped_image, results
            )
            st.session_state["result_crop_id"] = crop_id

        except Exception as error:
            st.error(f"Erkennung fehlgeschlagen: {error}")

    if (
        st.session_state.get("result_crop_id") == crop_id
        and "results" in st.session_state
    ):
        results = st.session_state["results"]

        st.subheader("Top 3 Erkennung")

        for name, score in results:
            st.write(f"**{name}** – {score * 100:.1f} %")

        st.caption(
            "KI-Modellwerte sind keine gesicherte Artbestimmung."
        )

        st.subheader("Ergebnisbild mit Bildausschnitt und Top 3")

        result_bytes = st.session_state["result_image"]

        st.image(result_bytes, use_container_width=True)

        st.download_button(
            "🧪 Ergebnisbild mit Top 3 auf iPhone speichern",
            data=result_bytes,
            file_name="LiveDetection_Ergebnis_Top3.jpg",
            mime="image/jpeg",
            key=f"download_result_{epoch}",
        )

    st.divider()

    if st.button(
        "↩️ Beenden / Neue Aufnahme",
        key=f"reset_{epoch}",
        use_container_width=True,
    ):
        st.session_state["input_epoch"] += 1
        st.session_state.pop("photo_id", None)
        st.session_state.pop("results", None)
        st.session_state.pop("result_image", None)
        st.session_state.pop("result_crop_id", None)
        st.rerun()
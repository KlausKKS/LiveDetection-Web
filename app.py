
import streamlit as st
import streamlit.components.v1 as components
from streamlit_cropper import st_cropper
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pathlib import Path
from io import BytesIO
from datetime import datetime
import hashlib
import base64
import html
import json


# --------------------------------------------------
# Seiteneinstellungen
# --------------------------------------------------

st.set_page_config(
    page_title="LiveDetection",
    page_icon="livedetection_icon.png",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "assets" / "model.tflite"
LABELS_PATH = BASE_DIR / "assets" / "labels.txt"


# --------------------------------------------------
# Sitzung und Zurücksetzen
# --------------------------------------------------

if "input_epoch" not in st.session_state:
    st.session_state.input_epoch = 0

if "results" not in st.session_state:
    st.session_state.results = None

if "result_image" not in st.session_state:
    st.session_state.result_image = None

if "result_crop_id" not in st.session_state:
    st.session_state.result_crop_id = None


def reset_app():
    st.session_state.input_epoch += 1
    st.session_state.results = None
    st.session_state.result_image = None
    st.session_state.result_crop_id = None
    st.rerun()


# --------------------------------------------------
# Modell und Artennamen laden
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden: {MODEL_PATH}"
        )

    interpreter = tf.lite.Interpreter(
        model_path=str(MODEL_PATH)
    )
    interpreter.allocate_tensors()
    return interpreter


@st.cache_data
def load_labels():
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Artenliste nicht gefunden: {LABELS_PATH}"
        )

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return labels


# --------------------------------------------------
# Bilder für das Modell vorbereiten und erkennen
# --------------------------------------------------

def recognize(image):
    interpreter = load_model()
    labels = load_labels()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    input_height = int(input_shape[1])
    input_width = int(input_shape[2])
    input_dtype = input_details[0]["dtype"]

    image = image.convert("RGB").resize(
        (input_width, input_height),
        Image.Resampling.LANCZOS,
    )

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    # Das Modell erwartet RGB-Werte im Bereich 0 bis 255.
    image_array = np.expand_dims(image_array, axis=0)

    if input_dtype == np.float32:
        model_input = image_array.astype(np.float32)

    elif input_dtype in (np.uint8, np.int8):
        scale, zero_point = input_details[0]["quantization"]

        if scale:
            model_input = np.round(
                image_array / scale + zero_point
            )
        else:
            model_input = image_array

        limits = np.iinfo(input_dtype)
        model_input = np.clip(
            model_input,
            limits.min,
            limits.max,
        ).astype(input_dtype)

    else:
        model_input = image_array.astype(input_dtype)

    interpreter.set_tensor(
        input_details[0]["index"],
        model_input,
    )
    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    scores = np.asarray(output, dtype=np.float64).flatten()

    # Falls das Modell Logits statt Wahrscheinlichkeiten liefert:
    if (
        np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(np.sum(scores), 1.0, atol=0.03)
    ):
        exp_scores = np.exp(scores - np.max(scores))
        scores = exp_scores / np.sum(exp_scores)

    number_of_classes = min(len(labels), len(scores))

    top_indices = np.argsort(scores[:number_of_classes])[::-1][:3]

    results = []

    for index in top_indices:
        results.append({
            "name": labels[index],
            "score": float(scores[index]),
        })

    return results


# --------------------------------------------------
# Schriftarten
# --------------------------------------------------

def load_font(size, bold=False):
    if bold:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        ]

    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue

    return ImageFont.load_default()


# --------------------------------------------------
# Bildfunktionen
# --------------------------------------------------

def image_to_jpeg(image):
    buffer = BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=95,
    )
    return buffer.getvalue()


def create_result_image(cropped_image, results):
    cropped_image = cropped_image.convert("RGB")

    image_width, image_height = cropped_image.size

    footer_height = 390

    result_image = Image.new(
        "RGB",
        (image_width, image_height + footer_height),
        "white",
    )

    result_image.paste(cropped_image, (0, 0))

    draw = ImageDraw.Draw(result_image)

    font_title = load_font(42, bold=True)
    font_regular = load_font(30)
    font_top3 = load_font(42, bold=True)
    font_small = load_font(25)

    y = image_height + 20

    draw.text(
        (30, y),
        "LiveDetection – Erkennung",
        fill="black",
        font=font_title,
    )

    y += 65

    draw.text(
        (30, y),
        "Die wahrscheinlichsten Arten:",
        fill="black",
        font=font_regular,
    )

    y += 55

    for position, result in enumerate(results, start=1):
        name = result["name"]
        percentage = result["score"] * 100

        text = (
            f"{position}. {name} – "
            f"{percentage:.1f} %"
        )

        draw.text(
            (30, y),
            text,
            fill="black",
            font=font_top3,
        )

        y += 85

    timestamp = datetime.now().strftime(
        "%d.%m.%Y  %H:%M"
    )

    draw.text(
        (30, image_height + footer_height - 45),
        timestamp,
        fill="gray",
        font=font_small,
    )

    return result_image


# --------------------------------------------------
# Teilen- und Speichern-Schaltfläche
# --------------------------------------------------

def show_share_button(
    image_bytes,
    button_id,
    filename,
    title,
    label,
):
    image_base64 = base64.b64encode(
        image_bytes
    ).decode("ascii")

    safe_button_id = json.dumps(button_id)
    safe_filename = json.dumps(filename)
    safe_title = json.dumps(title)
    safe_label = html.escape(label)

    share_html = """
    <html>
    <head>
        <meta name="viewport"
              content="width=device-width, initial-scale=1">
        <style>
            body {
                margin: 0;
                padding: 5px 0;
                font-family: Arial, sans-serif;
            }

            button {
                width: 100%;
                min-height: 52px;
                border: none;
                border-radius: 10px;
                background: #16834a;
                color: white;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                padding: 12px;
            }

            button:active {
                opacity: 0.8;
            }

            #status {
                font-size: 13px;
                margin-top: 5px;
                text-align: center;
                color: #555;
            }
        </style>
    </head>
    <body>
        <button id="SHARE_ID">BUTTON_LABEL</button>
        <div id="STATUS_ID"></div>

        <script>
            const buttonId = SHARE_ID;
            const filename = FILENAME;
            const title = TITLE;

            const imageData = "__IMAGE_DATA__";

            const button = document.getElementById(buttonId);
            const status = document.getElementById(
                "status-" + buttonId
            );

            button.addEventListener("click", async function () {
                try {
                    const binary = atob(imageData);
                    const bytes = new Uint8Array(binary.length);

                    for (let i = 0; i < binary.length; i++) {
                        bytes[i] = binary.charCodeAt(i);
                    }

                    const file = new File(
                        [bytes],
                        filename,
                        { type: "image/jpeg" }
                    );

                    if (
                        navigator.share &&
                        navigator.canShare &&
                        navigator.canShare({ files: [file] })
                    ) {
                        await navigator.share({
                            files: [file],
                            title: title
                        });

                        status.textContent = "Teilen abgeschlossen.";
                    } else {
                        const url = URL.createObjectURL(file);
                        const link = document.createElement("a");

                        link.href = url;
                        link.download = filename;
                        document.body.appendChild(link);
                        link.click();
                        link.remove();

                        setTimeout(
                            () => URL.revokeObjectURL(url),
                            10000
                        );

                        status.textContent =
                            "Teilen nicht verfügbar – Bild wurde zum Download bereitgestellt.";
                    }
                } catch (error) {
                    if (error.name === "AbortError") {
                        status.textContent = "Teilen abgebrochen.";
                    } else {
                        status.textContent =
                            "Teilen nicht möglich: " + error.message;
                    }
                }
            });
        </script>
    </body>
    </html>
    """

    share_html = share_html.replace(
        "SHARE_ID",
        safe_button_id,
    )
    share_html = share_html.replace(
        "FILENAME",
        safe_filename,
    )
    share_html = share_html.replace(
        "TITLE",
        safe_title,
    )
    share_html = share_html.replace(
        "BUTTON_LABEL",
        safe_label,
    )
    share_html = share_html.replace(
        "__IMAGE_DATA__",
        image_base64,
    )
    share_html = share_html.replace(
        'id="STATUS_ID"',
        f'id="status-{button_id}"',
    )

    components.html(
        share_html,
        height=85,
        scrolling=False,
    )


# --------------------------------------------------
# Benutzeroberfläche
# --------------------------------------------------

st.title("🔬 LiveDetection")
st.write(
    "Erkennung von Desmiden anhand eines Fotos."
)

st.caption(
    "Nimm ein Foto auf oder wähle ein vorhandenes Bild aus."
)

epoch = st.session_state.input_epoch


# --------------------------------------------------
# Foto aufnehmen oder auswählen
# --------------------------------------------------

camera_image = st.camera_input(
    "📷 Foto aufnehmen",
    key=f"camera_{epoch}",
)

uploaded_image = st.file_uploader(
    "🖼️ Bild auswählen",
    type=["jpg", "jpeg", "png"],
    key=f"upload_{epoch}",
)


# --------------------------------------------------
# Ausgewähltes Bild verarbeiten
# --------------------------------------------------

source_image = None

if camera_image is not None:
    source_image = Image.open(
        camera_image
    )

elif uploaded_image is not None:
    source_image = Image.open(
        uploaded_image
    )


if source_image is not None:

    source_image = ImageOps.exif_transpose(
        source_image
    ).convert("RGB")

    st.subheader("Originalbild")

    st.image(
        source_image,
        use_container_width=True,
    )

    original_bytes = image_to_jpeg(source_image)

    st.download_button(
        "⬇️ Originalfoto herunterladen",
        data=original_bytes,
        file_name="LiveDetection_Original.jpg",
        mime="image/jpeg",
        key=f"download-original-{epoch}",
        use_container_width=True,
    )

    st.divider()

    st.subheader("Bildausschnitt auswählen")

    st.write(
        "Verschiebe und vergrößere den Ausschnitt, "
        "bis die Desmide gut zu sehen ist."
    )

    cropped_image = st_cropper(
        source_image,
        aspect_ratio=(1, 1),
        return_type="image",
        key=f"desmiden_cropper_{epoch}",
    )

    if cropped_image is not None:

        cropped_image = cropped_image.convert("RGB")

        cropped_bytes = image_to_jpeg(
            cropped_image
        )

        crop_id = hashlib.sha256(
            cropped_bytes
        ).hexdigest()

        st.subheader("Gewählter Ausschnitt")

        st.image(
            cropped_image,
            use_container_width=True,
        )

        show_share_button(
            cropped_bytes,
            f"share-crop-{epoch}",
            "LiveDetection_Zuschnitt.jpg",
            "LiveDetection Bildausschnitt",
            "📲 Zuschnitt teilen / in Fotos sichern",
        )

        st.divider()

        # --------------------------------------------------
        # Erkennung starten
        # --------------------------------------------------

        if st.button(
            "🔬 Desmide erkennen",
            type="primary",
            use_container_width=True,
            key=f"recognize-{epoch}",
        ):
            try:
                with st.spinner(
                    "Das Bild wird analysiert ..."
                ):
                    results = recognize(cropped_image)

                    result_image = create_result_image(
                        cropped_image,
                        results,
                    )

                    st.session_state.results = results
                    st.session_state.result_image = (
                        image_to_jpeg(result_image)
                    )
                    st.session_state.result_crop_id = crop_id

            except Exception as error:
                st.error(
                    f"Bei der Erkennung ist ein Fehler aufgetreten: {error}"
                )

        # --------------------------------------------------
        # Ergebnisse anzeigen
        # --------------------------------------------------

        if (
            st.session_state.results is not None
            and st.session_state.result_crop_id == crop_id
        ):

            st.divider()
            st.subheader("🧪 Erkennungsergebnis")

            for position, result in enumerate(
                st.session_state.results,
                start=1,
            ):
                st.write(
                    f"**{position}. {result['name']}** "
                    f"– {result['score'] * 100:.1f} %"
                )

            result_bytes = st.session_state.result_image

            st.image(
                result_bytes,
                caption="Ergebnisbild mit den drei wahrscheinlichsten Arten",
                use_container_width=True,
            )

            show_share_button(
                result_bytes,
                f"share-result-{epoch}",
                "LiveDetection_Ergebnis_Top3.jpg",
                "LiveDetection Ergebnis mit Top 3",
                "📲 Ergebnisbild teilen / in Fotos sichern",
            )


# --------------------------------------------------
# Neue Aufnahme
# --------------------------------------------------

st.divider()

if st.button(
    "🔄 Beenden / Neue Aufnahme",
    use_container_width=True,
    key=f"reset-{epoch}",
):
    reset_app()

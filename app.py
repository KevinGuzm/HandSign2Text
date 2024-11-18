import joblib
import cv2
import gradio as gr
from gradio_webrtc import WebRTC
from utils import get_hand_landmarks

# Diccionario para mapear las predicciones a letras
letter_map = {"A": "A", "B": "B", "C": "C", "D": "D"}

# Cargar el modelo entrenado
model = joblib.load("./random_forest_model.pkl")

def process_landmarks(image, model, letter_map):
    """Procesa la imagen para detectar landmarks, clasificar la letra y dibujar bounding box."""
    h, w, _ = image.shape

    # Obtener landmarks de la mano
    landmarks, x_coords, y_coords = get_hand_landmarks(image, draw=True, return_coords=True)

    # Validar que se obtuvieron los landmarks necesarios
    if len(landmarks) == 63:
        try:
            output = model.predict([landmarks])
            letter = letter_map.get(output[0], "Unknown")

            # Dibujar bounding box
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Mostrar la predicción encima del bounding box
            cv2.putText(
                image, letter, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
        except Exception as e:
            print("Error en la predicción:", e)
            letter = "Error"
    else:
        cv2.putText(
            image, "No hand detected", (15, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

    return image

# Función principal para Gradio
def detection(image):
    processed_image = process_landmarks(image, model, letter_map)
    return processed_image

css = """.my-group {max-width: 600px !important; max-height: 600 !important;}
                      .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    Hand Gesture Recognition Webcam Stream (Powered by WebRTC ⚡️)
    </h1>
    """
    )
    gr.HTML(
        """
        <h3 style='text-align: center'>
        Detect Hand Gestures in Real-Time
        </h3>
        """
    )
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            image = WebRTC(label="Stream", rtc_configuration=None)

        # Configuración del stream para ejecutar la función de detección
        image.stream(
            fn=detection, inputs=[image], outputs=[image], time_limit=50
        )

if __name__ == "__main__":
    demo.launch(share=True)
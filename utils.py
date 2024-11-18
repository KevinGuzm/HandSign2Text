import mediapipe as mp
import cv2

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def get_hand_landmarks(image, draw=False, return_coords=False):
    """Obtiene los landmarks de la mano a partir de una imagen."""
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    landmarks = []
    x_coords = []
    y_coords = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                x_coords.append(x)
                y_coords.append(y)
                landmarks.extend([lm.x, lm.y, lm.z])
            if draw:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if return_coords:
        return landmarks, x_coords, y_coords
    return landmarks
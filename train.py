import cv2
import mediapipe as mp
import json


# Inicializa MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Inicializa captura de video
cap = cv2.VideoCapture(0)

dataset = []  # Lista para almacenar los datos capturados

print("Presiona 's' para guardar una muestra con etiqueta. Presiona 'Esc' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar video")
        break

    # Voltea la imagen horizontalmente y conviértela a RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesa el cuadro con MediaPipe
    result = hands.process(rgb_frame)

    # Verifica si se detectan manos
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Dibuja puntos clave en la mano detectada
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extrae coordenadas de landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            # Captura la seña cuando se presiona la tecla 's'
            if cv2.waitKey(1) & 0xFF == ord("s"):
                label = input("Introduce la etiqueta para esta sena: ")
                dataset.append({"landmarks": landmarks, "label": label})
                print(f"Se guardo la muestra con etiqueta: {label}")

    # Muestra el video en una ventana
    cv2.imshow("Capturador de Datos", frame)

    # Presiona 'Esc' para salir
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()

# Guarda el dataset en un archivo JSON
with open("senas.json", "w") as f:
    json.dump(dataset, f)

print("Dataset guardado en 'senas.json'")
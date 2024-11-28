import cv2
import mediapipe as mp
import joblib
import numpy as np

# Cargar el modelo entrenado
model = joblib.load("modelo_senhas.pkl")

# Obtener la lista de gestos (asumiendo que tu modelo tiene un método para recuperar clases)
trained_gestures = model.classes_

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Paleta de colores moderna y minimalista
COLORS = {
    'background': (92, 91, 91),  # Light gray
    'text': (255, 255, 255),  #white
    'active_gesture': (0 , 0, 0), #black
    'inactive_gesture': (140, 137, 137), # light dark gray
    'point_color': (30, 30, 30),  #dark gray
    'connection_color': (56, 55, 55) #medium drak grey
}

# Captura de video
cap = cv2.VideoCapture(0)

# Ajustar tamaño de la ventana
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def draw_gesture_list(frame, gestures, current_gesture):
    """Dibuja la lista de gestos entrenados"""
    height, width, _ = frame.shape
    list_width = 250
    
    # Fondo de la lista de gestos
    cv2.rectangle(frame, 
                  (width - list_width, 0), 
                  (width, height), 
                  COLORS['background'], -1)
    
    # Título de la lista
    cv2.putText(frame, "Gestos Entrenados", 
                (width - list_width + 20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, COLORS['text'], 2)
    
    # Dibujar lista de gestos
    for i, gesture in enumerate(gestures):
        color = COLORS['active_gesture'] if gesture == current_gesture else COLORS['inactive_gesture']
        cv2.rectangle(frame, 
                      (width - list_width + 20, 70 + i*40), 
                      (width - 20, 100 + i*40), 
                      color, -1)
        
        cv2.putText(frame, gesture, 
                    (width - list_width + 40, 95 + i*40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, COLORS['text'], 2)

def draw_status_bar(frame, status, gesture='N/A'):
    """Dibuja una barra de estado moderna"""
    height, width, _ = frame.shape
    status_bar_height = 40
    
    # Fondo de la barra de estado
    cv2.rectangle(frame, (0, height - status_bar_height), 
                  (width, height), 
                  COLORS['background'], -1)
    
    # Texto de estado
    status_text = f"Estado: {status} | Gesto: {gesture}"
    cv2.putText(frame, status_text, 
                (20, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, COLORS['text'], 2)

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

    # Estado inicial
    status = "Sin mano detectada"
    current_gesture = "N/A"

    # Verifica si se detectan manos
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Dibuja puntos clave en la mano detectada
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=COLORS['point_color'], thickness=5),
                mp_drawing.DrawingSpec(color=COLORS['connection_color'], thickness=3))

            # Extrae coordenadas de landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            # Aplana los landmarks y realiza la predicción
            flattened_landmarks = sum(landmarks, [])
            prediction = model.predict([flattened_landmarks])
            current_gesture = prediction[0]
            status = "Gesto detectado"

            # Muestra la predicción con estilo minimalista
            cv2.putText(frame, f"{current_gesture}", 
                        (frame.shape[1] - 250, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, COLORS['text'], 2)

    # Dibuja lista de gestos
    draw_gesture_list(frame, trained_gestures, current_gesture)

    # Dibuja barra de estado
    draw_status_bar(frame, status, current_gesture)

    # Muestra el video en una ventana
    cv2.imshow("Detector de Gestos", frame)

    # Presiona 'Esc' para salir
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Carregar o modelo treinado
model = load_model('model.keras')

# Função para processar a imagem e fazer a detecção
def process_frame(frame):
    imgPre = cv2.GaussianBlur(frame, (5, 5), 3)  # Aplicar desfoque para reduzir ruído
    imgPre = cv2.Canny(imgPre, 90, 140)  # Detectar bordas usando o Canny
    kernel = np.ones((4, 4), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations=2)
    imgPre = cv2.erode(imgPre, kernel, iterations=1)
    
    return imgPre

# Inicializar a captura de vídeo (webcam)
cap = cv2.VideoCapture(0)
imgIndex = 199

while True:
    ret, frame = cap.read()
    if not ret:
        break

    imgPreProcess = process_frame(frame)

    countours, _ = cv2.findContours(imgPreProcess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cut = None

    for countour in countours:
        area = cv2.contourArea(countour)

        if area < 2000:
            continue

        x, y, w, h = cv2.boundingRect(countour)

        cut = frame[y:y + h, x:x + w]

    cv2.imshow("Moeda Reconhecida", frame)
    cv2.imshow("", imgPreProcess)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite(f"/home/giulliano/repos/contador-de-moedas/images/25/{imgIndex}.jpg", cut)
        imgIndex += 1

cap.release()
cv2.destroyAllWindows()
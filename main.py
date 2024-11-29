import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

class_names = os.listdir("images")
class_names.sort()

print

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Carregar o modelo treinado
model = load_model('model.keras')

# Função para processar a imagem e fazer a detecção
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converter para escala de cinza
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Aplicar desfoque para reduzir ruído
    edges = cv2.Canny(blurred, 50, 150)  # Detectar bordas usando o Canny
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Encontrar os contornos
    
    return contours

# Inicializar a captura de vídeo (webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Processar o frame da webcam
    contours = process_frame(frame)

    for contour in contours:
        # Ignorar contornos pequenos
        if cv2.contourArea(contour) < 500:
            continue

        # Obter a caixa delimitadora do contorno
        x, y, w, h = cv2.boundingRect(contour)

        roi = frame[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (224, 224))  # Redimensionar para o tamanho esperado pelo modelo
        roi_normalized = roi_resized / 255.0  # Normalizar a imagem

        # Adicionar uma dimensão extra (para representar o batch dimension)
        roi_input = np.expand_dims(roi_normalized, axis=0)  # Adiciona a dimensão de batch

        # Prever a classe da moeda
        prediction = model.predict(roi_input)
        predicted_class = np.argmax(prediction)  # Obter a classe com maior probabilidade
        predicted_label = class_names[predicted_class]  # Obter o nome da classe


        # Desenhar o contorno e o nome da moeda na imagem original
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Desenhar o contorno da moeda
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Exibir o nome da moeda

    # Mostrar o vídeo com o contorno e nome da moeda
    cv2.imshow("Moeda Reconhecida", frame)

    # Sair ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura e fechar as janelas
cap.release()
cv2.destroyAllWindows()

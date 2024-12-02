import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

class_ids = os.listdir("images")
class_ids.sort()

classes_names = {
    '25': {
        "displayName": "25 centavos",
        "value": 0.25
    },
    '50': {
        "displayName": "50 centavos",
        "value": 0.5
    },
    '1': {
        "displayName": "1 real",
        "value": 1
    }
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Carregar o modelo treinado
model = load_model('model.keras')

# Função para processar a imagem e fazer a detecção
def capture_countours(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converter para escala de cinza
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Aplicar desfoque para reduzir ruído
    edges = cv2.Canny(blurred, 50, 150)  # Detectar bordas usando o Canny
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Encontrar os contornos
    
    return contours

def detect_coin(img):
    roi_resized = cv2.resize(img, (224, 224))  # Redimensionar para o tamanho esperado pelo modelo
    roi_normalized = roi_resized / 255.0  # Normalizar a imagem

    # Adicionar uma dimensão extra (para representar o batch dimension)
    roi_input = np.expand_dims(roi_normalized, axis=0)  # Adiciona a dimensão de batch

    # Prever a classe da moeda
    prediction = model.predict(roi_input)
    predicted_class = np.argmax(prediction)  # Obter a classe com maior probabilidade
    prediction_percent = prediction[0][predicted_class]
    predicted_label = class_ids[predicted_class]  # Obter o nome da classe

    return predicted_label, prediction_percent

def format_coin_value(value):
    a = '{:,.2f}'.format(float(value))
    b = a.replace(',','v')
    c = b.replace('.',',')
    return c.replace('v','.')

# Inicializar a captura de vídeo (webcam)
cap = cv2.VideoCapture(0)

totalValue = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Processar o frame da webcam
    contours = capture_countours(frame)

    amount = 0

    for contour in contours:
        # Ignorar contornos pequenos
        if cv2.contourArea(contour) < 500:
            continue

        # Obter a caixa delimitadora do contorno
        x, y, w, h = cv2.boundingRect(contour)

        roi = frame[y:y+h, x:x+w]
        
        predicted_label, percent = detect_coin(roi)

        if percent > 0.7:
            coinName = classes_names[predicted_label]

            amount += coinName["value"]

            # Desenhar o contorno e o nome da moeda na imagem original
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Desenhar o contorno da moeda
            cv2.putText(frame, coinName["displayName"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Exibir o nome da moeda

    cv2.rectangle(frame,(430,30),(600,80),(0,0,255),-1)
    cv2.putText(frame,f'R$ {format_coin_value(totalValue+amount)}',(440,67),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
    # Mostrar o vídeo com o contorno e nome da moeda
    cv2.imshow("Moeda Reconhecida", frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        totalValue += amount

    # Sair ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura e fechar as janelas
cap.release()
cv2.destroyAllWindows()
import cv2
import os
import numpy as np
import random

def preprocess_images(folder, size=(224, 224)):
    images = []
    labels = []
    class_names = os.listdir(folder)
    class_names.sort()  # Para manter a consistência
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(folder, class_name)
        for filename in os.listdir(class_folder):
            filepath = os.path.join(class_folder, filename)
            img = cv2.imread(filepath)
            img = crop_and_resize(img, size)

            # Salvar imagem de exemplo (opcional)
            os.makedirs(f"/home/giulliano/repos/contador-de-moedas/imagens_resized/{class_name}", exist_ok=True)
            cv2.imwrite(f"/home/giulliano/repos/contador-de-moedas/imagens_resized/{class_name}/{filename}", img)
            
            img = img / 255.0  # Normalização
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels), class_names

def crop_and_resize(img, size):
    # Mantem a proporção da imagem
    height, width = img.shape[:2]
    aspect = width / height
    
    if aspect > 1:
        new_width = size[0]
        new_height = int(size[0] / aspect)
    else:
        new_height = size[1]
        new_width = int(size[1] * aspect)
        
    resized = cv2.resize(img, (new_width, new_height))
    
    # Centraliza a imagem em um canvas do tamanho desejado
    result = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    y_offset = (size[1] - new_height) // 2
    x_offset = (size[0] - new_width) // 2
    result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return result

preprocess_images("images")
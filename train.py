from sklearn.model_selection import train_test_split
from resize_images import preprocess_images
import numpy as np
from model import create_model

images, labels, _ = preprocess_images("images")
model = create_model()

# Divida os dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Ajuste a forma das imagens para incluir o canal (grayscale -> canal único)
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

# Treine o modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

model.save("model.keras")

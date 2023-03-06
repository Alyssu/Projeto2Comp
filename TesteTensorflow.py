import tensorflow as tf
import numpy as np

# Define os dados de entrada e saída
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Define o modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='sigmoid', input_dim=2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compila o modelo
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Treina o modelo
model.fit(X, Y, epochs=1000, verbose=0)

# Faz a previsão com o modelo treinado
predictions = model.predict(X)

# Exibe as previsões
for i in range(len(predictions)):
    print(f"Entrada: {X[i]} Saída Esperada: {Y[i]} Saída do Modelo: {predictions[i][0]}")

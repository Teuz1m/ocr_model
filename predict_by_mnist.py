import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

"""Modelo de Rede Neural Convolucional para classificação de dígitos manuscritos do MNIST."""
# Carregar o modelo treinado
model = tf.keras.models.load_model('mnist_model_improved.h5')

# Recompilar o modelo para garantir que as métricas estejam configuradas
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Carregar dataset MNIST
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar os dados (média zero e desvio padrão 1)
mean = np.mean(x_test)
std = np.std(x_test)
x_test = (x_test - mean) / std

#Redimensionar para formato correto (batch, altura, largura, canais)
x_test = np.expand_dims(x_test, axis=-1)

# Função para carregar e pré-processar a imagem
def load_and_preprocess_image(image):
    image = image.astype('float32')  # Certificar-se de que a imagem está em float32
    image = np.expand_dims(image, axis=-1)  # Adicionar dimensão do canal (28, 28, 1)
    image = np.expand_dims(image, axis=0)  # Adicionar dimensão do batch (1, 28, 28, 1)
    return image
   
# Avaliar várias imagens do conjunto de teste
for index in range(10):  # Avaliar as primeiras 10 imagens
    mnist_image = x_test[index]
    true_label = y_test[index]

    # Pré-processar a imagem
    img_array = load_and_preprocess_image(mnist_image)

    # Fazer a previsão
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Mostrar a imagem
    plt.imshow(mnist_image.squeeze(), cmap='gray')
    plt.title(f'True Class: {true_label}, Predicted Class: {predicted_class[0]}')
    plt.axis('off')
    plt.show()

    print(f'True Class: {true_label}')
    print(f'Predicted Class: {predicted_class[0]}')

#Avaliar o modelo no conjunto de teste
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

"""Modelo que faz classificação de dígitos manuscritos do MNIST,com imagem vinda do computador.

Lembrando que a imagem a ser classificada precisa possuir fundo preto e dígitos brancos,tal qual o dataset MNIST.


"""
# Carregar o modelo treinado
model = tf.keras.models.load_model('mnist_model_improved.h5')

# Recompilar o modelo para garantir que as métricas estejam configuradas
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Função para carregar e pré-processar a imagem
def load_and_preprocess_image(image):
    # Carregar a imagem em escala de cinza
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img.astype('float32') #/ 255.0
    # Adicionar uma dimensão extra para o batch e canais
    img = np.expand_dims(img, axis=-1)  # Adicionar dimensão para canais
    img = np.expand_dims(img, axis=0)   # Adicionar dimensão para batch
    return img

# Caminho para a imagem que você quer testar
img_path = 'numero.png'  # Substitua pelo caminho da sua imagem

# Carregar e pré-processar a imagem
img_array = load_and_preprocess_image(img_path)

# Fazer a previsão
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# Mostrar a imagem
plt.imshow(img_array[0, :, :, 0], cmap='gray')
plt.title(f'Predicted Class: {predicted_class[0]}')
plt.axis('off')
plt.savefig('output.png')


print(f'Predicted Class: {predicted_class[0]}')

# Avaliar o modelo no conjunto de teste
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = (x_test - np.mean(x_test)) / np.std(x_test)
x_test = np.expand_dims(x_test, axis=-1)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

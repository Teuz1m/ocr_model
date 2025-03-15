import tensorflow as tf
import numpy as np

# Carregar dataset MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar os dados (média zero e desvio padrão 1)
mean = np.mean(x_train)
std = np.std(x_train)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Redimensionar para formato correto (batch, altura, largura, canais)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Data Augmentation (aumenta a diversidade das imagens de treino)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,       # Rotação aleatória até 10 graus
    zoom_range=0.1,          # Pequeno zoom
    width_shift_range=0.1,   # Deslocamento horizontal
    height_shift_range=0.1,  # Deslocamento vertical
    horizontal_flip=False    # MNIST não deve ser invertido horizontalmente
)
datagen.fit(x_train)

# Criar modelo aprimorado
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation=None, input_shape=(28, 28, 1)),  # CNN para extrair características
    tf.keras.layers.BatchNormalization(),  # Normalização para acelerar aprendizado
    tf.keras.layers.LeakyReLU(alpha=0.1),  # Ativação melhora aprendizado

    tf.keras.layers.Conv2D(64, (3, 3), activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Reduz dimensionalidade

    tf.keras.layers.Conv2D(128, (3, 3), activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(256, (3, 3), activation=None),  # Nova camada convolucional
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(0.3),  # Dropout adicional

    tf.keras.layers.Flatten(),  # Achata os dados para a rede densa

    tf.keras.layers.Dense(512, activation=None),  # Mais neurônios
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(0.5),  # Dropout maior para evitar overfitting

    tf.keras.layers.Dense(256, activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10, activation='softmax')  # Camada final com 10 classes (0 a 9)
])

# Compilar modelo com Adam e taxa de aprendizado menor
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks para melhorar o treinamento
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Treinar o modelo
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=15, validation_data=(x_test, y_test),
                    callbacks=[early_stopping, reduce_lr])

# Avaliação do modelo
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

# Salvar modelo treinado
model.save('mnist_model_improved.h5')
# -*- coding: utf-8 -*-

"""

Taller Optimizado: CNN y Fine-Tuning con CIFAR-10

Entorno: Spyder IDE

Estudiante: Daniel B

"""


# ==============================

# 1. LIBRERÍAS

# ==============================

import tensorflow as tf

from tensorflow.keras import layers, models

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt

import numpy as np

import os


# ==============================

# 2. CONFIG GPU

# ==============================

# Configuración para crecimiento dinámico de la memoria GPU

gpus = tf.config.list_physical_devices('GPU')

if gpus:

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(gpu, True)

        print("Crecimiento dinámico de memoria GPU activado.")

    except RuntimeError as e:

        print(f"Error al configurar la GPU: {e}")



# ==============================

# 3. CARGA Y PREPARACIÓN DEL DATASET

# ==============================

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# COnertir matriz multidimensional a matriz plana

y_train = y_train.flatten()

y_test = y_test.flatten()


# Normalización a rango [0, 1]

x_train = x_train / 255.0

x_test = x_test / 255.0


print("\n=== INFORMACIÓN DE LOS DATOS ===")

print(f"Imágenes de entrenamiento: {x_train.shape[0]} con dimensiones {x_train.shape[1:]}")

print(f"Imágenes de prueba (evaluación): {x_test.shape[0]}")


class_names = ['Avión', 'Auto', 'Pájaro', 'Gato', 'Ciervo', 'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']


# ==============================

# 4. DATA AUGMENTATION (FINE-TUNING)

# ==============================

# Definir el aumento de datos como un modelo secuencial de capas.
# Al ser capas nativas, TensorFlow maneja el prefetch automáticamente en GPU.
data_augmentation = models.Sequential([
    layers.RandomRotation(factor=0.055),  # ~20 grados (0.055 * 360)
    layers.RandomTranslation(height_factor=0.15, width_factor=0.15),
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(height_factor=0.1)
])


# ==============================

# 5. CONSTRUCCIÓN DE LA CNN OPTIMIZADA

# ==============================

model = models.Sequential()


# Bloque 1: Con padding='same' para preservar dimensiones espaciales

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))

model.add(layers.BatchNormalization())

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.25))  # Evita sobreajuste temprano


# Bloque 2

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.BatchNormalization())

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.3))


#Bloque 3

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(layers.BatchNormalization())

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.4))


# Bloque de Clasificación Densa

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation='softmax'))


# Compilación con Optimizador Adam

model.compile(

    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)


model.summary()


# Callback para reducir el Learning Rate dinámicamente si el modelo se estanca

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)


# ==============================

# 6. ENTRENAMIENTO DEL MODELO

# ==============================

print("\n--- Iniciando Entrenamiento Optimizado ---")

# Usar menos épocas por defecto para la prueba inicial, puede subirse a 20 o 25-> 30GPU

EPOCHS = 30 

BATCH_SIZE = 256

# TensorFlow se encarga de paralelizar el lote de 256 de forma asíncrona y eficiente.
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=[reduce_lr]
)

# ==============================

# 7. EVALUACIÓN DE PRECISIÓN EN CONSOLAa

# ==============================

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print("\n==============================================")

print("           MÉTRICAS FINALES DEL MODELO        ")

print("==============================================")

print(f"-> Precisión final en Entrenamiento: {history.history['accuracy'][-1]*100:.2f}%")

print(f"-> Precisión final en Validación:    {history.history['val_accuracy'][-1]*100:.2f}%")

print(f"-> PRECISIÓN GENERAL EN TEST (EXAMEN): {test_acc*100:.2f}%")

print(f"-> Pérdida (Loss) en Test:             {test_loss:.4f}")

print("==============================================")


# ==============================

# 8. GRÁFICAS COMPARATIVAS DE RENDIMIENTO

# ==============================

plt.figure(figsize=(14, 5))


# Gráfica de Precisión (Accuracy)

plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'], label='Entrenamiento (Train ACC)', color='blue', linewidth=2)

plt.plot(history.history['val_accuracy'], label='Validación (Val ACC)', color='orange', linewidth=2)

plt.axhline(y=test_acc, color='red', linestyle='--', label=f'Test ACC Final ({test_acc*100:.1f}%)')

plt.title("Evolución de la Precisión durante el Entrenamiento")

plt.xlabel("Épocas")

plt.ylabel("Porcentaje de Precisión")

plt.legend()

plt.grid(True)


# Gráfica de Pérdida (Loss)

plt.subplot(1, 2, 2)

plt.plot(history.history['loss'], label='Pérdida Entrenamiento', color='blue', linewidth=2)

plt.plot(history.history['val_loss'], label='Pérdida Validación', color='orange', linewidth=2)

plt.title("Evolución de la Pérdida (Función de Costo)")

plt.xlabel("Épocas")

plt.ylabel("Valor de Pérdida")

plt.legend()

plt.grid(True)


plt.tight_layout()

plt.show()


# ==============================

# 9. MÓDULO INTERACTIVO DE PREDICCIÓN

# ==============================

print("\n=== SISTEMA DE CLASIFICACIÓN DE IMÁGENES ===")

print("1. Usar una imagen aleatoria del dataset de prueba")

print("2. Cargar una imagen externa desde mi computadora (.jpg, .png)")


opcion = input("Seleccione una opción : ")


img_para_predecir = None

titulo_grafica = ""

clase_real_texto = "N/A (Imagen Externa)"


if opcion == "1":

    # Seleccionar un índice aleatorio de las 10,000 imágenes de prueba

    indice_aleatorio = np.random.randint(0, x_test.shape[0])

    img_para_predecir = x_test[indice_aleatorio]

    clase_real_texto = class_names[int(y_test[indice_aleatorio])]

    titulo_grafica = f"Muestra Dataset - Real: {clase_real_texto}"

    

elif opcion == "2":

    ruta_img = input("Ingrese la ruta completa de la imagen (ej: C:/imagenes/perro.jpg): ")

    if os.path.exists(ruta_img):

        try:

            # Cargar y forzar el redimensionamiento requerido de 32x32

            img_original = image.load_img(ruta_img, target_size=(32, 32))

            # Convertir a matriz y normalizar exactamente igual que el dataset

            img_para_predecir = image.img_to_array(img_original) / 255.0

            titulo_grafica = f"Externa: {os.path.basename(ruta_img)}"

        except Exception as e:

            print(f"Error al procesar el archivo: {e}")

    else:

        print("La ruta ingresada no existe. Se canceló el proceso.")


# Si se logró obtener la imagen , realiza la prediccion

if img_para_predecir is not None:

    # Preparar las dimensiones para la CNN añadiendo el eje del lote: (1, 32, 32, 3)

    datos_entrada = np.expand_dims(img_para_predecir, axis=0)

    

    # Predicción

    prediccion_cruda = model.predict(datos_entrada, verbose=0)

    indice_predicho = np.argmax(prediccion_cruda)

    clase_predicha_texto = class_names[indice_predicho]

    porcentaje_confianza = prediccion_cruda[0][indice_predicho] * 100

    

    # Imprimir resumen de datos relevantes en la consola

    print("\n==============================================")

    print("         DATOS RELEVANTES DE LA PREDICCIÓN    ")

    print("==============================================")

    print(f"Clase Real esperada:   {clase_real_texto}")

    print(f"Clase Predicha por CNN: {clase_predicha_texto}")

    print(f"Porcentaje de Certeza: {porcentaje_confianza:.2f}%")

    print(f"Dimensiones de entrada a la Red: {datos_entrada.shape}")

    print("==============================================")

    

    # Mostrar la imagen analizada con su resultado en la sección de gráficos

    plt.figure(figsize=(5, 5))

    plt.imshow(img_para_predecir)

    plt.title(f"{titulo_grafica}\nPredicción CNN: {clase_predicha_texto} ({porcentaje_confianza:.1f}%)")

    plt.axis('off')

    plt.show()

else:

    print("No se pudo ejecutar la predicción debido a un fallo en la selección de la imagen.")

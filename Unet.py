import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

## Metrik Fonksiyonları
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

## UNet Model Mimarisi
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    # Encoder (Downsampling)
    # Block 1
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Block 2
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Block 3
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Block 4 - Bottleneck
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Decoder (Upsampling)
    # Block 5
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = concatenate([up5, conv3], axis=-1)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    # Block 6
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = concatenate([up6, conv2], axis=-1)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    # Block 7
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = concatenate([up7, conv1], axis=-1)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
    
    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', precision, recall, f1_score, dice_coef])
    
    return model

## Veri Yükleme Fonksiyonu
def load_data(image_dir, mask_dir, target_size=(256, 256)):
    image_files = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])
    
    images = []
    masks = []
    
    for img_file, mask_file in zip(image_files, mask_files):
        # Görüntüyü yükle
        img_path = os.path.join(image_dir, img_file)
        img = load_img(img_path, target_size=target_size)
        img = img_to_array(img) / 255.0
        images.append(img)
        
        # Maskeyi yükle
        mask_path = os.path.join(mask_dir, mask_file)
        mask = load_img(mask_path, target_size=target_size, color_mode='grayscale')
        mask = img_to_array(mask) / 255.0
        masks.append(mask)
    
    return np.array(images), np.array(masks)

## Metrik Grafikleri
def plot_metrics(history):
    plt.figure(figsize=(15, 10))
    
    # Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Precision
    plt.subplot(2, 2, 2)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    # Recall
    plt.subplot(2, 2, 3)
    plt.plot(history.history['recall'], label='Training Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    # F1-Score
    plt.subplot(2, 2, 4)
    plt.plot(history.history['f1_score'], label='Training F1-Score')
    plt.plot(history.history['val_f1_score'], label='Validation F1-Score')
    plt.title('F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

## Ana Fonksiyon
def main():
    # Veri yolları - bunları kendi dosya yollarınızla değiştirin
    image_dir = '/content/drive/MyDrive/CVC-ClinicDB/images'  # Orijinal görüntülerin bulunduğu dizin
    mask_dir = '/content/drive/MyDrive/CVC-ClinicDB/masks'    # Maskelerin bulunduğu dizin
    
    # Verileri yükle
    X, y = load_data(image_dir, mask_dir)
    
    # Verileri train ve test olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modeli oluştur
    model = unet_model()
    model.summary()
    
    # Callback'ler
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ModelCheckpoint('best_unet_model.h5', verbose=1, save_best_only=True)
    ]
    
    # Modeli eğit
    history = model.fit(X_train, y_train,
                        batch_size=8,
                        epochs=50,
                        verbose=1,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks)
    
    # Metrikleri görselleştir
    plot_metrics(history)
    
    # Test verisi üzerinde değerlendirme
    results = model.evaluate(X_test, y_test)
    print("\nTest Sonuçları:")
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test Precision: {results[2]:.4f}")
    print(f"Test Recall: {results[3]:.4f}")
    print(f"Test F1-Score: {results[4]:.4f}")
    print(f"Test Dice Coefficient: {results[5]:.4f}")

if __name__ == "__main__":
    main()

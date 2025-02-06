import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint  # modeli belli aralıklarlar kaydetmek için

# Eğitim veri artırma işlemleri
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
# Eğitim veri seti
training_set = train_datagen.flow_from_directory(
    r' ', # train veri klasör yolu
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Doğrulama veri artırma işlemi
val_datagen = ImageDataGenerator(rescale=1./255)
# Yeni doğrulama veri seti
validation_set = val_datagen.flow_from_directory(
    r' ', # validation veri klasör yolu
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Test veri seti (gerekirse)
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    r'  ', # test verisi klasör yolu
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# CNN Modeli
cnn = tf.keras.models.Sequential()

# 1. Conv Bloğu
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.BatchNormalization())  # Yeni eklendi
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# 2. Conv Bloğu
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# 3. Conv Bloğu
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.BatchNormalization())
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flatten ve Fully Connected Katmanlar
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(128, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Belli aralıklarla modeli kaydetmek için
checkpoint = ModelCheckpoint(
    'CadDog_CNN_model.keras',  # Kaydedilecek dosya ismi
    monitor='val_loss',  # İzlenecek metrik
    save_best_only=True,  # Sadece en iyi modeli kaydet
    mode='min',  # Kayıt için 'val_loss' değerinin en küçük olması gerektiğini belirt
    verbose=1  # Hangi adımda kaydedildiğini yazdırmak için
)

# Modeli derleme
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme (validation verisi ile birlikte)
history = cnn.fit(
    x=training_set,
    validation_data=validation_set,  # Burada validation_set kullanılıyor
    epochs=50,
    callbacks=[checkpoint]
)

# Eğitim ve doğrulama kayıplarını (loss) çizdir
plt.figure(figsize=(12, 5))

# Loss Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Eğitim ve Validation Loss')
plt.legend()

# Accuracy Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Eğitim ve Validation Accuracy')
plt.legend()

plt.show()

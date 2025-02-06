
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1️⃣ Eğitilmiş modeli yükle
model = load_model("CadDog_CNN_model.keras")  # Model dosyanın adını buraya yaz

# 2️⃣ Test edilecek resimlerin bulunduğu klasörü belirt
img_folder = r"  "  # Klasör yolu

# 3️⃣ Klasördeki tüm resimleri döngü ile işle
for img_name in os.listdir(img_folder):
    img_path = os.path.join(img_folder, img_name)  # Her bir resmin tam yolu
    if img_name.endswith(('.jpg', '.jpeg', '.png')):  # Sadece resim dosyalarını al
        # Resmi yükle ve modele uygun hale getir
        test_image = image.load_img(img_path, target_size=(64, 64))  # Modelin giriş boyutuna getir

        # Orijinal görüntüyü sakla (görüntüleme için)
        original_image = image.load_img(img_path)

        # Görüntüyü modele uygun hale getir
        test_image_array = image.img_to_array(test_image)  # Görüntüyü numpy dizisine çevir
        test_image_array = np.expand_dims(test_image_array, axis=0)  # Modelin beklediği 4D forma getir (1, 64, 64, 3)
        test_image_array = test_image_array / 255.0  # Normalizasyon

        # Modelden tahmin al
        result = model.predict(test_image_array)

        # Sonucu yorumla (Binary Classification: 0 = Cat, 1 = Dog)
        if result[0][0] >= 0.5:
            prediction = "Dog"
        else:
            prediction = "Cat"

        # Sonucu ekrana yazdır ve resmi göster
        plt.imshow(original_image)
        plt.title(f"Tahmin: {prediction} ({img_name})")
        plt.axis("off")  # Eksenleri kaldır
        plt.show()

        print(f"Resim: {img_name} - Tahmin Edilen Sınıf: {prediction}")


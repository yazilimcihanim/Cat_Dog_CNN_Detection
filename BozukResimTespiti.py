from PIL import Image
import os

source_folder = r"  "  # Kendi klasör yolunu buraya yaz

valid_images = []
invalid_images = []

# Klasördeki dosyaları kontrol et
for filename in os.listdir(source_folder):
    file_path = os.path.join(source_folder, filename)

    try:
        with Image.open(file_path) as img:
            img.verify()  # Resmin geçerli olup olmadığını doğrula
            valid_images.append(filename)
    except (IOError, SyntaxError) as e:
        print(f"Geçersiz resim dosyası bulundu: {filename}")
        invalid_images.append(filename)

# Geçerli resimler
print("Geçerli resimler:", valid_images)
# Geçersiz resimler
print("Geçersiz resimler:", invalid_images)

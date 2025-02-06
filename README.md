# Cat_Dog_CNN_Detection

Kedi-Köpek Tespit Modeli
  Bu proje, bir Convolutional Neural Network (CNN) modeli kullanarak kedi ve köpekleri tespit etmek için geliştirilmiştir. 
Model, 35.000 görüntü verisi üzerinde eğitilmiş olup, %91 doğruluk oranı ile başarılı bir şekilde sınıflandırma yapmaktadır.
Ayrıca, farklı tipteki bozuk verileri tespit etmek için ek bir Python dosyası da içermektedir.

İçindekiler
    -Model Eğitimi
    -Bozuk Resim Tespiti
    -Model Testi
  
Model Eğitimi
  Proje, bir CNN (Convolutional Neural Network) modelini kullanarak kedi ve köpekleri tespit etmek amacıyla eğitilmiştir. 
Model, 35.000 görüntü verisi ile eğitilmiş olup, yüksek doğruluk oranına sahiptir (%91 doğruluk).

Eğitim süreci sırasında, modelin en iyi versiyonunun kaybolmaması için checkpoint yöntemi kullanılmıştır.
Bu, modelin en iyi performans gösteren ağırlıklarının kaydedilmesini ve eğitim süreci sırasında olası hatalardan dolayı kaybolmalarını önler.

Kullanılan Kütüphaneler
    -TensorFlow / Keras
    -Numpy
    -Matplotlib
    -OpenCV

Bozuk Resim Tespiti
  Projede, farklı tipteki bozuk verileri tespit etmek için bir Python dosyası geliştirilmiştir. 
Bu dosya, görsellerin bozuk olup olmadığını kontrol eder ve bozuk olanları ayıklar.

Bozuk görsellerin tespitini yapan dosyanın adı: BozukResimTespiti.py'dir. Bu dosya, görsellerin doğruluğunu kontrol ederek veri setindeki hatalı resimleri bulmanıza yardımcı olur.

Model Testi
  Modelin doğruluğunu test etmek ve yeni veriler üzerinde tahmin yapmak amacıyla, model_testi.py adlı bir dosya oluşturulmuştur. 
  Bu dosya, verilen bir klasördeki resimleri yükleyip, modelin bu resimleri sınıflandırmasını sağlar.

model_testi.py dosyasını kullanarak modelinizi test edebilir ve tahminler yapabilirsiniz.




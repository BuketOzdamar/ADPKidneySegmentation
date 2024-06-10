Otozomal Dominant Polikistik Böbrek Hastalığında Derin Öğrenme ile Böbrek Segmentasyonu

Polikistik böbrek hastalığı temel olarak otozomal dominant ve otozomal resesif olarak ikiye ayrılır. Biz bu proje için otozomal dominant polikistik böbrek hastalığını ele alıyoruz. Bu hastalığın tespit edilmesinde en büyük rol oynayan kriter böbreğin hacmidir. Bu böbrek hacmini hesaplayabilmek için de öncelikle alınan görüntülerde böbrek segmentasyonu yapmalıyız.

Medikal alanlarda segmentasyon işlemi için en çok tercih edilen modellerden U-Net modeli. U-Net modeli oldukça başarılı bir model olmasına karşın zaman geçtikçe yeni modüller eklenerek daha başarılı ve hızlı hale getirilmeye çalışılmıştır. Araştrmalarım sonucunda U-Net ile birlikte Attention-UNet, SpatialAttention-UNet ve Half-UNet modellerinde eğitim yapılarak bir karşılaştırma yapılmıştır. Ek olarak veri arttırmadan önce ve veri arttırma sonrası olarak da bir kıyas yaptık.

Model eğitimlerinde en başarılı parametreleri kullanabilmek için de "grid search" uygulaması gerçekleştirerek çeşitli parametrelerde eğitim yaparak en başarılı sonuçları veren parametreleri bulmaya çalıştık.
![resim](https://github.com/BuketOzdamar/ADPKidneySegmentation/assets/78095286/14ea1b7a-00d6-4693-9bcb-7074baf8f9d9)


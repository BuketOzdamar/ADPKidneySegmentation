Otozomal Dominant Polikistik Böbrek Hastalığında Derin Öğrenme ile Böbrek Segmentasyonu

Polikistik böbrek hastalığı temel olarak otozomal dominant ve otozomal resesif olarak ikiye ayrılır. Biz bu proje için otozomal dominant polikistik böbrek hastalığını ele alıyoruz. Bu hastalığın tespit edilmesinde en büyük rol oynayan kriter böbreğin hacmidir. Bu böbrek hacmini hesaplayabilmek için de öncelikle alınan görüntülerde böbrek segmentasyonu yapmalıyız.

Medikal alanlarda segmentasyon işlemi için en çok tercih edilen modellerden U-Net modeli. U-Net modeli oldukça başarılı bir model olmasına karşın zaman geçtikçe yeni modüller eklenerek daha başarılı ve hızlı hale getirilmeye çalışılmıştır. Araştrmalarım sonucunda U-Net ile birlikte Attention-UNet, SpatialAttention-UNet ve Half-UNet modellerinde eğitim yapılarak bir karşılaştırma yapılmıştır. Ek olarak veri arttırmadan önce ve veri arttırma sonrası olarak da bir kıyas yaptık.

Model eğitimlerinde en başarılı parametreleri kullanabilmek için de "grid search" uygulaması gerçekleştirerek çeşitli parametrelerde eğitim yaparak en başarılı sonuçları veren parametreleri bulmaya çalıştık.

![resim](https://github.com/BuketOzdamar/ADPKidneySegmentation/assets/78095286/14ea1b7a-00d6-4693-9bcb-7074baf8f9d9)

![resim](https://github.com/BuketOzdamar/ADPKidneySegmentation/assets/78095286/369ef8cb-a7fa-4dfe-a517-0d7cef27ca48)
![resim](https://github.com/BuketOzdamar/ADPKidneySegmentation/assets/78095286/de8f5aaa-d294-4274-b02d-15b8a8153459)
![resim](https://github.com/BuketOzdamar/ADPKidneySegmentation/assets/78095286/e0bb7470-0ae3-4b04-b5a4-42a8a0b0296d)
![resim](https://github.com/BuketOzdamar/ADPKidneySegmentation/assets/78095286/8334a019-700c-4aa8-935d-dce7f2da5a77)

Tahmin sonuçları
![resim](https://github.com/BuketOzdamar/ADPKidneySegmentation/assets/78095286/818202e4-c52b-45c6-9dc9-09e4fee7701d)
![resim](https://github.com/BuketOzdamar/ADPKidneySegmentation/assets/78095286/08fda0c7-4361-46f4-886b-446efa4cb6cd)
![resim](https://github.com/BuketOzdamar/ADPKidneySegmentation/assets/78095286/22851f2c-a9e3-4f51-ac7c-df16e0cf9d08)
![resim](https://github.com/BuketOzdamar/ADPKidneySegmentation/assets/78095286/4fbbfd5a-06ba-4231-8e90-806f2dd441dd)



# I/B/E/S Analist Verisi: Dosya ve Değişken Rehberi


## 1. Kapsam

- Başlangıç listesinde 935 kayıt, tekrarlar çıkarıldığında 804 farklı enstrüman kodu var. Enstrüman kodu sayısı, farklı şirket sayısıyla her zaman aynı değil.
- Toplam 29 değişken istendi: 26'sı tarihsel seri, 3'ü ise tek seferlik alınan bilgiler (snapshot).
- 737 enstrüman için tarihsel kayıt geldi. Bunların 563'ünde istenen alanların tamamı, 174'ünde ise bir kısmı döndü. Bir alanın dönmüş olması, o alanın bütün tarihlerde dolu olduğu anlamına gelmiyor.
- 67 enstrümanın tarihsel sorgusu sonuç vermedi: 65'inde geçersiz kod/ifade, 2'sinde veri bulunamaması hatası var.
- Tarihler 26 Ekim 2000 ile 26 Ekim 2026 arasında. Üç ayda bir olmak üzere 105 gözlem tarihi var.
- Tarihler Ocak, Nisan, Temmuz ve Ekim aylarının 26. gününe denk geliyor. Bunlar sorgunun gözlem tarihleri; çeyrek sonu veya EPS'nin ait olduğu mali dönem bitiş tarihleri değil.
- İndirme, başlangıçta verilen kod listesi üzerinden yapıldı.

## 2. Klasör Yapısı ve Dosyalar

```text
ibes_production_output/
    ibes_historical_unique.csv
    ibes_snapshot_unique.csv
    ibes_surdt_snapshot_unique.csv
    historical_field_coverage.csv
    query_status.csv
    original_input_manifest.csv
    run_config.json
    checkpoints/
        historical/
        snapshot/
        surdt_snapshot/
```

### ibes_historical_unique.csv

Tarihsel verilerin birleştirilmiş dosyası. Toplam 1.917.300 satır var. Her satır, bir enstrümanın belirli bir tarihteki tek bir değişkenini gösteriyor. Dosya uzun formatta: değişkenin adı `Field`, değeri ise `Value` sütununda.

Örneğin aynı şirketin aynı tarihteki ortalama ve medyan EPS tahminleri ayrı satırlarda. Aynı enstrüman, tarih ve değişken için tekrarlanan satır yok.

### ibes_snapshot_unique.csv

Burada `IBQ1CRDT`, `IBTKR` ve `IBNAME` bilgilerinin tek seferlik alınmış değerleri var. 804 enstrüman için toplam 2.412 satır bulunuyor; bazıları boş.

### ibes_surdt_snapshot_unique.csv

`SURDT` için ayrıca yapılan tek seferlik sorgunun sonuçları burada. Kılavuz bu alanı yalnızca güncel değeri tutulan bir alan olarak tanımlıyor. Tarihsel dosyada aynı kodla dönen kayıtlar, bu ayrı snapshot dosyasının parçası değil.

### historical_field_coverage.csv

Her enstrüman ve değişken için gelen kayıtların ve boş değerlerin sayısını gösteriyor:

| Sütun | Açıklama |
|---|---|
| `RequestedInstrument` | Sorgulanan enstrüman kodu. |
| `Field` | İstenen değişken kodu. |
| `DatesReturned` | İlgili alan için dönen tarih sayısı. |
| `NonEmpty` | Dolu değer sayısı. |
| `Missing` | Dönen kayıtlar içindeki eksik değer sayısı. |
| `APIErrorCells` | Hücre düzeyinde işaretlenmiş API hatası sayısı. |
| `FieldReturned` | Alanın yanıtta dönüp dönmediğini gösteren işaret. |

### query_status.csv

İndirme sırasında hangi sorguların çalıştığını, hangilerinin eksik veya hatalı sonuç verdiğini burada görebiliyoruz. `Mode` sütunu sorgunun türünü gösteriyor: `historical`, `snapshot` veya ayrı SURDT sorgusu için `surdt_snapshot`.

- `HAS_DATA`: Veri geldi; ancak arada boş değerler olabilir.
- `PARTIAL_FIELDS`: İstenen değişkenlerin bir kısmı gelmedi.
- `API_RESPONSE_ERROR`: Beklenen tarihsel veri alınamadı. Hata açıklaması `Detail` sütununda.

`Rows` satır sayısını, `NonEmptyCells` dolu hücre sayısını, `APIErrorCells` hücrelerdeki hata sayısını, `MissingFields` gelmeyen değişkenleri gösteriyor. Hiç veri getirmeyen sorguların hataları da burada tutuluyor.

### original_input_manifest.csv

İndirme için kullanılan başlangıç listesi. `InputPosition` listedeki sırayı, `RequestedInstrument` ise sorgulanan kodu gösteriyor. Aynı kod listede birden fazla kez yer alabiliyor.

### run_config.json

İndirmenin ayarları burada: 26 yıl geriye gitme (`start: -26Y`), üç aylık sıklık (`frequency: Q`), istenen değişkenler ve enstrüman listesinin özeti. İlk çalıştırma tarihi 27 Ekim 2026 olarak kaydedilmiş.

### checkpoints/

Bunlar indirme sırasında oluşturulan ara CSV dosyaları ve JSON sorgu kayıtları. `historical/`, `snapshot/` ve `surdt_snapshot/` alt klasörleri ilgili sorgu türlerine ait kayıtları içeriyor. Birleştirilmiş dosyalarla aynı sorgulara ait olduklarından, bağımsız ek gözlemler değiller.

## 3. Ana Veri Dosyasının Sütunları

| Sütun | Açıklama |
|---|---|
| `RequestedInstrument` | Sorguya gönderilen enstrüman kodu. |
| `ReturnedInstrument` | Veri sağlayıcısının yanıtta döndürdüğü enstrüman kodu. |
| `Date` | Tarihsel sorgunun gözlem tarihi. Snapshot dosyalarında bulunmaz. |
| `Field` | Değişkenin kodu. |
| `Currency` | Sağlayıcının döndürdüğü para birimi kodu; bazı alanlarda boş olabilir. |
| `Value` | Değişkenin değeri. Alana göre sayı, tarih veya metin içerir. |
| `IsAPIError` | Hücrenin API hatası olarak işaretlenip işaretlenmediği. |

`Value` sütununda değişkene göre sayı, tarih veya metin bulunuyor. Enstrüman kodları `@` ve `:` gibi karakterler içeren metin kodları.

## 4. Değişken Açıklamaları

EPS, hisse başına kâr anlamına geliyor. `MN` ortalama, `MD` medyan, `NE` tahmin sayısı, `SD` standart sapma.

**Yıllık tahminler:** FY1, sonucu henüz açıklanmamış sıradaki mali yıl. `EPS1…` kodları bu döneme ait. Tanımlar, Thomson Reuters'ın [I/B/E/S on Datastream kılavuzunda](https://libapp.lib.ncku.edu.tw/libref/handout/20110107_IBES_user_guide.pdf), s. 22–23'te yer alıyor.

| Kod | Tanım |
|---|---|
| `EPS1MN` | FY1 EPS tahminlerinin ortalaması. |
| `EPS1MD` | FY1 EPS tahminlerinin medyanı. |
| `EPS1NE` | FY1 konsensüs ortalamasına dahil edilen tahmin sayısı. |
| `EPS1SD` | FY1 konsensüs tahminlerinin standart sapması. |

**Ara dönem tahminleri:** INT1, INT2 ve INT3 sırasıyla sonucu açıklanacak birinci, ikinci ve üçüncü ara dönemi gösteriyor. ABD çeyreklik serilerinde bunlar sıradaki üç mali çeyrek. İndirme sıklığından farklı olarak, bu kodlar tahminin hangi döneme ait olduğunu belirtiyor. Refinitiv'in [2020 kılavuzu](https://www.unisg.ch/fileadmin/user_upload/HSG_ROOT/_Kernauftritt_HSG/Universitaet/Bibliothek/Suchen_und_Nutzen/Datenbanken/Datenbankseiten/A-Z/IBES_User_guide_DFO_202011.pdf), s. 8–9.

| Kod | Tanım |
|---|---|
| `EPSI1MN` | INT1 EPS tahmin ortalaması. |
| `EPSI1MD` | INT1 EPS tahmin medyanı. |
| `EPSI1SD` | INT1 EPS tahmin standart sapması. |
| `EPSI1NE` | INT1 EPS tahmin sayısı. |
| `EPSI2NE` | INT2 EPS tahmin sayısı. |
| `EPSI2MD` | INT2 EPS tahmin medyanı. |
| `EPSI2MN` | INT2 EPS tahmin ortalaması. |
| `EPSI2SD` | INT2 EPS tahmin standart sapması. |
| `EPSI3MN` | INT3 EPS tahmin ortalaması. |
| `EPSI3SD` | INT3 EPS tahmin standart sapması. |
| `EPSI3NE` | INT3 EPS tahmin sayısı. |
| `EPSI3MD` | INT3 EPS tahmin medyanı. |

**Açıklanmış EPS değerleri:** Aşağıdaki sekiz alan tahmin değil, raporlanmış EPS değerleri. Dönemler gözlem tarihine göre göreli: INT0 son açıklanan ara dönem, FY0 son açıklanan mali yıl. Thomson Reuters [kılavuzu](https://libapp.lib.ncku.edu.tw/libref/handout/20110107_IBES_user_guide.pdf), s. 26–27.

| Kod | Tanım |
|---|---|
| `I0EPS` | Son açıklanan ara dönem EPS'si (INT0). |
| `IP1EPS` | INT0'dan bir ara dönem önceki EPS (INT−1). |
| `IP2EPS` | INT0'dan iki ara dönem önceki EPS (INT−2). |
| `IP3EPS` | INT0'dan üç ara dönem önceki EPS (INT−3). |
| `F0EPS` | Son açıklanan yıllık EPS (FY0). |
| `FP1EPS` | FY0'dan bir mali yıl önceki yıllık EPS (FY−1). |
| `FP2EPS` | FY0'dan iki mali yıl önceki yıllık EPS (FY−2). |
| `FP3EPS` | FY0'dan üç mali yıl önceki yıllık EPS (FY−3). |

**Tarihler ve şirket bilgileri:**

| Kod | Tanım | Tanımın kaynağı |
|---|---|---|
| `SURDT` | Son ara dönem sonuçlarının açıklanma tarihi. | Refinitiv 2020 kılavuzu, s. 23. |
| `EPSANCDT` | Son gerçekleşen EPS'nin açıklanma tarihi; sisteme kayıt tarihi değil. Sağlayıcı, resmi duyuru tarihiyle her zaman aynı olmayabileceğini belirtiyor. | Thomson Reuters, Infostream Q1 2014, s. 35; aşağıdaki bağlantı. |
| `IBQ1CRDT` | Şirketin bildirdiği, sıradaki çeyrek sonuçları için beklenen açıklama tarihi. | Refinitiv 2020 kılavuzu, s. 12. |
| `IBTKR` | I/B/E/S'nin şirket için kullandığı benzersiz ticker kodu. | Refinitiv 2020 kılavuzu, s. 6. |
| `IBNAME` | Dosyada I/B/E/S şirket adı olarak dönüyor. Bu kodun resmi tanımı erişilebilir kılavuzda bulunamadı; açıklama, paylaşılan görsele ve dosya içeriğine dayanıyor. | Paylaşılan görsel ve snapshot dosyası; resmi kod tanımı doğrulanamadı. |

Tarih ve ticker tanımları: [Refinitiv 2020 kılavuzu](https://www.unisg.ch/fileadmin/user_upload/HSG_ROOT/_Kernauftritt_HSG/Universitaet/Bibliothek/Suchen_und_Nutzen/Datenbanken/Datenbankseiten/A-Z/IBES_User_guide_DFO_202011.pdf). `EPSANCDT` tanımı: [Thomson Reuters Infostream Q1 2014 bülteninin çevrimiçi kopyası](https://studylib.net/doc/8881770/macro-explorer-in-thomson-reuters-eikon-the-new-visualisa...).

Doğrulama özeti: 27 alanın tanımı sağlayıcının kılavuzlarında, `EPSANCDT` tanımı sağlayıcının bülteninin çevrimiçi kopyasında bulundu. `IBNAME` kodu için resmi tanım bulunamadı. Canlı Datastream veri sözlüğü oturum açma ekranı arkasında.

İlk 26 alan tarihsel sorgu listesindedir. Son üç alan (`IBQ1CRDT`, `IBTKR`, `IBNAME`) snapshot olarak istenmiştir. `SURDT` ayrıca ayrı bir snapshot dosyasında da bulunmaktadır.

## 5. Eksik Değerler

Tarihsel dosyada 634.051 dolu ve 1.283.249 boş değer vardır. Dönen satırlar içindeki boşluk oranı yaklaşık %66,9'dur. Bu oran, hiç satır dönmeyen alanları veya başarısız sorguları içermez.

Boş değerlerin dağılımı:

| Boşluğun konumu | Sayı | Tüm boş değerler içindeki payı |
|---|---:|---:|
| Enstrümanın herhangi bir alandaki ilk dolu değerinden önce | 1.131.495 | %88,17 |
| İlk ve son dolu değer arasındaki tarih aralığında | 151.437 | %11,80 |
| Son dolu değerden sonra | 317 | %0,02 |

Boşlukların çoğu ilgili enstrümanın ilk mevcut verisinden önceki tarihlerde. Örneğin verisi 2020'de başlayan bir enstrüman için sorguda 2000'e kadar uzanan tarihler de bulunuyor. Verinin geç başlamasının nedeni dosyalarda belirtilmiyor.

- İlk ve son mevcut değer arasındaki dönen kayıtlarda eksiklik yaklaşık %19,3'tür; bu hesap çok seyrek olan `SURDT` alanını da içerir.
- `SURDT`, aynı tarih aralığında yaklaşık %98 boş. Refinitiv'in 2020 kılavuzunda bu alanın yalnızca güncel değerinin tutulduğu, tarihsel serisinin bulunmadığı belirtiliyor. Bu, dosyada tarihsel sorgu listesine dahil edilmiş olmasından ayrı bir özellik.
- Kontrol edilen dört standart sapma/tahmin sayısı çiftinde, tahmin sayısı en az iki olduğunda standart sapma eksik değil. Tek veya daha az tahmin bulunan kayıtlarda eksik standart sapmalar var.
- Boş hücre, sıfır değeri değil; ilgili kayıt için bir değer dönmediğini gösteriyor.
- Başarısız sorgular `query_status.csv` içinde kayıtlı. Yukarıdaki boşluk oranı yalnızca tarihsel dosyada dönen satırları kapsıyor.

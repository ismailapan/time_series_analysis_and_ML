
# Haftalık Talep Tahmin Sistemi

Bu doküman, haftalık ürün bazlı talep tahmin sisteminin inşa sürecinde alınan kararları, bulunan bulguları ve bunların gerekçelerini özetler.

## Amaç

Mevcut aylık tahmin sisteminin yerine, ürün × hafta granülerliğinde **native (doğrudan haftalık)** bir talep tahmin sistemi kurmak. Aylık tahminleri haftaya bölmek değil, en baştan haftalık veri üzerinde eğitilen bir sistem.

---

## 1. Veri ve Feature Engineering

- Veri, `[product, date, quantity]` yapısında, Pazartesi başlangıçlı haftalara agregasyonla hazırlandı.
- Feature'lar, **pivot-wide** (ürün = kolon) bir yaklaşımla, tamamen `lag_1` (bir önceki haftanın gerçek değeri) üzerinden hesaplandı — bu yapı, groupby+shift tabanlı yaklaşımlarda görülebilecek ürün-sınırı sızıntısına karşı yapısal olarak bağışık.
- 25 temel + birkaç aday feature: lag, rolling mean/std (4 ve 12 hafta), EMA (fast/slow), MACD, SBC sınıflandırması (`adi`, `cv2`, `sbc_category`), Croston tahmini, talep-arası süre istatistikleri, Fourier terimleri (mevsimsellik), yakın dönem trend.
- **Warm-up filtresi**: Ürünün kendi geçmişinin ilk 12 haftası (`history_weeks >= 12`) modelden çıkarıldı — rolling/lag feature'ları bu dönemde güvenilmez.

## 2. Doğrulama Metodolojisi

### Bulunan ve düzeltilen hatalar
- İlk denemede train/val/test aralıkları **örtüşüyordu** (`df['week'] < X` hepsi için) — kronolojik, dışlayıcı (exclusive) aralıklara çevrildi.
- Warm-up filtresi (`df2`) hesaplanıyor ama hiç kullanılmıyordu (ölü kod) — gerçek filtreye bağlandı.
- `training.py`'de RMSSE hesaplama döngüsünde yanlış tablodan (`val` yerine `pred_df`) kolon çekiliyordu; ayrıca `rmsse_df` her fold'da eziliyordu — ikisi de düzeltildi.

### 3-fold expanding-window CV
- Train her fold'da genişliyor (bir önceki fold'un validation'ı, sonraki fold'un train'ine dahil oluyor), validation pencereleri örtüşmüyor.
- CV'nin amacı modele "öğretmek" değil — modelin farklı zaman dilimlerinde **tutarlı** performans gösterip göstermediğini doğrulamak.

### Ham WAPE'nin yanıltıcılığı
- Satır bazlı (ürün, hafta) WAPE, sıfır-tahmininden (WAPE=1.0) bile kötü çıktı (1.29).
- Sebep modelin kalitesi değil, **ölçüm granülerliği**: intermittent talep yapısında, modelin "doğru büyüklükte ama yanlış haftada" tahmin etmesi, ham haftalık karşılaştırmada ağır şekilde cezalandırılıyor.
- Çözüm: **4 haftalık blok** agregasyonu — modelin tahmin ufkunu (hâlâ haftalık) değiştirmeden, sadece değerlendirme granülerliğini değiştiriyor. Haftalık (çok gürültülü) ile yıllık toplam (gerçek sorunları gizleyebilen) arasında bir denge noktası.

### win_rate metriği
- Agregat WAPE/Teknorot, hacim-ağırlıklı olduğu için az sayıda yüksek hacimli ürün tarafından domine edilebiliyor.
- `win_rate`: her (ürün, blok) biriminde modelin mi baseline'ın mı daha iyi olduğunu eşit ağırlıklı sayan metrik — agregat sonucun "genele yayılmış mı yoksa birkaç büyük ürün mü sürüklüyor" sorusunu cevaplıyor.

## 3. Segment Bazlı Bulgu

3-fold CV'de, `sbc_category` (Smooth / Intermittent / Erratic / Lumpy / LowData) bazında win_rate tutarlı bir patern gösterdi:

| Segment | Model vs mean52 (baseline) |
|---|---|
| LowData | Baseline kazanıyor (~%7 win_rate, çok tutarlı) |
| Intermittent | Baseline kazanıyor (~%39-46) |
| Erratic | Model kazanıyor (~%52-58) |
| Lumpy | Model kazanıyor (~%54-62) |
| Smooth | Belirsiz (çok küçük örneklem) |

Bu bulgu, hem WAPE hem Teknorot (asimetrik iş metriği) altında ayrı ayrı doğrulandı ve neredeyse birebir aynı çıktı — metrik seçiminden bağımsız, yapısal bir sonuç.

## 4. Asimetrik İş Metriği: Teknorot

```
Teknorot = (UNDER_PENALTY × Σmax(actual-pred,0) + OVER_PENALTY × Σmax(pred-actual,0)) / Σactual
```

- Az-tahmin (stoksuzluk) 2x, çok-tahmin (fazla stok) 1x cezalandırılıyor.
- Sıfır-tahmin her zaman `Teknorot = UNDER_PENALTY` verir (algebrik sabit nokta, sanity-check).
- `UNDER_PENALTY`/`OVER_PENALTY` oranı bir iş politikası parametresi — gerçek stoksuzluk/fazla-stok maliyet oranı işletmeden doğrulanmalı, şu an varsayılan (2:1) kullanılıyor.

## 5. Segment Router

Global tek-model yaklaşımı yerine (tüm ürünlere aynı tahmin yöntemini uygulamak), segment bazında karar veren bir yönlendirme mekanizması kuruldu:

```
Erratic      → model
Intermittent → baseline (mean52)
LowData      → baseline (mean52)
Lumpy        → model
Smooth       → model
```

### RMSSE Guardrail
Her segment kararı, Teknorot'un yanı sıra **RMSSE** (ölçekten bağımsız, simetrik bir istatistiksel doğruluk metriği) ile çapraz doğrulandı — Teknorot'un seçtiği yöntem, istatistiksel olarak en iyi alternatiften %5'ten fazla kötüyse karar geçersiz kılınıyor. Mevcut veride guardrail hiç tetiklenmedi — iş metriği (Teknorot) ile istatistiksel metrik (RMSSE) her segmentte aynı yöne işaret etti.

### Doğrulama sonucu (3-fold CV ortalaması)

| Yöntem | Ort. WAPE | Ort. Teknorot |
|---|---|---|
| **Router** | **0.944** | **1.479** |
| Model (saf) | 0.957 | 1.481 |
| %50/%50 blend | 0.965 | 1.485 |
| Baseline (saf) | 0.985 | 1.508 |

Router, WAPE'de 3 fold'un 3'ünde de en düşük hatayı verdi.

## 6. Final Model ve Tek Kullanımlık Test

- CV'den sonra kullanılmamış ~31 haftalık veri üçe bölündü: `final_train` (gerçek eğitim), `internal_val` (8 hafta, sadece early-stopping için), `test` (~14 hafta, tek kullanımlık).
- Early stopping ile optimal ağaç sayısı bulundu: **92** (varsayılan `n_estimators=1000`'in ~%9'u — erken durdurma olmadan ciddi overfitting riski olduğunu gösteriyor).
- Final model, bu sabit iterasyon sayısıyla, tüm mevcut veri (`internal_val` dahil) üzerinde yeniden eğitildi.
- **Test sonucu (tek seferlik ölçüm):**

| Yöntem | WAPE | Teknorot |
|---|---|---|
| **Router** | **0.879** (en düşük) | 1.370 |
| Model (saf) | 0.890 | **1.362** (en düşük) |
| %50/%50 blend | 0.894 | 1.392 |
| Baseline (saf) | 0.912 | 1.441 |

Router'ın WAPE üstünlüğü, hiç görmediği veride de teyit edildi. Teknorot'ta ise saf model, router'dan çok küçük bir farkla (~%0.6) daha iyi çıktı — bu, tek bir test penceresinin doğal varyansı olarak değerlendirildi; test setinin disiplinini korumak için bu sonuca göre geriye dönük ayar değişikliği yapılmadı.

---

## Henüz Yapılmayanlar / Bilinen Eksikler

- **Recursive (52 haftalık) tahmin mekanizması kurulmadı.** Şu ana kadarki tüm değerlendirmeler tek-adım (gerçek lag değerleriyle) tahmine dayanıyor. Çok-adımlı, modelin kendi çıktısını besleyeceği bir zincirleme tahmin sistemi ve bununla gelen hata-birikimi stabilizasyonu (feature dondurma/blend) henüz tasarlanmadı.
- **Gerçek UNDER_PENALTY/OVER_PENALTY oranı doğrulanmadı.** Şu an varsayılan (2:1) kullanılıyor; gerçek stoksuzluk vs fazla-stok maliyet oranı işletmeden teyit edilmeli.
- **Smooth segmenti için yeterli örneklem yok** — mevcut router kararı (model) düşük güvenle verilmiş durumda, daha fazla veri biriktikçe tekrar değerlendirilmeli.

📈 Forecasting Agent
Overview
Forecasting Agent, kullanıcıların kendi veri setleri üzerinden zaman serisi tahminleri (time series forecasting) yapabilmesini sağlayan, teknik bilgi gerektirmeyen, uçtan uca bir tahminleme çözümüdür.
Bu agent; veri yükleme, otomatik veri analizi, çoklu model karşılaştırması, tahmin üretimi ve sonuçların görselleştirilmesi süreçlerini tek bir arayüz üzerinden yönetir.
AI Hub içerisinde, veri odaklı karar alma süreçlerini hızlandırmak ve iş birimlerini self-service AI ile güçlendirmek amacıyla geliştirilmiştir.
🎯 Project Purpose
Kullanıcıların geçmiş verilerden geleceğe yönelik tahminler üretmesini sağlamak
Tahmin süreçlerini teknik ekip bağımlılığı olmadan erişilebilir hale getirmek
Farklı forecasting modellerini otomatik olarak karşılaştırmak
Belirsizlik aralıkları ile birlikte güvenilir tahminler sunmak
İş kararlarını veri odaklı hale getirmek
👤 Target Users
İş birimleri (satış, pazarlama, operasyon, finans)
Veri analistleri
Teknik olmayan kullanıcılar
Karar vericiler
⚙️ Core Capabilities
1. Data Upload
CSV / XLSX veri yükleme desteği
Otomatik veri okuma ve ön analiz
2. Smart Column Selection
Tarih kolonu seçimi
Hedef (forecast edilecek) kolon seçimi
Opsiyonel segment ve dışsal değişken seçimi
3. Data Quality Checks
Eksik veri kontrolü
Tarih sıralama kontrolü
Eksik zaman aralığı tespiti
Duplicate kayıt kontrolü
Outlier tespiti
Veri uzunluğu yeterlilik analizi
4. Automatic Feature Engineering
Tarih bazlı feature üretimi (ay, gün, hafta vb.)
Lag feature oluşturma
Rolling average / trend feature’ları
Sezonluk pattern çıkarımı
5. Multi-Model Forecasting
Aynı veri üzerinde birden fazla model çalıştırılır:
Naive Baseline
SARIMA
Prophet
XGBoost (feature-based forecasting)
6. Model Evaluation
Her model aşağıdaki metriklerle değerlendirilir:
MAE
RMSE
MAPE
En iyi model otomatik olarak seçilir.
7. Forecast Output
Gelecek dönem tahminleri
Confidence interval (alt-üst bant)
Actual vs Forecast grafikleri
🧠 Advanced Forecasting Features (Phase 2 Included)
📅 Holiday Calendar Integration
Ülkeye özel resmi tatiller otomatik olarak modele dahil edilir
Tatil etkileri forecast üzerinde dikkate alınır
🎯 Special Events / Campaign Flags
Kampanya, indirim, özel gün gibi etkiler modele feature olarak eklenir
Kullanıcı manuel event girişi yapabilir
🏢 Hierarchical Forecasting
Üst-alt kırılım desteklenir
Örnek:
Toplam satış → mağaza bazlı → ürün bazlı
Hem global hem segment bazlı forecast üretilebilir
📊 Multi-Series Batch Forecasting
Aynı anda birden fazla zaman serisi için tahmin üretimi
Örnek:
Her mağaza için ayrı forecast
Her müşteri segmenti için ayrı forecast
⚠️ Anomaly-Aware Forecasting
Geçmiş verideki anomaliler tespit edilir
Model bu anomalilere göre:
ignore edebilir
düzeltebilir
kullanıcıya raporlayabilir
🔮 What-if Simulation
Kullanıcı senaryo bazlı tahmin yapabilir:
“Satış %20 artarsa ne olur?”
“Kampanya eklenirse forecast nasıl değişir?”
“Fiyat düşerse etkisi ne olur?”
Bu sayede forecasting → decision support sistemine dönüşür
📊 Output & Visualization
Forecast grafikleri (actual vs predicted)
Confidence interval görselleştirmesi
Model karşılaştırma tablosu
Trend & seasonality analizi
📁 Export Options
Excel çıktısı aşağıdaki sheet’leri içerir:
Forecast Results
Model Metrics
Data Quality Report
Forecast Summary
🖥️ UI Flow
Veri yükleme (CSV/XLSX)
Tarih ve hedef kolon seçimi
Veri kalite analiz ekranı
Forecast horizon belirleme
Model çalıştırma
Sonuç ekranı:
Grafikler
Model karşılaştırma
Tahmin çıktıları
Excel export
⚠️ Limitations & Considerations
Veri çok kısa ise tahmin güvenilirliği düşer
Ani kırılmalar (kampanya, kriz vb.) modeli yanıltabilir
Uzun vadeli forecastlarda hata payı artar
Eksik veya düzensiz veri sonuçları etkiler
Agent gerektiğinde kullanıcıyı bu riskler hakkında uyarır
🚀 Future Improvements
LLM destekli forecast yorumlama
Otomatik anomaly düzeltme
Real-time forecasting
API entegrasyonu
Dashboard entegrasyonu
Auto retraining pipeline
🧩 AI Hub Integration
Forecasting Agent, AI Hub içerisindeki diğer agentlar ile entegre çalışabilir:
Data Quality Agent → veri temizliği
Feature Engineering Agent → feature üretimi
Anomaly Detection Agent → anomali analizi
Bu yapı ile uçtan uca bir veri işleme ve tahmin pipeline’ı oluşturulabilir.
🏁 Conclusion
Forecasting Agent, sadece tahmin yapan bir araç değil; veri analizi, modelleme, senaryo simülasyonu ve karar destek süreçlerini bir araya getiren kapsamlı bir AI çözümüdür. Kurum içinde veri odaklı karar alma süreçlerini hızlandırır ve teknik olmayan kullanıcıların da ileri seviye analizler yapabilmesini sağlar.

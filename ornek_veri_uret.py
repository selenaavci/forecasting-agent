"""
Ornek test veri seti uretici.
Calistirmak icin: python ornek_veri_uret.py
Olusan dosyalar: ornek_aylik_satis.csv, ornek_gunluk_satis.csv, ornek_haftalik_ciro.xlsx
"""

import numpy as np
import pandas as pd


def aylik_satis_verisi():
    """48 ay = 4 yil aylik perakende satis verisi.
    Artan trend + yillik sezonsallik (Aralik yuksek, Temmuz-Agustos dusuk) + gurultu.
    """
    rng = np.random.RandomState(42)
    n = 48
    dates = pd.date_range("2022-01-01", periods=n, freq="MS")

    trend = np.linspace(120000, 210000, n)
    month = dates.month
    seasonal_factor = np.where(month == 12, 1.35,
                       np.where(month == 11, 1.20,
                       np.where(month == 1, 0.85,
                       np.where(month.isin([7, 8]), 0.80,
                       np.where(month == 4, 1.10, 1.0)))))

    campaign = np.zeros(n)
    for i, d in enumerate(dates):
        if (d.month == 11 and d.day == 1):
            campaign[i] = 15000
        if (d.month == 4 and d.year >= 2023):
            campaign[i] = 8000

    noise = rng.normal(0, 6000, n)
    satis = (trend * seasonal_factor + campaign + noise).round(0).astype(int)

    df = pd.DataFrame({
        "Tarih": dates.strftime("%Y-%m-%d"),
        "Satis_Tutari_TL": satis,
    })
    return df


def gunluk_satis_verisi():
    """365 gunluk e-ticaret siparis verisi.
    Haftalik sezonsallik (hafta sonu yuksek) + yillik trend + ozel gunler.
    """
    rng = np.random.RandomState(7)
    n = 365
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    base = 400 + np.arange(n) * 0.4
    weekday = dates.dayofweek
    weekly = np.where(weekday >= 5, 1.45, np.where(weekday == 4, 1.15, 1.0))
    month_factor = np.where(dates.month == 12, 1.30,
                   np.where(dates.month == 11, 1.25,
                   np.where(dates.month.isin([7, 8]), 0.85, 1.0)))

    special = np.zeros(n)
    for i, d in enumerate(dates):
        if d.month == 11 and d.day in (11, 22, 23, 24, 25):
            special[i] = 500
        if d.month == 2 and d.day == 14:
            special[i] = 250
        if d.month == 3 and d.day == 8:
            special[i] = 200

    noise = rng.normal(0, 35, n)
    orders = (base * weekly * month_factor + special + noise).round(0).astype(int)
    orders = np.clip(orders, 0, None)

    df = pd.DataFrame({
        "Tarih": dates.strftime("%Y-%m-%d"),
        "Siparis_Adedi": orders,
    })
    return df


def haftalik_ciro_verisi():
    """156 hafta = 3 yil haftalik B2B ciro verisi (bulusik, outlier icerir)."""
    rng = np.random.RandomState(11)
    n = 156
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    base = np.linspace(45000, 62000, n)
    season = 6000 * np.sin(2 * np.pi * np.arange(n) / 52)
    noise = rng.normal(0, 3000, n)
    ciro = base + season + noise

    for idx in [25, 70, 120]:
        ciro[idx] *= 1.8
    for idx in [50, 100]:
        ciro[idx] *= 0.4

    df = pd.DataFrame({
        "Hafta_Baslangic": dates.strftime("%Y-%m-%d"),
        "Ciro_TL": ciro.round(2),
        "Aktif_Musteri": (200 + np.arange(n) * 0.5 + rng.normal(0, 10, n)).round(0).astype(int),
    })
    return df


if __name__ == "__main__":
    aylik = aylik_satis_verisi()
    gunluk = gunluk_satis_verisi()
    haftalik = haftalik_ciro_verisi()

    aylik.to_csv("ornek_aylik_satis.csv", index=False, encoding="utf-8-sig")
    gunluk.to_csv("ornek_gunluk_satis.csv", index=False, encoding="utf-8-sig")
    haftalik.to_excel("ornek_haftalik_ciro.xlsx", index=False)

    print("Uretilen dosyalar:")
    print(f"  - ornek_aylik_satis.csv    ({len(aylik)} satir, aylik satis TL)")
    print(f"  - ornek_gunluk_satis.csv   ({len(gunluk)} satir, gunluk siparis adedi)")
    print(f"  - ornek_haftalik_ciro.xlsx ({len(haftalik)} satir, haftalik ciro + musteri)")

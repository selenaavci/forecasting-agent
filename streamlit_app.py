from __future__ import annotations

import io
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")


def safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


st.set_page_config(
    page_title="Forecasting Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Forecasting Agent")
st.caption(
    "Veri setinizi yükleyin, tarih ve hedef kolonunu seçin, birden fazla "
    "zaman serisi modelini otomatik olarak çalıştırıp karşılaştırın."
)

st.warning(
    "**Bu uygulama bir hızlı tahminleme / keşif aracıdır.** "
    "Üretilen tahminler iş kararlarına destek niteliğindedir; "
    "**doğrudan operasyonel karar verme için tek başına kullanılmamalıdır.** "
    "Tahmin sonuçları; veri kalitesi, kampanya/özel gün etkileri ve dış "
    "faktörler dikkate alınarak iş birimleriyle birlikte değerlendirilmelidir."
)


_DEFAULTS = {
    "df": None,
    "file_name": None,
    "date_col": None,
    "target_col": None,
    "freq": None,
    "horizon": 12,
    "selected_models": ["Naive", "SARIMA", "Prophet", "XGBoost"],
    "holidays_country": "TR",
    "quality_report": None,
    "forecast_results": None,
    "best_model": None,
    "series": None,
}
for _key, _value in _DEFAULTS.items():
    st.session_state.setdefault(_key, _value)


@st.cache_data(show_spinner=False)
def load_dataframe(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=";")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return None


def detect_frequency(dates):
    dates = pd.to_datetime(dates).sort_values().drop_duplicates()
    if len(dates) < 2:
        return "D"
    median_days = dates.diff().dropna().dt.days.median()
    if median_days <= 1:
        return "D"
    if median_days <= 8:
        return "W"
    if median_days <= 31:
        return "MS"
    if median_days <= 95:
        return "QS"
    return "YS"


def freq_label(f):
    return {
        "D": "Günlük",
        "W": "Haftalık",
        "MS": "Aylık",
        "QS": "Çeyreklik",
        "YS": "Yıllık",
    }.get(f, "Bilinmeyen")


def quality_check(df, date_col, target_col):
    report = {}
    n = len(df)
    report["satir_sayisi"] = n
    report["eksik_hedef"] = int(df[target_col].isna().sum())
    report["eksik_tarih"] = int(df[date_col].isna().sum())
    report["duplicate_tarih"] = int(df[date_col].duplicated().sum())

    dates = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values()
    if len(dates) >= 2:
        freq = detect_frequency(dates)
        expected = pd.date_range(dates.min(), dates.max(), freq=freq)
        report["beklenen_satir"] = len(expected)
        report["eksik_donem"] = max(0, len(expected) - len(dates.drop_duplicates()))
        report["freq"] = freq
    else:
        report["beklenen_satir"] = n
        report["eksik_donem"] = 0
        report["freq"] = "D"

    series = pd.to_numeric(df[target_col], errors="coerce").dropna()
    if len(series) > 10:
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
        report["aykiri_deger"] = int(mask.sum())
    else:
        report["aykiri_deger"] = 0

    if n < 24:
        report["yeterlilik"] = "Düşük"
        report["yeterlilik_aciklama"] = "Veri çok kısa, sonuçlar güvenilir olmayabilir."
    elif n < 60:
        report["yeterlilik"] = "Orta"
        report["yeterlilik_aciklama"] = "Tahmin yapılabilir ancak uzun vadede risk var."
    else:
        report["yeterlilik"] = "Yeterli"
        report["yeterlilik_aciklama"] = "Güvenilir tahmin için yeterli veri var."
    return report


def prepare_series(df, date_col, target_col, freq):
    s = df[[date_col, target_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col])
    s[target_col] = pd.to_numeric(s[target_col], errors="coerce")
    s = s.groupby(date_col, as_index=False)[target_col].mean()
    s = s.sort_values(date_col).set_index(date_col)
    s = s.asfreq(freq)
    s[target_col] = s[target_col].interpolate(method="linear").bfill().ffill()
    return s[target_col]


def make_features(series):
    df = pd.DataFrame({"y": series.values}, index=series.index)
    df["ay"] = df.index.month
    df["gun"] = df.index.day
    df["haftagunu"] = df.index.dayofweek
    df["yil"] = df.index.year
    df["ceyrek"] = df.index.quarter
    for lag in [1, 2, 3, 7, 12]:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for w in [3, 7, 12]:
        df[f"rolling_{w}"] = df["y"].shift(1).rolling(w).mean()
    return df


def calc_metrics(actual, pred):
    actual = np.array(actual, dtype=float)
    pred = np.array(pred, dtype=float)
    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mask = actual != 0
    mape = (
        float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)
        if mask.any()
        else float("nan")
    )
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def model_naive(train, horizon):
    last = train.iloc[-1]
    step = train.index[-1] - train.index[-2]
    future = pd.date_range(train.index[-1] + step, periods=horizon, freq=train.index.freq)
    pred = pd.Series([last] * horizon, index=future)
    std = train.diff().std()
    return pred, pred - 1.96 * std, pred + 1.96 * std


def model_sarima(train, horizon, seasonal=12):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        m = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, seasonal),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = m.fit(disp=False)
        f = fit.get_forecast(steps=horizon)
        ci = f.conf_int()
        return f.predicted_mean, ci.iloc[:, 0], ci.iloc[:, 1]
    except Exception as e:
        st.warning(f"SARIMA çalıştırılamadı: {e}")
        return None, None, None


def model_prophet(train, horizon, freq, holidays_country=None):
    try:
        from prophet import Prophet

        dfp = pd.DataFrame({"ds": train.index, "y": train.values})
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=(freq in ["D", "W"]),
            daily_seasonality=False,
            interval_width=0.95,
        )
        if holidays_country and holidays_country != "Yok":
            try:
                m.add_country_holidays(country_name=holidays_country)
            except Exception:
                pass
        m.fit(dfp)
        future = m.make_future_dataframe(periods=horizon, freq=freq)
        fc = m.predict(future).tail(horizon)
        pred = pd.Series(fc["yhat"].values, index=pd.to_datetime(fc["ds"].values))
        lower = pd.Series(fc["yhat_lower"].values, index=pred.index)
        upper = pd.Series(fc["yhat_upper"].values, index=pred.index)
        return pred, lower, upper
    except Exception as e:
        st.warning(f"Prophet çalıştırılamadı: {e}")
        return None, None, None


def model_xgboost(train, horizon):
    try:
        from xgboost import XGBRegressor

        feats = make_features(train).dropna()
        X = feats.drop(columns=["y"])
        y = feats["y"]
        model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, verbosity=0)
        model.fit(X, y)

        history = train.copy()
        preds = []
        for _ in range(horizon):
            if train.index.freq is not None:
                next_dt = history.index[-1] + pd.tseries.frequencies.to_offset(train.index.freq)
            else:
                next_dt = history.index[-1] + (history.index[-1] - history.index[-2])
            row = {
                "ay": next_dt.month,
                "gun": next_dt.day,
                "haftagunu": next_dt.dayofweek,
                "yil": next_dt.year,
                "ceyrek": next_dt.quarter,
            }
            for lag in [1, 2, 3, 7, 12]:
                row[f"lag_{lag}"] = history.iloc[-lag] if len(history) >= lag else history.mean()
            for w in [3, 7, 12]:
                row[f"rolling_{w}"] = history.tail(w).mean()
            X_next = pd.DataFrame([row])[X.columns]
            yhat = float(model.predict(X_next)[0])
            preds.append((next_dt, yhat))
            history = pd.concat([history, pd.Series([yhat], index=[next_dt])])

        idx = [d for d, _ in preds]
        vals = [v for _, v in preds]
        pred = pd.Series(vals, index=idx)
        resid_std = (train - train.rolling(3).mean()).std()
        return pred, pred - 1.96 * resid_std, pred + 1.96 * resid_std
    except Exception as e:
        st.warning(f"XGBoost çalıştırılamadı: {e}")
        return None, None, None


def run_all_models(series, horizon, freq, holidays_country, selected_models):
    n = len(series)
    split = max(int(n * 0.8), n - horizon)
    train = series.iloc[:split]
    test = series.iloc[split:]

    seasonal_period = {"D": 7, "W": 52, "MS": 12, "QS": 4, "YS": 1}.get(freq, 12)
    results = {}

    def evaluate(name, fn, *args):
        pred, lo, up = fn(train, len(test), *args)
        if pred is None:
            return
        if len(test) > 0:
            m = calc_metrics(test.values, pred.iloc[: len(test)].values)
        else:
            m = {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
        full_pred, full_lo, full_up = fn(series, horizon, *args)
        results[name] = {
            "metrics": m,
            "forecast": full_pred,
            "lower": full_lo,
            "upper": full_up,
        }

    if "Naive" in selected_models:
        evaluate("Naive", model_naive)
    if "SARIMA" in selected_models:
        evaluate("SARIMA", model_sarima, seasonal_period)
    if "Prophet" in selected_models:
        evaluate("Prophet", model_prophet, freq, holidays_country)
    if "XGBoost" in selected_models:
        evaluate("XGBoost", model_xgboost)

    return results, test


with st.sidebar:
    st.header("Dosya Yükleme")
    uploaded_file = st.file_uploader(
        "CSV veya Excel dosyası yükleyin",
        type=["csv", "xlsx", "xls"],
        help="Dosyada tarih ve tahmin edilecek sayı kolonları olmalı.",
    )

    st.divider()
    st.markdown(
        "**Desteklenen formatlar:** CSV, XLSX, XLS\n\n"
        "**Özellikler:**\n"
        "- Otomatik veri sıklığı tespiti (günlük/haftalık/aylık)\n"
        "- Eksik değer ve aykırı değer analizi\n"
        "- Çoklu model karşılaştırması (Naive, SARIMA, Prophet, XGBoost)\n"
        "- Ülkeye özel resmi tatil takvimi\n"
        "- Güven aralığı (confidence interval)\n"
        "- What-if senaryo simülasyonu\n"
        "- Excel rapor çıktısı"
    )

    st.divider()
    if st.button("Örnek veri ile dene", use_container_width=True):
        dates = pd.date_range("2022-01-01", periods=48, freq="MS")
        trend = np.linspace(120000, 210000, 48)
        season = 20000 * np.sin(2 * np.pi * np.arange(48) / 12)
        noise = np.random.RandomState(42).normal(0, 6000, 48)
        sample = pd.DataFrame(
            {"Tarih": dates, "Satis_Tutari_TL": (trend + season + noise).round(0).astype(int)}
        )
        st.session_state["df"] = sample
        st.session_state["file_name"] = "ornek_veri.csv"
        st.session_state["forecast_results"] = None
        safe_rerun()

    if st.button("Oturumu Sıfırla", use_container_width=True):
        for _k in list(st.session_state.keys()):
            del st.session_state[_k]
        safe_rerun()


if uploaded_file is not None:
    try:
        if st.session_state.get("file_name") != uploaded_file.name:
            with st.spinner("Veri okunuyor..."):
                df_new = load_dataframe(uploaded_file)
            if df_new is None or df_new.empty:
                st.error("Dosya okunamadı veya boş.")
                st.stop()
            st.session_state["df"] = df_new
            st.session_state["file_name"] = uploaded_file.name
            st.session_state["forecast_results"] = None
            st.session_state["date_col"] = None
            st.session_state["target_col"] = None
    except Exception as exc:
        st.error(f"Dosya okunamadı: {exc}")
        st.stop()


df = st.session_state.get("df")

if df is None:
    st.info("Başlamak için kenar çubuğundan bir dosya yükleyin veya örnek veriyi deneyin.")
    st.stop()


tab_preview, tab_profile, tab_setup, tab_train, tab_results = st.tabs(
    [
        "Veri Önizleme",
        "Veri Profili",
        "Tahmin Tanımı",
        "Model Eğitimi",
        "Sonuçlar",
    ]
)


with tab_preview:
    st.subheader("Veri Seti Önizleme")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Satır Sayısı", f"{len(df):,}")
    c2.metric("Sütun Sayısı", len(df.columns))
    c3.metric("Eksik Değer", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Numerik Sütun", df.select_dtypes(include="number").shape[1])

    st.dataframe(df.head(100), use_container_width=True, height=400)

    with st.expander("Sütun Bilgileri"):
        rows = []
        for col in df.columns:
            samples = df[col].dropna()
            rows.append({
                "Sütun": col,
                "Veri Tipi": str(df[col].dtype),
                "Benzersiz Değer": int(df[col].nunique(dropna=True)),
                "Eksik Değer": int(df[col].isna().sum()),
                "Örnek Değer": str(samples.iloc[0]) if len(samples) > 0 else "-",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


with tab_profile:
    st.subheader("Veri Profili")

    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        st.markdown("#### Numerik Sütun İstatistikleri")
        st.dataframe(num_df.describe().T, use_container_width=True)

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.markdown("#### Eksik Değerler")
        miss_df = pd.DataFrame(
            {"Adet": missing.values, "Oran (%)": (missing.values / len(df) * 100).round(2)},
            index=missing.index,
        )
        miss_df.index.name = "Sütun"
        st.dataframe(miss_df, use_container_width=True)
    else:
        st.success("Eksik değer bulunmuyor.")

    date_col = st.session_state.get("date_col")
    target_col = st.session_state.get("target_col")

    if date_col and target_col:
        st.markdown("#### Zaman Serisi Kalite Raporu")
        st.caption(
            "Tahmin Tanımı sekmesinde seçilen tarih ve hedef kolonu üzerinden "
            "zaman serisine özel kalite analizi."
        )
        report = quality_check(df, date_col, target_col)
        st.session_state["quality_report"] = report

        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("Satır Sayısı", report["satir_sayisi"])
        qc2.metric("Eksik Hedef", report["eksik_hedef"])
        qc3.metric("Eksik Dönem", report["eksik_donem"])
        qc4.metric("Aykırı Değer", report["aykiri_deger"])

        yet = report["yeterlilik"]
        if yet == "Düşük":
            st.error(f"**Veri Yeterliliği: {yet}** — {report['yeterlilik_aciklama']}")
        elif yet == "Orta":
            st.warning(f"**Veri Yeterliliği: {yet}** — {report['yeterlilik_aciklama']}")
        else:
            st.success(f"**Veri Yeterliliği: {yet}** — {report['yeterlilik_aciklama']}")

        if report["duplicate_tarih"] > 0:
            st.warning(
                f"{report['duplicate_tarih']} adet tekrar eden tarih bulundu. "
                "Model eğitilirken ortalaması alınacak."
            )
        if report["eksik_donem"] > 0:
            st.warning(
                f"{report['eksik_donem']} adet eksik dönem var. "
                "Linear interpolation ile otomatik doldurulacak."
            )

        series = prepare_series(df, date_col, target_col, report["freq"])
        st.session_state["series"] = series

        st.markdown("#### Zaman Serisi Grafiği")
        fig = px.line(
            x=series.index, y=series.values,
            labels={"x": "Tarih", "y": target_col},
        )
        fig.update_traces(line_color="#667eea", line_width=2)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Zaman serisi kalite raporunu görmek için önce **Tahmin Tanımı** "
            "sekmesinden tarih ve hedef kolonlarını seçin."
        )


with tab_setup:
    st.subheader("Tahmin Tanımı")

    st.markdown("#### Tarih Kolonu")
    st.caption("Zaman bilgisi taşıyan kolonu seçin (örn. Tarih, Date, Month).")
    cols = df.columns.tolist()
    date_candidates = [c for c in cols if any(k in c.lower() for k in ["tarih", "date", "time", "ay", "month"])]
    current_date = st.session_state.get("date_col") or (date_candidates[0] if date_candidates else cols[0])
    try:
        date_idx = cols.index(current_date)
    except ValueError:
        date_idx = 0
    date_col = st.selectbox("Tarih Sütunu", cols, index=date_idx)
    st.session_state["date_col"] = date_col

    try:
        test_dates = pd.to_datetime(df[date_col], errors="coerce")
        if test_dates.isna().all():
            st.error("Seçilen kolon tarih formatında değil. Farklı bir kolon seçin.")
            st.stop()
        freq = detect_frequency(test_dates.dropna())
        st.session_state["freq"] = freq
        st.info(f"Algılanan veri sıklığı: **{freq_label(freq)}**")
    except Exception as e:
        st.error(f"Tarih kolonu okunamadı: {e}")
        st.stop()

    st.markdown("#### Hedef Kolon")
    st.caption("Tahmin etmek istediğiniz sayı kolonunu seçin (örn. Satış, Ciro, Stok).")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    target_options = num_cols if num_cols else [c for c in cols if c != date_col]
    current_target = st.session_state.get("target_col")
    if current_target not in target_options:
        current_target = target_options[0] if target_options else None
    target_col = st.selectbox(
        "Hedef Sütun",
        target_options,
        index=target_options.index(current_target) if current_target in target_options else 0,
    )
    st.session_state["target_col"] = target_col

    st.markdown("#### Tahmin Ufku")
    st.caption("Kaç dönem ileriyi tahmin etmek istediğinizi seçin.")
    defaults = {"D": 30, "W": 12, "MS": 6, "QS": 4, "YS": 3}
    horizon = st.slider(
        f"İleri dönem sayısı ({freq_label(freq).lower()})",
        min_value=1,
        max_value=120,
        value=st.session_state.get("horizon") or defaults.get(freq, 12),
        help="Örneğin aylık veride 6 = 6 ay ilerisi.",
    )
    st.session_state["horizon"] = horizon

    st.markdown("#### Resmi Tatil Takvimi")
    st.caption("Prophet modeli için ülkeye özel resmi tatiller modele dahil edilir.")
    holiday_options = ["Yok", "TR", "US", "DE", "GB", "FR", "IT", "ES", "NL"]
    current_h = st.session_state.get("holidays_country") or "TR"
    hc = st.selectbox(
        "Ülke",
        holiday_options,
        index=holiday_options.index(current_h) if current_h in holiday_options else 1,
    )
    st.session_state["holidays_country"] = hc


with tab_train:
    st.subheader("Model Eğitimi")

    date_col = st.session_state.get("date_col")
    target_col = st.session_state.get("target_col")
    series = st.session_state.get("series")

    if not date_col or not target_col:
        st.info("Önce **Tahmin Tanımı** sekmesinden tarih ve hedef kolonlarını seçin.")
    else:
        st.markdown("#### Model Seçimi")
        st.caption("Birden fazla model aynı veri üzerinde çalıştırılır, en düşük hatalı olan en iyi model seçilir.")

        model_options = ["Naive", "SARIMA", "Prophet", "XGBoost"]
        selected_models = st.multiselect(
            "Denenecek modeller",
            options=model_options,
            default=st.session_state.get("selected_models") or model_options,
        )
        st.session_state["selected_models"] = selected_models

        with st.expander("Model Açıklamaları"):
            st.markdown(
                "- **Naive** — En son değeri tekrarlar. Diğer modellerin kıyaslanacağı referans model.\n"
                "- **SARIMA** — Klasik istatistiksel model; trend ve sezonsallığı yakalar.\n"
                "- **Prophet** — Facebook tarafından geliştirilen model; resmi tatilleri, çoklu sezonsallığı destekler.\n"
                "- **XGBoost** — Gradient boosting tabanlı makine öğrenmesi; geçmiş verilerden feature üretip öğrenir."
            )

        st.markdown("#### Eğitim Parametreleri")
        info_cols = st.columns(3)
        info_cols[0].info(f"Veri sıklığı: **{freq_label(st.session_state.get('freq'))}**")
        info_cols[1].info(f"Tahmin ufku: **{st.session_state.get('horizon')} dönem**")
        info_cols[2].info(f"Tatil takvimi: **{st.session_state.get('holidays_country')}**")

        if st.button("Tahmini Başlat", type="primary", use_container_width=True):
            if not selected_models:
                st.warning("En az bir model seçilmelidir.")
            elif series is None:
                report = quality_check(df, date_col, target_col)
                st.session_state["quality_report"] = report
                series = prepare_series(df, date_col, target_col, report["freq"])
                st.session_state["series"] = series

            if selected_models and series is not None:
                try:
                    progress = st.progress(0, text="Veri hazırlanıyor...")
                    progress.progress(25, text="Modeller eğitiliyor...")
                    results, test = run_all_models(
                        series,
                        st.session_state["horizon"],
                        st.session_state["freq"],
                        st.session_state["holidays_country"],
                        selected_models,
                    )
                    progress.progress(80, text="Metrikler hesaplanıyor...")

                    if not results:
                        st.error("Hiçbir model çalıştırılamadı.")
                    else:
                        def score(m):
                            v = m["metrics"].get("MAPE", float("nan"))
                            return v if not np.isnan(v) else m["metrics"].get("MAE", float("inf"))

                        best = min(results.keys(), key=lambda k: score(results[k]))
                        st.session_state["forecast_results"] = results
                        st.session_state["best_model"] = best
                        progress.progress(100, text="Tamamlandı.")
                        st.success(
                            f"Eğitim tamamlandı. En iyi model: **{best}**. "
                            "**Sonuçlar** sekmesinden detaylar incelenebilir."
                        )
                except Exception as exc:
                    st.error(f"Eğitim sırasında hata: {exc}")


with tab_results:
    st.subheader("Sonuçlar ve Dışa Aktarma")

    results = st.session_state.get("forecast_results")
    best = st.session_state.get("best_model")
    series = st.session_state.get("series")
    target_col = st.session_state.get("target_col")

    if not results:
        st.info("Henüz bir tahmin çalıştırılmadı. **Model Eğitimi** sekmesinden başlatılabilir.")
    else:
        rows = []
        for name, info in results.items():
            m = info["metrics"]
            rows.append({
                "Model": name,
                "MAE": round(m["MAE"], 4),
                "RMSE": round(m["RMSE"], 4),
                "MAPE (%)": round(m["MAPE"], 2) if not np.isnan(m["MAPE"]) else None,
            })
        comp_df = pd.DataFrame(rows)

        st.markdown("#### Model Karşılaştırma")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.success(f"En iyi model: **{best}**")

        with st.expander("Metrik Açıklamaları"):
            st.markdown(
                "- **MAE (Mean Absolute Error)** — Ortalama mutlak hata. Tahmin ve gerçek değer farklarının ortalaması.\n"
                "- **RMSE (Root Mean Squared Error)** — Kök ortalama kare hata. Büyük hataları daha çok cezalandırır.\n"
                "- **MAPE (%)** — Ortalama mutlak yüzde hata. Tahminin gerçek değere oranla yüzde kaç saptığını gösterir."
            )

        st.markdown("#### Model Detayları")
        model_names = list(results.keys())
        chosen = st.selectbox(
            "Detayı görüntülenecek model",
            model_names,
            index=model_names.index(best),
        )
        r = results[chosen]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, mode="lines",
            name="Geçmiş Veri", line=dict(color="#2c3e50", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=r["forecast"].index, y=r["forecast"].values, mode="lines",
            name="Tahmin", line=dict(color="#e74c3c", width=3, dash="dash"),
        ))
        if r["upper"] is not None and r["lower"] is not None:
            fig.add_trace(go.Scatter(
                x=list(r["upper"].index) + list(r["lower"].index[::-1]),
                y=list(r["upper"].values) + list(r["lower"].values[::-1]),
                fill="toself", fillcolor="rgba(231,76,60,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Güven Aralığı (%95)",
            ))
        fig.update_layout(
            xaxis_title="Tarih", yaxis_title=target_col,
            hovermode="x unified", height=480,
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Tahmin Tablosu")
        forecast_df = pd.DataFrame({
            "Tarih": r["forecast"].index,
            "Tahmin": r["forecast"].values.round(2),
            "Alt Sınır": r["lower"].values.round(2) if r["lower"] is not None else np.nan,
            "Üst Sınır": r["upper"].values.round(2) if r["upper"] is not None else np.nan,
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        st.markdown("#### What-If Senaryo Simülasyonu")
        st.caption("Tahmini belli bir oranla artırıp/azaltarak farklı senaryoları inceleyin.")
        change_pct = st.slider("Senaryo etkisi (%)", -50, 50, 0, step=5)
        if change_pct != 0:
            scen = r["forecast"] * (1 + change_pct / 100)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=series.index, y=series.values, name="Geçmiş", line=dict(color="#2c3e50")))
            fig2.add_trace(go.Scatter(
                x=r["forecast"].index, y=r["forecast"].values,
                name="Baz Tahmin", line=dict(color="#3498db", dash="dash"),
            ))
            fig2.add_trace(go.Scatter(
                x=scen.index, y=scen.values,
                name=f"Senaryo (%{change_pct:+d})", line=dict(color="#e67e22", width=3),
            ))
            fig2.update_layout(height=380, hovermode="x unified", xaxis_title="Tarih", yaxis_title=target_col)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Dışa Aktarma")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as w:
            forecast_df.to_excel(w, sheet_name="Tahmin Sonuclari", index=False)
            comp_df.to_excel(w, sheet_name="Model Karsilastirma", index=False)
            qr = st.session_state.get("quality_report") or {}
            if qr:
                pd.DataFrame([qr]).T.reset_index().rename(
                    columns={"index": "Metrik", 0: "Değer"}
                ).to_excel(w, sheet_name="Veri Kalite Raporu", index=False)
            summary = pd.DataFrame([{
                "Zaman Damgasi": datetime.now().isoformat(timespec="seconds"),
                "En Iyi Model": best,
                "Tahmin Ufku": len(r["forecast"]),
                "Veri Sikligi": freq_label(st.session_state["freq"]),
                "Toplam Gecmis Satir": len(series),
                "Ortalama Tahmin": round(float(r["forecast"].mean()), 2),
                "Min Tahmin": round(float(r["forecast"].min()), 2),
                "Max Tahmin": round(float(r["forecast"].max()), 2),
            }])
            summary.to_excel(w, sheet_name="Ozet", index=False)

        st.download_button(
            "Sonuçları Excel olarak indir",
            data=buffer.getvalue(),
            file_name=f"forecast_sonuclari_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

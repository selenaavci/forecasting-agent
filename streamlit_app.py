import io
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Forecasting Agent | AI Hub",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1rem;
    }
    .step-box {
        background: #f8f9fa; padding: 1rem; border-radius: 8px;
        border-left: 4px solid #667eea; margin-bottom: 1rem;
    }
    .success-box { background: #d4edda; padding: 0.75rem; border-radius: 6px; color: #155724; }
    .warn-box { background: #fff3cd; padding: 0.75rem; border-radius: 6px; color: #856404; }
    .err-box { background: #f8d7da; padding: 0.75rem; border-radius: 6px; color: #721c24; }
    </style>
    <div class="main-header">
        <h1>Forecasting Agent</h1>
        <p>Verinizi yükleyin, birkaç tıkla geleceğe bakın. Teknik bilgi gerekmez.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def init_state():
    defaults = {
        "df_raw": None,
        "df_clean": None,
        "date_col": None,
        "target_col": None,
        "freq": None,
        "quality_report": None,
        "forecast_results": None,
        "best_model": None,
        "step": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


with st.sidebar:
    st.markdown("### Adımlar")
    steps = [
        "1. Veri Yükle",
        "2. Kolon Seçimi",
        "3. Veri Kalitesi",
        "4. Tahmin Ayarları",
        "5. Model Çalıştır",
        "6. Sonuçlar",
    ]
    selected_step = st.radio(
        "Hangi adımdasınız?", steps, index=st.session_state["step"] - 1
    )
    st.session_state["step"] = steps.index(selected_step) + 1

    st.markdown("---")
    st.markdown("### Nasıl Çalışır?")
    st.info(
        "1. Excel/CSV dosyanızı yükleyin\n"
        "2. Tarih ve tahmin edeceğiniz kolonu seçin\n"
        "3. Veri kalite raporunu inceleyin\n"
        "4. Kaç dönem ileri tahmin istediğinizi belirtin\n"
        "5. Tahminleri görüntüleyin ve Excel'e aktarın"
    )

    st.markdown("---")
    if st.button("Sıfırla", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


@st.cache_data(show_spinner=False)
def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=";")
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        return None


def detect_frequency(dates):
    dates = pd.to_datetime(dates).sort_values().drop_duplicates()
    if len(dates) < 2:
        return "D"
    diffs = dates.diff().dropna()
    median_days = diffs.dt.days.median()
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
        report["yeterlilik"] = "Düşük - Veri çok kısa, sonuçlar güvenilir olmayabilir."
    elif n < 60:
        report["yeterlilik"] = "Orta - Tahmin yapılabilir ancak uzun vadede risk var."
    else:
        report["yeterlilik"] = "Yeterli - Güvenilir tahmin için yeterli veri var."
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
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mask = actual != 0
    mape = (
        np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100
        if mask.any()
        else np.nan
    )
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def model_naive(train, horizon):
    last = train.iloc[-1]
    future = pd.date_range(
        train.index[-1] + (train.index[-1] - train.index[-2]), periods=horizon, freq=train.index.freq
    )
    pred = pd.Series([last] * horizon, index=future)
    std = train.diff().std()
    lower = pred - 1.96 * std
    upper = pred + 1.96 * std
    return pred, lower, upper


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
        pred = f.predicted_mean
        ci = f.conf_int()
        return pred, ci.iloc[:, 0], ci.iloc[:, 1]
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
        model = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, verbosity=0
        )
        model.fit(X, y)

        history = train.copy()
        preds = []
        freq = train.index.freq
        for i in range(horizon):
            next_dt = history.index[-1] + (history.index[-1] - history.index[-2])
            if freq is not None:
                next_dt = history.index[-1] + pd.tseries.frequencies.to_offset(freq)
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
        lower = pred - 1.96 * resid_std
        upper = pred + 1.96 * resid_std
        return pred, lower, upper
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

    def evaluate(name, fn, *args, **kwargs):
        pred, lo, up = fn(train, len(test), *args, **kwargs)
        if pred is None:
            return
        if len(test) > 0:
            pred_t = pred.iloc[: len(test)].values
            m = calc_metrics(test.values, pred_t)
        else:
            m = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
        full_pred, full_lo, full_up = fn(series, horizon, *args, **kwargs)
        results[name] = {
            "metrics": m,
            "forecast": full_pred,
            "lower": full_lo,
            "upper": full_up,
            "validation_pred": pred,
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


step = st.session_state["step"]

if step == 1:
    st.markdown("## 1. Veri Dosyanızı Yükleyin")
    st.markdown(
        '<div class="step-box">Excel (<b>.xlsx</b>) veya CSV (<b>.csv</b>) dosyanızı sürükleyip bırakabilirsiniz. '
        "Dosyada <b>tarih</b> ve <b>tahmin etmek istediğiniz sayı</b> (örneğin satış adedi, ciro, stok) olmalı.</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Dosya seçin", type=["csv", "xlsx", "xls"], help="Maksimum 200MB"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Örnek veri ile dene", use_container_width=True):
            dates = pd.date_range("2021-01-01", periods=48, freq="MS")
            trend = np.linspace(100, 200, 48)
            season = 20 * np.sin(2 * np.pi * np.arange(48) / 12)
            noise = np.random.RandomState(42).normal(0, 8, 48)
            sample = pd.DataFrame(
                {"Tarih": dates, "Satis": (trend + season + noise).round(2)}
            )
            st.session_state["df_raw"] = sample
            st.success("Örnek veri yüklendi. Bir sonraki adıma geçebilirsiniz.")

    if uploaded is not None:
        df = load_file(uploaded)
        if df is None or df.empty:
            st.error("Dosya okunamadı. Lütfen Excel veya CSV formatında olduğundan emin olun.")
        else:
            st.session_state["df_raw"] = df
            st.success(f"Dosya başarıyla yüklendi: **{len(df)} satır, {len(df.columns)} kolon**")

    if st.session_state["df_raw"] is not None:
        st.markdown("### Verinin İlk 10 Satırı")
        st.dataframe(st.session_state["df_raw"].head(10), use_container_width=True)
        if st.button("Devam Et: Kolon Seçimi", type="primary", use_container_width=True):
            st.session_state["step"] = 2
            st.rerun()


elif step == 2:
    st.markdown("## 2. Kolonları Seçin")
    df = st.session_state["df_raw"]
    if df is None:
        st.warning("Önce veri yüklemeniz gerekir. Lütfen 1. adıma dönün.")
        st.stop()

    st.markdown(
        '<div class="step-box"><b>Tarih Kolonu:</b> Verinizde tarihlerin tutulduğu kolon (örn. <i>Tarih, Date, Month</i>).<br>'
        "<b>Hedef Kolon:</b> Tahmin etmek istediğiniz sayı kolonu (örn. <i>Satış, Ciro, Stok</i>).</div>",
        unsafe_allow_html=True,
    )

    cols = df.columns.tolist()
    date_candidates = [c for c in cols if any(k in c.lower() for k in ["tarih", "date", "time", "ay", "month"])]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    c1, c2 = st.columns(2)
    with c1:
        date_col = st.selectbox(
            "Tarih Kolonu",
            cols,
            index=cols.index(date_candidates[0]) if date_candidates else 0,
        )
    with c2:
        target_col = st.selectbox(
            "Hedef Kolon (tahmin edilecek sayı)",
            num_cols if num_cols else cols,
            index=0,
        )

    try:
        test_dates = pd.to_datetime(df[date_col], errors="coerce")
        if test_dates.isna().all():
            st.error("Seçtiğiniz tarih kolonu tarih formatında değil. Lütfen farklı bir kolon seçin.")
            st.stop()
        freq = detect_frequency(test_dates.dropna())
        st.info(f"Algılanan veri sıklığı: **{freq_label(freq)}** (`{freq}`)")
    except Exception as e:
        st.error(f"Tarih kolonu okunamadı: {e}")
        st.stop()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Geri", use_container_width=True):
            st.session_state["step"] = 1
            st.rerun()
    with col_b:
        if st.button("Devam Et: Veri Kalitesi", type="primary", use_container_width=True):
            st.session_state["date_col"] = date_col
            st.session_state["target_col"] = target_col
            st.session_state["freq"] = freq
            st.session_state["step"] = 3
            st.rerun()


elif step == 3:
    st.markdown("## 3. Veri Kalite Analizi")
    df = st.session_state["df_raw"]
    date_col = st.session_state["date_col"]
    target_col = st.session_state["target_col"]
    if df is None or date_col is None:
        st.warning("Önceki adımları tamamlayın.")
        st.stop()

    report = quality_check(df, date_col, target_col)
    st.session_state["quality_report"] = report

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Satır Sayısı", report["satir_sayisi"])
    c2.metric("Eksik Hedef", report["eksik_hedef"])
    c3.metric("Eksik Dönem", report["eksik_donem"])
    c4.metric("Aykırı Değer", report["aykiri_deger"])

    y = report["yeterlilik"]
    if y.startswith("Düşük"):
        st.markdown(f'<div class="err-box"><b>Veri Yeterliliği:</b> {y}</div>', unsafe_allow_html=True)
    elif y.startswith("Orta"):
        st.markdown(f'<div class="warn-box"><b>Veri Yeterliliği:</b> {y}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="success-box"><b>Veri Yeterliliği:</b> {y}</div>', unsafe_allow_html=True)

    if report["duplicate_tarih"] > 0:
        st.warning(f"{report['duplicate_tarih']} adet tekrar eden tarih bulundu. Ortalaması alınacak.")
    if report["eksik_donem"] > 0:
        st.warning(f"{report['eksik_donem']} adet eksik dönem var. Otomatik doldurulacak (linear interpolation).")

    series = prepare_series(df, date_col, target_col, report["freq"])
    st.session_state["df_clean"] = series

    st.markdown("### Verinizin Grafiği")
    fig = px.line(
        x=series.index,
        y=series.values,
        labels={"x": "Tarih", "y": target_col},
        title=f"{target_col} Zaman Serisi",
    )
    fig.update_traces(line_color="#667eea", line_width=2)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Geri", use_container_width=True):
            st.session_state["step"] = 2
            st.rerun()
    with col_b:
        if st.button("Devam Et: Tahmin Ayarları", type="primary", use_container_width=True):
            st.session_state["step"] = 4
            st.rerun()


elif step == 4:
    st.markdown("## 4. Tahmin Ayarları")
    if st.session_state["df_clean"] is None:
        st.warning("Önceki adımları tamamlayın.")
        st.stop()

    freq = st.session_state["freq"]
    fl = freq_label(freq)
    st.markdown(
        f'<div class="step-box">Veriniz <b>{fl}</b> olarak algılandı. '
        f"Kaç dönem ilerisini tahmin etmek istiyorsunuz?</div>",
        unsafe_allow_html=True,
    )

    defaults = {"D": 30, "W": 12, "MS": 6, "QS": 4, "YS": 3}
    horizon = st.slider(
        "Tahmin Ufku (ileri dönem sayısı)",
        min_value=1,
        max_value=120,
        value=defaults.get(freq, 12),
        help="Örneğin aylık veride 6 = 6 ay ilerisi",
    )

    st.markdown("### Hangi Modeller Çalışsın?")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        use_naive = st.checkbox("Naive (Basit)", value=True, help="En son değeri tekrarlar - referans model")
    with c2:
        use_sarima = st.checkbox("SARIMA", value=True, help="Klasik istatistiksel model, sezonsallık yakalar")
    with c3:
        use_prophet = st.checkbox("Prophet", value=True, help="Facebook modeli, tatil ve trend destekler")
    with c4:
        use_xgb = st.checkbox("XGBoost", value=True, help="Gelişmiş makine öğrenmesi modeli")

    selected = []
    if use_naive: selected.append("Naive")
    if use_sarima: selected.append("SARIMA")
    if use_prophet: selected.append("Prophet")
    if use_xgb: selected.append("XGBoost")

    st.markdown("### Resmi Tatil Takvimi")
    holidays_country = st.selectbox(
        "Ülke seçin (Prophet modeli için)",
        ["Yok", "TR", "US", "DE", "GB", "FR", "IT", "ES", "NL"],
        index=1,
        help="Resmi tatiller modele dahil edilir",
    )

    st.session_state["_horizon"] = horizon
    st.session_state["_selected_models"] = selected
    st.session_state["_holidays"] = holidays_country

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Geri", use_container_width=True):
            st.session_state["step"] = 3
            st.rerun()
    with col_b:
        if st.button("Modelleri Çalıştır", type="primary", use_container_width=True, disabled=not selected):
            st.session_state["step"] = 5
            st.rerun()


elif step == 5:
    st.markdown("## 5. Modeller Çalışıyor...")
    series = st.session_state["df_clean"]
    if series is None:
        st.warning("Veri bulunamadı.")
        st.stop()

    horizon = st.session_state.get("_horizon", 12)
    selected = st.session_state.get("_selected_models", ["Naive"])
    holidays_country = st.session_state.get("_holidays", "TR")
    freq = st.session_state["freq"]

    with st.spinner("Modeller eğitiliyor ve tahminler üretiliyor... Bu işlem birkaç dakika sürebilir."):
        progress = st.progress(0)
        progress.progress(20)
        results, test = run_all_models(series, horizon, freq, holidays_country, selected)
        progress.progress(100)

    if not results:
        st.error("Hiçbir model çalıştırılamadı. Lütfen veri boyutunu ve kütüphane kurulumunu kontrol edin.")
        st.stop()

    def score(m):
        v = m["metrics"].get("MAPE", np.nan)
        return v if not np.isnan(v) else m["metrics"].get("MAE", np.inf)

    best = min(results.keys(), key=lambda k: score(results[k]))
    st.session_state["forecast_results"] = results
    st.session_state["best_model"] = best
    st.session_state["_test_set"] = test

    st.success(f"{len(results)} model başarıyla çalıştırıldı. En iyi model: **{best}**")
    st.session_state["step"] = 6
    st.rerun()


elif step == 6:
    st.markdown("## 6. Tahmin Sonuçları")
    results = st.session_state.get("forecast_results")
    best = st.session_state.get("best_model")
    series = st.session_state["df_clean"]
    target_col = st.session_state["target_col"]

    if not results:
        st.warning("Henüz model çalıştırılmadı. 5. adıma geçin.")
        st.stop()

    st.markdown("### Model Karşılaştırması")
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name + (" (en iyi)" if name == best else ""),
            "MAE": f"{r['metrics']['MAE']:.2f}",
            "RMSE": f"{r['metrics']['RMSE']:.2f}",
            "MAPE (%)": f"{r['metrics']['MAPE']:.2f}" if not np.isnan(r['metrics']['MAPE']) else "-",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Daha düşük değer daha iyidir. MAPE hata yüzdesidir.")

    chosen = st.selectbox("Görüntülemek istediğiniz model:", list(results.keys()),
                          index=list(results.keys()).index(best))
    r = results[chosen]

    st.markdown(f"### {chosen} Modeli - Tahmin Grafiği")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                             name="Geçmiş Veri", line=dict(color="#2c3e50", width=2)))
    fig.add_trace(go.Scatter(x=r["forecast"].index, y=r["forecast"].values, mode="lines",
                             name="Tahmin", line=dict(color="#e74c3c", width=3, dash="dash")))
    if r["upper"] is not None and r["lower"] is not None:
        fig.add_trace(go.Scatter(
            x=list(r["upper"].index) + list(r["lower"].index[::-1]),
            y=list(r["upper"].values) + list(r["lower"].values[::-1]),
            fill="toself", fillcolor="rgba(231,76,60,0.15)",
            line=dict(color="rgba(255,255,255,0)"), name="Güven Aralığı (%95)",
        ))
    fig.update_layout(
        xaxis_title="Tarih", yaxis_title=target_col,
        hovermode="x unified", height=500,
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tahmin Değerleri")
    forecast_df = pd.DataFrame({
        "Tarih": r["forecast"].index,
        "Tahmin": r["forecast"].values.round(2),
        "Alt Sınır": r["lower"].values.round(2) if r["lower"] is not None else np.nan,
        "Üst Sınır": r["upper"].values.round(2) if r["upper"] is not None else np.nan,
    })
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    st.markdown("### What-If Senaryo Simülasyonu")
    st.caption("Tahmini belli bir oran ile artırıp/azaltarak farklı senaryoları inceleyin.")
    change_pct = st.slider("Senaryo etkisi (%)", -50, 50, 0, step=5)
    if change_pct != 0:
        scen = r["forecast"] * (1 + change_pct / 100)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=series.index, y=series.values, name="Geçmiş", line=dict(color="#2c3e50")))
        fig2.add_trace(go.Scatter(x=r["forecast"].index, y=r["forecast"].values, name="Baz Tahmin", line=dict(color="#3498db", dash="dash")))
        fig2.add_trace(go.Scatter(x=scen.index, y=scen.values, name=f"Senaryo (%{change_pct:+d})", line=dict(color="#e67e22", width=3)))
        fig2.update_layout(height=400, hovermode="x unified", xaxis_title="Tarih", yaxis_title=target_col)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Sonuçları İndir")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as w:
        forecast_df.to_excel(w, sheet_name="Tahmin Sonuclari", index=False)
        metrics_df = pd.DataFrame([
            {"Model": n, **rr["metrics"]} for n, rr in results.items()
        ])
        metrics_df.to_excel(w, sheet_name="Model Metrikleri", index=False)
        qr = st.session_state.get("quality_report") or {}
        pd.DataFrame([qr]).T.reset_index().rename(columns={"index": "Metrik", 0: "Değer"}).to_excel(
            w, sheet_name="Veri Kalite Raporu", index=False
        )
        summary = pd.DataFrame([{
            "En İyi Model": best,
            "Tahmin Ufku": len(r["forecast"]),
            "Veri Sıklığı": freq_label(st.session_state["freq"]),
            "Toplam Geçmiş Satır": len(series),
            "Ortalama Tahmin": round(r["forecast"].mean(), 2),
        }])
        summary.to_excel(w, sheet_name="Ozet", index=False)

    st.download_button(
        "Excel olarak indir",
        buffer.getvalue(),
        file_name="forecast_sonuclari.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary",
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Yeni Ayarlar", use_container_width=True):
            st.session_state["step"] = 4
            st.rerun()
    with col_b:
        if st.button("Yeni Tahmin", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


st.markdown("---")
st.caption("AI Hub - Forecasting Agent | Streamlit Cloud Versiyonu")

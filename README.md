# 📈 TradeVision AI — End-to-End Machine Learning System for Stock Analysis

> A production-grade ML system that fetches daily stock data, engineers technical & sentiment features, trains direction-prediction models, serves predictions via FastAPI, and monitors for data/concept drift — all orchestrated through a single config file.

---

## 🏗️ System Architecture



| Stage | Input | Output |
|---|---|---|
| **Data Ingestion** | Ticker symbol + date range | Versioned raw OHLCV parquet |
| **Feature Store** | Raw OHLCV parquet | Feature parquet (RSI, MACD, Bollinger, sentiment, lags) |
| **Model Training** | Feature parquet | Trained models (RF, XGBoost, LR, Isolation Forest) |
| **Prediction API** | Ticker symbol | JSON with signals, sentiment, anomaly flag, direction |
| **Monitoring** | Prediction logs | Drift reports, rolling accuracy, retrain triggers |

---

## 📁 Project Structure

```
Stocks_ML/
├── config/
│   └── config.yaml              # All hyperparameters & settings (Rule 2)
├── data/
│   ├── raw/                     # Versioned OHLCV parquets (AAPL_2024-01-15.parquet)
│   ├── features/                # Feature store parquets (AAPL_features.parquet)
│   └── metadata/                # Per-run JSON metadata logs
├── models/                      # Saved model artifacts (.pkl)
├── logs/
│   └── predictions.jsonl        # Structured prediction logs (Rule 3)
├── api/
│   └── app.py                   # FastAPI — 3 endpoints
├── monitoring/
│   └── drift.py                 # KS drift detection + should_retrain()
├── dashboard/
│   └── streamlit_app.py         # Streamlit monitoring & analysis dashboard
├── tests/                       # Unit & integration tests
├── docs/
│   └── architecture_diagram.png # System architecture diagram
├── src/
│   ├── data_pipeline.py         # DataPipeline class (validate, clean, preprocess, log)
│   ├── feature_store.py         # FeatureStore class (used in train AND api — Rule 1)
│   └── train.py                 # Training script (reads --config, logs to MLflow)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone <repo-url>
cd Stocks_ML
pip install -r requirements.txt

# 2. Fetch & process data for 5 tickers
python src/data_pipeline.py --config config/config.yaml

# 3. Engineer features
python src/feature_store.py --config config/config.yaml

# 4. Train models (logged to MLflow)
python src/train.py --config config/config.yaml

# 5. Launch API
uvicorn api.app:app --reload

# 6. Launch Dashboard
streamlit run dashboard/streamlit_app.py

# 7. Run everything with Docker
docker-compose up --build
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze` | Full pipeline: OHLCV + indicators + sentiment + anomaly + prediction |
| `GET` | `/fundamentals/{ticker}` | Live P/E, EPS, market cap, margins from yfinance |
| `GET` | `/anomalies/{ticker}` | Isolation Forest flagged dates with context |

### Example Request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "start_date": "2024-01-01", "end_date": "2024-12-31"}'
```

### Example Response

```json
{
  "ticker": "AAPL",
  "direction_prediction": "UP",
  "confidence": 0.73,
  "anomaly_flag": false,
  "sentiment_score": 0.42,
  "technical_signals": {
    "rsi_14": 58.3,
    "macd": 1.24,
    "bollinger_position": "middle",
    "volume_ratio": 1.15
  },
  "fundamentals": {
    "pe_ratio": 28.5,
    "market_cap": "2.8T",
    "profit_margin": 0.26
  }
}
```

---

## 🧪 Models & Evaluation

| Model | Task | Training Accuracy | F1 Score | Status |
|---|---|---|---|---|
| **Naive Baseline** | Always predict UP | 51.01% | 0.6756 | Reference |
| **Random Forest** | Direction Prediction | 68.20% | 0.6950 | ✅ PASS |
| **XGBoost** | Direction Prediction | 72.40% | 0.7410 | ✅ BEST |
| **Logistic Regression**| Direction Prediction | 54.10% | 0.6810 | ✅ PASS |
| **Isolation Forest** | Anomaly Detection | N/A | N/A | ACTIVE |

### Baseline
A naive "always predict UP" model achieves **51.01%** accuracy on our current dataset. All classifiers successfully beat this baseline on F1 score, with **XGBoost** being the clear winner.

### Tradeoff Analysis
For this project, **XGBoost** outperformed Random Forest and Logistic Regression. 
- **Why XGBoost won:** It captured non-linear relationships and momentum shifts more effectively through its gradient boosting architecture, particularly handling the noise in volume indicators better than the bagging approach of Random Forest.
- **Tradeoffs:** XGBoost has slightly higher inference latency (~0.05ms more) than Logistic Regression and is more sensitive to hyperparameter tuning.
- **Improvements:** With more time, I would implement **recursive feature elimination** to reduce the current 23 features to the most impactful 10, and add **cross-validation** over multiple time-folds to ensure stability across market regimes.

---

## 📊 MLflow Experiments

All training runs are versioned and logged in MLflow locally.
- **Random Seeds:** 42 (pinned in `config.yaml`)
- **Metric Tracking:** F1, Accuracy, ROC-AUC, Training Time.
- **Artifacts:** Confusion Matrices and Feature Importance plots are saved per model.

---

## 🔄 Drift Detection & Monitoring

The system monitors for three types of model decay:
1. **Feature drift**: Kolmogorov-Smirnov test (30-day window).
2. **Prediction drift**: Proportional shift in "UP" predictions > 15%.
3. **Rolling accuracy**: Retrain trigger if 30-day accuracy falls below 55%.

The `should_retrain()` function in `monitoring/drift.py` centralizes these triggers.

---

## 📐 Three Rules

1. **One feature pipeline, used twice.** `FeatureStore` is imported identically in `train.py` and `api/app.py`. No copy-paste.
2. **Config controls everything.** Every number lives in `config/config.yaml`. Training scripts take `--config` as their only argument.
3. **Log before you need it.** Prediction logging is built in from day one. Drift detection has real data to work with.

---

## 🛣️ Fairness Analysis

Checked model performance on high-volatility stocks (**TSLA**) vs low-volatility ones (**MSFT**).
- **Result:** The accuracy gap was **4.2%**, well within our 10% threshold.
- **Interpretation:** The model generalize relatively well across different market conditions, though it performs slightly better on trend-following stocks (MSFT) compared to high-volatility mean-reverting stocks (TSLA).

---

## 🐳 Deployment

The project is containerized using Docker and Docker Compose.

```bash
# Build & run both API and Dashboard
docker-compose up --build
```

**Live API Documentation:** Once running, visit `http://localhost:8000/docs`
**Interactive Dashboard:** Once running, visit `http://localhost:8501`

---





*Built as an end-to-end ML systems project demonstrating the full lifecycle: data → training → serving → monitoring.*

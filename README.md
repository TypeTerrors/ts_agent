# ts_trade Neural Trading Engine

This project builds a TypeScript-based neural trading engine that consumes KuCoin trade data, aggregates it into feature-rich windows, feeds those windows into a temporal convolutional network (TCN), and adjusts model-driven signals with a risk map to produce bounded exposure guidance. The training runtime lives under `training/`, while a Go websocket API under `api/` streams new predictions to subscribers.

The sections below walk through the entire flow—data ingestion, feature construction, model training, risk management, and decision output—in the exact order that the application executes. Setup and run instructions follow at the end.

---

## 1. Environment & Tooling Setup

1. **System prerequisites**
   - Node.js 18+ (the repo was developed with Node 22.13.1).
   - npm (bundled with Node installations).

2. **Install dependencies**
   ```bash
   cd training
   npm install
   ```
   This pulls the runtime libraries (`@tensorflow/tfjs-node`, `axios`) and development tooling (`typescript`, `ts-node`, `@types/node`).

3. **Project layout essentials**
   - `training/src/` – TypeScript trading engine (client, data, model, risk, orchestration).
   - `training/package.json` – npm scripts and dependencies for the trainer.
   - `api/` – Go websocket API that relays prediction inserts from Postgres to connected clients (`/ws`).
   - `fed/` – React + Tailwind dashboard (shadcn-inspired UI) that subscribes to the websocket feed.
   - `docker-compose.yml` – Spins up the trainer, Postgres, and websocket API on a single network.

---

## 2. Data Acquisition (KuCoin REST Client)

**File:** `training/src/client/kucoinClient.ts`

1. Builds an `AxiosInstance` that defaults to `https://api.kucoin.com`.
2. Exposes `fetchRecentTrades(symbol, limit)`:
   - Enforces KuCoin’s 1–500 trade limit.
   - Calls `/api/v1/market/histories` with the requested symbol.
   - Coerces price and size fields to floating-point numbers.
   - Returns an array of `KucoinTrade` records (`sequence`, `price`, `size`, `side`, `time`).

This is the only network-aware component; everything onward manipulates in-memory data structures.

---

## 3. Trade-to-Bar Aggregation

**File:** `src/data/barBuilder.ts`

1. Sorts raw KuCoin trades by ascending timestamp.
2. Buckets trades into fixed-duration candles (`intervalMs`, defaulted later to one minute).
3. Tracks mutable accumulator state per bucket:
   - OHLC price fields (`open`, `high`, `low`, `close`).
   - Volume, buy volume, sell volume, notional amount, trade count.
4. On bucket rollover or at the end of the series, flushes the accumulator into a `TradeBar`:
   ```ts
   {
     startTime, endTime,
     open, high, low, close,
     volume, buyVolume, sellVolume,
     notional, tradeCount,
     vwap
   }
   ```
5. Optionally trims to the latest `maxBars` if requested.

This step transforms tick-level trades into evenly spaced bars that downstream components consume.

---

## 4. Feature Engineering & Labeling

**File:** `training/src/data/featureExtractor.ts`

For each contiguous sequence of bars:
1. Compute per-bar log returns and rolling volatility estimates.
2. Construct a 22-dimensional feature vector comprising:
   - Price action ratios: close, VWAP, log return, spread/close, body/open, and upper/lower shadows.
   - Volume analytics: raw volume, notional value, buy-volume share, signed volume delta, and volume vs. 20-bar SMA.
   - Volatility metrics: rolling return stdev, ATR/close, and Bollinger-percent position.
   - Trend/oscillator signals: close vs. SMA20/EMA12, RSI14, MACD line & signal, TSI (25/13), and 5-bar momentum.
3. Slice the bar series into overlapping `windowSize` sequences (default 64 bars).
4. Optionally z-score each column within the window to normalize features.
5. Derive a binary label for the next bar (`1` if `nextClose > lastClose`, else `0`).
6. Package each window into:
   ```ts
   {
     window: number[][], // shape [windowSize, FEATURE_COUNT]
     label: number,      // 0 or 1
     lastClose: number,
     nextClose: number,
     bars: TradeBar[]
   }
   ```

The final window is reserved for inference; earlier windows feed model training.

---

## 5. Neural Model Definition (TCN)

**File:** `training/src/model/tcn.ts`

1. Defines `TcnConfig` with tunable hyperparameters:
   - `filtersPerLayer`, `kernelSize`, `dropoutRate`, `denseUnits`, `learningRate`.
2. Builds a causal temporal convolutional network:
   - Stacks residual Conv1D blocks (dilation fixed at 1 in tfjs-node for gradient stability).
   - Applies layer normalization and ReLU activations inside each block.
   - Aggregates skip connections, performs global average pooling.
   - Adds a dense “embedding” layer before a single-unit sigmoid output for probability.
3. Compiles the model with Adam optimizer and binary cross-entropy loss.

The architecture outputs the probability that the next bar closes higher than the current bar.

---

## 6. Training Helpers

**File:** `training/src/model/trainer.ts`

1. `tensorsFromWindows`:
   - Converts an array of feature windows into `tf.Tensor3D` inputs (`batch × window × features`) and `tf.Tensor2D` labels.
   - Provides shape validation—throws if the feature vectors are empty.
2. `trainModel`:
   - Wraps `model.fit` with configurable epochs, batch size, validation split, and shuffle flag.
   - Includes early stopping (patience = 3) to avoid overfitting on limited data.

The trading engine uses these helpers to train (or fine-tune) the TCN on recent windows.

---

## 7. Risk Map & Signal Scaling

**File:** `training/src/risk/riskMap.ts`

1. `computeReturns` – converts bars to log returns (for volatility assessment).
2. `computeForecastVolatility` – calculates trailing volatility using a configurable lookback, defaulting to the target if insufficient data exists.
3. `applyRiskMap` – transforms model probabilities into real-valued exposure:
   - Converts probability to a centered signal (`p ∈ [0,1] → signal ∈ [-1,1]`).
   - Applies a neutral-zone deadband (returns zero inside the band).
   - Scales inversely with forecast volatility (capped at 1×).
   - Clamps the final value to `[-maxExposure, maxExposure]`.

The risk map is where the “secret sauce” modulates raw model output according to market conditions.

---

## 8. Trading Engine Orchestration

**File:** `training/src/tradingEngine.ts`

Detailed execution order when `runTradingCycle` is invoked:

1. **Configuration unpacking** – reads symbol, history length, bar interval, window size, client/model/training/risk configs, and volatility lookback.
2. **Trade fetch** – retrieves the requested history (default 500, but the engine can page back thousands of trades when `TRADE_HISTORY_LIMIT` is higher).
3. **Bar construction** – aggregates trades into `TradeBar`s using the requested interval.
4. **Feature window generation** – builds overlapping normalized windows and labels, with 22 engineered features per bar (price/volume ratios, moving-average differentials, RSI, MACD, TSI, ATR scaling, momentum, Bollinger position, and more).
5. **Training sample split** – uses all but the last window for training; reserves the last for inference.
6. **Model instantiation** – builds a TCN tailored to the effective window size and feature count, overlaying any user-defined hyperparameters.
7. **Model training (optional)** – if training samples exist:
   - materialize tensors,
   - fit the model with early stopping,
   - dispose of tensors to release native memory.
8. **Inference** – feed the latest window through the model to obtain `probability`.
9. **Volatility & risk adaptation**:
   - compute log returns,
   - estimate forecast volatility (respecting `volatilityLookback`),
   - apply the risk map to produce a bounded `exposure`.
10. **Return payload** – packages probability, exposure, volatility, latest window, bar count, and number of trained samples.

If insufficient data is available at any checkpoint, the engine returns a neutral decision (probability=0.5, exposure=0).

---

## 9. CLI Entry, Scheduling & Configuration

**File:** `src/index.ts`

1. Reads environment variables to build a `TradingEngineConfig`:
   - `TICKET` (preferred) or `SYMBOL`, `TRADE_HISTORY_LIMIT`, `BAR_INTERVAL_MINUTES`, `FEATURE_WINDOW_SIZE`.
   - TCN overrides (`TCN_FILTERS`, `TCN_KERNEL`, `TCN_DROPOUT`, `TCN_DENSE_UNITS`, `TCN_LEARNING_RATE`).
   - Training overrides (`TRAINING_EPOCHS`, `TRAINING_BATCH_SIZE`, `TRAINING_VAL_SPLIT`, `TRAINING_SHUFFLE`).
   - Risk overrides (`RISK_NEUTRAL_ZONE`, `RISK_VOL_TARGET`, `RISK_VOL_FLOOR`, `RISK_MAX_EXPOSURE`).
   - Persistence overrides (`MODEL_STORE_PATH` defaults to `models/`).
   - Client overrides (`KUCOIN_BASE_URL`, `KUCOIN_TIMEOUT_MS`).
   - Scheduler & logging overrides (`RUN_INTERVAL_MINUTES`, `LOG_FILE`).
2. Runs trading cycles in perpetuity on a 15-minute interval by default:
   - Each cycle generates descriptive logs (start, configuration summary, prediction summary).
   - Logs are streamed to STDOUT and appended to the configured log file (default `logs/trading.log`).
   - A detailed JSON payload of every prediction is written inside the log file.
3. The console still outputs the same messages that are persisted to disk, making live monitoring and post-mortem reviews consistent:
   ```json
   {
     "probability": 0.63,
     "exposure": 0.28,
     "forecastVolatility": 0.012,
     "barsCount": 256,
     "trainedSamples": 191,
     "windowShape": { "rows": 64, "cols": 12 },
     "symbol": "BTC-USDT"
   }
   ```

This entry point is how you interact with the system in batch mode.

---

## 10. Running & Monitoring the Application

1. **Compile TypeScript to JavaScript**
   ```bash
   cd training
   npm run build
   ```

2. **Run the trading cycle directly (on-the-fly compilation)**
   ```bash
   cd training
   npm start
   ```
   or provide custom environment variables:
   ```bash
   TICKET=ETH-USDT RUN_INTERVAL_MINUTES=5 LOG_FILE=logs/eth.log npm start
   ```
   For symbols with sparse trade data, you can shrink the feature window and the candle interval using the provided helper script:
   ```bash
   cd training
   npm run start:short-window
   ```
   This script sets `FEATURE_WINDOW_SIZE=32`, `BAR_INTERVAL_MINUTES=0.25`, `TRAINING_EPOCHS=12`, and `RUN_INTERVAL_MINUTES=5`, which helps ensure enough samples are available for training.

   To gather a larger historical lookback each cycle, use the deep-lookback helper which pages through the KuCoin trade history endpoint until it amasses ~5,000 trades:
   ```bash
   cd training
   npm run start:deep-lookback
   ```

   The trained model is saved under `models/<symbol>/` by default; override with `MODEL_STORE_PATH` if you need a different location. When running in Docker, mount a host volume to this directory so weights persist across container restarts. The process will continue to execute every configured interval until interrupted (Ctrl+C).
   When running with Postgres (see docker-compose), provide `PG_HOST`, `PG_PORT`, `PG_USER`, `PG_PASSWORD`, and `PG_DATABASE` so prediction payloads are persisted automatically.

3. **Execute the compiled output (after `npm run build`)**
   ```bash
   node training/dist/index.js
   ```

4. **Optional verifications**
   - Ensure KuCoin API credentials and rate limits are respected (public endpoints do not require keys, but high-frequency usage should be considerate).
   - Monitor console output for JSON decisions; integrate this output into downstream strategy components as needed.
   - Inspect the log file for historical predictions, configuration summaries, and failure diagnostics.
   - Check the `MODEL_STORE_PATH` directory to confirm `model.json` and `weights.bin` are being updated per symbol.
   - If the websocket API is running, connect to `ws://<host>:<API_PORT>/ws` to receive live prediction payloads as they are inserted into Postgres.

### Docker One-Liner

```bash
docker compose up --build
```

This boots the trainer (Node.js), Postgres, the websocket API (Go), and the React dashboard. By default the dashboard is reachable at `http://localhost:4173` and the websocket at `ws://localhost:8080/ws` (configurable via compose environment variables).

Or use the provided `Makefile`:

```bash
make up      # build images and start the stack
make down    # stop everything
make logs    # tail all service logs
```

Sample environment files are provided as `.env.docker.local.example` and `.env.docker.prod.example`. Copy one to `.env` (Docker Compose loads it automatically) and adjust values as needed—particularly `VITE_WS_URL` when exposing the websocket through Cloudflare tunnels.

## Websocket API

- Build locally via `cd api && go build ./...`.
- Run the server (for local testing) with `go run ./api/cmd/wsapi`.
- When running through Docker Compose, the service listens on `/ws` (default port 8080). Every insert into the `predictions` table triggers a Postgres `NOTIFY`, which the API broadcasts to subscribed websocket clients as raw JSON payloads.
- A simple HTTP endpoint is also available at `GET /recent` to fetch the most recent predictions (default `limit=10`; configurable via `?limit=`). Example: `curl http://localhost:8080/recent?limit=10`.

### Heartbeat configuration

- The server sends websocket PING control frames on an interval to keep intermediaries (e.g., Cloudflare tunnels) from closing idle connections, and expects a PONG in response. These are configurable via environment variables:
  - `WS_PING_SECONDS` – interval between server pings (default `25` seconds). Accepts either an integer (seconds) or a Go duration string (e.g., `30s`, `1m`).
  - `WS_PONG_WAIT_SECONDS` – read deadline extension after receiving a PONG (default `60` seconds, but at least `2× WS_PING_SECONDS`).

The frontend also sends a lightweight JSON heartbeat `{type:"ping"}` on the same cadence and logs `{type:"pong"}` responses for visibility.

## React Dashboard (`fed/`)

- Install dependencies with `cd fed && npm install`.
- Set the websocket endpoint via `VITE_WS_URL` (defaults to `ws://localhost:8080/ws`).
- Optional heartbeat interval via `VITE_WS_PING_MS` (defaults to `25000` ms). This controls how often the UI sends `{type:"ping"}`.
- Start the dev server using `npm run dev`; the UI shows descriptive context on the left and a live feed of prediction cards on the right, with the newest payload inserted at the top.
- Build for production using `npm run build` (outputs to `fed/dist/`).

### Prediction Cards Explained

Each card represents one model inference persisted to Postgres and streamed to the UI.

- Symbol: Trading pair for which the inference was produced (e.g., `BTC-USDT`).
- Timestamp: The time the payload was created or received.
- Probability: Model-estimated probability that the next bar will close higher than the current bar. Displayed as a percentage (e.g., `13.59%`). Values near 50% indicate low directional edge; values closer to 0%/100% indicate stronger directional conviction. This is not a price forecast—only directional likelihood for the immediate next bar.
- Exposure: Final position sizing after the risk map is applied, within `[-100%, 100%]`. Positive values indicate long bias, negative values indicate short bias. Exposure scales with conviction and inversely with forecast volatility, and is clipped to the configured max exposure. This number is what you would trade in a simple proportional strategy.
- Forecast volatility: A near-term volatility estimate (e.g., `3.67e-5`) computed from recent log returns. Lower values imply calmer markets; higher values imply choppier markets. The risk map reduces exposure as this value rises.
- Bars: How many candles were available after aggregation for this cycle (e.g., `68`). A larger number generally provides more context for features, but does not directly indicate training length.
- Samples: The number of windowed samples used to train or fine‑tune the model before inference (e.g., `3`). `0` means the model skipped training in that cycle and only performed inference on the latest window.
- Window: The shape of the input matrix `[rows × cols]` used for inference (e.g., `64×22`). `rows` equals the time steps per window; `cols` equals engineered feature count. This determines the temporal receptive field the model sees per inference.

Front-end layout

- Left panel: Context about the system, current connection status, and where the websocket is connected.
- Right panel: A scrollable feed of the newest predictions at the top (limited to the viewport height for usability). When the page first loads, the UI queries `GET /recent` to seed the list, then listens for live updates via the websocket.

---

## 11. Extending or Troubleshooting

- **Hyperparameter tuning** – Adjust TCN filter stacks or training epochs via environment variables; the engine rebuilds the model dynamically each run.
- **Alternative data intervals** – Change `BAR_INTERVAL_MINUTES` to create longer/shorter bars (e.g., 5-minute candles).
- **Custom risk logic** – Modify `applyRiskMap` if you need advanced exposure curves.
- **Adding persistence** – Persist trained weights by calling `model.save` inside `tradingEngine.ts` if you want continuity between runs.
- **Debugging data issues** – Log the intermediate `TradeBar`s or feature matrices to confirm normalization and labeling when experimenting with new symbols.
- **Database persistence** – The TypeScript runtime can insert each prediction into Postgres. Configure `PG_HOST`, `PG_PORT`, `PG_USER`, `PG_PASSWORD`, and `PG_DATABASE` (defaults provided in `docker-compose.yml`).
- **Websocket API** – The `api/` service (Go) listens on `/ws` and streams new prediction payloads using Postgres `LISTEN/NOTIFY`. Build locally with `cd api && go build ./...` or run via Docker Compose.
- **React dashboard** – The `fed/` app consumes the websocket stream. Start with `npm run dev` (after configuring `VITE_WS_URL`) to visualise predictions in real time.

By following the steps above, you can understand, configure, and run the neural trading application from raw KuCoin trades through to exposure-adjusted decisions. Feel free to build additional tooling or dashboards on top of the JSON output emitted by the CLI.

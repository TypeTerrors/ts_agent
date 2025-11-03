import { appendFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { runTradingCycle } from "./tradingEngine.js";
import type { TradingEngineConfig } from "./tradingEngine.js";
import type { KucoinClientConfig } from "./client/kucoinClient.js";

type Env = NodeJS.ProcessEnv;

const envNumber = (env: Env, key: string, fallback: number): number => {
  const value = env[key];
  if (!value) {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const envList = (env: Env, key: string): number[] | undefined => {
  const value = env[key];
  if (!value) {
    return undefined;
  }
  const parsed = value
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((num) => Number.isFinite(num));
  return parsed.length > 0 ? parsed : undefined;
};

const LOG_FILE = resolve(process.env.LOG_FILE ?? "logs/trading.log");
let logDirEnsured = false;

const ensureLogDir = async () => {
  if (logDirEnsured) {
    return;
  }
  await mkdir(dirname(LOG_FILE), { recursive: true });
  logDirEnsured = true;
};

const writeLog = async (message: string) => {
  const timestamp = new Date().toISOString();
  const formatted = `[${timestamp}] ${message}`;
  console.log(formatted);
  await ensureLogDir();
  await appendFile(LOG_FILE, `${formatted}\n`);
};

const buildClientConfig = (env: Env): KucoinClientConfig | undefined => {
  const client: KucoinClientConfig = {};
  if (env.KUCOIN_BASE_URL) {
    client.baseURL = env.KUCOIN_BASE_URL;
  }
  if (env.KUCOIN_TIMEOUT_MS) {
    client.timeoutMs = envNumber(env, "KUCOIN_TIMEOUT_MS", 10_000);
  }
  return Object.keys(client).length > 0 ? client : undefined;
};

const buildConfig = (env: Env): TradingEngineConfig => {
  const symbol = env.TICKET ?? env.SYMBOL ?? "BTC-USDT";

  const config: TradingEngineConfig = {
    symbol,
    tradeHistoryLimit: envNumber(env, "TRADE_HISTORY_LIMIT", 500),
    barIntervalMs: envNumber(env, "BAR_INTERVAL_MINUTES", 1) * 60_000,
    featureWindowSize: envNumber(env, "FEATURE_WINDOW_SIZE", 64),
    training: {
      epochs: envNumber(env, "TRAINING_EPOCHS", 8),
      batchSize: envNumber(env, "TRAINING_BATCH_SIZE", 32),
      validationSplit: Number(env.TRAINING_VAL_SPLIT ?? 0.1),
      shuffle:
        env.TRAINING_SHUFFLE?.toLowerCase() === "false" ? false : true,
    },
    volatilityLookback: envNumber(env, "VOLATILITY_LOOKBACK", 64),
    risk: {
      neutralZone: Number(env.RISK_NEUTRAL_ZONE ?? 0.05),
      volatilityTarget: Number(env.RISK_VOL_TARGET ?? 0.015),
      volatilityFloor: Number(env.RISK_VOL_FLOOR ?? 1e-4),
      maxExposure: Number(env.RISK_MAX_EXPOSURE ?? 0.5),
    },
  };

  const filters = envList(env, "TCN_FILTERS");
  const tcnConfig = {
    filtersPerLayer: filters,
    kernelSize: envNumber(env, "TCN_KERNEL", 3),
    dropoutRate: Number(env.TCN_DROPOUT ?? 0.1),
    denseUnits: envNumber(env, "TCN_DENSE_UNITS", 64),
    learningRate: Number(env.TCN_LEARNING_RATE ?? 1e-3),
  };

  config.tcn = {
    ...(tcnConfig.filtersPerLayer ? { filtersPerLayer: tcnConfig.filtersPerLayer } : {}),
    kernelSize: tcnConfig.kernelSize,
    dropoutRate: tcnConfig.dropoutRate,
    denseUnits: tcnConfig.denseUnits,
    learningRate: tcnConfig.learningRate,
  };

  const client = buildClientConfig(env);
  if (client) {
    config.client = client;
  }

  return config;
};

const buildPredictionPayload = (symbol: string, result: Awaited<ReturnType<typeof runTradingCycle>>) => {
  const { latestWindow, ...rest } = result;
  return {
    ...rest,
    windowShape: latestWindow
      ? { rows: latestWindow.window.length, cols: latestWindow.window[0]?.length ?? 0 }
      : null,
    symbol,
  };
};

const start = async () => {
  const env = process.env;
  const config = buildConfig(env);
  const runIntervalMinutes = envNumber(env, "RUN_INTERVAL_MINUTES", 15);
  const runIntervalMs = runIntervalMinutes * 60_000;
  const ticket = config.symbol;

  await writeLog(
    `Initialized trading engine for ticket=${ticket}, interval=${runIntervalMinutes}min, log=${LOG_FILE}`,
  );

  let isRunning = false;
  let cycle = 0;

  const executeCycle = async () => {
    if (isRunning) {
      await writeLog("Previous cycle still running; skipping this interval.");
      return;
    }
    isRunning = true;
    cycle += 1;
    const startTime = new Date();
    try {
      await writeLog(
        `Cycle #${cycle} starting at ${startTime.toISOString()} for ticket=${ticket}`,
      );
      await writeLog(
        `Configuration summary: barsLimit=${config.tradeHistoryLimit}, windowSize=${config.featureWindowSize}, volatilityLookback=${config.volatilityLookback}`,
      );
      const result = await runTradingCycle(config);
      const payload = buildPredictionPayload(ticket, result);
      const summary = [
        `prob=${payload.probability.toFixed(4)}`,
        `exposure=${payload.exposure.toFixed(4)}`,
        `vol=${payload.forecastVolatility.toFixed(6)}`,
        `bars=${payload.barsCount}`,
        `trained=${payload.trainedSamples}`,
      ].join(", ");
      await writeLog(`Cycle #${cycle} prediction summary → ${summary}`);
      await writeLog(
        `Cycle #${cycle} prediction payload:\n${JSON.stringify(payload, null, 2)}`,
      );
    } catch (error) {
      const err = error as Error;
      await writeLog(
        `Cycle #${cycle} failed: ${err?.stack ?? err?.message ?? String(err)}`,
      );
    } finally {
      isRunning = false;
    }
  };

  // run immediately
  await executeCycle();

  // schedule subsequent runs
  const interval = setInterval(() => {
    void executeCycle();
  }, runIntervalMs);

  const shutdown = async (signal: NodeJS.Signals) => {
    clearInterval(interval);
    await writeLog(`Received ${signal}. Shutting down gracefully.`);
    process.exit(0);
  };

  process.on("SIGINT", (signal) => {
    void shutdown(signal);
  });
  process.on("SIGTERM", (signal) => {
    void shutdown(signal);
  });

  // keep the process alive
  await new Promise<void>(() => {
    // intentionally empty: promise never resolves
  });
};

start().catch(async (error) => {
  await writeLog(
    `Fatal error during startup: ${(error as Error)?.stack ?? (error as Error)?.message ?? String(
      error,
    )}`,
  );
  process.exitCode = 1;
});

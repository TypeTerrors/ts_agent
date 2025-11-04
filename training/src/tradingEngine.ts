import * as tf from "@tensorflow/tfjs-node";
import { access, mkdir } from "node:fs/promises";
import { constants } from "node:fs";
import { join } from "node:path";
import { KucoinClient } from "./client/kucoinClient.js";
import type { KucoinClientConfig } from "./client/kucoinClient.js";
import { buildBarsFromTrades } from "./data/barBuilder.js";
import {
  buildFeatureWindows,
  FEATURE_COUNT,
} from "./data/featureExtractor.js";
import type { FeatureWindow, TradeBar } from "./types.js";
import { buildTcnModel } from "./model/tcn.js";
import type { TcnConfig } from "./model/tcn.js";
import { tensorsFromWindows, trainModel } from "./model/trainer.js";
import type { TrainingOptions } from "./model/trainer.js";
import {
  applyRiskMap,
  computeForecastVolatility,
  computeReturns,
} from "./risk/riskMap.js";
import type { RiskMapConfig } from "./risk/riskMap.js";

const DEFAULT_BAR_INTERVAL_MS = 60_000;

type PartialTcnConfig = Partial<Omit<TcnConfig, "windowSize" | "featureCount">>;

export interface TradingEngineConfig {
  symbol: string;
  tradeHistoryLimit?: number;
  barIntervalMs?: number;
  featureWindowSize?: number;
  client?: KucoinClientConfig;
  tcn?: PartialTcnConfig;
  training?: TrainingOptions;
  risk?: RiskMapConfig;
  volatilityLookback?: number;
}

export interface TradingDecision {
  probability: number;
  exposure: number;
  forecastVolatility: number;
  latestWindow: FeatureWindow | null;
  barsCount: number;
  trainedSamples: number;
}

const defaultDecision = (barsCount: number): TradingDecision => ({
  probability: 0.5,
  exposure: 0,
  forecastVolatility: 0,
  latestWindow: null,
  barsCount,
  trainedSamples: 0,
});

const pathExists = async (targetPath: string): Promise<boolean> => {
  try {
    await access(targetPath, constants.F_OK);
    return true;
  } catch {
    return false;
  }
};

const sanitizeSymbol = (value: string) =>
  value.replace(/[^a-zA-Z0-9_\-]/g, "_");

const compileModel = (model: tf.LayersModel, learningRate: number) => {
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });
};

export const runTradingCycle = async (
  config: TradingEngineConfig,
): Promise<TradingDecision> => {
  const {
    symbol,
    tradeHistoryLimit = 500,
    barIntervalMs = DEFAULT_BAR_INTERVAL_MS,
    featureWindowSize = 64,
    client: clientConfig,
    tcn: tcnConfig,
    training: trainingConfig,
    risk: riskConfig,
    volatilityLookback = 64,
  } = config;

  const client = new KucoinClient(clientConfig);
  const trades = await client.fetchRecentTrades(symbol, tradeHistoryLimit);

  const bars = buildBarsFromTrades(
    trades,
    barIntervalMs,
    tradeHistoryLimit,
  );

  if (bars.length <= featureWindowSize) {
    return defaultDecision(bars.length);
  }

  const featureWindows = buildFeatureWindows(bars, {
    windowSize: featureWindowSize,
    normalizePerFeature: true,
    volatilityLookback: Math.min(featureWindowSize, 16),
  });

  if (featureWindows.length === 0) {
    return defaultDecision(bars.length);
  }

  const trainingSamples = featureWindows.slice(0, -1);
  const latestWindow = featureWindows.at(-1) ?? null;

  if (!latestWindow) {
    return defaultDecision(bars.length);
  }

  const firstRow = latestWindow.window[0];
  if (!firstRow) {
    return defaultDecision(bars.length);
  }

  const featureCount = firstRow.length;
  const effectiveWindowSize = latestWindow.window.length;

  const modelStorePath = process.env.MODEL_STORE_PATH ?? "models";
  const symbolKey = sanitizeSymbol(symbol);
  const modelDir = join(modelStorePath, symbolKey);
  const modelJsonPath = join(modelDir, "model.json");
  await mkdir(modelStorePath, { recursive: true });

  const learningRate = tcnConfig?.learningRate ?? 1e-3;
  let model: tf.LayersModel | null = null;
  let loadedFromDisk = false;

  if (await pathExists(modelJsonPath)) {
    try {
      const loaded = await tf.loadLayersModel(`file://${modelJsonPath}`);
      const inputShape = loaded.inputs[0]?.shape ?? [];
      const [, loadedWindowSize, loadedFeatureCount] = inputShape;
      if (
        loadedWindowSize === effectiveWindowSize &&
        loadedFeatureCount === featureCount
      ) {
        compileModel(loaded, learningRate);
        model = loaded;
        loadedFromDisk = true;
      } else {
        console.warn(
          `Stored model shape mismatch (expected ${effectiveWindowSize}x${featureCount}, got ${loadedWindowSize}x${loadedFeatureCount}). Rebuilding.`,
        );
        loaded.dispose();
      }
    } catch (error) {
      console.warn(
        `Failed to load stored model for ${symbol}: ${
          (error as Error)?.message ?? String(error)
        }`,
      );
    }
  }

  if (!model) {
    model = buildTcnModel({
      ...(tcnConfig ?? {}),
      windowSize: effectiveWindowSize,
      featureCount,
    });
    compileModel(model, learningRate);
  }

  let trainedSamples = 0;

  if (trainingSamples.length > 0) {
    const tensors = tensorsFromWindows(trainingSamples);
    trainedSamples = trainingSamples.length;
    try {
      await trainModel(model, tensors, trainingConfig);
    } finally {
      tensors.xs.dispose();
      tensors.ys.dispose();
    }
  }

  const windowTensor = tf.tensor3d(latestWindow.window.flat(), [
    1,
    effectiveWindowSize,
    featureCount,
  ]);
  const prediction = model.predict(windowTensor) as tf.Tensor;
  const probabilities = await prediction.data();
  const probability = probabilities[0] ?? 0.5;
  prediction.dispose();
  windowTensor.dispose();

  const returns = computeReturns(bars as TradeBar[]);
  const forecastVolatility = computeForecastVolatility(
    returns,
    Math.min(volatilityLookback, bars.length),
  );
  const exposure = applyRiskMap(probability, forecastVolatility, riskConfig);

  if (trainedSamples > 0 || !loadedFromDisk) {
    await mkdir(modelDir, { recursive: true });
    await model.save(`file://${modelDir}`);
  }

  model.dispose();

  return {
    probability,
    exposure,
    forecastVolatility,
    latestWindow,
    barsCount: bars.length,
    trainedSamples,
  };
};

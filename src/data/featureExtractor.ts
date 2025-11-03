import type { FeatureWindow, TradeBar } from "../types.js";

export const FEATURE_COUNT = 12;

export interface FeatureExtractionOptions {
  windowSize: number;
  normalizePerFeature?: boolean;
  volatilityLookback?: number;
}

type FeatureVector = number[];

const safeDivide = (numerator: number, denominator: number, fallback = 0) => {
  if (!Number.isFinite(numerator) || !Number.isFinite(denominator)) {
    return fallback;
  }
  if (Math.abs(denominator) < 1e-12) {
    return fallback;
  }
  return numerator / denominator;
};

const computeLogReturn = (currentClose: number, prevClose: number) =>
  prevClose > 0 ? Math.log(currentClose / prevClose) : 0;

const rollingStdDev = (
  values: number[],
  index: number,
  lookback: number,
): number => {
  const start = Math.max(0, index - lookback + 1);
  const slice = values.slice(start, index + 1);
  if (slice.length === 0) {
    return 0;
  }
  const mean = slice.reduce((acc, val) => acc + val, 0) / Math.max(slice.length, 1);
  const variance =
    slice.reduce((acc, val) => acc + (val - mean) ** 2, 0) /
    Math.max(slice.length - 1, 1);
  return Math.sqrt(Math.max(variance, 0));
};

const normalizeWindow = (window: number[][]): number[][] => {
  const rows = window.length;
  if (rows === 0) {
    return window;
  }
  const cols = window[0]?.length ?? 0;
  if (cols === 0) {
    return window;
  }

  const means = new Array(cols).fill(0);
  const variances = new Array(cols).fill(0);

  for (const row of window) {
    for (let c = 0; c < cols; c += 1) {
      means[c] += row[c] ?? 0;
    }
  }

  for (let c = 0; c < cols; c += 1) {
    means[c] /= rows;
  }

  for (const row of window) {
    for (let c = 0; c < cols; c += 1) {
      const value = row[c] ?? 0;
      const diff = value - means[c];
      variances[c] += diff * diff;
    }
  }

  for (let c = 0; c < cols; c += 1) {
    variances[c] = Math.sqrt(variances[c] / Math.max(rows - 1, 1));
  }

  return window.map((row) =>
    row.map((value, c) => {
      const stdev = variances[c];
      return stdev < 1e-6 ? 0 : ((value ?? 0) - means[c]) / stdev;
    }),
  );
};

const buildFeatureVector = (
  bars: TradeBar[],
  returns: number[],
  index: number,
  volatilityLookback: number,
): FeatureVector => {
  const bar = bars[index];
  if (!bar) {
    return new Array(FEATURE_COUNT).fill(0);
  }
  const prevBar = index > 0 ? bars[index - 1] ?? bar : bar;
  const prevClose = prevBar.close;
  const logReturn = computeLogReturn(bar.close, prevClose);
  const spread = Math.max(bar.high - bar.low, 1e-12);
  const body = bar.close - bar.open;
  const upperShadow = bar.high - Math.max(bar.open, bar.close);
  const lowerShadow = Math.min(bar.open, bar.close) - bar.low;
  const signVolume = bar.buyVolume - bar.sellVolume;
  const totalVolume = bar.volume || 1e-12;
  const buyRatio = safeDivide(bar.buyVolume, totalVolume, 0);
  const volumeDelta = safeDivide(signVolume, totalVolume, 0);
  const volatility = rollingStdDev(
    returns,
    index,
    Math.max(volatilityLookback, 1),
  );

  return [
    bar.close,
    bar.vwap,
    logReturn,
    safeDivide(spread, bar.close, 0),
    safeDivide(body, bar.open, 0),
    safeDivide(upperShadow, bar.close, 0),
    safeDivide(lowerShadow, bar.close, 0),
    bar.volume,
    bar.notional,
    buyRatio,
    volumeDelta,
    volatility,
  ];
};

export const buildFeatureWindows = (
  bars: TradeBar[],
  options: FeatureExtractionOptions,
): FeatureWindow[] => {
  const { windowSize, normalizePerFeature = true, volatilityLookback = 16 } =
    options;

  if (bars.length <= windowSize) {
    return [];
  }

  const returns = bars.map((bar, index) => {
    const prev = index > 0 ? bars[index - 1] ?? bar : bar;
    return computeLogReturn(bar.close, prev.close);
  });

  const featureWindows: FeatureWindow[] = [];

  for (let idx = windowSize - 1; idx < bars.length - 1; idx += 1) {
    const windowBars = bars.slice(idx - windowSize + 1, idx + 1);
    const featureMatrix = windowBars.map((_, offset) =>
      buildFeatureVector(
        bars,
        returns,
        idx - windowSize + 1 + offset,
        volatilityLookback,
      ),
    );

    const normalized = normalizePerFeature
      ? normalizeWindow(featureMatrix)
      : featureMatrix;
    const currentBar = bars[idx];
    const nextBar = bars[idx + 1];
    if (!currentBar || !nextBar) {
      continue;
    }
    const nextClose = nextBar.close;
    const lastClose = currentBar.close;
    const label = nextClose > lastClose ? 1 : 0;

    featureWindows.push({
      window: normalized,
      label,
      lastClose,
      nextClose,
      bars: windowBars,
    });
  }

  return featureWindows;
};

export const buildFeatureWindow = (
  bars: TradeBar[],
  windowSize: number,
): FeatureWindow | null => {
  const windows = buildFeatureWindows(bars, {
    windowSize,
    normalizePerFeature: true,
  });
  return windows.at(-1) ?? null;
};

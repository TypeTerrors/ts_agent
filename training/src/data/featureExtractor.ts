import type { FeatureWindow, TradeBar } from "../types.js";

export const FEATURE_COUNT = 22;

export interface FeatureExtractionOptions {
  windowSize: number;
  normalizePerFeature?: boolean;
  volatilityLookback?: number;
}

type FeatureVector = number[];

type NumberArray = number[];

const getBar = (bars: TradeBar[], index: number): TradeBar | null =>
  index >= 0 && index < bars.length ? bars[index]! : null;

const getValue = (arr: NumberArray, index: number, fallback = 0): number =>
  index >= 0 && index < arr.length && Number.isFinite(arr[index])
    ? arr[index]!
    : fallback;

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
  values: NumberArray,
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

const computeSMA = (values: NumberArray, period: number): NumberArray => {
  const result = new Array(values.length).fill(0);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += getValue(values, i);
    if (i >= period) {
      sum -= getValue(values, i - period);
    }
    if (i >= period - 1) {
      result[i] = sum / period;
    }
  }
  return result;
};

const computeEMA = (values: NumberArray, period: number): NumberArray => {
  const result = new Array(values.length).fill(0);
  if (values.length === 0) {
    return result;
  }
  const alpha = 2 / (period + 1);
  result[0] = getValue(values, 0);
  for (let i = 1; i < values.length; i += 1) {
    result[i] = alpha * getValue(values, i) + (1 - alpha) * result[i - 1];
  }
  return result;
};

const computeRSI = (values: NumberArray, period: number): NumberArray => {
  const result = new Array(values.length).fill(0.5);
  if (values.length < 2) {
    return result;
  }
  let gain = 0;
  let loss = 0;
  for (let i = 1; i <= period && i < values.length; i += 1) {
    const diff = getValue(values, i) - getValue(values, i - 1);
    if (diff >= 0) gain += diff;
    else loss -= diff;
  }
  let avgGain = gain / period;
  let avgLoss = loss / period;
  if (period < values.length) {
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    result[period] = 1 / (1 + 1 / rs);
  }
  for (let i = period + 1; i < values.length; i += 1) {
    const diff = getValue(values, i) - getValue(values, i - 1);
    const currentGain = diff > 0 ? diff : 0;
    const currentLoss = diff < 0 ? -diff : 0;
    avgGain = (avgGain * (period - 1) + currentGain) / period;
    avgLoss = (avgLoss * (period - 1) + currentLoss) / period;
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    result[i] = 1 / (1 + 1 / rs);
  }
  return result;
};

const computeStdDevSeries = (values: NumberArray, period: number): NumberArray =>
  values.map((_, idx) => rollingStdDev(values, idx, period));

const computeMACD = (
  values: NumberArray,
  fastPeriod: number,
  slowPeriod: number,
  signalPeriod: number,
): { macd: NumberArray; signal: NumberArray } => {
  const fast = computeEMA(values, fastPeriod);
  const slow = computeEMA(values, slowPeriod);
  const macd = fast.map((value, index) => value - getValue(slow, index));
  const signal = computeEMA(macd, signalPeriod);
  return { macd, signal };
};

const computeTSI = (
  values: NumberArray,
  longPeriod: number,
  shortPeriod: number,
): NumberArray => {
  const result = new Array(values.length).fill(0);
  if (values.length < 2) {
    return result;
  }
  const momentum = values.map((value, index) =>
    index === 0 ? 0 : value - getValue(values, index - 1),
  );
  const absMomentum = momentum.map((value) => Math.abs(value));
  const emaMomentum1 = computeEMA(momentum, shortPeriod);
  const emaAbsMomentum1 = computeEMA(absMomentum, shortPeriod);
  const emaMomentum2 = computeEMA(emaMomentum1, longPeriod);
  const emaAbsMomentum2 = computeEMA(emaAbsMomentum1, longPeriod);
  return emaMomentum2.map((value, index) => {
    const denominator = getValue(emaAbsMomentum2, index);
    return denominator === 0 ? 0 : value / denominator;
  });
};

const computeATR = (bars: TradeBar[], period: number): NumberArray => {
  const tr = new Array(bars.length).fill(0);
  for (let i = 0; i < bars.length; i += 1) {
    const bar = getBar(bars, i);
    if (!bar) {
      continue;
    }
    const prevBar = getBar(bars, i - 1) ?? bar;
    const prevClose = prevBar.close;
    const highLow = bar.high - bar.low;
    const highPrev = Math.abs(bar.high - prevClose);
    const lowPrev = Math.abs(bar.low - prevClose);
    tr[i] = Math.max(highLow, highPrev, lowPrev);
  }
  const atr = computeEMA(tr, period);
  return atr.map((value, index) => {
    const close = getBar(bars, index)?.close ?? 1;
    return close === 0 ? 0 : value / close;
  });
};

const computeMomentum = (values: NumberArray, lookback: number): NumberArray => {
  const result = new Array(values.length).fill(0);
  for (let i = lookback; i < values.length; i += 1) {
    const base = getValue(values, i - lookback);
    result[i] = base === 0 ? 0 : getValue(values, i) / base - 1;
  }
  return result;
};

const computeBollingerPercent = (
  values: NumberArray,
  period: number,
  deviation: number,
): NumberArray => {
  const sma = computeSMA(values, period);
  const std = computeStdDevSeries(values, period);
  return values.map((value, index) => {
    const denom = deviation * getValue(std, index);
    if (!denom) {
      return 0;
    }
    return (value - getValue(sma, index)) / denom;
  });
};

const computeVolumeSmaRatio = (volumes: NumberArray, period: number): NumberArray => {
  const sma = computeSMA(volumes, period);
  return volumes.map((value, index) => {
    const base = getValue(sma, index);
    return !base ? 0 : value / base - 1;
  });
};

interface IndicatorContext {
  volatility: NumberArray;
  smaRatio: NumberArray;
  emaRatio: NumberArray;
  rsi: NumberArray;
  macd: NumberArray;
  macdSignal: NumberArray;
  tsi: NumberArray;
  atrRatio: NumberArray;
  volumeSmaRatio: NumberArray;
  momentum: NumberArray;
  bollingerPercent: NumberArray;
}

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
  returns: NumberArray,
  index: number,
  volatilityLookback: number,
  indicators: IndicatorContext,
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
  const volatility =
    indicators.volatility[index] ?? rollingStdDev(returns, index, Math.max(volatilityLookback, 1));

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
    indicators.smaRatio[index] ?? 0,
    indicators.emaRatio[index] ?? 0,
    indicators.rsi[index] ?? 0.5,
    indicators.macd[index] ?? 0,
    indicators.macdSignal[index] ?? 0,
    indicators.tsi[index] ?? 0,
    indicators.atrRatio[index] ?? 0,
    indicators.volumeSmaRatio[index] ?? 0,
    indicators.momentum[index] ?? 0,
    indicators.bollingerPercent[index] ?? 0,
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

  const closes = bars.map((bar) => bar.close);
  const volumes = bars.map((bar) => bar.volume);
  const volatilitySeries = returns.map((_, idx) =>
    rollingStdDev(returns, idx, Math.max(volatilityLookback, 1)),
  );
  const sma20 = computeSMA(closes, 20);
  const ema12 = computeEMA(closes, 12);
  const smaRatio = closes.map((value, index) => {
    const base = getValue(sma20, index);
    return !base ? 0 : value / base - 1;
  });
  const emaRatio = closes.map((value, index) => {
    const base = getValue(ema12, index);
    return !base ? 0 : value / base - 1;
  });
  const rsi14 = computeRSI(closes, 14);
  const { macd, signal: macdSignal } = computeMACD(closes, 12, 26, 9);
  const tsi = computeTSI(closes, 25, 13);
  const atrRatio = computeATR(bars, 14);
  const volumeSmaRatio = computeVolumeSmaRatio(volumes, 20);
  const momentum5 = computeMomentum(closes, 5);
  const bollingerPercent = computeBollingerPercent(closes, 20, 2);

  const indicators: IndicatorContext = {
    volatility: volatilitySeries,
    smaRatio,
    emaRatio,
    rsi: rsi14,
    macd,
    macdSignal,
    tsi,
    atrRatio,
    volumeSmaRatio,
    momentum: momentum5,
    bollingerPercent,
  };

  const featureWindows: FeatureWindow[] = [];

  for (let idx = windowSize - 1; idx < bars.length - 1; idx += 1) {
    const windowBars = bars.slice(idx - windowSize + 1, idx + 1);
    const featureMatrix = windowBars.map((_, offset) =>
      buildFeatureVector(
        bars,
        returns,
        idx - windowSize + 1 + offset,
        volatilityLookback,
        indicators,
      ),
    );

    const normalized = normalizePerFeature
      ? normalizeWindow(featureMatrix)
      : featureMatrix;
    const currentBar = getBar(bars, idx);
    const nextBar = getBar(bars, idx + 1);
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

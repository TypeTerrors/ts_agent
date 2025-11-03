import type { TradeBar } from "../types.js";

export interface RiskMapConfig {
  neutralZone?: number;
  volatilityTarget?: number;
  volatilityFloor?: number;
  maxExposure?: number;
}

const defaultConfig: Required<RiskMapConfig> = {
  neutralZone: 0.05,
  volatilityTarget: 0.015,
  volatilityFloor: 1e-4,
  maxExposure: 0.5,
};

export const computeForecastVolatility = (
  returns: number[],
  lookback: number,
): number => {
  if (returns.length === 0) {
    return defaultConfig.volatilityTarget;
  }
  const start = Math.max(0, returns.length - lookback);
  const slice = returns.slice(start);
  if (slice.length === 0) {
    return defaultConfig.volatilityTarget;
  }
  const mean = slice.reduce((acc, value) => acc + value, 0) / slice.length;
  const variance =
    slice.reduce((acc, value) => acc + (value - mean) ** 2, 0) /
    Math.max(slice.length - 1, 1);
  return Math.sqrt(Math.max(variance, 0));
};

export const applyRiskMap = (
  probability: number,
  forecastVolatility: number,
  config: RiskMapConfig = {},
): number => {
  const merged = { ...defaultConfig, ...config };

  const clippedProb = Math.min(Math.max(probability, 0), 1);
  const rawSignal = clippedProb * 2 - 1; // map to [-1, 1]

  if (Math.abs(rawSignal) < merged.neutralZone) {
    return 0;
  }

  const vol = Math.max(forecastVolatility, merged.volatilityFloor);
  const scale = Math.min(merged.volatilityTarget / vol, 1);
  const scaledSignal = rawSignal * scale;

  return Math.max(-merged.maxExposure, Math.min(merged.maxExposure, scaledSignal));
};

export const computeReturns = (bars: TradeBar[]): number[] => {
  return bars.map((bar, index) => {
    if (index === 0) {
      return 0;
    }
    const prev = bars[index - 1];
    if (!prev || prev.close <= 0 || bar.close <= 0) {
      return 0;
    }
    return Math.log(bar.close / prev.close);
  });
};

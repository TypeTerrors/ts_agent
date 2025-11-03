import type { KucoinTrade, TradeBar } from "../types.js";

interface MutableBarAccumulator {
  startTime: number;
  endTime: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  buyVolume: number;
  sellVolume: number;
  tradeCount: number;
  notional: number;
}

const roundTo = (value: number, decimals: number) => {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
};

const floorToInterval = (timestampMs: number, intervalMs: number): number => {
  return Math.floor(timestampMs / intervalMs) * intervalMs;
};

const createAccumulator = (
  bucketStart: number,
  bucketEnd: number,
  trade: KucoinTrade,
): MutableBarAccumulator => ({
  startTime: bucketStart,
  endTime: bucketEnd,
  open: trade.price,
  high: trade.price,
  low: trade.price,
  close: trade.price,
  volume: trade.size,
  buyVolume: trade.side === "buy" ? trade.size : 0,
  sellVolume: trade.side === "sell" ? trade.size : 0,
  tradeCount: 1,
  notional: trade.price * trade.size,
});

const finalizeAccumulator = (accumulator: MutableBarAccumulator): TradeBar => {
  const volume = accumulator.volume || 1e-12;
  return {
    startTime: accumulator.startTime,
    endTime: accumulator.endTime,
    open: accumulator.open,
    high: accumulator.high,
    low: accumulator.low,
    close: accumulator.close,
    volume: roundTo(accumulator.volume, 8),
    buyVolume: roundTo(accumulator.buyVolume, 8),
    sellVolume: roundTo(accumulator.sellVolume, 8),
    tradeCount: accumulator.tradeCount,
    notional: roundTo(accumulator.notional, 8),
    vwap: roundTo(accumulator.notional / volume, 8),
  };
};

export const buildBarsFromTrades = (
  trades: KucoinTrade[],
  intervalMs: number,
  maxBars?: number,
): TradeBar[] => {
  if (trades.length === 0) {
    return [];
  }

  const sorted = [...trades].sort((a, b) => a.time - b.time);
  const firstTrade = sorted[0];
  if (!firstTrade) {
    return [];
  }

  const bars: TradeBar[] = [];
  let bucketStart = floorToInterval(firstTrade.time, intervalMs);
  let bucketEnd = bucketStart + intervalMs;
  let accumulator: MutableBarAccumulator | null = null;

  const flush = () => {
    if (!accumulator) {
      return;
    }
    bars.push(finalizeAccumulator(accumulator));
    accumulator = null;
  };

  for (const trade of sorted) {
    while (trade.time >= bucketEnd) {
      flush();
      bucketStart = bucketEnd;
      bucketEnd += intervalMs;
    }

    if (!accumulator) {
      accumulator = createAccumulator(bucketStart, bucketEnd, trade);
      continue;
    }

    accumulator.high = Math.max(accumulator.high, trade.price);
    accumulator.low = Math.min(accumulator.low, trade.price);
    accumulator.close = trade.price;
    accumulator.volume += trade.size;
    accumulator.notional += trade.price * trade.size;
    accumulator.tradeCount += 1;
    if (trade.side === "buy") {
      accumulator.buyVolume += trade.size;
    } else {
      accumulator.sellVolume += trade.size;
    }
  }

  flush();

  if (maxBars && bars.length > maxBars) {
    return bars.slice(-maxBars);
  }

  return bars;
};

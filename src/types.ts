export interface KucoinTrade {
  sequence: string;
  price: number;
  size: number;
  side: "buy" | "sell";
  time: number;
}

export interface TradeBar {
  startTime: number;
  endTime: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  vwap: number;
  buyVolume: number;
  sellVolume: number;
  notional: number;
  tradeCount: number;
}

export type FeatureVector = number[];

export interface FeatureWindow {
  window: FeatureVector[];
  label: number;
  lastClose: number;
  nextClose: number;
  bars: TradeBar[];
}

import axios from "axios";

import type { AxiosInstance } from "axios";
import type { KucoinTrade } from "../types.js";
import { writeLog } from "../index.js";

export interface KucoinClientConfig {
  baseURL?: string;
  timeoutMs?: number;
}

interface KucoinTradeResponse {
  data: Array<{
    sequence: string;
    price: string;
    size: string;
    side: "buy" | "sell";
    time: number;
  }>;
}

export class KucoinClient {
  private static readonly MAX_BATCH_SIZE = 500;
  private readonly http: AxiosInstance;

  constructor(config: KucoinClientConfig = {}) {
    this.http = axios.create({
      baseURL: config.baseURL ?? "https://api.kucoin.com",
      timeout: config.timeoutMs ?? 10_000,
    });
  }

  private async fetchBatch(
    symbol: string,
    limit: number,
    endAt?: number,
  ): Promise<KucoinTrade[]> {
    const response = await this.http.get<KucoinTradeResponse>(
      "/api/v1/market/histories",
      {
        params: {
          symbol,
          limit: Math.min(limit, KucoinClient.MAX_BATCH_SIZE),
          ...(endAt ? { endAt } : {}),
        },
      },
    );

    writeLog(`Kucoin response status ${response.status} Kucoin message ${response.statusText} Kucoin Data ${response.data.data}`)

    return response.data.data.map((trade) => ({
      sequence: trade.sequence,
      price: Number(trade.price),
      size: Number(trade.size),
      side: trade.side,
      time: trade.time,
    }));
  }

  async fetchRecentTrades(
    symbol: string,
    desiredCount = KucoinClient.MAX_BATCH_SIZE,
  ): Promise<KucoinTrade[]> {
    if (desiredCount <= 0) {
      throw new Error("desiredCount must be positive");
    }

    const collected: KucoinTrade[] = [];
    const seenSequences = new Set<string>();

    let remaining = desiredCount;
    let endAt: number | undefined;

    while (remaining > 0) {
      const requestLimit = Math.min(remaining, KucoinClient.MAX_BATCH_SIZE);
      const batch = await this.fetchBatch(symbol, requestLimit, endAt);
      if (batch.length === 0) {
        break;
      }

      for (const trade of batch) {
        if (!seenSequences.has(trade.sequence)) {
          collected.push(trade);
          seenSequences.add(trade.sequence);
        }
      }

      remaining = desiredCount - collected.length;

      if (batch.length < requestLimit) {
        break;
      }

      const oldestTrade = batch[batch.length - 1]!;
      endAt = oldestTrade.time - 1;
    }

    return collected.sort((a, b) => a.time - b.time);
  }
}

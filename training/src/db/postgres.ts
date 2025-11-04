import { Pool } from "pg";

export interface PostgresConfig {
  host: string;
  port: number;
  user: string;
  password: string;
  database: string;
  ssl?: boolean;
}

const CREATE_TABLE_SQL = `
  CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    exposure DOUBLE PRECISION NOT NULL,
    forecast_volatility DOUBLE PRECISION NOT NULL,
    bars_count INTEGER NOT NULL,
    trained_samples INTEGER NOT NULL,
    window_rows INTEGER,
    window_cols INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  );
`;

const INSERT_PREDICTION_SQL = `
  INSERT INTO predictions
    (symbol, probability, exposure, forecast_volatility, bars_count, trained_samples, window_rows, window_cols)
  VALUES
    ($1, $2, $3, $4, $5, $6, $7, $8);
`;

const NOTIFY_SQL = "SELECT pg_notify('predictions', $1::text);";

export interface PredictionPayload {
  symbol: string;
  probability: number;
  exposure: number;
  forecastVolatility: number;
  barsCount: number;
  trainedSamples: number;
  windowShape: { rows: number; cols: number } | null;
}

export class PredictionRepository {
  private readonly pool: Pool;

  constructor(pool: Pool) {
    this.pool = pool;
  }

  static async initialize(config: PostgresConfig): Promise<PredictionRepository> {
    const pool = new Pool({
      host: config.host,
      port: config.port,
      user: config.user,
      password: config.password,
      database: config.database,
      ssl: config.ssl ?? false,
    });

    await pool.query(CREATE_TABLE_SQL);
    return new PredictionRepository(pool);
  }

  async savePrediction(payload: PredictionPayload): Promise<void> {
    const windowRows = payload.windowShape?.rows ?? null;
    const windowCols = payload.windowShape?.cols ?? null;

    await this.pool.query(INSERT_PREDICTION_SQL, [
      payload.symbol,
      payload.probability,
      payload.exposure,
      payload.forecastVolatility,
      payload.barsCount,
      payload.trainedSamples,
      windowRows,
      windowCols,
    ]);

    try {
      await this.pool.query(NOTIFY_SQL, [JSON.stringify(payload)]);
    } catch (error) {
      // Notification failures shouldn't block the training loop; log to stderr.
      console.error("pg_notify failed", error);
    }
  }

  async close(): Promise<void> {
    await this.pool.end();
  }
}

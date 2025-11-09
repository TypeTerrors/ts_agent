package postgres

import (
    "context"
    "fmt"
    "time"

    "github.com/charmbracelet/log"
    "github.com/jackc/pgx/v5/pgxpool"
    "github.com/jackc/pgx/v5/pgtype"
)

// Repository provides read access to recent predictions.
type Repository struct {
    pool *pgxpool.Pool
}

// WindowShape represents the optional rows/cols shape of the input window.
type WindowShape struct {
    Rows int32 `json:"rows"`
    Cols int32 `json:"cols"`
}

// Prediction is a JSON-friendly representation of a prediction row.
type Prediction struct {
    Symbol             string       `json:"symbol"`
    Probability        float64      `json:"probability"`
    Exposure           float64      `json:"exposure"`
    ForecastVolatility float64      `json:"forecastVolatility"`
    BarsCount          int32        `json:"barsCount"`
    TrainedSamples     int32        `json:"trainedSamples"`
    WindowShape        *WindowShape `json:"windowShape"`
    CreatedAt          time.Time    `json:"createdAt"`
}

// NewRepository initialises a connection pool for read queries using env config.
func NewRepository(ctx context.Context) (*Repository, error) {
    cfg := fromEnv()
    connString := fmt.Sprintf("postgres://%s:%s@%s:%s/%s?sslmode=%s",
        cfg.User,
        cfg.Password,
        cfg.Host,
        cfg.Port,
        cfg.Database,
        cfg.SSLMode,
    )
    pool, err := pgxpool.New(ctx, connString)
    if err != nil {
        return nil, err
    }
    log.Info("postgres pool initialised", "host", cfg.Host, "port", cfg.Port, "database", cfg.Database)
    return &Repository{pool: pool}, nil
}

// Close releases the underlying pool resources.
func (r *Repository) Close() {
    r.pool.Close()
}

// GetRecentPredictions returns the most recent N predictions ordered by created_at DESC.
func (r *Repository) GetRecentPredictions(ctx context.Context, limit int) ([]Prediction, error) {
    if limit <= 0 {
        limit = 10
    }
    if limit > 100 {
        limit = 100
    }
    const q = `
        SELECT
            symbol,
            probability,
            exposure,
            forecast_volatility,
            bars_count,
            trained_samples,
            window_rows,
            window_cols,
            created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT $1
    `
    rows, err := r.pool.Query(ctx, q, limit)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    out := make([]Prediction, 0, limit)
    for rows.Next() {
        var (
            symbol string
            probability float64
            exposure float64
            forecastVol float64
            barsCount int32
            trainedSamples int32
            windowRows pgtype.Int4
            windowCols pgtype.Int4
            createdAt time.Time
        )
        if err := rows.Scan(
            &symbol,
            &probability,
            &exposure,
            &forecastVol,
            &barsCount,
            &trainedSamples,
            &windowRows,
            &windowCols,
            &createdAt,
        ); err != nil {
            return nil, err
        }
        var ws *WindowShape
        if windowRows.Valid && windowCols.Valid {
            ws = &WindowShape{Rows: windowRows.Int32, Cols: windowCols.Int32}
        } else {
            ws = nil
        }
        out = append(out, Prediction{
            Symbol:             symbol,
            Probability:        probability,
            Exposure:           exposure,
            ForecastVolatility: forecastVol,
            BarsCount:          barsCount,
            TrainedSamples:     trainedSamples,
            WindowShape:        ws,
            CreatedAt:          createdAt,
        })
    }
    if rows.Err() != nil {
        return nil, rows.Err()
    }
    return out, nil
}


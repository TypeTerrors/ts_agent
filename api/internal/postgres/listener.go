package postgres

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/charmbracelet/log"
	"github.com/jackc/pgx/v5"
)

type Listener struct {
	conn   *pgx.Conn
	ctx    context.Context
	cancel context.CancelFunc
}

type Config struct {
	Host     string
	Port     string
	User     string
	Password string
	Database string
	SSLMode  string
}

func fromEnv() Config {
	return Config{
		Host:     getenv("PG_HOST", "localhost"),
		Port:     getenv("PG_PORT", "5432"),
		User:     getenv("PG_USER", "trader"),
		Password: getenv("PG_PASSWORD", "traderpass"),
		Database: getenv("PG_DATABASE", "trading"),
		SSLMode:  getenv("PG_SSLMODE", "disable"),
	}
}

func getenv(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return fallback
}

func New(ctx context.Context) (*Listener, error) {
	cfg := fromEnv()
	connString := fmt.Sprintf("postgres://%s:%s@%s:%s/%s?sslmode=%s",
		cfg.User,
		cfg.Password,
		cfg.Host,
		cfg.Port,
		cfg.Database,
		cfg.SSLMode,
	)
	conn, err := pgx.Connect(ctx, connString)
	if err != nil {
		return nil, err
	}
	log.Info("connected to postgres", "host", cfg.Host, "port", cfg.Port, "database", cfg.Database)
	listenCtx, cancel := context.WithCancel(ctx)
	return &Listener{conn: conn, ctx: listenCtx, cancel: cancel}, nil
}

func (l *Listener) Listen(channel string) error {
	identifier := pgx.Identifier{channel}
	sql := fmt.Sprintf("LISTEN %s", identifier.Sanitize())
	_, err := l.conn.Exec(l.ctx, sql)
	return err
}

func (l *Listener) Wait(handler func(string)) error {
	for {
		notification, err := l.conn.WaitForNotification(l.ctx)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return nil
			}
			return err
		}
		handler(notification.Payload)
	}
}

func (l *Listener) Close() error {
	l.cancel()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	err := l.conn.Close(ctx)
	if err != nil {
		log.Error("failed to close postgres connection", "err", err)
		return err
	}
	log.Info("postgres listener connection closed")
	return nil
}

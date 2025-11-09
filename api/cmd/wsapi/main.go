package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/charmbracelet/log"

	"ts_trade_ws/internal/server"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	srv, err := server.New(ctx)
	if err != nil {
		log.Fatal("failed to start server", "err", err)
	}

	http.HandleFunc("/ws", srv.HandleWS)
	http.HandleFunc("/recent", srv.HandleRecent)

	port := os.Getenv("API_PORT")
	if port == "" {
		port = "8080"
	}

	httpSrv := &http.Server{
		Addr:              ":" + port,
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		log.Info("websocket API listening", "port", port)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal("http server error", "err", err)
		}
	}()

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	<-sigs
	log.Info("shutdown signal received")

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		log.Warn("http shutdown error", "err", err)
	}
	srv.Close()
	cancel()
}

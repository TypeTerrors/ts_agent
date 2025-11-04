package server

import (
	"context"
	"net/http"
	"sync"

	"github.com/charmbracelet/log"
	"github.com/gorilla/websocket"

	"ts_trade_ws/internal/hub"
	"ts_trade_ws/internal/postgres"
)

type Server struct {
	upgrader websocket.Upgrader
	hub      *hub.Hub
	listener *postgres.Listener
	ctx      context.Context
	cancel   context.CancelFunc
	wg       sync.WaitGroup
}

func New(ctx context.Context) (*Server, error) {
	h := hub.New()
	listener, err := postgres.New(ctx)
	if err != nil {
		return nil, err
	}
	if err := listener.Listen("predictions"); err != nil {
		listener.Close()
		return nil, err
	}
	ctx, cancel := context.WithCancel(ctx)
	s := &Server{
		hub:      h,
		listener: listener,
		ctx:      ctx,
		cancel:   cancel,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		},
	}
	s.wg.Add(1)
	go s.listenForNotifications()
	log.Info("websocket server initialised", "channel", "predictions")
	return s, nil
}

func (s *Server) listenForNotifications() {
	defer s.wg.Done()
	err := s.listener.Wait(func(payload string) {
		s.hub.Broadcast(payload)
		log.Debug("broadcast payload", "length", len(payload))
	})
	if err != nil {
		log.Error("postgres listener stopped", "err", err)
	}
}

func (s *Server) HandleWS(w http.ResponseWriter, r *http.Request) {
	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Error("websocket upgrade failed", "err", err)
		return
	}
	client := hub.NewWSClient(conn)
	s.hub.Register(client)
	log.Info("client connected", "remote", r.RemoteAddr)
	go client.Run(func() {
		s.hub.Unregister(client)
		log.Info("client disconnected", "remote", r.RemoteAddr)
	})
}

func (s *Server) Close() {
	s.cancel()
	s.listener.Close()
	s.hub.Shutdown(context.Background())
	s.wg.Wait()
	log.Info("websocket server shut down")
}

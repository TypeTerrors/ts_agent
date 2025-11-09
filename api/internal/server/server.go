package server

import (
    "context"
    "encoding/json"
    "net/http"
    "strconv"
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
    repo     *postgres.Repository
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
    repo, err := postgres.NewRepository(ctx)
    if err != nil {
        listener.Close()
        return nil, err
    }
    ctx, cancel := context.WithCancel(ctx)
    s := &Server{
        hub:      h,
        listener: listener,
        repo:     repo,
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
    if s.repo != nil {
        s.repo.Close()
    }
    s.hub.Shutdown(context.Background())
    s.wg.Wait()
    log.Info("websocket server shut down")
}

// HandleRecent writes the most recent predictions as JSON.
func (s *Server) HandleRecent(w http.ResponseWriter, r *http.Request) {
    // Default limit is 10, optional ?limit= query param
    limit := 10
    if v := r.URL.Query().Get("limit"); v != "" {
        if n, err := strconv.Atoi(v); err == nil {
            limit = n
        }
    }
    preds, err := s.repo.GetRecentPredictions(r.Context(), limit)
    if err != nil {
        log.Error("failed to fetch recent predictions", "err", err)
        http.Error(w, "failed to fetch predictions", http.StatusInternalServerError)
        return
    }
    w.Header().Set("Content-Type", "application/json")
    // Basic CORS for simple GET requests (frontend may be on another origin)
    w.Header().Set("Access-Control-Allow-Origin", "*")
    enc := json.NewEncoder(w)
    if err := enc.Encode(preds); err != nil {
        log.Error("failed to encode predictions", "err", err)
    }
}

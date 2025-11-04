package hub

import (
	"context"
	"sync"
)

type Message struct {
	Payload string
}

type Client interface {
	Send(msg Message)
	Close()
}

type Hub struct {
	clients    map[Client]struct{}
	register   chan Client
	unregister chan Client
	broadcast  chan Message
	mu         sync.RWMutex
}

func New() *Hub {
	h := &Hub{
		clients:    make(map[Client]struct{}),
		register:   make(chan Client),
		unregister: make(chan Client),
		broadcast:  make(chan Message, 32),
	}
	go h.run()
	return h
}

func (h *Hub) run() {
	for {
		select {
		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = struct{}{}
			h.mu.Unlock()
		case client := <-h.unregister:
			h.removeClient(client)
		case msg := <-h.broadcast:
			h.mu.RLock()
			for c := range h.clients {
				c.Send(msg)
			}
			h.mu.RUnlock()
		}
	}
}

func (h *Hub) removeClient(client Client) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if _, ok := h.clients[client]; ok {
		delete(h.clients, client)
		client.Close()
	}
}

func (h *Hub) Broadcast(payload string) {
	h.broadcast <- Message{Payload: payload}
}

func (h *Hub) Register(client Client) {
	h.register <- client
}

func (h *Hub) Unregister(client Client) {
	h.unregister <- client
}

func (h *Hub) Shutdown(ctx context.Context) {
	done := make(chan struct{})
	go func() {
		h.mu.Lock()
		for c := range h.clients {
			c.Close()
			delete(h.clients, c)
		}
		h.mu.Unlock()
		close(done)
	}()
	select {
	case <-ctx.Done():
	case <-done:
	}
}

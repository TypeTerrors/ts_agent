package hub

import (
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

const (
	writeWait      = 10 * time.Second
	pongWait       = 2 * time.Minute
	pingPeriod     = 50 * time.Second
	maxMessageSize = 64 * 1024
)

type WSClient struct {
	conn      *websocket.Conn
	mu        sync.Mutex
	done      chan struct{}
	closeOnce sync.Once
}

func NewWSClient(conn *websocket.Conn) *WSClient {
	return &WSClient{
		conn: conn,
		done: make(chan struct{}),
	}
}

func (c *WSClient) Send(msg Message) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.conn == nil {
		return
	}

	if err := c.conn.SetWriteDeadline(time.Now().Add(writeWait)); err != nil {
		go c.Close()
		return
	}

	if err := c.conn.WriteMessage(websocket.TextMessage, []byte(msg.Payload)); err != nil {
		go c.Close()
	}
}

func (c *WSClient) Close() {
	c.closeOnce.Do(func() {
		close(c.done)

		c.mu.Lock()
		defer c.mu.Unlock()

		if c.conn != nil {
			_ = c.conn.WriteControl(
				websocket.CloseMessage,
				websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""),
				time.Now().Add(writeWait),
			)
			_ = c.conn.Close()
			c.conn = nil
		}
	})
}

func (c *WSClient) Run(onClose func()) {
	c.mu.Lock()
	if c.conn == nil {
		c.mu.Unlock()
		if onClose != nil {
			onClose()
		}
		return
	}

	c.conn.SetReadLimit(maxMessageSize)
	_ = c.conn.SetReadDeadline(time.Now().Add(pongWait))
	c.conn.SetPongHandler(func(string) error {
		return c.conn.SetReadDeadline(time.Now().Add(pongWait))
	})
	c.mu.Unlock()

	go c.keepAlive()

	defer func() {
		c.Close()
		if onClose != nil {
			onClose()
		}
	}()

	for {
		if _, _, err := c.conn.ReadMessage(); err != nil {
			return
		}
	}
}

func (c *WSClient) keepAlive() {
	ticker := time.NewTicker(pingPeriod)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			c.mu.Lock()
			if c.conn == nil {
				c.mu.Unlock()
				return
			}

			if err := c.conn.SetWriteDeadline(time.Now().Add(writeWait)); err != nil {
				c.mu.Unlock()
				go c.Close()
				return
			}

			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				c.mu.Unlock()
				go c.Close()
				return
			}
			c.mu.Unlock()
		case <-c.done:
			return
		}
	}
}

package hub

import (
    "encoding/json"
    "os"
    "strconv"
    "strings"
    "sync"
    "time"

    "github.com/charmbracelet/log"
    "github.com/gorilla/websocket"
)

const (
    writeWait      = 10 * time.Second
    maxMessageSize = 64 * 1024
)

// Configurable at runtime via env vars:
//   WS_PING_SECONDS (or Go duration string)
//   WS_PONG_WAIT_SECONDS (or Go duration string)
var (
    // How frequently we send ping control frames to keep intermediaries alive
    pingPeriod = 25 * time.Second
    // How long we wait for the next pong from the client
    pongWait   = 60 * time.Second
)

func init() {
    pingPeriod = envDuration("WS_PING_SECONDS", pingPeriod)
    pongWait = envDuration("WS_PONG_WAIT_SECONDS", pongWait)
    if pongWait < pingPeriod*2 {
        // Ensure reasonable slack beyond ping cadence
        pongWait = pingPeriod * 2
    }
    log.Info("ws heartbeat configured", "ping", pingPeriod.String(), "pongWait", pongWait.String())
}

func envDuration(key string, fallback time.Duration) time.Duration {
    v, ok := os.LookupEnv(key)
    if !ok || strings.TrimSpace(v) == "" {
        return fallback
    }
    if d, err := time.ParseDuration(v); err == nil {
        return d
    }
    if n, err := strconv.Atoi(v); err == nil {
        return time.Duration(n) * time.Second
    }
    return fallback
}

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
    c.conn.SetPongHandler(func(appData string) error {
        remote := c.conn.RemoteAddr().String()
        log.Debug("ws pong received", "remote", remote, "appData", appData)
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
        msgType, data, err := c.conn.ReadMessage()
        if err != nil {
            return
        }
        // Handle simple application-level ping/pong
        if msgType == websocket.TextMessage {
            payload := strings.TrimSpace(string(data))
            // Detect app-level ping either as raw "ping" or JSON {"type":"ping"}
            isPing := false
            if payload == "ping" {
                isPing = true
            } else {
                var tmp struct{ Type string `json:"type"` }
                if json.Unmarshal([]byte(payload), &tmp) == nil && strings.EqualFold(tmp.Type, "ping") {
                    isPing = true
                }
            }
            if isPing {
                remote := c.conn.RemoteAddr().String()
                log.Debug("app ping received", "remote", remote)
                // Respond with a JSON pong message
                resp := map[string]any{
                    "type": "pong",
                    "ts":   time.Now().UTC().Format(time.RFC3339Nano),
                }
                b, _ := json.Marshal(resp)
                c.mu.Lock()
                if c.conn != nil {
                    _ = c.conn.SetWriteDeadline(time.Now().Add(writeWait))
                    if err := c.conn.WriteMessage(websocket.TextMessage, b); err != nil {
                        c.mu.Unlock()
                        go c.Close()
                        return
                    }
                    log.Debug("app pong sent", "remote", remote)
                }
                c.mu.Unlock()
            }
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
            remote := c.conn.RemoteAddr().String()
            log.Debug("ws ping sent", "remote", remote)
            c.mu.Unlock()
        case <-c.done:
            return
        }
    }
}

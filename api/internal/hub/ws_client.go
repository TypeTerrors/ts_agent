package hub

import (
	"github.com/gorilla/websocket"
)

type WSClient struct {
	conn *websocket.Conn
}

func NewWSClient(conn *websocket.Conn) *WSClient {
	return &WSClient{conn: conn}
}

func (c *WSClient) Send(msg Message) {
	if c.conn == nil {
		return
	}
	_ = c.conn.WriteMessage(websocket.TextMessage, []byte(msg.Payload))
}

func (c *WSClient) Close() {
	if c.conn != nil {
		_ = c.conn.Close()
	}
}

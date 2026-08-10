package httpx

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/coder/websocket"
)

type WSMessageType int

const (
	WSMessageUnknown WSMessageType = iota + 1
	WSMessageText
	WSMessageBinary
)

type WSEventHandler func(ctx context.Context, messageType WSMessageType, data []byte) error

var (
	ErrUnsupportedMessageType = errors.New("unsupported message type")
)

type WSHandler func(messageType WSMessageType, data []byte)
type WSClientOptions struct {
	Header     http.Header
	HTTPClient *http.Client
}

type WSClient struct {
	options WSClientOptions
	conn    *websocket.Conn
	resp    *http.Response
}

func WSConnect(ctx context.Context, url string, options WSClientOptions) (*WSClient, error) {

	dialOptions := websocket.DialOptions{HTTPClient: options.HTTPClient}
	if options.Header != nil {
		dialOptions.HTTPHeader = options.Header
	}

	conn, resp, err := websocket.Dial(ctx, url, &dialOptions)
	if err != nil {
		return nil, err
	}

	client := &WSClient{
		options: options,
		conn:    conn,
		resp:    resp,
	}

	return client, nil
}

func (w *WSClient) SetReadLimit(n int64) {
	w.conn.SetReadLimit(n)
}

func (w *WSClient) Close() {
	w.conn.Close(websocket.StatusNormalClosure, "")
}

func (w *WSClient) Ping(ctx context.Context) {
	w.conn.Ping(ctx)
}

func (w *WSClient) Send(ctx context.Context, messageType WSMessageType, data []byte) error {
	switch messageType {
	case WSMessageText:
		return w.conn.Write(ctx, websocket.MessageText, data)
	case WSMessageBinary:
		return w.conn.Write(ctx, websocket.MessageBinary, data)
	default:
		return ErrUnsupportedMessageType
	}
}

func (w *WSClient) SendJsonMessage(ctx context.Context, msg any) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}

	return w.Send(ctx, WSMessageText, data)
}

func (w *WSClient) Recv(ctx context.Context) (WSMessageType, []byte, error) {
	messageType, r, err := w.conn.Reader(ctx)
	if err != nil {
		return WSMessageUnknown, nil, err
	}

	var msgType WSMessageType
	switch messageType {
	case websocket.MessageText:
		msgType = WSMessageText
	case websocket.MessageBinary:
		msgType = WSMessageBinary
	default:
		return WSMessageUnknown, nil, ErrUnsupportedMessageType
	}

	data, err := io.ReadAll(r)
	if err != nil {
		return msgType, nil, err
	}

	return msgType, data, nil
}

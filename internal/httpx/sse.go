package httpx

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"
)

type JsonSseEvent[T any] struct {
	Err  error
	Data *T
}

func PostJsonSseWithConfig[T any](ctx context.Context, conf *Config, url string, body any, opts ...Option) (chan *JsonSseEvent[T], error) {

	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	opts = append(opts, WithHeader("Content-Type", "application/json"))
	opts = append(opts, WithHeader("Accept", "text/event-stream"))
	opts = append(opts, WithHeader("Cache-Control", "no-cache"))
	opts = append(opts, WithHeader("Connection", "keep-alive"))

	rsp, err := PostWithConfig(ctx, conf, url, bytes.NewReader(data), opts...)
	if err != nil {
		return nil, err
	}

	ch := make(chan *JsonSseEvent[T])
	sseRsp := NewSseResponse(rsp.Body)
	go func() {
		defer rsp.Body.Close()
		defer close(ch)

		done := make(chan struct{})
		go func() {
			select {
			case <-ctx.Done():
				_ = rsp.Body.Close()
			case <-done:
			}
		}()
		defer close(done)

		send := func(event *JsonSseEvent[T]) bool {
			select {
			case ch <- event:
				return true
			case <-ctx.Done():
				return false
			}
		}

		for {
			event, err := sseRsp.Recv()
			if err != nil {
				if ctx.Err() != nil && !errors.Is(err, io.EOF) {
					send(&JsonSseEvent[T]{Err: context.Cause(ctx)})
				}
				return
			}

			switch event.Event {
			case "message":
				if event.Data == "[DONE]" {
					return
				}

				if event.Data == `"[DONE]"` {
					return
				}

				var data T
				if err := json.Unmarshal([]byte(event.Data), &data); err != nil {
					send(&JsonSseEvent[T]{Err: fmt.Errorf("decode SSE JSON event: %w", err)})
					return
				}
				if !send(&JsonSseEvent[T]{Data: &data}) {
					return
				}
			}
		}
	}()

	return ch, nil
}

func PostJsonSse[T any](ctx context.Context, url string, body any, opts ...Option) (chan *JsonSseEvent[T], error) {
	return PostJsonSseWithConfig[T](ctx, &Config{}, url, body, opts...)
}

type SseEvent struct {
	ID    string
	Event string
	Data  string
	Retry time.Duration
}

type SseResponse struct {
	reader      *bufio.Reader
	lastEventID string
	retryDelay  time.Duration
	bomSkipped  bool
	isFinished  bool
}

func NewSseResponse(r io.Reader) *SseResponse {
	return &SseResponse{
		reader: bufio.NewReaderSize(r, 32<<10), // 32 KiB 缓冲区
	}
}

// Recv 读取并返回下一个解析好的泛型对象
func (s *SseResponse) Recv() (*SseEvent, error) {
	if s.isFinished {
		return nil, io.EOF
	}

	for {
		ev, err := s.decodeNextEvent()
		if err != nil {
			if errors.Is(err, io.EOF) {
				s.isFinished = true
			}
			return nil, err
		}

		// SSE 规范：如果 data 为空，则不触发事件
		if ev == nil {
			continue
		}

		return ev, nil
	}
}

// decodeNextEvent 读取流直到形成一个完整的 SSE 事件
func (s *SseResponse) decodeNextEvent() (*SseEvent, error) {
	var (
		dataBuf   strings.Builder
		eventType string = "message"
		hasData   bool
	)

	for {
		line, err := s.readRawLine()
		if err != nil {
			if errors.Is(err, io.EOF) && dataBuf.Len() > 0 {
				return nil, io.ErrUnexpectedEOF
			}
			return nil, err
		}

		// 遇到空行：提交当前事件
		if line == "" {
			if !hasData {
				return nil, nil // 这是一个心跳或空帧，继续下一次循环
			}

			// 组装 Event
			dataStr := strings.TrimSuffix(dataBuf.String(), "\n")
			return &SseEvent{
				ID:    s.lastEventID,
				Event: eventType,
				Data:  dataStr,
				Retry: s.retryDelay,
			}, nil
		}

		// 注释行跳过
		if strings.HasPrefix(line, ":") {
			continue
		}

		field, val, _ := strings.Cut(line, ":")
		if len(val) > 0 && val[0] == ' ' {
			val = val[1:]
		}

		switch field {
		case "event":
			eventType = val
		case "data":
			dataBuf.WriteString(val)
			dataBuf.WriteByte('\n')
			hasData = true
		case "id":
			if !strings.ContainsRune(val, '\x00') {
				s.lastEventID = val
			}
		case "retry":
			if ms, err := strconv.Atoi(val); err == nil && ms >= 0 {
				s.retryDelay = time.Duration(ms) * time.Millisecond
			}
		}
	}
}

// readRawLine 处理多种换行符并剥离 BOM
func (s *SseResponse) readRawLine() (string, error) {
	if !s.bomSkipped {
		s.bomSkipped = true
		if r, _, err := s.reader.ReadRune(); err == nil && r != '\uFEFF' {
			_ = s.reader.UnreadRune()
		}
	}

	var line []byte
	for {
		b, err := s.reader.ReadByte()
		if err != nil {
			if len(line) > 0 {
				return string(line), nil
			}
			return "", err
		}
		if b == '\n' {
			break
		}
		if b == '\r' {
			next, _ := s.reader.Peek(1)
			if len(next) > 0 && next[0] == '\n' {
				_, _ = s.reader.Discard(1)
			}
			break
		}
		line = append(line, b)
	}
	return string(line), nil
}

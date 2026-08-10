package v1

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/google/uuid"
	"github.com/labstack/echo/v4"
	"github.com/xucx/gox/logx"
	"github.com/xucx/llmx/types"
)

// See https://docs.anthropic.com/en/api/messages
type ClaudeMessageRequest struct {
	Model         string            `json:"model"`
	Messages      []ClaudeMessage   `json:"messages"`
	System        interface{}       `json:"system,omitempty"` // string or []ClaudeContent
	MaxTokens     int64             `json:"max_tokens,omitempty"`
	Metadata      *ClaudeMetadata   `json:"metadata,omitempty"`
	StopSequences []string          `json:"stop_sequences,omitempty"`
	Stream        bool              `json:"stream,omitempty"`
	Temperature   *float32          `json:"temperature,omitempty"`
	TopP          *float32          `json:"top_p,omitempty"`
	TopK          *int              `json:"top_k,omitempty"`
	Tools         []ClaudeTool      `json:"tools,omitempty"`
	ToolChoice    *ClaudeToolChoice `json:"tool_choice,omitempty"`
	Thinking      *ClaudeThinking   `json:"thinking,omitempty"`
}

type ClaudeMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string or []ClaudeContent
}

type ClaudeContent struct {
	Type      string             `json:"type"`
	Text      string             `json:"text,omitempty"`
	Thinking  string             `json:"thinking,omitempty"`
	Signature string             `json:"signature,omitempty"`
	Source    *ClaudeImageSource `json:"source,omitempty"`
	ID        string             `json:"id,omitempty"`
	Name      string             `json:"name,omitempty"`
	Input     interface{}        `json:"input,omitempty"`
	ToolUseID string             `json:"tool_use_id,omitempty"`
	Content   interface{}        `json:"content,omitempty"` // For tool_result
}

type ClaudeImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type ClaudeMetadata struct {
	UserID string `json:"user_id,omitempty"`
}

type ClaudeTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	InputSchema map[string]interface{} `json:"input_schema"`
}

type ClaudeToolChoice struct {
	Type                   string `json:"type"`
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse *bool  `json:"disable_parallel_tool_use,omitempty"`
}

type ClaudeThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

type ClaudeMessageResponse struct {
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Role         string          `json:"role"`
	Content      []ClaudeContent `json:"content"`
	Model        string          `json:"model"`
	StopReason   *string         `json:"stop_reason"`
	StopSequence *string         `json:"stop_sequence"`
	Usage        ClaudeUsage     `json:"usage"`
}

type ClaudeUsage struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
}

// Streaming events
type ClaudeStreamEvent struct {
	Type         string                 `json:"type"`
	Message      *ClaudeMessageResponse `json:"message,omitempty"`
	Index        int                    `json:"index,omitempty"`
	ContentBlock *ClaudeContent         `json:"content_block,omitempty"`
	Delta        *ClaudeDelta           `json:"delta,omitempty"`
	Usage        *ClaudeUsage           `json:"usage,omitempty"`
}

type ClaudeDelta struct {
	Type         string  `json:"type"`
	Text         string  `json:"text,omitempty"`
	Thinking     string  `json:"thinking,omitempty"`
	Signature    string  `json:"signature,omitempty"`
	PartialJson  string  `json:"partial_json,omitempty"`
	StopReason   *string `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
}

func (s *ApiService) ClaudeCreateMessage(c echo.Context) error {
	req := &ClaudeMessageRequest{}
	if err := c.Bind(req); err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, err.Error())
	}

	ctx := c.Request().Context()

	// Convert messages
	messages := []*types.Message{}

	// Handle System Prompt
	if req.System != nil {
		sysMsg := types.NewMessage(types.MessageRoleSystem)
		switch v := req.System.(type) {
		case string:
			sysMsg.Parts = append(sysMsg.Parts, &types.MessagePart{Text: &types.MessageText{Text: v}})
		case []interface{}:
			for _, item := range v {
				if itemMap, ok := item.(map[string]interface{}); ok {
					if typeVal, ok := itemMap["type"].(string); ok && typeVal == "text" {
						if text, ok := itemMap["text"].(string); ok {
							sysMsg.Parts = append(sysMsg.Parts, &types.MessagePart{Text: &types.MessageText{Text: text}})
						}
					}
				}
			}
		}
		messages = append(messages, sysMsg)
	}

	for _, m := range req.Messages {
		msg, err := fromClaudeMessage(m)
		if err != nil {
			return echo.NewHTTPError(http.StatusBadRequest, err.Error())
		}
		messages = append(messages, msg)
	}

	// Convert options
	options := []types.CompletionOption{}
	if len(req.Tools) > 0 {
		tools, err := fromClaudeTools(req.Tools)
		if err != nil {
			return echo.NewHTTPError(http.StatusBadRequest, err.Error())
		}
		options = append(options, types.CompletionWithTools(tools))
	}

	// Temperature
	if req.Temperature != nil {
		options = append(options, func(opts *types.CompletionOptions) *types.CompletionOptions {
			opts.Temperature = req.Temperature
			return opts
		})
	}

	// MaxTokens
	if req.MaxTokens > 0 {
		options = append(options, func(opts *types.CompletionOptions) *types.CompletionOptions {
			opts.MaxTokens = &req.MaxTokens
			return opts
		})
	}
	if req.Thinking != nil {
		level := 0
		if req.Thinking.Type == "enabled" {
			if req.MaxTokens <= 0 {
				return echo.NewHTTPError(http.StatusBadRequest, "max_tokens is required when thinking is enabled")
			}
			level = int((int64(req.Thinking.BudgetTokens)*10 + req.MaxTokens - 1) / req.MaxTokens)
			if level < 1 {
				level = 1
			}
			if level > 10 {
				level = 10
			}
		}
		options = append(options, types.CompletionWithReasoningLevel(level))
	}

	// TopP
	if req.TopP != nil {
		options = append(options, func(opts *types.CompletionOptions) *types.CompletionOptions {
			opts.TopP = req.TopP
			return opts
		})
	}

	// TopK
	if req.TopK != nil {
		options = append(options, func(opts *types.CompletionOptions) *types.CompletionOptions {
			opts.TopK = req.TopK
			return opts
		})
	}

	// StopSequences
	if len(req.StopSequences) > 0 {
		options = append(options, func(opts *types.CompletionOptions) *types.CompletionOptions {
			opts.StopSequences = req.StopSequences
			return opts
		})
	}

	// Generate
	if req.Stream {
		c.Response().Header().Set(echo.HeaderContentType, "text/event-stream")
		c.Response().Header().Set(echo.HeaderCacheControl, "no-cache")
		c.Response().Header().Set(echo.HeaderConnection, "keep-alive")
		c.Response().WriteHeader(http.StatusOK)

		options = append(options, types.CompletionWithStreamingFunc(func(ctx context.Context, completion *types.Completion) error {
			events, err := toClaudeStreamEvents(completion)
			if err != nil {
				return err
			}

			for _, event := range events {
				eventData, err := json.Marshal(event)
				if err != nil {
					return err
				}
				fmt.Fprintf(c.Response(), "event: %s\ndata: %s\n\n", event.Type, eventData)
			}
			c.Response().Flush()
			return nil
		}))

		// Send message_start event first (simplified)
		// Since we don't have the initial response structure available before Generate returns,
		// we might need to send a dummy message_start or wait for first chunk if it contained metadata.
		// For now, we rely on content_block_delta being acceptable by some clients,
		// but standard Claude stream starts with message_start.

		msgID := "msg_" + uuid.NewString()
		msgStart := ClaudeStreamEvent{
			Type: "message_start",
			Message: &ClaudeMessageResponse{
				ID:      msgID,
				Type:    "message",
				Role:    "assistant",
				Model:   req.Model,
				Content: []ClaudeContent{},
				Usage:   ClaudeUsage{InputTokens: 0, OutputTokens: 0}, // Placeholder
			},
		}

		if startData, err := json.Marshal(msgStart); err == nil {
			fmt.Fprintf(c.Response(), "event: message_start\ndata: %s\n\n", startData)

			// Also send content_block_start for text
			blockStart := ClaudeStreamEvent{
				Type:  "content_block_start",
				Index: 0,
				ContentBlock: &ClaudeContent{
					Type: "text",
					Text: "",
				},
			}
			if blockData, err := json.Marshal(blockStart); err == nil {
				fmt.Fprintf(c.Response(), "event: content_block_start\ndata: %s\n\n", blockData)
			}
			c.Response().Flush()
		}

		_, err := s.models.Completion(ctx, req.Model, messages, options...)
		if err != nil {
			logx.Errorw("llm chat fail", "model", req.Model, "error", err)
			return nil
		}

		// Send message_stop event at the end
		msgStop := ClaudeStreamEvent{
			Type: "message_stop",
		}
		if stopData, err := json.Marshal(msgStop); err == nil {
			fmt.Fprintf(c.Response(), "event: message_stop\ndata: %s\n\n", stopData)
			c.Response().Flush()
		}

		return nil

	} else {
		completion, err := s.models.Completion(ctx, req.Model, messages, options...)
		if err != nil {
			logx.Errorw("llm chat fail", "model", req.Model, "error", err)
			return echo.NewHTTPError(http.StatusInternalServerError, err.Error())
		}

		resp, err := toClaudeMessageResponse(completion)
		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, err.Error())
		}

		return c.JSON(http.StatusOK, resp)
	}
}

func fromClaudeMessage(m ClaudeMessage) (*types.Message, error) {
	role := types.MessageRoleUser
	switch m.Role {
	case "assistant":
		role = types.MessageRoleAssistant
	case "user":
		role = types.MessageRoleUser
	default:
		return nil, fmt.Errorf("%w: claude role %q", types.ErrInvalidMessage, m.Role)
	}

	msg := types.NewMessage(role)

	// Handle Content
	if m.Content != nil {
		switch c := m.Content.(type) {
		case string:
			msg.Parts = append(msg.Parts, &types.MessagePart{Text: &types.MessageText{Text: c}})
		case []interface{}:
			for i, item := range c {
				itemMap, ok := item.(map[string]interface{})
				if !ok {
					return nil, fmt.Errorf("claude content %d is not an object", i)
				}
				itemType, _ := itemMap["type"].(string)
				switch itemType {
				case "text":
					if text, ok := itemMap["text"].(string); ok {
						msg.Parts = append(msg.Parts, &types.MessagePart{Text: &types.MessageText{Text: text}})
					}
				case "image":
					sourceMap, ok := itemMap["source"].(map[string]interface{})
					if !ok {
						return nil, fmt.Errorf("claude image content %d source is missing", i)
					}
					sourceType, _ := sourceMap["type"].(string)
					switch sourceType {
					case "base64":
						mediaType, _ := sourceMap["media_type"].(string)
						data, _ := sourceMap["data"].(string)
						if mediaType == "" || data == "" {
							return nil, fmt.Errorf("claude image content %d base64 source is incomplete", i)
						}
						msg.Parts = append(msg.Parts, &types.MessagePart{File: &types.MessageFile{
							MIMEType: mediaType,
							Data:     data,
						}})
					case "url":
						imageURL, _ := sourceMap["url"].(string)
						if imageURL == "" {
							return nil, fmt.Errorf("claude image content %d URL is empty", i)
						}
						msg.Parts = append(msg.Parts, &types.MessagePart{ImageURL: &types.MessageImageURL{URL: imageURL}})
					default:
						return nil, fmt.Errorf("%w: claude image source type %q", types.ErrUnsupported, sourceType)
					}
				case "thinking":
					thinking, _ := itemMap["thinking"].(string)
					signature, _ := itemMap["signature"].(string)
					msg.Parts = append(msg.Parts, &types.MessagePart{Reasoning: &types.MessageReasoning{
						Text:             thinking,
						ThoughtSignature: signature,
					}})
				case "tool_use":
					id, _ := itemMap["id"].(string)
					name, _ := itemMap["name"].(string)
					input, _ := itemMap["input"] // keep as map/interface

					inputJSON, err := json.Marshal(input)
					if err != nil {
						return nil, fmt.Errorf("marshal claude tool input: %w", err)
					}

					msg.Parts = append(msg.Parts, &types.MessagePart{ToolCall: &types.MessageToolCall{
						ID:        id,
						Type:      types.ToolTypeFunction,
						Name:      name,
						Arguments: string(inputJSON),
					}})
				case "tool_result":
					toolUseID, _ := itemMap["tool_use_id"].(string)
					// Content can be string or array of blocks
					contentVal := ""
					if contentStr, ok := itemMap["content"].(string); ok {
						contentVal = contentStr
					} else {
						// simplistic handling for complex content in tool result
						contentBytes, err := json.Marshal(itemMap["content"])
						if err != nil {
							return nil, fmt.Errorf("marshal claude tool result: %w", err)
						}
						contentVal = string(contentBytes)
					}

					msg.Parts = append(msg.Parts, &types.MessagePart{ToolResult: &types.MessageToolResult{
						ID:     toolUseID,
						Result: contentVal,
					}})
					msg.Role = types.MessageRoleTool
				default:
					return nil, fmt.Errorf("%w: claude content type %q", types.ErrUnsupported, itemType)
				}
			}
		default:
			return nil, fmt.Errorf("claude message content has unsupported shape %T", m.Content)
		}
	}

	if err := msg.Validate(); err != nil {
		return nil, err
	}
	return msg, nil
}

func fromClaudeTools(tools []ClaudeTool) ([]*types.Tool, error) {
	ts := []*types.Tool{}
	for _, t := range tools {
		ts = append(ts, &types.Tool{
			Type: types.ToolTypeFunction,
			Function: &types.ToolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		})
	}
	return ts, nil
}

func toClaudeMessageResponse(c *types.Completion) (*ClaudeMessageResponse, error) {
	resp := &ClaudeMessageResponse{
		ID:      c.Message.ID,
		Type:    "message",
		Role:    string(c.Message.Role),
		Model:   c.Model,
		Content: []ClaudeContent{},
		Usage: ClaudeUsage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
		},
	}

	// Ensure ID starts with msg_
	if resp.ID == "" {
		resp.ID = "msg_" + uuid.NewString()
	} else if len(resp.ID) < 4 || resp.ID[:4] != "msg_" {
		resp.ID = "msg_" + resp.ID
	}

	// Content
	text := c.Message.Text()
	if text != "" {
		resp.Content = append(resp.Content, ClaudeContent{
			Type: "text",
			Text: text,
		})
	}
	for _, part := range c.Message.Parts {
		if part != nil && part.Reasoning != nil {
			resp.Content = append(resp.Content, ClaudeContent{
				Type:      "thinking",
				Thinking:  part.Reasoning.Text,
				Signature: part.Reasoning.ThoughtSignature,
			})
		}
	}

	toolCalls := c.Message.ToolCalls()
	for _, tc := range toolCalls {
		// Parse arguments back to map
		var inputMap map[string]interface{}
		if err := json.Unmarshal([]byte(tc.Arguments), &inputMap); err != nil {
			return nil, fmt.Errorf("decode tool call %q arguments: %w", tc.Name, err)
		}

		resp.Content = append(resp.Content, ClaudeContent{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Name,
			Input: inputMap,
		})
	}

	stopReason := "end_turn"
	if len(toolCalls) > 0 {
		stopReason = "tool_use"
	}
	resp.StopReason = &stopReason

	return resp, nil
}

func toClaudeStreamEvents(c *types.Completion) ([]*ClaudeStreamEvent, error) {
	events := []*ClaudeStreamEvent{}

	if c.Delta {
		reasoning := c.Message.Reasoning()
		if reasoning != "" {
			events = append(events, &ClaudeStreamEvent{
				Type:  "content_block_delta",
				Index: 0,
				Delta: &ClaudeDelta{
					Type:     "thinking_delta",
					Thinking: reasoning,
				},
			})
		}

		text := c.Message.Text()
		if text != "" {
			events = append(events, &ClaudeStreamEvent{
				Type:  "content_block_delta",
				Index: 0,
				Delta: &ClaudeDelta{
					Type: "text_delta",
					Text: text,
				},
			})
		}

		// 检查工具调用增量（需要额外支持，目前types.Message的ToolCalls可能不是增量的）
		// 如果需要完整支持Claude流式协议，这里需要更复杂的状态机来生成:
		// message_start, content_block_start, content_block_delta, content_block_stop, message_delta, message_stop

		// 简单的模拟结束：如果是流的最后一帧（如何判断？）
		// 目前的types.Completion没有明确的“流结束”标志，除了Stream回调结束。
		// 但OpenAI兼容接口通过 "[DONE]" 标记。
		// Claude接口需要在回调外发送 message_stop。
	}

	return events, nil
}

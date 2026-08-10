package v1

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/labstack/echo/v4"
	"github.com/xucx/gox/logx"
	"github.com/xucx/llmx/internal/providers/openai"
	"github.com/xucx/llmx/types"
)

func (s *ApiService) OpenaiCompletion(c echo.Context) error {
	req := &openai.CompletionReq{}
	if err := c.Bind(req); err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, err.Error())
	}

	ctx := c.Request().Context()

	// Convert messages
	messages := []*types.Message{}
	for _, m := range req.Messages {
		msg, err := fromOpenaiMessage(m)
		if err != nil {
			return echo.NewHTTPError(http.StatusBadRequest, err.Error())
		}
		messages = append(messages, msg)
	}

	// Convert options
	options := []types.CompletionOption{}
	if len(req.Tools) > 0 {
		tools, err := fromOpenaiTools(req.Tools)
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
	if req.TopP != nil {
		options = append(options, func(opts *types.CompletionOptions) *types.CompletionOptions {
			opts.TopP = req.TopP
			return opts
		})
	}
	if req.MaxCompletionTokens != nil || req.MaxTokens != nil {
		maxTokens := req.MaxCompletionTokens
		if maxTokens == nil {
			maxTokens = req.MaxTokens
		}
		options = append(options, func(opts *types.CompletionOptions) *types.CompletionOptions {
			opts.MaxTokens = maxTokens
			return opts
		})
	}
	if len(req.Stop) > 0 {
		options = append(options, func(opts *types.CompletionOptions) *types.CompletionOptions {
			opts.StopSequences = append([]string(nil), req.Stop...)
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
			resp, err := toOpenaiCompletionResponse(completion)
			if err != nil {
				return err
			}

			chunkData, err := json.Marshal(resp)
			if err != nil {
				return err
			}

			fmt.Fprintf(c.Response(), "data: %s\n\n", chunkData)
			c.Response().Flush()
			return nil
		}))

		_, err := s.models.Completion(ctx, req.Model, messages, options...)
		if err != nil {
			logx.Errorw("llm chat fail", "model", req.Model, "error", err)
			return nil
		}

		fmt.Fprintf(c.Response(), "data: [DONE]\n\n")
		c.Response().Flush()
		return nil

	} else {
		completion, err := s.models.Completion(ctx, req.Model, messages, options...)
		if err != nil {
			logx.Errorw("llm chat fail", "model", req.Model, "error", err)
			return echo.NewHTTPError(http.StatusInternalServerError, err.Error())
		}

		resp, err := toOpenaiCompletionResponse(completion)
		if err != nil {
			return echo.NewHTTPError(http.StatusInternalServerError, err.Error())
		}

		return c.JSON(http.StatusOK, resp)
	}
}

func fromOpenaiMessage(m *openai.CompletionMessageParam) (*types.Message, error) {
	if m == nil {
		return nil, fmt.Errorf("%w: openai message is nil", types.ErrInvalidMessage)
	}
	role := types.MessageRoleUser
	switch m.Role {
	case "system":
		role = types.MessageRoleSystem
	case "assistant":
		role = types.MessageRoleAssistant
	case "user":
		role = types.MessageRoleUser
	case "tool":
		role = types.MessageRoleTool
	default:
		if m.Role != "" {
			role = types.MessageRole(m.Role)
		}
	}

	msg := types.NewMessage(role)

	if role != types.MessageRoleTool && m.Content != "" {
		msg.Parts = append(msg.Parts, &types.MessagePart{Text: &types.MessageText{Text: m.Content}})
	}
	if m.ReasoningContent != "" {
		msg.Parts = append(msg.Parts, &types.MessagePart{Reasoning: &types.MessageReasoning{Text: m.ReasoningContent}})
	}
	if m.Refusal != "" {
		msg.Parts = append(msg.Parts, &types.MessagePart{Refusal: &types.MessageRefusal{Text: m.Refusal}})
	}

	// Handle Tool Calls
	if len(m.ToolCalls) > 0 {
		for i, tc := range m.ToolCalls {
			if tc == nil || tc.Function == nil {
				return nil, fmt.Errorf("%w: openai tool call %d is nil", types.ErrInvalidMessage, i)
			}
			msg.Parts = append(msg.Parts, &types.MessagePart{ToolCall: &types.MessageToolCall{
				ID:        tc.ID,
				Type:      types.ToolTypeFunction,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			}})
		}
	}

	// Handle Tool Result
	if role == types.MessageRoleTool {
		msg.Parts = append(msg.Parts, &types.MessagePart{ToolResult: &types.MessageToolResult{
			ID:     m.ToolCallID,
			Result: m.Content,
		}})
	}

	if err := msg.Validate(); err != nil {
		return nil, err
	}
	return msg, nil
}

func fromOpenaiTools(tools []*openai.Tool) ([]*types.Tool, error) {
	ts := []*types.Tool{}
	for i, t := range tools {
		if t == nil || t.Function == nil {
			return nil, fmt.Errorf("%w: openai tool %d is nil", types.ErrInvalidMessage, i)
		}
		if t.Type == "function" {
			ts = append(ts, &types.Tool{
				Type: types.ToolTypeFunction,
				Function: &types.ToolFunction{
					Name:        t.Function.Name,
					Description: t.Function.Description,
					Parameters:  t.Function.Parameters,
				},
			})
		} else {
			return nil, fmt.Errorf("%w: openai tool type %q", types.ErrUnsupported, t.Type)
		}
	}
	return ts, nil
}

func toOpenaiCompletionResponse(c *types.Completion) (*openai.CompletionRsp, error) {
	resp := &openai.CompletionRsp{
		ID:      c.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   c.Model,
		Choices: []*openai.CompletionChoise{},
	}
	if resp.ID == "" {
		resp.ID = c.Message.ID
	}

	if c.Delta {
		resp.Object = "chat.completion.chunk"
	}

	choice := &openai.CompletionChoise{
		Index: 0,
	}

	// Use OpenaiCompletion for response message
	msg := &openai.CompletionMessage{}

	content := c.Message.Text()
	if content != "" {
		msg.Content = content
	}
	msg.ReasoningContent = c.Message.Reasoning()
	for _, part := range c.Message.Parts {
		if part != nil && part.Refusal != nil {
			msg.Refusal += part.Refusal.Text
		}
	}

	toolCalls := c.Message.ToolCalls()
	if len(toolCalls) > 0 {
		openaiToolCalls := []openai.CompletionToolCall{}
		for _, tc := range toolCalls {
			openaiToolCalls = append(openaiToolCalls, openai.CompletionToolCall{
				ID:   tc.ID,
				Type: openai.ToolTypeFucntion,
				Function: &openai.CompletionToolCallFunction{
					Name:      tc.Name,
					Arguments: tc.Arguments,
				},
			})
		}
		msg.ToolCalls = openaiToolCalls
	}

	finishReason := "stop"
	if len(toolCalls) > 0 {
		finishReason = "tool_calls"
	}

	if c.Delta {
		choice.Delta = msg
		choice.FinishReason = ""
	} else {
		choice.Message = msg
		choice.FinishReason = finishReason
	}

	resp.Choices = append(resp.Choices, choice)

	if c.Usage.TotalTokens > 0 {
		resp.Usage = openai.CompletionUsage{
			PromptTokens:     c.Usage.PromptTokens,
			CompletionTokens: c.Usage.CompletionTokens,
			TotalTokens:      c.Usage.TotalTokens,
			TokenDetail: openai.CompletionTokenDetail{
				ReasoningTokens: c.Usage.ReasoningTokens,
			},
		}
	}

	return resp, nil
}

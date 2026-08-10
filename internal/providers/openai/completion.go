package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/xucx/llmx/internal/httpx"
	"github.com/xucx/llmx/internal/utils"
	"github.com/xucx/llmx/types"
)

const (
	CompletionModelDefault = openai.ChatModelGPT5_4
	CompletionPath         = "/chat/completions"
)

type CompletionReq struct {
	Model               string                      `json:"model,omitempty"`
	Messages            []*CompletionMessageParam   `json:"messages,omitempty"`
	Stream              bool                        `json:"stream,omitempty"`
	StreamOptions       *CompletionReqStreamOptions `json:"stream_options,omitempty"`
	Tools               []*Tool                     `json:"tools,omitempty"`
	Temperature         *float32                    `json:"temperature,omitempty"`
	TopP                *float32                    `json:"top_p,omitempty"`
	MaxTokens           *int64                      `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int64                      `json:"max_completion_tokens,omitempty"`
	Stop                []string                    `json:"stop,omitempty"`
	ReasoningEffort     *ReasoningEffort            `json:"reasoning_effort,omitempty"`
	ExtraBody           map[string]any              `json:"-"`
}

func (c *CompletionReq) MarshalJSON() ([]byte, error) {
	type Alias CompletionReq
	b, err := json.Marshal((*Alias)(c))
	if err != nil {
		return nil, err
	}

	if c.ExtraBody == nil {
		return b, nil
	}

	m := map[string]any{}
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	for k, v := range c.ExtraBody {
		m[k] = v
	}

	return json.Marshal(m)
}

type CompletionReqStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}
type CompletionReqExtra struct {
	Thinking           any `json:"thinking,omitempty"`
	ChatTemplateKwargs any `json:"chat_template_kwargs,omitempty"`
}

type CompletionRsp struct {
	ID                string              `json:"id,omitempty"`
	Object            string              `json:"object,omitempty"`
	Created           int64               `json:"created,omitempty"`
	Model             string              `json:"model,omitempty"`
	SystemFingerprint string              `json:"system_fingerprint,omitempty"`
	Choices           []*CompletionChoise `json:"choices,omitempty"`
	Usage             CompletionUsage     `json:"usage,omitempty"`
}

func (p *OpenaiProvider) Completion(ctx context.Context, model string, messages []*types.Message, options ...types.CompletionOption) (*types.Completion, error) {
	if err := types.ValidateMessages(messages); err != nil {
		return nil, err
	}
	opts := types.ApplyCompletionOptions(&types.CompletionOptions{}, options...)

	req, err := p.toCompletion(model, messages, opts)
	if err != nil {
		return nil, fmt.Errorf("build openai completion request: %w", err)
	}

	httpOptions := []httpx.Option{}
	if opts.Retry != nil {
		httpOptions = append(httpOptions, httpx.WithRetry(opts.Retry.InitInterval, opts.Retry.MaxInterval, opts.Retry.Max, opts.Retry.Multiplier))
	}

	if opts.StreamingFunc == nil && opts.StreamingAccFunc == nil {
		rsp, err := httpx.PostJsonWithConfig[CompletionRsp](ctx, p.httpConfig, CompletionPath, &req, httpOptions...)
		if err != nil {
			return nil, err
		}
		return p.fromCompletion(rsp, false)
	}

	req.Stream = true
	req.StreamOptions = &CompletionReqStreamOptions{
		IncludeUsage: true,
	}

	ch, err := httpx.PostJsonSseWithConfig[CompletionRsp](ctx, p.httpConfig, CompletionPath, &req, httpOptions...)
	if err != nil {
		return nil, err
	}

	rsp := &CompletionRsp{}

loop:
	for {
		select {
		case <-ctx.Done():
			// 取消必须显式失败：不返回 partial completion，调用方不落库、不执行工具
			return nil, context.Cause(ctx)
		case event, ok := <-ch:
			if !ok {
				break loop
			}

			if event.Err != nil {
				return nil, event.Err
			}

			mergeDeltaCompletion(rsp, event.Data)
			if len(event.Data.Choices) == 0 {
				continue
			}

			if opts.StreamingFunc != nil {
				completion, err := p.fromCompletion(event.Data, true)
				if err != nil {
					return nil, err
				}
				if len(completion.Message.Parts) > 0 {
					if err := opts.StreamingFunc(ctx, completion); err != nil {
						return nil, err
					}
				}
			}

			if opts.StreamingAccFunc != nil {
				completion, err := p.fromCompletion(rsp, true)
				if err != nil {
					return nil, err
				}
				if err := opts.StreamingAccFunc(ctx, completion); err != nil {
					return nil, err
				}
			}
		}
	}

	return p.fromCompletion(rsp, false)
}

func (p *OpenaiProvider) toCompletion(model string, messages []*types.Message, opts *types.CompletionOptions) (*CompletionReq, error) {
	isOpenai := strings.TrimRight(p.options.Url, "/") == DefaultBaseUrl

	req := &CompletionReq{
		Model: model,
	}

	if opts.Temperature != nil {
		req.Temperature = opts.Temperature
	}
	req.TopP = opts.TopP
	req.Stop = append([]string(nil), opts.StopSequences...)
	if opts.TopK != nil {
		return nil, fmt.Errorf("%w: openai completion option top_k", types.ErrUnsupported)
	}
	if opts.MaxTokens != nil {
		if *opts.MaxTokens <= 0 {
			return nil, fmt.Errorf("openai max tokens must be positive")
		}
		if isOpenai {
			req.MaxCompletionTokens = opts.MaxTokens
		} else {
			req.MaxTokens = opts.MaxTokens
		}
	}

	if opts.ReasoningLevel != nil {
		level := *opts.ReasoningLevel

		if isOpenai {
			var effort ReasoningEffort
			switch {
			case level <= 0:
				effort = ReasoningEffortNone
			case level > 0 && level <= 2:
				effort = ReasoningEffortMinimal
			case level > 2 && level <= 4:
				effort = ReasoningEffortLow
			case level > 4 && level <= 8:
				effort = ReasoningEffortMedium
			case level > 8 && level < 10:
				effort = ReasoningEffortHigh
			case level >= 10:
				effort = ReasoningEffortXhigh
			}
			req.ReasoningEffort = &effort
		} else {
			if level <= 0 {
				if p.options.ThinkArgs != nil && p.options.ThinkArgs.Disable != nil {
					req.ExtraBody = p.options.ThinkArgs.Disable
				}
			} else {
				if p.options.ThinkArgs != nil && p.options.ThinkArgs.Enable != nil {
					req.ExtraBody = p.options.ThinkArgs.Enable
				}
			}
		}
	}

	if opts.Instructions != "" {
		req.Messages = append(req.Messages, &CompletionMessageParam{
			Role:    string(RoleSystem),
			Content: opts.Instructions,
		})
	}

	for _, msg := range messages {
		switch msg.Role {
		case types.MessageRoleSystem:
			var content strings.Builder
			for _, part := range msg.Parts {
				switch {
				case part.Text != nil:
					content.WriteString(part.Text.TextWithInstruction())
				default:
					return nil, fmt.Errorf("%w: openai system message part", types.ErrUnsupported)
				}
			}
			req.Messages = append(req.Messages, &CompletionMessageParam{
				Role:    string(RoleSystem),
				Content: content.String(),
			})
		case types.MessageRoleUser:
			var content strings.Builder
			for _, part := range msg.Parts {
				switch {
				case part.Text != nil:
					content.WriteString(part.Text.TextWithInstruction())
				default:
					return nil, fmt.Errorf("%w: openai user message part", types.ErrUnsupported)
				}
			}
			req.Messages = append(req.Messages, &CompletionMessageParam{
				Role:    string(RoleUser),
				Content: content.String(),
			})
		case types.MessageRoleAssistant:
			toolCalls := []*CompletionToolCall{}
			message := &CompletionMessageParam{Role: string(RoleAssistant)}
			for _, part := range msg.Parts {
				switch {
				case part.Text != nil:
					message.Content += part.Text.Text
				case part.Reasoning != nil:
					message.ReasoningContent += part.Reasoning.Text
				case part.Refusal != nil:
					message.Refusal += part.Refusal.Text
				case part.ToolCall != nil:
					if part.ToolCall.Type == "" || part.ToolCall.Type == types.ToolTypeFunction {
						toolCalls = append(toolCalls, &CompletionToolCall{
							ID:   part.ToolCall.ID,
							Type: ToolTypeFucntion,
							Function: &CompletionToolCallFunction{
								Name:      part.ToolCall.Name,
								Arguments: part.ToolCall.Arguments,
							},
						})
					} else {
						return nil, fmt.Errorf("%w: openai tool type %q", types.ErrUnsupported, part.ToolCall.Type)
					}
				default:
					return nil, fmt.Errorf("%w: openai assistant message part", types.ErrUnsupported)
				}
			}
			message.ToolCalls = toolCalls
			req.Messages = append(req.Messages, message)
		case types.MessageRoleTool:
			for _, part := range msg.Parts {
				switch {
				case part.ToolResult != nil:
					req.Messages = append(req.Messages, &CompletionMessageParam{
						Role:       string(RoleTool),
						ToolCallID: part.ToolResult.ID,
						Content:    part.ToolResult.ContentWithInstruction(),
					})
				default:
					return nil, fmt.Errorf("%w: openai tool message part", types.ErrUnsupported)
				}
			}
		}
	}

	for _, tool := range opts.Tools {
		if tool == nil || tool.Function == nil {
			return nil, fmt.Errorf("%w: openai function tool is nil", types.ErrInvalidMessage)
		}

		function := &ToolFunction{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			Parameters:  tool.Function.Parameters,
		}

		req.Tools = append(req.Tools, &Tool{
			Type:     ToolTypeFucntion,
			Function: function,
		})
	}

	return req, nil
}

func (p *OpenaiProvider) fromCompletion(from *CompletionRsp, delta bool) (*types.Completion, error) {
	if from == nil {
		return nil, fmt.Errorf("openai completion is nil")
	}
	to := &types.Completion{
		ID:    from.ID,
		Delta: delta,
		Model: from.Model,
		Message: &types.Message{
			ID:          from.ID,
			Role:        types.MessageRoleAssistant,
			Delta:       delta,
			Token:       from.Usage.CompletionTokens,
			ReasonToken: from.Usage.TokenDetail.ReasoningTokens,
		},
		Usage: types.CompletionUsage{
			PromptTokens:     from.Usage.PromptTokens,
			CompletionTokens: from.Usage.CompletionTokens,
			TotalTokens:      from.Usage.TotalTokens,
			ReasoningTokens:  from.Usage.TokenDetail.ReasoningTokens,
		},
	}

	if len(from.Choices) > 0 {
		choice := from.Choices[0]
		if choice == nil {
			return nil, fmt.Errorf("openai completion choice is nil")
		}

		message := choice.Message
		if message == nil {
			message = choice.Delta
		}

		if message != nil {

			if message.ReasoningContent != "" {
				to.Message.Parts = append(to.Message.Parts, &types.MessagePart{
					Reasoning: &types.MessageReasoning{
						Text: message.ReasoningContent,
					},
				})
			}

			if message.Content != "" {
				to.Message.Parts = append(to.Message.Parts, &types.MessagePart{
					Text: &types.MessageText{
						Text: message.Content,
					},
				})
			}

			if message.Refusal != "" {
				to.Message.Parts = append(to.Message.Parts, &types.MessagePart{
					Refusal: &types.MessageRefusal{
						Text: message.Refusal,
					},
				})
			}

			if !delta {
				for _, toolCall := range message.ToolCalls {
					if toolCall.Type == ToolTypeFucntion && toolCall.Function != nil {
						to.Message.Parts = append(to.Message.Parts, &types.MessagePart{
							ToolCall: &types.MessageToolCall{
								ID:        toolCall.ID,
								Type:      types.ToolTypeFunction,
								Name:      toolCall.Function.Name,
								Arguments: toolCall.Function.Arguments,
							},
						})
					}
				}
			}

		}

	}

	return to, nil
}

func mergeDeltaCompletion(to *CompletionRsp, from *CompletionRsp) {
	to.ID = from.ID
	to.Object = from.Object
	to.Created = from.Created
	if from.Model != "" {
		to.Model = from.Model
	}
	to.Usage.CompletionTokens += from.Usage.CompletionTokens
	to.Usage.PromptTokens += from.Usage.PromptTokens
	to.Usage.TotalTokens += from.Usage.TotalTokens
	to.Usage.TokenDetail.ReasoningTokens += from.Usage.TokenDetail.ReasoningTokens

	// only merge choice index == 0

	var choice *CompletionChoise
	for _, c := range from.Choices {
		if c == nil {
			continue
		}
		if c.Index == 0 {
			choice = c
			break
		}
	}

	if choice == nil || choice.Delta == nil {
		return
	}

	if len(to.Choices) == 0 {
		to.Choices = append(to.Choices, &CompletionChoise{
			Index:   0,
			Message: &CompletionMessage{},
		})
	}
	message := to.Choices[0].Message

	message.Content += choice.Delta.Content
	message.ReasoningContent += choice.Delta.ReasoningContent
	message.Refusal += choice.Delta.Refusal
	for i := range choice.Delta.ToolCalls {
		deltaTool := &choice.Delta.ToolCalls[i]
		toolIndex := deltaTool.Index
		message.ToolCalls = utils.ExpandSliceToFit(message.ToolCalls, toolIndex)
		tool := &message.ToolCalls[toolIndex]
		if deltaTool.ID != "" {
			tool.ID = deltaTool.ID
		}
		if deltaTool.Type != "" {
			tool.Type = deltaTool.Type
		}
		if deltaTool.Function != nil {
			if tool.Function == nil {
				tool.Function = &CompletionToolCallFunction{}
			}
			tool.Function.Name += deltaTool.Function.Name
			tool.Function.Arguments += deltaTool.Function.Arguments
		}
	}

}

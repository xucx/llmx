package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/xucx/llmx/types"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
)

func (p *AnthropicProvider) Completion(ctx context.Context, model string, messages []*types.Message, options ...types.CompletionOption) (*types.Completion, error) {
	if err := types.ValidateMessages(messages); err != nil {
		return nil, err
	}
	opts := types.ApplyCompletionOptions(&types.CompletionOptions{}, options...)
	if opts.StreamingFunc != nil || opts.StreamingAccFunc != nil {
		return nil, fmt.Errorf("%w: anthropic streaming completion", types.ErrUnsupported)
	}

	params, err := toChatParams(model, messages, opts)
	if err != nil {
		return nil, err
	}

	var (
		rsp *anthropic.Message
	)

	rsp, err = p.client.Messages.New(ctx, *params)
	if err != nil {
		return nil, err
	}

	return fromChatCompletion(rsp)
}

func toChatParams(model string, messages []*types.Message, opts *types.CompletionOptions) (*anthropic.MessageNewParams, error) {
	params := &anthropic.MessageNewParams{}

	params.Model = anthropic.Model(model)
	params.MaxTokens = 4096
	if opts.MaxTokens != nil {
		if *opts.MaxTokens <= 0 {
			return nil, fmt.Errorf("anthropic max tokens must be positive")
		}
		params.MaxTokens = *opts.MaxTokens
	}
	if opts.Temperature != nil {
		params.Temperature = anthropic.Float(float64(*opts.Temperature))
	}
	if opts.TopP != nil {
		params.TopP = anthropic.Float(float64(*opts.TopP))
	}
	if opts.TopK != nil {
		if *opts.TopK < 0 {
			return nil, fmt.Errorf("anthropic top_k must not be negative")
		}
		params.TopK = anthropic.Int(int64(*opts.TopK))
	}
	params.StopSequences = append([]string(nil), opts.StopSequences...)
	if opts.Instructions != "" {
		params.System = append(params.System, anthropic.TextBlockParam{Text: opts.Instructions})
	}
	if opts.ReasoningLevel != nil {
		level := *opts.ReasoningLevel
		if level <= 0 {
			disabled := anthropic.NewThinkingConfigDisabledParam()
			params.Thinking.OfDisabled = &disabled
		} else {
			if params.MaxTokens <= 1024 {
				return nil, fmt.Errorf("anthropic max tokens must exceed 1024 when reasoning is enabled")
			}
			if level > 10 {
				level = 10
			}
			budget := params.MaxTokens * int64(level) / 10
			if budget < 1024 {
				budget = 1024
			}
			if budget >= params.MaxTokens {
				budget = params.MaxTokens - 1
			}
			params.Thinking = anthropic.ThinkingConfigParamOfEnabled(budget)
		}
	}

	if err := toChatMessages(params, messages); err != nil {
		return nil, err
	}

	if err := toChatTools(params, opts); err != nil {
		return nil, err
	}

	return params, nil
}

func toChatMessages(params *anthropic.MessageNewParams, messages []*types.Message) error {
	for _, msg := range messages {
		toMessage := anthropic.MessageParam{}

		switch msg.Role {
		case types.MessageRoleSystem:
			toMessage.Role = RoleSystem
		case types.MessageRoleAssistant:
			toMessage.Role = RoleAssistant
		case types.MessageRoleUser:
			toMessage.Role = RoleUser
		case types.MessageRoleTool:
			toMessage.Role = RoleUser
		default:
			return fmt.Errorf("role %v not supported", msg.Role)
		}

		for _, part := range msg.Parts {
			var toPart *anthropic.ContentBlockParamUnion
			switch {
			case part.Text != nil:
				toPart = &anthropic.ContentBlockParamUnion{
					OfText: &anthropic.TextBlockParam{
						Text: part.Text.Text,
					},
				}
			case part.Reasoning != nil:
				if part.Reasoning.ThoughtSignature == "" {
					return fmt.Errorf("%w: anthropic reasoning signature is missing", types.ErrInvalidMessage)
				}
				toPart = &anthropic.ContentBlockParamUnion{
					OfThinking: &anthropic.ThinkingBlockParam{
						Thinking:  part.Reasoning.Text,
						Signature: part.Reasoning.ThoughtSignature,
					},
				}
			case part.Refusal != nil:
				toPart = &anthropic.ContentBlockParamUnion{
					OfText: &anthropic.TextBlockParam{Text: part.Refusal.Text},
				}
			case part.ImageURL != nil:
				toPart = &anthropic.ContentBlockParamUnion{
					OfImage: &anthropic.ImageBlockParam{
						Source: anthropic.ImageBlockParamSourceUnion{
							OfURL: &anthropic.URLImageSourceParam{
								URL: part.ImageURL.URL,
							},
						},
					},
				}
			case part.File != nil:
				if strings.HasPrefix(part.File.MIMEType, "image/") {
					toPart = &anthropic.ContentBlockParamUnion{
						OfImage: &anthropic.ImageBlockParam{
							Source: anthropic.ImageBlockParamSourceUnion{
								OfBase64: &anthropic.Base64ImageSourceParam{
									Data:      part.File.Data,
									MediaType: anthropic.Base64ImageSourceMediaType(part.File.MIMEType),
								},
							},
						},
					}
					break
				}
				if part.File.MIMEType != "application/pdf" {
					return fmt.Errorf("%w: anthropic file MIME type %q", types.ErrUnsupported, part.File.MIMEType)
				}
				toPart = &anthropic.ContentBlockParamUnion{
					OfDocument: &anthropic.DocumentBlockParam{
						Source: anthropic.DocumentBlockParamSourceUnion{
							OfBase64: &anthropic.Base64PDFSourceParam{
								Data:      part.File.Data,
								MediaType: constant.ApplicationPDF(part.File.MIMEType), //?? check
							},
						},
					},
				}

			case part.Audio != nil:
				return fmt.Errorf("%w: anthropic audio input", types.ErrUnsupported)
			case part.ToolCall != nil:
				if part.ToolCall.Type != "" && part.ToolCall.Type != types.ToolTypeFunction {
					return fmt.Errorf("%w: anthropic tool type %q", types.ErrUnsupported, part.ToolCall.Type)
				}
				var input any
				if err := json.Unmarshal([]byte(part.ToolCall.Arguments), &input); err != nil {
					return fmt.Errorf("decode anthropic tool arguments: %w", err)
				}
				toPart = &anthropic.ContentBlockParamUnion{
					OfToolUse: &anthropic.ToolUseBlockParam{
						ID:    part.ToolCall.ID,
						Name:  part.ToolCall.Name,
						Input: input,
					},
				}
			case part.ToolResult != nil:
				toPart = &anthropic.ContentBlockParamUnion{
					OfToolResult: &anthropic.ToolResultBlockParam{
						ToolUseID: part.ToolResult.ID,
						Content: []anthropic.ToolResultBlockParamContentUnion{
							{
								OfText: &anthropic.TextBlockParam{
									Text: part.ToolResult.ContentWithInstruction(),
								},
							},
						},
					},
				}
			}

			if toPart == nil {
				return fmt.Errorf("%w: anthropic message part", types.ErrUnsupported)
			}
			toMessage.Content = append(toMessage.Content, *toPart)
		}

		if toMessage.Role == RoleSystem {
			for _, part := range toMessage.Content {
				if part.OfText != nil {
					params.System = append(params.System, *part.OfText)
				}
			}
		} else {
			params.Messages = append(params.Messages, toMessage)
		}
	}

	return nil
}

func toChatTools(params *anthropic.MessageNewParams, opts *types.CompletionOptions) error {

	for _, tool := range opts.Tools {
		if tool == nil || tool.Function == nil {
			return fmt.Errorf("%w: anthropic function tool is nil", types.ErrInvalidMessage)
		}

		inputSchema := anthropic.ToolInputSchemaParam{}
		if v, ok := tool.Function.Parameters["properties"]; ok {
			inputSchema.Properties = v
		}
		if v, ok := tool.Function.Parameters["type"]; ok {
			if vv, ok := v.(string); ok {
				inputSchema.Type = constant.Object(vv)
			}
		}
		if v, ok := tool.Function.Parameters["required"]; ok {
			switch vv := v.(type) {
			case []string:
				inputSchema.Required = vv
			case []any:
				for _, value := range vv {
					required, ok := value.(string)
					if !ok {
						return fmt.Errorf("anthropic tool %q has non-string required field", tool.Function.Name)
					}
					inputSchema.Required = append(inputSchema.Required, required)
				}
			}
		}

		toTool := anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        tool.Function.Name,
				Description: anthropic.String(tool.Function.Description),
				InputSchema: inputSchema,
			},
		}

		params.Tools = append(params.Tools, toTool)
	}

	return nil
}

func fromChatCompletion(msg *anthropic.Message) (*types.Completion, error) {
	if msg == nil {
		return nil, fmt.Errorf("anthropic completion is nil")
	}

	completion := &types.Completion{
		ID:    msg.ID,
		Model: string(msg.Model),
		Message: &types.Message{
			ID:   msg.ID,
			Role: types.MessageRoleAssistant,
		},
		Usage: types.CompletionUsage{
			CompletionTokens: msg.Usage.OutputTokens,
			PromptTokens:     msg.Usage.InputTokens,
			TotalTokens:      msg.Usage.InputTokens + msg.Usage.OutputTokens,
		},
	}

	var (
		contentBuf   = strings.Builder{}
		reasoningBuf = strings.Builder{}
	)

	for _, c := range msg.Content {
		switch variant := c.AsAny().(type) {
		case anthropic.ThinkingBlock:
			reasoningBuf.WriteString(variant.Thinking)
			if variant.Signature != "" {
				completion.Message.Parts = append(completion.Message.Parts, &types.MessagePart{
					Reasoning: &types.MessageReasoning{
						Text:             variant.Thinking,
						ThoughtSignature: variant.Signature,
					},
				})
			}
		case anthropic.TextBlock:
			contentBuf.WriteString(variant.Text)
		case anthropic.ToolUseBlock:
			completion.Message.Parts = append(completion.Message.Parts, &types.MessagePart{
				ToolCall: &types.MessageToolCall{
					ID:        variant.ID,
					Type:      types.ToolTypeFunction,
					Name:      variant.Name,
					Arguments: string(variant.Input),
				},
			})

		case anthropic.RedactedThinkingBlock:
			//
		}
	}

	reasoning := reasoningBuf.String()
	if reasoning != "" && completion.Message.Reasoning() == "" {
		completion.Message.Parts = append(completion.Message.Parts, &types.MessagePart{Reasoning: &types.MessageReasoning{Text: reasoning}})
	}
	content := contentBuf.String()
	if content != "" {
		completion.Message.Parts = append(completion.Message.Parts, &types.MessagePart{Text: &types.MessageText{Text: content}})
	}

	return completion, nil
}

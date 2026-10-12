package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	apiv1 "github.com/xucx/llmx/api/v1"
	v1 "github.com/xucx/llmx/api/v1"
	"github.com/xucx/llmx/types"
)

func (p *ProxyProvider) Completion(ctx context.Context, model string, messages []*types.Message, options ...types.CompletionOption) (*types.Completion, error) {
	if err := types.ValidateMessages(messages); err != nil {
		return nil, err
	}

	client, err := p.getApiClient()
	if err != nil {
		return nil, err
	}

	opts := types.ApplyCompletionOptions(&types.CompletionOptions{}, options...)

	chatParams, err := ChatOptionsToParams(model, messages, opts)
	if err != nil {
		return nil, err
	}

	if opts.StreamingFunc == nil && opts.StreamingAccFunc == nil {
		rsp, err := client.Chat(ctx, &v1.ChatRequest{ChatParams: chatParams})
		if err != nil {
			return nil, err
		}
		return ToChatCompletion(rsp.ChatCompletion)
	} else {
		stream, err := client.ChatStream(ctx, &v1.ChatStreamRequest{ChatParams: chatParams})
		if err != nil {
			return nil, fmt.Errorf("api chat stream fail: %w", err)
		}

		acc := accChatCompletion{
			Completion: &v1.ChatCompletion{},
		}
		// 取消必须显式失败，同时带回已累计的内容，由调用方决定是否落库
		canceled := func() (*types.Completion, error) {
			completion, err := ToChatCompletion(acc.Completion)
			if err != nil {
				return nil, context.Cause(ctx)
			}
			return completion, context.Cause(ctx)
		}

		for {
			rsp, err := stream.Recv()
			if err != nil {
				if ctx.Err() != nil {
					return canceled()
				}
				if errors.Is(err, io.EOF) {
					return nil, errors.New("not finished")
				}
				return nil, fmt.Errorf("api chat stream recv fail: %w", err)
			}

			if rsp.ChatCompletion.Delta {

				acc.add(rsp.ChatCompletion)

				if opts.StreamingFunc != nil {
					streamCompletion, err := ToChatCompletion(rsp.ChatCompletion)
					if err != nil {
						return nil, err
					}
					if err := opts.StreamingFunc(ctx, streamCompletion); err != nil {
						return nil, err
					}
				}

				if opts.StreamingAccFunc != nil {
					accCompletion, err := ToChatCompletion(acc.Completion)
					if err != nil {
						return nil, err
					}
					if err := opts.StreamingAccFunc(ctx, accCompletion); err != nil {
						return nil, err
					}
				}

				continue
			}

			return ToChatCompletion(rsp.ChatCompletion)

		}
	}
}

// Notice: we only merge reasoning and text now
type accChatCompletion struct {
	Completion *v1.ChatCompletion
	reasoning  string
	text       string
}

func (acc *accChatCompletion) add(c *v1.ChatCompletion) {
	if !c.Delta {
		return
	}

	acc.Completion.Delta = true
	acc.Completion.Model = c.Model
	acc.Completion.Message = &v1.ChatMessage{
		Role: "assistant",
	}

	if c.Message != nil {

		for _, c := range c.Message.Contents {
			if reasoningContent := c.GetReasoning(); reasoningContent != nil {
				acc.reasoning += reasoningContent.Text
			}

			if textContent := c.GetText(); textContent != nil {
				acc.text += textContent.Text
			}
		}

		acc.Completion.Message.Id = c.Message.Id
		acc.Completion.Message.Contents = []*v1.ChatContent{}
		if acc.reasoning != "" {
			acc.Completion.Message.Contents = append(acc.Completion.Message.Contents, &v1.ChatContent{Content: &v1.ChatContent_Reasoning{Reasoning: &v1.ChatContentReasoning{
				Text: acc.reasoning,
			}}})
		}

		if acc.text != "" {
			acc.Completion.Message.Contents = append(acc.Completion.Message.Contents, &v1.ChatContent{Content: &v1.ChatContent_Text{Text: &v1.ChatContentText{
				Text: acc.text,
			}}})
		}
	}
}

func ToChatCompletion(from *apiv1.ChatCompletion) (*types.Completion, error) {
	if from == nil {
		return nil, errors.New("chat completion is nil")
	}
	message, err := ToMessage(from.Message)
	if err != nil {
		return nil, err
	}

	usage := types.CompletionUsage{}
	if from.Usage != nil {
		usage.PromptTokens = from.Usage.PromptTokens
		usage.CompletionTokens = from.Usage.CompletionTokens
		usage.TotalTokens = from.Usage.TotalTokens
		usage.ReasoningTokens = from.Usage.ReasoningTokens
	}

	return &types.Completion{
		ID:      from.Id,
		Delta:   from.Delta,
		Model:   from.Model,
		Message: message,
		Usage:   usage,
	}, nil
}

func FromChatCompletion(completion *types.Completion) (*apiv1.ChatCompletion, error) {
	if completion.Message == nil {
		return nil, errors.New("no completion message")
	}

	message, err := FromMessage(completion.Message)
	if err != nil {
		return nil, err
	}

	return &apiv1.ChatCompletion{
		Id:      completion.ID,
		Model:   completion.Model,
		Delta:   completion.Delta,
		Message: message,
		Usage: &apiv1.ChatUsage{
			PromptTokens:     completion.Usage.PromptTokens,
			CompletionTokens: completion.Usage.CompletionTokens,
			TotalTokens:      completion.Usage.TotalTokens,
			ReasoningTokens:  completion.Usage.ReasoningTokens,
		},
	}, nil
}

func ToMessage(from *apiv1.ChatMessage) (*types.Message, error) {
	if from == nil {
		return nil, fmt.Errorf("%w: proxy message is nil", types.ErrInvalidMessage)
	}
	to := &types.Message{
		ID: from.Id,
	}

	switch from.Role {
	case "system":
		to.Role = types.MessageRoleSystem
	case "assistant":
		to.Role = types.MessageRoleAssistant
	case "user":
		to.Role = types.MessageRoleUser
	case "tool":
		to.Role = types.MessageRoleTool
	default:
		return nil, fmt.Errorf("unknown role %s", from.Role)
	}

	for i, p := range from.Contents {
		if p == nil {
			return nil, fmt.Errorf("%w: proxy message content %d is nil", types.ErrInvalidMessage, i)
		}
		if text := p.GetText(); text != nil {
			to.Parts = append(to.Parts, &types.MessagePart{Text: &types.MessageText{
				Text: text.Text,
			}})
		} else if reasoning := p.GetReasoning(); reasoning != nil {
			to.Parts = append(to.Parts, &types.MessagePart{Reasoning: &types.MessageReasoning{
				Text:             reasoning.Text,
				ThoughtSignature: reasoning.ThoughtSignature,
			}})
		} else if refusal := p.GetRefusal(); refusal != nil {
			to.Parts = append(to.Parts, &types.MessagePart{Refusal: &types.MessageRefusal{
				Text: refusal.Text,
			}})
		} else if toolCall := p.GetToolCall(); toolCall != nil {
			to.Parts = append(to.Parts, &types.MessagePart{ToolCall: &types.MessageToolCall{
				ID:        toolCall.Id,
				Type:      types.ToolTypeFunction,
				Name:      toolCall.Name,
				Arguments: toolCall.Arguments,
			}})
		} else if toolResult := p.GetToolResult(); toolResult != nil {
			to.Parts = append(to.Parts, &types.MessagePart{ToolResult: &types.MessageToolResult{
				ID:     toolResult.Id,
				Name:   toolResult.Name,
				Result: toolResult.Result,
			}})
		} else if audio := p.GetAudio(); audio != nil {
			to.Parts = append(to.Parts, &types.MessagePart{Audio: &types.MessageAudio{
				Delta:      audio.Delta,
				Data:       audio.Data,
				Format:     audio.Format,
				Transcript: audio.Transcript,
			}})
		} else {
			return nil, fmt.Errorf("%w: proxy message content %d", types.ErrUnsupported, i)
		}
	}

	if err := to.Validate(); err != nil {
		return nil, err
	}
	return to, nil
}

func FromMessage(from *types.Message) (*apiv1.ChatMessage, error) {
	if err := from.Validate(); err != nil {
		return nil, err
	}
	to := &apiv1.ChatMessage{
		Id:   from.ID,
		Role: string(from.Role),
	}

	for _, part := range from.Parts {
		switch {
		case part.Text != nil:
			to.Contents = append(to.Contents, &apiv1.ChatContent{Content: &apiv1.ChatContent_Text{
				Text: &apiv1.ChatContentText{
					Text: part.Text.Text,
				},
			}})
		case part.Reasoning != nil:
			to.Contents = append(to.Contents, &apiv1.ChatContent{Content: &apiv1.ChatContent_Reasoning{
				Reasoning: &apiv1.ChatContentReasoning{
					Text:             part.Reasoning.Text,
					ThoughtSignature: part.Reasoning.ThoughtSignature,
				},
			}})
		case part.Refusal != nil:
			to.Contents = append(to.Contents, &apiv1.ChatContent{Content: &apiv1.ChatContent_Refusal{
				Refusal: &apiv1.ChatContentRefusal{
					Text: part.Refusal.Text,
				},
			}})
		case part.ToolCall != nil:
			to.Contents = append(to.Contents, &apiv1.ChatContent{Content: &apiv1.ChatContent_ToolCall{
				ToolCall: &apiv1.ChatContentToolCall{
					Id:        part.ToolCall.ID,
					Name:      part.ToolCall.Name,
					Arguments: part.ToolCall.Arguments,
				},
			}})
		case part.ToolResult != nil:
			to.Contents = append(to.Contents, &apiv1.ChatContent{Content: &apiv1.ChatContent_ToolResult{
				ToolResult: &apiv1.ChatContentToolResult{
					Id:     part.ToolResult.ID,
					Name:   part.ToolResult.Name,
					Result: part.ToolResult.ContentWithInstruction(),
				},
			}})
		case part.Audio != nil:
			to.Contents = append(to.Contents, &apiv1.ChatContent{Content: &apiv1.ChatContent_Audio{
				Audio: &apiv1.ChatContentAudio{
					Delta:      part.Audio.Delta,
					Data:       part.Audio.Data,
					Format:     part.Audio.Format,
					Transcript: part.Audio.Transcript,
				},
			}})
		default:
			return nil, fmt.Errorf("%w: proxy message part", types.ErrUnsupported)
		}
	}

	return to, nil
}

func ChatParamsToOptions(req *apiv1.ChatParams) (*types.CompletionOptions, error) {
	tools, err := ToChatTools(req.Tools)
	if err != nil {
		return nil, err
	}

	options := &types.CompletionOptions{
		Instructions:  req.Instructions,
		Tools:         tools,
		AudioVoice:    types.AudioVoiceType(req.Voice),
		StopSequences: append([]string(nil), req.StopSequences...),
	}
	if req.Temperature != nil {
		value := *req.Temperature
		options.Temperature = &value
	}
	if req.TopP != nil {
		value := *req.TopP
		options.TopP = &value
	}
	if req.TopK != nil {
		value := int(*req.TopK)
		options.TopK = &value
	}
	if req.MaxTokens != nil {
		value := *req.MaxTokens
		options.MaxTokens = &value
	}
	if req.ReasoningLevel != nil {
		value := int(*req.ReasoningLevel)
		options.ReasoningLevel = &value
	}
	return options, nil
}

func ToChatTools(tools []*apiv1.ChatTool) ([]*types.Tool, error) {
	all := []*types.Tool{}
	for i, t := range tools {
		if t == nil {
			return nil, fmt.Errorf("chat tool %d is nil", i)
		}
		params := map[string]any{}
		if err := json.Unmarshal([]byte(t.Params), &params); err != nil {
			return nil, err
		}

		all = append(all, &types.Tool{
			Type: types.ToolTypeFunction,
			Function: &types.ToolFunction{
				Name:        t.Name,
				Description: t.Desc,
				Parameters:  params,
			},
		})
	}

	return all, nil
}

func ToChatVoice(voice string) (types.AudioVoiceType, error) {
	switch voice {
	case "women":
		return types.AudioVoiceWomen, nil
	case "men":
		return types.AudioVoiceMen, nil
	default:
		return "", fmt.Errorf("%w: voice %q", types.ErrUnsupported, voice)
	}
}

func ToTurnDetection(turnDetection string) (types.TurnDetectionType, error) {
	switch turnDetection {
	case "server":
		return types.TurnDetectionServer, nil
	case "semantic":
		return types.TurnDetectionSemantic, nil
	default:
		return "", fmt.Errorf("turnDetection %s not support", turnDetection)
	}
}

func ChatOptionsToParams(model string, messages []*types.Message, opts *types.CompletionOptions) (*apiv1.ChatParams, error) {
	chatParams := &apiv1.ChatParams{
		Model:         model,
		Instructions:  opts.Instructions,
		Voice:         string(opts.AudioVoice),
		StopSequences: append([]string(nil), opts.StopSequences...),
	}
	if opts.Temperature != nil {
		value := *opts.Temperature
		chatParams.Temperature = &value
	}
	if opts.TopP != nil {
		value := *opts.TopP
		chatParams.TopP = &value
	}
	if opts.TopK != nil {
		if int64(*opts.TopK) > int64(^uint32(0)>>1) || int64(*opts.TopK) < -int64(^uint32(0)>>1)-1 {
			return nil, fmt.Errorf("top_k is out of int32 range: %d", *opts.TopK)
		}
		value := int32(*opts.TopK)
		chatParams.TopK = &value
	}
	if opts.MaxTokens != nil {
		value := *opts.MaxTokens
		chatParams.MaxTokens = &value
	}
	if opts.ReasoningLevel != nil {
		if int64(*opts.ReasoningLevel) > int64(^uint32(0)>>1) ||
			int64(*opts.ReasoningLevel) < -int64(^uint32(0)>>1)-1 {
			return nil, fmt.Errorf("reasoning level is out of int32 range: %d", *opts.ReasoningLevel)
		}
		value := int32(*opts.ReasoningLevel)
		chatParams.ReasoningLevel = &value
	}

	for _, msg := range messages {
		m, err := FromMessage(msg)
		if err != nil {
			return nil, err
		}
		chatParams.Messages = append(chatParams.Messages, m)
	}

	for i, tool := range opts.Tools {
		if tool == nil || tool.Function == nil {
			return nil, fmt.Errorf("chat tool %d function is nil", i)
		}
		params, err := json.Marshal(tool.Function.Parameters)
		if err != nil {
			return nil, fmt.Errorf("marshal chat tool %q parameters: %w", tool.Function.Name, err)
		}
		chatParams.Tools = append(chatParams.Tools, &apiv1.ChatTool{
			Name:   tool.Function.Name,
			Desc:   tool.Function.Description,
			Params: string(params),
		})
	}

	return chatParams, nil
}

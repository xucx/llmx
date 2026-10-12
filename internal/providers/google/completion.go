package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/xucx/llmx/internal/utils"
	"github.com/xucx/llmx/types"

	"github.com/google/uuid"
	"google.golang.org/genai"
)

func (p *GoogleProvider) Completion(ctx context.Context, model string, messages []*types.Message, options ...types.CompletionOption) (*types.Completion, error) {
	if err := types.ValidateMessages(messages); err != nil {
		return nil, err
	}
	opts := types.ApplyCompletionOptions(&types.CompletionOptions{}, options...)

	contents, contentsOpts, err := toChatContents(messages)
	if err != nil {
		return nil, err
	}

	config, err := toChatConfig(opts, contentsOpts)
	if err != nil {
		return nil, err
	}

	var (
		rsp *genai.GenerateContentResponse
	)

	if opts.StreamingFunc == nil && opts.StreamingAccFunc == nil {
		rsp, err = p.client.Models.GenerateContent(ctx, model, contents, config)
		if err != nil {
			return nil, err
		}
	} else {
		rsp = &genai.GenerateContentResponse{}

		// 取消必须显式失败，同时带回已累计的内容，由调用方决定是否落库
		canceled := func() (*types.Completion, error) {
			completion, err := fromChatCompletion(rsp, false)
			if err != nil {
				return nil, context.Cause(ctx)
			}
			return completion, context.Cause(ctx)
		}

		stream := p.client.Models.GenerateContentStream(ctx, model, contents, config)
		for chunck, err := range stream {
			if err != nil {
				if ctx.Err() != nil {
					return canceled()
				}
				return nil, err
			}

			mergeChatStreamCompletion(rsp, chunck)
			if len(chunck.Candidates) == 0 {
				continue
			}

			if opts.StreamingFunc != nil {
				chunckComplete, err := fromChatCompletion(chunck, true)
				if err != nil {
					return nil, err
				}
				if err := opts.StreamingFunc(ctx, chunckComplete); err != nil {
					return nil, err
				}
			}

			if opts.StreamingAccFunc != nil {
				accComplete, err := fromChatCompletion(rsp, true)
				if err != nil {
					return nil, err
				}
				if err := opts.StreamingAccFunc(ctx, accComplete); err != nil {
					return nil, err
				}
			}

		}
		if ctx.Err() != nil {
			return canceled()
		}
	}

	return fromChatCompletion(rsp, false)
}

func toChatConfig(opts *types.CompletionOptions, contentOpts *contentsOpt) (*genai.GenerateContentConfig, error) {
	config := &genai.GenerateContentConfig{
		ResponseModalities: []string{"TEXT"},
	}

	instructions := make([]string, 0, 1+len(contentOpts.systemInstructions))
	if opts.Instructions != "" {
		instructions = append(instructions, opts.Instructions)
	}
	instructions = append(instructions, contentOpts.systemInstructions...)
	if len(instructions) > 0 {
		config.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{
				{Text: strings.Join(instructions, "\n\n")},
			},
		}
	}

	if opts.Temperature != nil {
		config.Temperature = opts.Temperature
	}
	config.TopP = opts.TopP
	if opts.TopK != nil {
		if *opts.TopK < 0 {
			return nil, fmt.Errorf("google top_k must not be negative")
		}
		topK := float32(*opts.TopK)
		config.TopK = &topK
	}
	if opts.MaxTokens != nil {
		if *opts.MaxTokens <= 0 || *opts.MaxTokens > int64(^uint32(0)>>1) {
			return nil, fmt.Errorf("google max tokens is out of range: %d", *opts.MaxTokens)
		}
		config.MaxOutputTokens = int32(*opts.MaxTokens)
	}
	config.StopSequences = append([]string(nil), opts.StopSequences...)
	if opts.ReasoningLevel != nil {
		level := *opts.ReasoningLevel
		thinking := &genai.ThinkingConfig{}
		if level > 0 {
			if level > 10 {
				level = 10
			}
			budget := int32(level * 8192 / 10)
			if budget < 1024 {
				budget = 1024
			}
			thinking.IncludeThoughts = true
			thinking.ThinkingBudget = &budget
		} else {
			thinking.ThinkingBudget = utils.Ptr[int32](0)
		}
		config.ThinkingConfig = thinking
	}

	for i, tool := range opts.Tools {
		if tool == nil || tool.Function == nil {
			return nil, fmt.Errorf("%w: google tool [%d] function is nil", types.ErrInvalidMessage, i)
		}
		if tool.Type != types.ToolTypeFunction {
			return nil, fmt.Errorf("tool [%d]: unsupported type %q, want 'function'", i, tool.Type)
		}

		config.Tools = append(config.Tools, &genai.Tool{FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name:                 tool.Function.Name,
				Description:          tool.Function.Description,
				ParametersJsonSchema: tool.Function.Parameters,
			},
		}})
	}

	if opts.AudioVoice != "" {
		config.ResponseModalities = append(config.ResponseModalities, "AUDIO")
		config.SpeechConfig = &genai.SpeechConfig{}
		voice, err := ToVoice(opts.AudioVoice)
		if err != nil {
			return nil, err
		}
		config.SpeechConfig.VoiceConfig = &genai.VoiceConfig{
			PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
				VoiceName: voice,
			},
		}
	}

	return config, nil
}

type contentsOpt struct {
	systemInstructions []string
}

func toChatContents(messages []*types.Message) ([]*genai.Content, *contentsOpt, error) {

	contents := []*genai.Content{}
	opts := &contentsOpt{}

	for _, msg := range messages {
		content := &genai.Content{}

		switch msg.Role {
		case types.MessageRoleSystem:
			for _, part := range msg.Parts {
				if part.Text == nil {
					return nil, nil, fmt.Errorf("%w: google system message part", types.ErrUnsupported)
				}
				opts.systemInstructions = append(opts.systemInstructions, part.Text.TextWithInstruction())
			}
			continue
		case types.MessageRoleAssistant:
			content.Role = RoleModel
		case types.MessageRoleUser:
			content.Role = RoleUser
		case types.MessageRoleTool:
			content.Role = RoleUser
		default:
			return nil, nil, fmt.Errorf("role %v not supported", msg.Role)
		}

		var thoughtSignature []byte

		for _, part := range msg.Parts {
			var toPart *genai.Part
			switch {
			case part.Reasoning != nil:
				if part.Reasoning.ThoughtSignature != "" {
					var err error
					thoughtSignature, err = base64.StdEncoding.DecodeString(part.Reasoning.ThoughtSignature)
					if err != nil {
						return nil, nil, fmt.Errorf("decode google thought signature: %w", err)
					}
				}
				toPart = &genai.Part{
					Thought:          true,
					ThoughtSignature: thoughtSignature,
					Text:             part.Reasoning.Text,
				}
			case part.Text != nil:
				toPart = &genai.Part{
					Text:             part.Text.Text,
					ThoughtSignature: thoughtSignature,
				}
			case part.Refusal != nil:
				toPart = &genai.Part{Text: part.Refusal.Text}
			case part.ImageURL != nil:
				toPart = &genai.Part{
					FileData: &genai.FileData{
						FileURI: part.ImageURL.URL,
					},
				}
			case part.File != nil:
				data, err := base64.StdEncoding.DecodeString(part.File.Data)
				if err != nil {
					data, err = base64.RawStdEncoding.DecodeString(part.File.Data)
					if err != nil {
						return nil, nil, fmt.Errorf("decode google file data: %w", err)
					}
				}
				toPart = &genai.Part{
					InlineData: &genai.Blob{
						MIMEType:    part.File.MIMEType,
						Data:        data,
						DisplayName: part.File.Name,
					},
				}
			case part.Audio != nil:
				if part.Audio.Delta {
					return nil, nil, fmt.Errorf("%w: google delta audio input", types.ErrUnsupported)
				}
				data, err := base64.StdEncoding.DecodeString(part.Audio.Data)
				if err != nil {
					return nil, nil, fmt.Errorf("decode google audio data: %w", err)
				}
				toPart = &genai.Part{
					InlineData: &genai.Blob{
						MIMEType: fmt.Sprintf("audio/%s", part.Audio.Format),
						Data:     data,
					},
				}

			case part.ToolCall != nil:
				if part.ToolCall.Type != "" && part.ToolCall.Type != types.ToolTypeFunction {
					return nil, nil, fmt.Errorf("%w: google tool type %q", types.ErrUnsupported, part.ToolCall.Type)
				}

				args := map[string]any{}
				if err := json.Unmarshal([]byte(part.ToolCall.Arguments), &args); err != nil {
					return nil, nil, err
				}

				toPart = &genai.Part{
					FunctionCall: &genai.FunctionCall{
						ID:   part.ToolCall.ID,
						Name: part.ToolCall.Name,
						Args: args,
					},
					ThoughtSignature: thoughtSignature,
				}
			case part.ToolResult != nil:
				toPart = &genai.Part{
					FunctionResponse: &genai.FunctionResponse{
						ID:       part.ToolResult.ID,
						Name:     part.ToolResult.Name,
						Response: map[string]any{"output": part.ToolResult.ContentWithInstruction()},
					},
				}
			}

			if toPart == nil {
				return nil, nil, fmt.Errorf("%w: google message part", types.ErrUnsupported)
			}
			content.Parts = append(content.Parts, toPart)
		}

		if len(content.Parts) > 0 {
			contents = append(contents, content)
		}
	}

	return contents, opts, nil
}

func mergeChatStreamCompletion(to *genai.GenerateContentResponse, delta *genai.GenerateContentResponse) bool {

	if len(to.ResponseID) == 0 {
		to.ResponseID = delta.ResponseID
	} else if to.ResponseID != delta.ResponseID {
		return false
	}

	to.SDKHTTPResponse = delta.SDKHTTPResponse
	to.CreateTime = delta.CreateTime
	to.ModelVersion = delta.ModelVersion
	to.PromptFeedback = delta.PromptFeedback
	to.ResponseID = delta.ResponseID
	to.UsageMetadata = delta.UsageMetadata

	for _, c := range delta.Candidates {
		if c == nil {
			continue
		}
		//expand
		to.Candidates = expandSliceToFit(to.Candidates, int(c.Index))
		candidate := to.Candidates[c.Index]
		if candidate == nil {
			candidate = &genai.Candidate{}
			to.Candidates[c.Index] = candidate
		}

		candidate.CitationMetadata = c.CitationMetadata
		candidate.FinishMessage = c.FinishMessage
		candidate.TokenCount = c.TokenCount
		candidate.FinishReason = c.FinishReason
		candidate.URLContextMetadata = c.URLContextMetadata
		candidate.AvgLogprobs = c.AvgLogprobs
		candidate.GroundingMetadata = c.GroundingMetadata
		candidate.Index = c.Index
		candidate.LogprobsResult = c.LogprobsResult
		candidate.SafetyRatings = c.SafetyRatings

		if c.Content != nil {
			if candidate.Content == nil {
				candidate.Content = &genai.Content{}
			}

			candidate.Content.Role = c.Content.Role

			parts := candidate.Content.Parts
			for _, p := range c.Content.Parts {

				//merge text
				if p.Text != "" || len(p.ThoughtSignature) != 0 {
					mergerd := false
					for _, v := range parts {
						if v.Thought == p.Thought {
							v.Text += p.Text
							v.ThoughtSignature = p.ThoughtSignature
							mergerd = true
							break
						}
					}

					if !mergerd {
						parts = append(parts, &genai.Part{
							Thought:          p.Thought,
							ThoughtSignature: p.ThoughtSignature,
							Text:             p.Text,
						})
					}
				}

				if p.FunctionCall != nil || p.InlineData != nil || p.FileData != nil {
					parts = append(parts, p)
				}
			}
			candidate.Content.Parts = parts
		}
	}

	return true
}

func expandSliceToFit[T any](slice []T, index int) []T {
	if index < len(slice) {
		return slice
	}
	if index < cap(slice) {
		return slice[:index+1]
	}
	newSlice := make([]T, index+1)
	copy(newSlice, slice)
	return newSlice
}

func fromChatCompletion(rsp *genai.GenerateContentResponse, delta bool) (*types.Completion, error) {
	if rsp == nil {
		return nil, fmt.Errorf("google completion is nil")
	}
	if len(rsp.Candidates) < 1 {
		return nil, fmt.Errorf("completion no candidates")
	}

	candidate := rsp.Candidates[0]
	if candidate == nil || candidate.Content == nil {
		return nil, fmt.Errorf("completion candidate has no content")
	}

	message := &types.Message{
		ID:   rsp.ResponseID,
		Role: types.MessageRoleAssistant,
	}

	for _, part := range candidate.Content.Parts {

		thoughtSignature := ""
		if len(part.ThoughtSignature) > 0 {
			thoughtSignature = base64.StdEncoding.EncodeToString(part.ThoughtSignature)
		}

		if part.Text != "" {
			if part.Thought {
				message.Parts = append(message.Parts, &types.MessagePart{Reasoning: &types.MessageReasoning{
					Text:             part.Text,
					ThoughtSignature: thoughtSignature,
				}})
			} else {
				message.Parts = append(message.Parts, &types.MessagePart{Text: &types.MessageText{Text: part.Text}})
			}
		}

		if part.InlineData != nil {
			if strings.HasPrefix(part.InlineData.MIMEType, "audio/") {
				format := strings.TrimPrefix(part.InlineData.MIMEType, "audio/")
				data := base64.StdEncoding.EncodeToString(part.InlineData.Data)
				message.Parts = append(message.Parts, &types.MessagePart{Audio: &types.MessageAudio{
					Format: format,
					Data:   data,
				}})
			}
		}

		if !delta {
			if part.FunctionCall != nil {
				args, err := json.Marshal(part.FunctionCall.Args)
				if err != nil {
					return nil, err
				}
				id := part.FunctionCall.ID
				if id == "" {
					id = uuid.NewString()
				}
				toolCall := &types.MessageToolCall{
					ID:        id,
					Type:      types.ToolTypeFunction,
					Name:      part.FunctionCall.Name,
					Arguments: string(args),
				}
				message.Parts = append(message.Parts, &types.MessagePart{ToolCall: toolCall})
			}
		}
	}

	completion := &types.Completion{
		ID:      rsp.ResponseID,
		Delta:   delta,
		Model:   rsp.ModelVersion,
		Message: message,
		Usage:   types.CompletionUsage{},
	}

	if rsp.UsageMetadata != nil {
		completion.Usage.CompletionTokens = int64(rsp.UsageMetadata.CandidatesTokenCount)
		completion.Usage.PromptTokens = int64(rsp.UsageMetadata.PromptTokenCount)
		completion.Usage.TotalTokens = int64(rsp.UsageMetadata.TotalTokenCount)
		completion.Usage.ReasoningTokens = int64(rsp.UsageMetadata.ThoughtsTokenCount)
	}

	return completion, nil
}

func ToVoice(in types.AudioVoiceType) (string, error) {
	switch in {
	case types.AudioVoiceWomen:
		return "Zephyr", nil
	case types.AudioVoiceMen:
		return "Gacrux", nil
	default:
		return "", fmt.Errorf("voice %s not support", in)
	}
}

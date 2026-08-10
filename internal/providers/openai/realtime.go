package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/xucx/llmx/internal/httpx"
	"github.com/xucx/llmx/types"

	"go.uber.org/multierr"
)

const (
	DefaultRealTimeModel = "gpt-4o-realtime-preview"
)

func (p *OpenaiProvider) Realtime(ctx context.Context, model string, messages []*types.Message, options ...types.RealTimeOption) (types.RealTimeSession, error) {
	if err := types.ValidateMessages(messages); err != nil {
		return nil, err
	}
	option := types.ApplyRealTimeOptions(nil, options...)

	url, err := p.getRealTimeUrl(model, option)
	if err != nil {
		return nil, err
	}

	header := http.Header{}
	header.Set("Authorization", "Bearer "+p.options.Sk)

	client, err := httpx.WSConnect(ctx, url, httpx.WSClientOptions{
		Header:     header,
		HTTPClient: p.options.HTTPClient,
	})
	if err != nil {
		return nil, err
	}

	client.SetReadLimit(10000000) //10M

	//update session
	sessionUpdateEvent, err := CreateSessionUpdateEvent(model, option)
	if err != nil {
		client.Close()
		return nil, err
	}
	err = client.SendJsonMessage(ctx, sessionUpdateEvent)
	if err != nil {
		client.Close()
		return nil, err
	}

	session := ClientSession{client: client}

	//send messages
	for _, msg := range messages {
		if err := session.Send(ctx, msg); err != nil {
			session.Close()
			return nil, err
		}
	}

	return &session, nil
}

func (p *OpenaiProvider) getRealTimeUrl(model string, opts *types.RealTimeOptions) (string, error) {
	query := url.Values{}
	query.Set("model", model)
	url, err := url.Parse(p.options.Url + "?" + query.Encode())
	if err != nil {
		return "", err
	}
	url = url.JoinPath("/realtime")
	switch url.Scheme {
	case "http":
		url.Scheme = "ws"
	default:
		url.Scheme = "wss"
	}

	return url.String(), nil
}

type ClientSession struct {
	client *httpx.WSClient
}

func (r *ClientSession) Send(ctx context.Context, msg *types.Message) error {
	if err := msg.Validate(); err != nil {
		return err
	}

	var (
		allErr       error
		messageEvent = NewClientEventConversationItemCreate()
		otherEvents  = []any{}
	)

	messageEvent.Item.Type = "message"

	switch msg.Role {
	case types.MessageRoleSystem:
		messageEvent.Item.Role = "system"
		for _, part := range msg.Parts {
			switch {
			case part.Text != nil:
				messageEvent.Item.Content = append(messageEvent.Item.Content, &RealTimeContent{
					Type: "input_text",
					Text: part.Text.Text,
				})
			default:
				return fmt.Errorf("not support message for realtime system role")
			}
		}
	case types.MessageRoleUser:
		messageEvent.Item.Role = "user"
		for _, part := range msg.Parts {
			switch {
			case part.Text != nil:
				messageEvent.Item.Content = append(messageEvent.Item.Content, &RealTimeContent{
					Type: "input_text",
					Text: part.Text.Text,
				})

			case part.Audio != nil:
				if part.Audio.Format != "pcm16" {
					return fmt.Errorf("realtime audio format %s not support", part.Audio.Format)
				}

				if !part.Audio.Delta {
					messageEvent.Item.Content = append(messageEvent.Item.Content, &RealTimeContent{
						Type:  "input_audio",
						Audio: part.Audio.Data,
					})
				} else {
					audioDeltaEvent := NewClientEventInputAudioBufferAppend()
					audioDeltaEvent.Audio = part.Audio.Data
					otherEvents = append(otherEvents, audioDeltaEvent)
				}
			case part.ImageURL != nil:
				if !strings.EqualFold(part.ImageURL.Format, "PNG") || !strings.EqualFold(part.ImageURL.Format, "JPEG") {
					return fmt.Errorf("realtime image url format %s not support", part.ImageURL.Format)
				}
				messageEvent.Item.Content = append(messageEvent.Item.Content, &RealTimeContent{
					Type:     "input_image",
					ImageUrl: part.ImageURL.URL,
				})
			case part.RealtimeResponse != nil:
				createRspEvent := NewClientEventResponseCreate()
				otherEvents = append(otherEvents, createRspEvent)
			default:
				return fmt.Errorf("unsupport realtime user message")
			}
		}

	case types.MessageRoleAssistant:
		messageEvent.Item.Role = "assistant"
		for _, part := range msg.Parts {
			switch {
			case part.Text != nil:
				messageEvent.Item.Content = append(messageEvent.Item.Content, &RealTimeContent{
					Type: "output_text",
					Text: part.Text.Text,
				})
			case part.ToolCall != nil:
				toolCallEvent := NewClientEventConversationItemCreate()
				toolCallEvent.Item.Type = "function_call"
				toolCallEvent.Item.CallID = part.ToolCall.ID
				toolCallEvent.Item.Name = part.ToolCall.Name
				toolCallEvent.Item.Arguments = part.ToolCall.Arguments
				otherEvents = append(otherEvents, toolCallEvent)
			default:
				return fmt.Errorf("unsupport realtime assistant message")
			}
		}
	case types.MessageRoleTool:
		for _, part := range msg.Parts {
			switch {
			case part.ToolResult != nil:
				toolOutputEvent := NewClientEventConversationItemCreate()
				toolOutputEvent.Item.Type = "function_call_output"
				toolOutputEvent.Item.CallID = part.ToolResult.ID
				toolOutputEvent.Item.Output = part.ToolResult.ContentWithInstruction()
				otherEvents = append(otherEvents, toolOutputEvent)
			default:
				return fmt.Errorf("unsupport realtime tool message")
			}
		}
	default:
		return fmt.Errorf("realtime unsupport role %s", msg.Role)
	}

	if len(messageEvent.Item.Content) > 0 {
		if err := r.client.SendJsonMessage(ctx, messageEvent); err != nil {
			allErr = multierr.Append(allErr, err)
		}
	}

	for _, event := range otherEvents {
		if err := r.client.SendJsonMessage(ctx, event); err != nil {
			allErr = multierr.Append(allErr, err)
		}
	}

	return allErr
}

func (r *ClientSession) Recv(ctx context.Context) (*types.Completion, error) {
	messageType, data, err := r.client.Recv(ctx)
	if err != nil {
		return nil, err
	}
	if messageType != httpx.WSMessageText {
		return nil, fmt.Errorf("expected text message, got %d", messageType)
	}

	_, event, err := UnmarshalServerEvent(data)
	if err != nil {
		return nil, err
	}

	completion := &types.Completion{
		Message: &types.Message{
			Role: types.MessageRoleAssistant,
		},
	}

	switch p := event.(type) {
	case *ServerEventResponseOutputAudioDelta:
		completion.Delta = true
		completion.Message.Parts = append(completion.Message.Parts, &types.MessagePart{Audio: &types.MessageAudio{
			Data:   p.Delta,
			Format: "pcm16",
			Delta:  true,
		}})
	case *ServerEventResponseDone:
		if p.Response.Status != "completed" {
			break
		}

		for _, output := range p.Response.Output {
			switch output.Type {
			case "message":
				for _, content := range output.Content {
					switch content.Type {
					case "output_text":
						completion.Message.Parts = append(completion.Message.Parts, &types.MessagePart{Text: &types.MessageText{
							Text: content.Text,
						}})
					case "output_audio":
						completion.Message.Parts = append(completion.Message.Parts, &types.MessagePart{Audio: &types.MessageAudio{
							Transcript: content.Transcript,
						}})
					}
				}
			case "function_call":
				completion.Message.Parts = append(completion.Message.Parts, &types.MessagePart{ToolCall: &types.MessageToolCall{
					ID:        output.CallID,
					Type:      types.ToolTypeFunction,
					Name:      output.Name,
					Arguments: output.Arguments,
				}})
			}
		}
	case *Event:
		//
	default:

	}

	return completion, nil
}

func (r *ClientSession) Close() {
	r.client.Close()
}

func CreateSessionUpdateEvent(model string, opts *types.RealTimeOptions) (*ClientEventSessionUpdate, error) {
	event := NewClientEventSessionUpdate()
	event.Session.Type = "realtime"
	event.Session.Model = model
	event.Session.Instructions = opts.Instructions
	event.Session.OutputModalities = []Modality{ModalityAudio}
	event.Session.Audio = &ClientEventSessionUpdateSessionAudio{
		Input: &ClientEventSessionUpdateSessionAudioInput{
			TurnDetection: &ClientEventSessionUpdateSessionAudioInputTurnDetection{
				Type:              ClientTurnDetectionTypeSemanticVad,
				CreateResponse:    true,
				InterruptResponse: true,
			},
		},
		Output: &ClientEventSessionUpdateSessionAudioOutput{
			Voice: VoiceAlloy,
		},
	}

	if opts.AudioVoice != nil {
		voice, err := ToVoice(*opts.AudioVoice)
		if err != nil {
			return nil, err
		}
		event.Session.Audio.Output.Voice = voice
	}

	for _, tool := range opts.Tools {
		if tool == nil || tool.Function == nil {
			return nil, fmt.Errorf("%w: openai realtime function tool is nil", types.ErrInvalidMessage)
		}
		if tool.Type == types.ToolTypeFunction {
			event.Session.Tools = append(event.Session.Tools, &ClientEventSessionUpdateSessionToolFunctions{
				Type:        "function",
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
			})
		} else {
			return nil, fmt.Errorf("%w: openai realtime tool type %q", types.ErrUnsupported, tool.Type)
		}
	}

	return event, nil
}

func ToModality(in types.Modality) (Modality, error) {
	switch in {
	case types.ModalityText:
		return ModalityText, nil
	case types.ModalityAudio:
		return ModalityAudio, nil
	default:
		return "", fmt.Errorf("unsupport modality %s ", in)
	}
}

func ToVoice(in types.AudioVoiceType) (Voice, error) {
	switch in {
	case types.AudioVoiceWomen:
		return VoiceAlloy, nil
	case types.AudioVoiceMen:
		return VoiceAsh, nil
	default:
		return "", fmt.Errorf("voice %s not support", in)
	}
}

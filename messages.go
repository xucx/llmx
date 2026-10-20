package llmx

import (
	"github.com/xucx/llmx/types"
)

// BuildMessages renders the admitted events as the messages a provider
// takes. Message IDs are the event IDs, so a message maps back to its event
// (interceptors such as compress rely on it). It is the only place events
// become provider messages.
func BuildMessages(s *State) []*types.Message {
	if s == nil {
		return nil
	}
	messages := make([]*types.Message, 0, len(s.Admitted))
	for _, event := range s.Admitted {
		if message := eventMessage(event); message != nil {
			messages = append(messages, message)
		}
		if images := toolImagesMessage(event); images != nil {
			messages = append(messages, images)
		}
	}
	return messages
}

// ImagesMessageSuffix is appended to a ToolResult event's ID to name the
// user message that carries the images it returned.
const ImagesMessageSuffix = ":images"

// toolImagesMessage shows the images a tool returned. Providers accept
// pictures only in user messages, so they follow the tool message as one
// with the event's ID plus ImagesMessageSuffix.
func toolImagesMessage(event *Event) *types.Message {
	if event.ToolResult == nil || len(event.ToolResult.Images) == 0 || event.ToolResult.Error != "" {
		return nil
	}
	name := event.ToolResult.Name
	if name == "" {
		name = event.ToolResult.CallID
	}
	message := &types.Message{ID: event.ID + ImagesMessageSuffix, Role: types.MessageRoleUser}
	message.Parts = append(message.Parts, &types.MessagePart{
		Text: &types.MessageText{Text: "Images returned by tool " + name + ":"},
	})
	for _, url := range event.ToolResult.Images {
		if url == "" {
			continue
		}
		message.Parts = append(message.Parts, &types.MessagePart{
			ImageURL: &types.MessageImageURL{URL: url},
		})
	}
	return message
}

func eventMessage(event *Event) *types.Message {
	switch {
	case event.Input != nil:
		message := &types.Message{ID: event.ID, Role: types.MessageRoleUser}
		for _, content := range event.Input.Content {
			switch {
			case content.Text != "":
				message.Parts = append(message.Parts, &types.MessagePart{
					Text: &types.MessageText{Text: content.Text, Instruction: instructionMap(content.Instruction)},
				})
			case content.ImageURL != "":
				message.Parts = append(message.Parts, &types.MessagePart{
					ImageURL: &types.MessageImageURL{URL: content.ImageURL},
				})
			}
		}
		if len(message.Parts) == 0 {
			return nil
		}
		return message
	case event.Reply != nil:
		reply := event.Reply
		message := &types.Message{ID: event.ID, Role: types.MessageRoleAssistant}
		if reply.Reasoning != "" {
			message.Parts = append(message.Parts, &types.MessagePart{
				Reasoning: &types.MessageReasoning{Text: reply.Reasoning},
			})
		}
		if reply.Text != "" {
			message.Parts = append(message.Parts, &types.MessagePart{
				Text: &types.MessageText{Text: reply.Text},
			})
		}
		for _, call := range reply.ToolCalls {
			message.Parts = append(message.Parts, &types.MessagePart{
				ToolCall: &types.MessageToolCall{
					ID:        call.ID,
					Type:      types.ToolTypeFunction,
					Name:      call.Name,
					Arguments: call.Arguments,
				},
			})
		}
		if len(message.Parts) == 0 {
			return nil
		}
		return message
	case event.ToolResult != nil:
		result := event.ToolResult
		instruction := instructionMap(result.Instruction)
		return &types.Message{ID: event.ID, Role: types.MessageRoleTool, Parts: []*types.MessagePart{{
			ToolResult: &types.MessageToolResult{
				ID:          result.CallID,
				Name:        result.Name,
				Result:      result.Value,
				Error:       result.Error,
				Instruction: instruction,
				Final:       result.Final,
			},
		}}}
	}
	return nil
}

// instructionMap converts event instructions to the provider form.
func instructionMap(source map[string]string) map[string][]string {
	if len(source) == 0 {
		return nil
	}
	instruction := make(map[string][]string, len(source))
	for key, value := range source {
		instruction[key] = []string{value}
	}
	return instruction
}

// ToolSpecsToTypes converts tool declarations to the provider tool type.
func ToolSpecsToTypes(specs []ToolSpec) []*types.Tool {
	tools := make([]*types.Tool, 0, len(specs))
	for _, spec := range specs {
		tools = append(tools, types.NewFunctionTool(spec.Name, spec.Description, spec.Parameters))
	}
	return tools
}

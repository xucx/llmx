package llmx

import (
	"context"
	"fmt"
	"strings"

	"github.com/xucx/llmx/internal/utils"
	"github.com/xucx/llmx/types"
)

const OutputSchemaInstruction = "<|output_json_schema|>"

func RunCompletionTask[T any](ctx context.Context, m *Model, prompt string, failRetry int, options ...types.CompletionOption) (*T, error) {
	if m == nil {
		return nil, fmt.Errorf("model is nil")
	}
	if failRetry < 0 {
		return nil, fmt.Errorf("failRetry must not be negative")
	}
	opts := types.ApplyCompletionOptions(nil, options...)

	resolved, schema, err := utils.JsonSchemaResolve[T]()
	if err != nil {
		return nil, err
	}

	if strings.Contains(opts.Instructions, OutputSchemaInstruction) {
		opts.Instructions = strings.ReplaceAll(opts.Instructions, OutputSchemaInstruction, schema)
	} else {
		opts.Instructions += fmt.Sprintf("\n\nThe response MUST be output strictly in a valid JSON format, adhering to the following JSON schema constraints:\n\n%s\n", schema)
	}

	var (
		lastText string
		lastErr  error
	)

	messages := []*types.Message{
		types.NewTextMessage(types.MessageRoleUser, prompt),
	}

	for attempt := 0; attempt <= failRetry; attempt++ {
		completion, err := m.Completion(ctx, messages, types.CompletionWithOptions(opts))
		if err != nil {
			return nil, err
		}

		if completion.Message == nil {
			return nil, fmt.Errorf("completion message nil")
		}

		lastText = completion.Message.Text()
		messages = append(messages, types.NewTextMessage(types.MessageRoleAssistant, lastText))

		var result T
		lastErr = utils.JsonSchemaValidate(resolved, lastText, &result)
		if lastErr == nil {
			return &result, nil
		}
		if attempt < failRetry {
			messages = append(messages, types.NewTextMessage(types.MessageRoleUser, fmt.Sprintf(
				"Output unmarshalling to JSON failed with error: [%s]. You must fix this and re-output the correct JSON directly. The output MUST strictly comply with the provided JSON schema definition. If there are any other formatting or structural issues, fix them all together.",
				lastErr.Error(),
			)))
		}
	}
	return nil, fmt.Errorf("completion returned invalid structured output after %d attempt(s): %w", failRetry+1, lastErr)
}

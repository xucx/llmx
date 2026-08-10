package openai

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/xucx/llmx/types"
)

func TestCompletion(t *testing.T) {
	provider, err := InitTestOpenaiProvider()
	if err != nil {
		//skip test
		if errors.Is(err, errorNoOpenApiKey) {
			return
		}
		t.Fatalf("create openai provider fail: %v", err)
	}

	completion, err := provider.Completion(
		context.Background(),
		"gpt-5.4",
		[]*types.Message{types.NewTextMessage(types.MessageRoleUser, "hi")},
		types.CompletionWithReasoningLevel(10),
		types.CompletionWithStreamingAccFunc(func(ctx context.Context, completion *types.Completion) error {
			fmt.Printf("acc stream: %s", completion.Message.String())
			return nil
		}),
	)
	if err != nil {
		t.Fatalf("call completion fail: %v", err)
	}

	fmt.Println(completion.Message.String())
}

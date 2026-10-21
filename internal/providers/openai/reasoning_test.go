package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/xucx/llmx/types"
)

// Newer vLLM returns the reasoning in "reasoning" and leaves "reasoning_content" empty.
func TestReasoningFieldIsParsed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, delta := range []string{
			`{"reasoning":"think ","reasoning_content":""}`,
			`{"reasoning":"hard","reasoning_content":""}`,
			`{"content":"answer","reasoning_content":""}`,
		} {
			fmt.Fprintf(w, "data: {\"id\":\"r\",\"choices\":[{\"index\":0,\"delta\":%s}]}\n\n", delta)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	provider, err := NewOpenaiProvider(types.ProviderWithUrl(server.URL), types.ProviderWithSk("test"))
	if err != nil {
		t.Fatal(err)
	}
	completion, err := provider.Completion(
		context.Background(),
		"model",
		[]*types.Message{types.NewTextMessage(types.MessageRoleUser, "hi")},
		types.CompletionWithStreamingFunc(func(context.Context, *types.Completion) error { return nil }),
	)
	if err != nil {
		t.Fatal(err)
	}
	if got := completion.Message.Reasoning(); got != "think hard" {
		t.Fatalf("reasoning lost: %q", got)
	}
	if got := completion.Message.Text(); got != "answer" {
		t.Fatalf("text changed: %q", got)
	}
}

func TestReasoningContentTakesPrecedence(t *testing.T) {
	m := &CompletionMessage{ReasoningContent: "a", Reasoning: "a"}
	if got := m.reasoningText(); got != "a" {
		t.Fatalf("reasoning duplicated: %q", got)
	}
}

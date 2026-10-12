package openai

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/xucx/llmx/types"
)

func TestCanceledStreamReturnsTheCompletionReceivedSoFar(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// One chunk of text and the start of a tool call, then the stream stalls.
		fmt.Fprint(w, `data: {"id":"r","choices":[{"index":0,"delta":{"content":"partial",`+
			`"tool_calls":[{"index":0,"id":"call","type":"function","function":{"name":"f","arguments":"{\"a"}}]}}]}`+"\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer server.Close()

	provider, err := NewOpenaiProvider(types.ProviderWithUrl(server.URL), types.ProviderWithSk("test"))
	if err != nil {
		t.Fatal(err)
	}
	stop := errors.New("stop")
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(nil)
	completion, err := provider.Completion(
		ctx,
		"model",
		[]*types.Message{types.NewTextMessage(types.MessageRoleUser, "hi")},
		types.CompletionWithStreamingFunc(func(context.Context, *types.Completion) error {
			cancel(stop) // interrupt once the first chunk arrived
			return nil
		}),
	)
	if !errors.Is(err, stop) {
		t.Fatalf("expected the cancellation cause, got %v", err)
	}
	if completion == nil || completion.Message.Text() != "partial" {
		t.Fatalf("the text received so far was lost: %+v", completion)
	}
	if calls := completion.Message.ToolCalls(); len(calls) != 1 || calls[0].Arguments != `{"a` {
		t.Fatalf("expected the truncated tool call as received: %+v", calls)
	}
}

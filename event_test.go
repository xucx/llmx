package llmx

import (
	"strings"
	"testing"

	"github.com/xucx/llmx/types"
)

func TestEventValidateRequiresExactlyOnePayload(t *testing.T) {
	event := NewEventMessage(
		types.NewTextMessage(types.MessageRoleUser, "hello"),
	)
	event.Usage = &EventUsage{}
	err := event.Validate()
	if err == nil || !strings.Contains(err.Error(), "payloads") {
		t.Fatalf("expected oneof validation error, got %v", err)
	}

	empty := &Event{ID: "empty"}
	err = empty.Validate()
	if err == nil || !strings.Contains(err.Error(), "payloads") {
		t.Fatalf("expected empty oneof validation error, got %v", err)
	}
}

func TestEventStateRejectsEmptyKey(t *testing.T) {
	err := NewEventState(map[string]string{"": "value"}, true).Validate()
	if err == nil || !strings.Contains(err.Error(), "state key is empty") {
		t.Fatalf("expected invalid state key error, got %v", err)
	}
}

func TestEventStateAllowsEmptyReplacementButNotEmptyPatch(t *testing.T) {
	if err := NewEventState(map[string]string{}, false).Validate(); err != nil {
		t.Fatalf("empty replacement must be valid: %v", err)
	}
	err := NewEventState(map[string]string{}, true).Validate()
	if err == nil || !strings.Contains(err.Error(), "state patch is empty") {
		t.Fatalf("expected empty state patch error, got %v", err)
	}
}

func TestNewSessionAppliesCompressionAndIgnoresDelta(t *testing.T) {
	original := types.NewTextMessage(types.MessageRoleUser, "original")
	delta := types.NewTextMessage(types.MessageRoleAssistant, "partial")
	delta.Delta = true
	kept := types.NewTextMessage(types.MessageRoleAssistant, "kept")
	summary := types.NewTextMessage(types.MessageRoleUser, "summary")
	latest := types.NewTextMessage(types.MessageRoleUser, "latest")

	events := []*Event{
		NewEventMessage(original),
		NewEventMessage(delta),
		NewEventMessage(kept),
		NewEventUsage(types.CompletionUsage{PromptTokens: 9, TotalTokens: 9}),
		NewEventCompression(&EventCompression{
			Summary: &CompressResultSummary{
				Message: summary,
				// The reduction is derived from this usage on apply:
				// 8 prompt - 2 completion = 6 tokens removed.
				Usage: types.CompletionUsage{
					PromptTokens:     8,
					CompletionTokens: 2,
					TotalTokens:      10,
				},
			},
			UntilMessageID: original.ID,
		}),
		NewEventMessage(latest),
		NewEventState(map[string]string{"key": "value"}, true),
	}

	agent, err := NewAgent("test", "", NewModels(), WithModel("test"))
	if err != nil {
		t.Fatal(err)
	}
	session, err := NewSession(agent, events)
	if err != nil {
		t.Fatal(err)
	}
	// The summary covers the original message only, so the message after the
	// boundary survives it.
	if len(session.history) != 3 ||
		session.history[0].Text() != "summary" ||
		session.history[1].Text() != "kept" ||
		session.history[2].Text() != "latest" {
		t.Fatalf("unexpected session history: %+v", session.history)
	}
	// A compression is not a reported context size: it only says how much it
	// removed, so restoring it subtracts that from the last reported size.
	if session.contextTokens != 3 {
		t.Fatalf("unexpected session context tokens: %d", session.contextTokens)
	}
	if session.state["key"] != "value" {
		t.Fatalf("unexpected session state: %+v", session.state)
	}
	for index, event := range events {
		if session.events[index] != event {
			t.Fatalf("history event %d was cloned", index)
		}
	}
}

func TestNewSessionValidatesHumanCorrelation(t *testing.T) {
	agent, err := NewAgent("test", "", NewModels(), WithModel("test"))
	if err != nil {
		t.Fatal(err)
	}
	unknown := NewEventHumanResponse(&EventHumanResponse{
		ID: "missing",
	})
	if _, err := NewSession(
		agent,
		[]*Event{unknown},
	); err == nil {
		t.Fatal("expected unknown human request error")
	}

	request := NewEventHumanRequest(&EventHumanRequest{
		ID: "request",
		ToolCall: &types.MessageToolCall{
			ID: "call", Name: "tool", Arguments: "{}",
		},
	})
	assistant := types.NewMessage(types.MessageRoleAssistant)
	assistant.Parts = append(assistant.Parts, &types.MessagePart{
		ToolCall: &types.MessageToolCall{
			ID: "call", Name: "tool", Arguments: "{}",
		},
	})
	response := NewEventHumanResponse(&EventHumanResponse{
		ID: "request",
	})
	duplicate := NewEventHumanResponse(&EventHumanResponse{
		ID: "request",
	})
	if _, err := NewSession(
		agent,
		[]*Event{
			NewEventMessage(assistant),
			request,
			response,
			duplicate,
		},
	); err == nil {
		t.Fatal("expected duplicate human response error")
	}
}

func TestEventCloneOwnsNestedInput(t *testing.T) {
	message := types.NewTextMessage(types.MessageRoleUser, "before")
	event := NewEventMessage(message)
	message.Parts[0].Text.Text = "after"
	if event.Message.Text() != "before" {
		t.Fatalf("constructor retained caller message: %+v", event)
	}

	response := &EventHumanResponse{
		ID: "request",
		Result: map[string]any{
			"nested": map[string]any{"value": "before"},
		},
	}
	responseEvent := NewEventHumanResponse(response)
	response.Result["nested"].(map[string]any)["value"] = "after"
	nested := responseEvent.HumanResponse.Result["nested"].(map[string]any)
	if nested["value"] != "before" {
		t.Fatalf("constructor retained caller result: %+v", responseEvent)
	}
}

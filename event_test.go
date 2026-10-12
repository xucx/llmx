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

func TestNewAgentRestoresHistoryAndIgnoresDelta(t *testing.T) {
	original := types.NewTextMessage(types.MessageRoleUser, "original")
	delta := types.NewTextMessage(types.MessageRoleAssistant, "partial")
	delta.Delta = true
	kept := types.NewTextMessage(types.MessageRoleAssistant, "kept")
	latest := types.NewTextMessage(types.MessageRoleUser, "latest")

	events := []*Event{
		NewEventMessage(original),
		NewEventMessage(delta),
		NewEventMessage(kept),
		NewEventUsage(types.CompletionUsage{PromptTokens: 9, TotalTokens: 9}),
		NewEventMessage(latest),
		NewEventState(map[string]string{"key": "value"}, true),
	}

	factory, err := newAgentFactory(NewModels(), WithModel("test"))
	if err != nil {
		t.Fatal(err)
	}
	agent, err := factory.build(events)
	if err != nil {
		t.Fatal(err)
	}
	// A delta is a streaming fragment, never history.
	history := agent.session.history()
	if len(history) != 3 ||
		history[0].Text() != "original" ||
		history[1].Text() != "kept" ||
		history[2].Text() != "latest" {
		t.Fatalf("unexpected agent history: %+v", history)
	}
	if agent.session.state["key"] != "value" {
		t.Fatalf("unexpected agent state: %+v", agent.session.state)
	}
	// The delta is dropped; every other event is retained by reference.
	restored := append(append([]*Event(nil), events[:1]...), events[2:]...)
	if len(agent.session.events) != len(restored) {
		t.Fatalf("unexpected restored events: %+v", agent.session.events)
	}
	for index, event := range restored {
		if agent.session.events[index] != event {
			t.Fatalf("history event %d was cloned", index)
		}
	}
}

func TestNewAgentValidatesHumanCorrelation(t *testing.T) {
	agent, err := newAgentFactory(NewModels(), WithModel("test"))
	if err != nil {
		t.Fatal(err)
	}
	unknown := NewEventHumanResponse(&EventHumanResponse{
		ID: "missing",
	})
	if _, err := agent.build(
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
	if _, err := agent.build(
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

func TestStatusEventStopReasonExactlyWhenNotRunning(t *testing.T) {
	cases := []struct {
		status EventStatus
		valid  bool
	}{
		{EventStatus{Status: StatusRunning}, true},
		{EventStatus{Status: StatusWaiting, StopReason: StopInterrupted, Turns: 1}, true},
		{EventStatus{Status: StatusStopped, StopReason: StopMaxTurn}, true},
		{EventStatus{}, false},
		{EventStatus{Status: StatusRunning, StopReason: StopDone}, false},
		{EventStatus{Status: StatusWaiting, Turns: 1}, false},
		{EventStatus{Status: StatusStopped}, false},
	}
	for _, tc := range cases {
		status := tc.status
		err := (&Event{ID: "status", Status: &status}).Validate()
		if (err == nil) != tc.valid {
			t.Fatalf("status %+v: valid=%v, got err %v", tc.status, tc.valid, err)
		}
	}
}

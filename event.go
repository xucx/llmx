package llmx

import (
	"fmt"
	"maps"

	"github.com/google/uuid"
	"github.com/xucx/llmx/types"
)

// Event is a oneof envelope. Exactly one payload field must be non-nil. Once
// accepted by a Session, an Event and its nested values are immutable; callers
// must Clone before modifying them.
type Event struct {
	ID string `json:"id,omitempty" yaml:"id,omitempty"`

	Message       *types.Message      `json:"message,omitempty" yaml:"message,omitempty"`
	HumanRequest  *EventHumanRequest  `json:"humanRequest,omitempty" yaml:"humanRequest,omitempty"`
	HumanResponse *EventHumanResponse `json:"humanResponse,omitempty" yaml:"humanResponse,omitempty"`
	Compression   *EventCompression   `json:"compression,omitempty" yaml:"compression,omitempty"`
	Usage         *EventUsage         `json:"usage,omitempty" yaml:"usage,omitempty"`
	State         *EventState         `json:"state,omitempty" yaml:"state,omitempty"`
}

type EventHumanRequest struct {
	ID        string                 `json:"id,omitempty" yaml:"id,omitempty"`
	Arguments map[string]any         `json:"arguments,omitempty" yaml:"arguments,omitempty"`
	ToolCall  *types.MessageToolCall `json:"toolCall,omitempty" yaml:"toolCall,omitempty"`
}

type EventHumanResponse struct {
	ID     string         `json:"id,omitempty" yaml:"id,omitempty"`
	Result map[string]any `json:"result,omitempty" yaml:"result,omitempty"`
}

// EventCompression is one self-contained compression: the range it covers and
// the transform it applies to that range. How much context the transform
// removed is not stored; ApplyCompression derives it from the event, so
// generating and replaying a compression always report the same reduction.
type EventCompression struct {
	//one of
	ToolResult *CompressResultToolResult `json:"toolResult,omitempty" yaml:"toolResult,omitempty"`
	Summary    *CompressResultSummary    `json:"summary,omitempty" yaml:"summary,omitempty"`

	// UntilMessageID is the last history message this compression covers,
	// inclusive. Messages after it are left unchanged.
	UntilMessageID string `json:"untilMessageID,omitempty" yaml:"untilMessageID,omitempty"`
}

type CompressResultToolResult struct {
	Result       string            `json:"result,omitempty" yaml:"result,omitempty"`
	Instructions map[string]string `json:"instructions,omitempty" yaml:"instructions,omitempty"`
}

// CompressResultSummary replaces the whole covered range with one bridge
// message. The messages it replaces stay in the event log, so only the bridge
// itself has to be stored here. Usage is from the summary call that wrote the
// bridge: it read the range the bridge replaces, so the reduction this
// compression achieves is derived from it when the event is applied.
type CompressResultSummary struct {
	Message *types.Message        `json:"message,omitempty" yaml:"message,omitempty"`
	Usage   types.CompletionUsage `json:"usage,omitempty" yaml:"usage,omitempty"`
}
type EventUsage struct {
	Usage types.CompletionUsage `json:"usage,omitempty" yaml:"usage,omitempty"`
}

type EventState struct {
	Patch bool              `json:"patch,omitempty" yaml:"patch,omitempty"`
	State map[string]string `json:"state,omitempty" yaml:"state,omitempty"`
}

// NewEventMessage wraps one message into one event: a message is the unit of
// idempotency and ordering, so it is never batched with others.
func NewEventMessage(message *types.Message) *Event {
	return &Event{
		ID:      uuid.NewString(),
		Message: message.Clone(),
	}
}

func NewEventHumanRequest(request *EventHumanRequest) *Event {
	return &Event{ID: uuid.NewString(), HumanRequest: request.Clone()}
}

func NewEventHumanResponse(response *EventHumanResponse) *Event {
	return &Event{ID: uuid.NewString(), HumanResponse: response.Clone()}
}

func NewEventCompression(compression *EventCompression) *Event {
	return &Event{ID: uuid.NewString(), Compression: compression.Clone()}
}

func NewEventUsage(usage types.CompletionUsage) *Event {
	return &Event{ID: uuid.NewString(), Usage: &EventUsage{Usage: usage}}
}

func NewEventState(state map[string]string, patch bool) *Event {
	return &Event{
		ID:    uuid.NewString(),
		State: &EventState{State: maps.Clone(state), Patch: patch},
	}
}

func (e *Event) Validate() error {
	if e == nil {
		return fmt.Errorf("event is nil")
	}
	if e.ID == "" {
		return fmt.Errorf("event id is empty")
	}

	fields := 0
	for _, present := range []bool{
		e.Message != nil,
		e.HumanRequest != nil,
		e.HumanResponse != nil,
		e.Compression != nil,
		e.Usage != nil,
		e.State != nil,
	} {
		if present {
			fields++
		}
	}
	if fields != 1 {
		return fmt.Errorf("event %q has %d payloads, want 1", e.ID, fields)
	}

	switch {
	case e.Message != nil:
		if err := types.ValidateMessages([]*types.Message{e.Message}); err != nil {
			return fmt.Errorf("event %q: %w", e.ID, err)
		}
	case e.HumanRequest != nil:
		if e.HumanRequest.ID == "" {
			return fmt.Errorf("event %q human request id is empty", e.ID)
		}
		if e.HumanRequest.ToolCall == nil {
			return fmt.Errorf("event %q human request tool call is nil", e.ID)
		}
		if e.HumanRequest.ToolCall.ID == "" {
			return fmt.Errorf("event %q human request tool call id is empty", e.ID)
		}
	case e.HumanResponse != nil:
		if e.HumanResponse.ID == "" {
			return fmt.Errorf("event %q human response request id is empty", e.ID)
		}
	case e.Compression != nil:
		if e.Compression.ToolResult == nil && e.Compression.Summary == nil {
			return fmt.Errorf("event %q compression messages are empty", e.ID)
		}
		if e.Compression.ToolResult != nil && e.Compression.Summary != nil {
			return fmt.Errorf("event %q compression has 2 results, want 1", e.ID)
		}
		if e.Compression.UntilMessageID == "" {
			return fmt.Errorf("event %q compression until message id is empty", e.ID)
		}
		if e.Compression.Summary != nil && e.Compression.Summary.Message == nil {
			return fmt.Errorf("event %q compression summary message is nil", e.ID)
		}
	case e.State != nil:
		if e.State.Patch && len(e.State.State) == 0 {
			return fmt.Errorf("event %q state patch is empty", e.ID)
		}
		if err := validateState(e.State.State); err != nil {
			return fmt.Errorf("event %q state: %w", e.ID, err)
		}
	}
	return nil
}

func (e *Event) Clone() *Event {
	if e == nil {
		return nil
	}
	clone := &Event{ID: e.ID}
	if e.Message != nil {
		clone.Message = e.Message.Clone()
	}
	if e.HumanRequest != nil {
		clone.HumanRequest = e.HumanRequest.Clone()
	}
	if e.HumanResponse != nil {
		clone.HumanResponse = e.HumanResponse.Clone()
	}
	if e.Compression != nil {
		clone.Compression = e.Compression.Clone()
	}
	if e.Usage != nil {
		value := *e.Usage
		clone.Usage = &value
	}
	if e.State != nil {
		clone.State = &EventState{State: maps.Clone(e.State.State), Patch: e.State.Patch}
	}
	return clone
}

func (e *EventHumanRequest) Clone() *EventHumanRequest {
	if e == nil {
		return nil
	}
	clone := *e
	clone.Arguments = cloneAnyMap(e.Arguments)
	if e.ToolCall != nil {
		call := *e.ToolCall
		clone.ToolCall = &call
	}
	return &clone
}

func (e *EventHumanResponse) Clone() *EventHumanResponse {
	if e == nil {
		return nil
	}
	clone := *e
	clone.Result = cloneAnyMap(e.Result)
	return &clone
}

func (e *EventCompression) Clone() *EventCompression {
	if e == nil {
		return nil
	}
	clone := *e
	if e.ToolResult != nil {
		clone.ToolResult = &CompressResultToolResult{
			Result:       e.ToolResult.Result,
			Instructions: maps.Clone(e.ToolResult.Instructions),
		}
	}
	if e.Summary != nil {
		clone.Summary = &CompressResultSummary{
			Message: e.Summary.Message.Clone(),
			Usage:   e.Summary.Usage,
		}
	}

	return &clone
}

func cloneMessages(messages []*types.Message) []*types.Message {
	if messages == nil {
		return nil
	}

	clones := make([]*types.Message, len(messages))
	for index, message := range messages {
		clones[index] = message.Clone()
	}
	return clones
}

func cloneAnyMap(source map[string]any) map[string]any {
	if source == nil {
		return nil
	}
	clone := make(map[string]any, len(source))
	for key, value := range source {
		clone[key] = cloneAny(value)
	}
	return clone
}

func cloneAny(value any) any {
	switch value := value.(type) {
	case map[string]any:
		return cloneAnyMap(value)
	case []any:
		clone := make([]any, len(value))
		for index, item := range value {
			clone[index] = cloneAny(item)
		}
		return clone
	default:
		return value
	}
}

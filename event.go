package llmx

import (
	"fmt"
	"maps"

	"github.com/google/uuid"
	"github.com/xucx/llmx/types"
)

// Event is a oneof envelope. Exactly one payload field must be non-nil. Once
// appended to an event stream, an Event and its nested values are immutable;
// callers must Clone before modifying them.
type Event struct {
	ID string `json:"id,omitempty" yaml:"id,omitempty"`

	Status        *EventStatus        `json:"status,omitempty" yaml:"status,omitempty"`
	Message       *types.Message      `json:"message,omitempty" yaml:"message,omitempty"`
	HumanRequest  *EventHumanRequest  `json:"humanRequest,omitempty" yaml:"humanRequest,omitempty"`
	HumanResponse *EventHumanResponse `json:"humanResponse,omitempty" yaml:"humanResponse,omitempty"`
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

type EventUsage struct {
	Usage types.CompletionUsage `json:"usage,omitempty" yaml:"usage,omitempty"`
}

type EventState struct {
	Patch bool              `json:"patch,omitempty" yaml:"patch,omitempty"`
	State map[string]string `json:"state,omitempty" yaml:"state,omitempty"`
}

// Status is the run state of an agent.
type Status string

const (
	// StatusRunning means a task started: it appends its inputs, resumes
	// answered human requests and runs model turns.
	StatusRunning Status = "running"
	// StatusWaiting means the task ended and the agent waits for the next one;
	// StopReason tells how the task ended.
	StatusWaiting Status = "waiting"
	// StatusStopped means the agent closes; StopReason tells why.
	StatusStopped Status = "stopped"
)

// StopReason tells why a task or an agent stopped.
type StopReason string

const (
	// StopDone means Loop finished the task, or stopped the agent.
	StopDone StopReason = "done"
	// StopMaxTurn means the task used up MaxTurn; the agent closes.
	StopMaxTurn StopReason = "maxTurn"
	// StopInterjected means an interjection ended the task at a turn boundary.
	StopInterjected StopReason = "interjected"
	// StopInterrupted means an interruption canceled the task in flight.
	StopInterrupted StopReason = "interrupted"
	// StopRejected means the stream rejected the inputs of the task, which
	// appended nothing; the agent waits for the next one.
	StopRejected StopReason = "rejected"
	// StopCanceled and StopError only appear in a Result: the agent closed
	// because its context was canceled or an error occurred.
	StopCanceled StopReason = "canceled"
	StopError    StopReason = "error"
)

// EventStatus records one change of the agent run state. Every task is
// bracketed by a running status and a waiting (or stopped) status, so a caller
// rendering the event stream knows when a reply is complete. An agent closed by
// an error or a canceled context appends no closing status; Runner.Wait or the
// Run result report it instead. It changes no state, so replaying it is a no-op.
type EventStatus struct {
	Status Status `json:"status,omitempty" yaml:"status,omitempty"`
	// StopReason is set exactly when Status is waiting or stopped.
	StopReason StopReason `json:"stopReason,omitempty" yaml:"stopReason,omitempty"`
	// Turns is the number of model turns the task ran, the count MaxTurn
	// bounds. It is zero for StatusRunning.
	Turns int `json:"turns,omitempty" yaml:"turns,omitempty"`
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
		e.Usage != nil,
		e.State != nil,
		e.Status != nil,
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
	case e.State != nil:
		if e.State.Patch && len(e.State.State) == 0 {
			return fmt.Errorf("event %q state patch is empty", e.ID)
		}
		if err := validateState(e.State.State); err != nil {
			return fmt.Errorf("event %q state: %w", e.ID, err)
		}
	case e.Status != nil:
		if e.Status.Status == "" {
			return fmt.Errorf("event %q status is empty", e.ID)
		}
		if (e.Status.Status != StatusRunning) != (e.Status.StopReason != "") {
			return fmt.Errorf("event %q stop reason must be set exactly when not running", e.ID)
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
	if e.Usage != nil {
		value := *e.Usage
		clone.Usage = &value
	}
	if e.State != nil {
		clone.State = &EventState{State: maps.Clone(e.State.State), Patch: e.State.Patch}
	}
	if e.Status != nil {
		value := *e.Status
		clone.Status = &value
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

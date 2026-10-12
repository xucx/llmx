package llmx

import (
	"fmt"
	"maps"

	"github.com/xucx/llmx/types"
)

// Input is the only way a caller feeds an agent. Every input starts a task,
// through Agent.Run or through a Runner.
//
// Callers never build input events themselves: the agent turns one input into
// events, so every event a caller observes comes from the agent.
//
// A task runs model turns only when its inputs give the model something to
// answer: a system or user message, or a tool result -- passed in or produced
// by resuming a human request -- that is not marked final. Any other input,
// such as a state change alone, is appended and the task is done.
type Input struct {
	// ID identifies this input while it waits in a Runner, so it can be
	// withdrawn. The Runner fills it in when it is empty. Agent.Run ignores it.
	ID string `json:"id,omitempty" yaml:"id,omitempty"`

	// Messages are appended to the model history, one message event each.
	Messages []*types.Message `json:"messages,omitempty" yaml:"messages,omitempty"`

	// HumanResponses answer human requests that are still pending. The ID of
	// each must match the ID of an EventHumanRequest in the event stream.
	HumanResponses []*EventHumanResponse `json:"humanResponses,omitempty" yaml:"humanResponses,omitempty"`

	// State replaces the agent state before the rest of this input.
	// StateDelta merges into the current agent state instead.
	State      map[string]string `json:"state,omitempty" yaml:"state,omitempty"`
	StateDelta map[string]string `json:"stateDelta,omitempty" yaml:"stateDelta,omitempty"`
}

// NewInput creates one input carrying messages.
func NewInput(messages ...*types.Message) *Input {
	return &Input{Messages: cloneMessages(messages)}
}

// NewUserInput creates one input carrying a single user text message.
func NewUserInput(text string) *Input {
	return NewInput(types.NewTextMessage(types.MessageRoleUser, text))
}

// NewHumanResponseInput creates one input answering a pending human request.
func NewHumanResponseInput(
	requestID string,
	result map[string]any,
) *Input {
	return new(Input).WithHumanResponse(requestID, result)
}

// WithMessages appends messages to this input.
func (i *Input) WithMessages(messages ...*types.Message) *Input {
	if i == nil {
		i = &Input{}
	}
	i.Messages = append(i.Messages, cloneMessages(messages)...)
	return i
}

// WithHumanResponse appends one human response to this input.
func (i *Input) WithHumanResponse(
	id string,
	result map[string]any,
) *Input {
	if i == nil {
		i = &Input{}
	}
	i.HumanResponses = append(i.HumanResponses, &EventHumanResponse{
		ID:     id,
		Result: cloneAnyMap(result),
	})
	return i
}

// WithState sets either a full state replacement or a state patch on this
// input. When delta is true, state is merged; otherwise it replaces the state,
// and a nil state clears it.
func (i *Input) WithState(state map[string]string, delta bool) *Input {
	if i == nil {
		i = &Input{}
	}

	clone := maps.Clone(state)
	if delta {
		i.StateDelta = clone
		return i
	}
	if clone == nil {
		clone = map[string]string{}
	}
	i.State = clone
	return i
}

// IsEmpty reports whether this input carries nothing for the agent to accept.
func (i *Input) IsEmpty() bool {
	return i == nil ||
		(i.State == nil &&
			len(i.StateDelta) == 0 &&
			len(i.HumanResponses) == 0 &&
			len(i.Messages) == 0)
}

func (i *Input) Clone() *Input {
	if i == nil {
		return nil
	}

	clone := &Input{
		ID:         i.ID,
		State:      maps.Clone(i.State),
		StateDelta: maps.Clone(i.StateDelta),
		Messages:   cloneMessages(i.Messages),
	}
	if len(i.HumanResponses) == 0 {
		return clone
	}
	clone.HumanResponses = make([]*EventHumanResponse, len(i.HumanResponses))
	for index, response := range i.HumanResponses {
		clone.HumanResponses[index] = response.Clone()
	}
	return clone
}

func (i *Input) Validate() error {
	if i.IsEmpty() {
		return nil
	}
	if i.State != nil && i.StateDelta != nil {
		return fmt.Errorf("state and stateDelta both set")
	}

	if err := validateState(i.State); err != nil {
		return fmt.Errorf("state: %w", err)
	}
	if err := validateState(i.StateDelta); err != nil {
		return fmt.Errorf("stateDelta: %w", err)
	}
	for index, response := range i.HumanResponses {
		if response == nil {
			return fmt.Errorf("human response %d is nil", index)
		}
		if response.ID == "" {
			return fmt.Errorf("human response %d request id is empty", index)
		}
	}
	for index, message := range i.Messages {
		if message == nil {
			return fmt.Errorf("message %d is nil", index)
		}
		if message.Delta {
			return fmt.Errorf("message %d is delta", index)
		}
	}
	return types.ValidateMessages(i.Messages)
}

// events converts this input into the ordered events a task appends. State
// lands first so tools observe it, human responses next so pending tools can
// resume, and messages last so they are the newest history.
//
// Responses are not deduplicated here: a second response for the same request
// is rejected when it is appended, so the contradiction fails loudly instead of
// being silently dropped.
func (i *Input) events() ([]*Event, error) {
	if err := i.Validate(); err != nil {
		return nil, err
	}
	if i.IsEmpty() {
		return nil, nil
	}

	events := make([]*Event, 0, len(i.HumanResponses)+len(i.Messages)+2)
	if i.State != nil {
		events = append(events, NewEventState(i.State, false))
	}
	if len(i.StateDelta) > 0 {
		events = append(events, NewEventState(i.StateDelta, true))
	}
	for _, response := range i.HumanResponses {
		events = append(events, NewEventHumanResponse(response))
	}
	for _, message := range i.Messages {
		events = append(events, NewEventMessage(message))
	}
	return events, nil
}

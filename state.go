package llmx

import (
	"maps"
)

// PendingCall is a tool call of the newest reply that has no result yet.
type PendingCall struct {
	Call ToolCall
	// Since is the append time of the reply that made the call.
	Since int64
	// Ask is set while the tool waits for an answer; Answer once it arrived
	// and the tool has not run again yet.
	Ask    *ToolAsk
	Answer *ToolAnswer
}

// State is the view Fold derives from the events of a Session. It is never
// stored: every driver folds it again from the events it loads.
type State struct {
	// Seq is the seq of the last folded event.
	Seq int64
	// Events are all the folded events, in order.
	Events []*Event
	// Admitted holds, in order, the Input, Reply and ToolResult events that
	// make up the conversation the model sees. BuildMessages renders it.
	Admitted []*Event
	// Tools are the tools declared by the newest admitted input that
	// declared any.
	Tools []ToolSpec
	// Pending are the tool calls of the newest reply without a result.
	Pending []PendingCall
	// Inputs are the input events not admitted yet, in order.
	Inputs []*Event
	// Turns counts the model calls since the newest admitted input.
	Turns int
	// Stopped is set while a Stop closes the round: the model is not called
	// again until a new input is admitted.
	Stopped *Stop
	// KV is the key-value state patched by inputs and Patch events.
	KV map[string]string

	calls    map[string]ToolCall
	resolved map[string]struct{}
	// held marks the inputs that arrived while the model was being asked:
	// they wait for the reply (or stop) that was in flight, so the reply is
	// never shown as an answer to them.
	held map[string]struct{}
}

// Fold derives the State from events, which must be in seq order.
func Fold(events []*Event) *State {
	s := &State{
		KV:       map[string]string{},
		calls:    map[string]ToolCall{},
		resolved: map[string]struct{}{},
		held:     map[string]struct{}{},
	}
	for _, event := range events {
		s.apply(event)
	}
	return s
}

func (s *State) apply(event *Event) {
	if event == nil {
		return
	}
	s.Seq = event.Seq
	s.Events = append(s.Events, event)
	switch {
	case event.Input != nil:
		s.Inputs = append(s.Inputs, event)
		// The model is being asked (or about to be): an interjection or a
		// queued input must not slip in before its reply lands, or the
		// reply would read as an answer to the newcomer.
		if event.Input.Mode != InputInterrupt && len(s.Pending) == 0 && s.NeedsAnswer() {
			s.held[event.ID] = struct{}{}
		}
	case event.Reply != nil:
		clear(s.held)
		s.Admitted = append(s.Admitted, event)
		s.Turns++
		s.Pending = s.Pending[:0]
		for _, call := range event.Reply.ToolCalls {
			s.calls[call.ID] = call
			s.Pending = append(s.Pending, PendingCall{Call: call, Since: event.At})
		}
	case event.ToolResult != nil:
		id := event.ToolResult.CallID
		if _, known := s.calls[id]; !known {
			return
		}
		if _, done := s.resolved[id]; done {
			return
		}
		s.resolved[id] = struct{}{}
		s.Admitted = append(s.Admitted, event)
		for index, pending := range s.Pending {
			if pending.Call.ID == id {
				s.Pending = append(s.Pending[:index:index], s.Pending[index+1:]...)
				break
			}
		}
	case event.ToolAsk != nil:
		if pending := s.pending(event.ToolAsk.CallID); pending != nil && pending.Answer == nil {
			pending.Ask = event.ToolAsk
		}
	case event.ToolAnswer != nil:
		if pending := s.pending(event.ToolAnswer.CallID); pending != nil && pending.Ask != nil {
			pending.Answer = event.ToolAnswer
		}
	case event.Patch != nil:
		s.patch(event.Patch.KV, event.Patch.Reset)
	case event.Stop != nil:
		clear(s.held)
		s.Stopped = event.Stop
	}
	s.admit()
}

func (s *State) pending(callID string) *PendingCall {
	for index := range s.Pending {
		if s.Pending[index].Call.ID == callID {
			return &s.Pending[index]
		}
	}
	return nil
}

func (s *State) patch(kv map[string]string, reset bool) {
	if reset {
		s.KV = map[string]string{}
	}
	maps.Copy(s.KV, kv)
}

// admit moves the inputs that may join the conversation now from Inputs to
// Admitted. The rules only depend on the events folded so far, so every
// driver agrees:
//
//   - an input appended while the model was being asked waits for the reply;
//   - an interrupt joins once nothing is active (Plan cancels what is first)
//     and discards the inputs waiting before it;
//   - an interjection joins once no tool call is pending, ahead of queued
//     inputs;
//   - a queued input joins once the conversation is idle.
func (s *State) admit() {
	for {
		if len(s.Inputs) == 0 {
			return
		}
		held := func(event *Event) bool { _, ok := s.held[event.ID]; return ok }

		if index := s.firstInput(InputInterrupt); index >= 0 && !held(s.Inputs[index]) {
			if s.Active() {
				return
			}
			event := s.Inputs[index]
			s.Inputs = s.Inputs[index+1:]
			s.admitEvent(event)
			continue
		}
		if index := s.firstInput(InputInterject); index >= 0 && !held(s.Inputs[index]) {
			if len(s.Pending) > 0 {
				return
			}
			event := s.Inputs[index]
			s.Inputs = append(s.Inputs[:index:index], s.Inputs[index+1:]...)
			s.admitEvent(event)
			continue
		}
		if index := s.firstInput(InputQueue); index >= 0 && !held(s.Inputs[index]) {
			if s.Active() {
				return
			}
			event := s.Inputs[index]
			s.Inputs = append(s.Inputs[:index:index], s.Inputs[index+1:]...)
			s.admitEvent(event)
			continue
		}
		return
	}
}

func (s *State) firstInput(mode InputMode) int {
	for index, event := range s.Inputs {
		if event.Input.Mode == mode {
			return index
		}
	}
	return -1
}

func (s *State) admitEvent(event *Event) {
	input := event.Input
	s.Admitted = append(s.Admitted, event)
	s.Turns = 0
	s.Stopped = nil
	if input.Tools != nil {
		s.Tools = input.Tools
	}
	if input.State != nil || input.Reset {
		s.patch(input.State, input.Reset)
	}
}

// FirstInput returns the oldest input event waiting in the given mode, or
// nil.
func (s *State) FirstInput(mode InputMode) *Event {
	if index := s.firstInput(mode); index >= 0 {
		return s.Inputs[index]
	}
	return nil
}

// NeedsAnswer reports whether the newest admitted event gives the model
// something to answer: an input with content, or a tool result not marked
// final. A Stop closes the round until the next input.
func (s *State) NeedsAnswer() bool {
	if s.Stopped != nil || len(s.Admitted) == 0 {
		return false
	}
	last := s.Admitted[len(s.Admitted)-1]
	switch {
	case last.Input != nil:
		return last.Input.HasContent()
	case last.ToolResult != nil:
		return !last.ToolResult.Final
	}
	return false
}

// Active reports whether a round is in progress: a tool call is pending or
// the model has something to answer.
func (s *State) Active() bool {
	return len(s.Pending) > 0 || s.NeedsAnswer()
}

// LastReply returns the newest reply event, or nil.
func (s *State) LastReply() *Event {
	for index := len(s.Admitted) - 1; index >= 0; index-- {
		if s.Admitted[index].Reply != nil {
			return s.Admitted[index]
		}
	}
	return nil
}

// Call returns a tool call the conversation made, by ID.
func (s *State) Call(id string) (ToolCall, bool) {
	call, ok := s.calls[id]
	return call, ok
}

// Resolved reports whether the tool call has a result.
func (s *State) Resolved(callID string) bool {
	_, ok := s.resolved[callID]
	return ok
}

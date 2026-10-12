package llmx

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/xucx/llmx/types"
)

// session keeps the event stream of an agent, its single source of truth. It
// derives, incrementally as events are appended, what keeps the stream
// consistent: the state and the correlation of tool calls, human requests and
// human responses. The model history is derived only when a turn needs it.
// Appends are safe for concurrent use.
//
// Streaming fragments (delta messages) are written to the sink as they arrive
// but never kept: the complete message follows them, so they carry nothing
// the stream needs.
//
// Appended events and their nested values are immutable; the event sink and
// every reader share them by reference.
type session struct {
	mu   sync.Mutex
	sink EventSink
	// sinkCtx is the context events are written to the sink with. It outlives
	// task cancellation, so a canceled task can still close its turn.
	sinkCtx context.Context

	events         []*Event
	ids            map[string]struct{}
	state          map[string]string
	toolCalls      map[string]*types.MessageToolCall
	toolResults    map[string]*types.MessageToolResult
	humanRequests  map[string]*EventHumanRequest
	humanResponses map[string]*EventHumanResponse
	// unresolved holds, in order, the human responses whose tool call has no
	// result yet: answers that arrived but whose tool never resumed.
	unresolved []*EventHumanResponse
}

// newSession restores a session from history. Historical events are retained by
// reference and never written to the sink again.
func newSession(history []*Event, sink EventSink) (*session, error) {
	s := &session{
		sink:           sink,
		sinkCtx:        context.Background(),
		ids:            map[string]struct{}{},
		state:          map[string]string{},
		toolCalls:      map[string]*types.MessageToolCall{},
		toolResults:    map[string]*types.MessageToolResult{},
		humanRequests:  map[string]*EventHumanRequest{},
		humanResponses: map[string]*EventHumanResponse{},
	}
	for _, event := range history {
		if isDelta(event) {
			continue
		}
		if err := s.admit(event); err != nil {
			return nil, err
		}
		s.apply(event)
	}
	return s, nil
}

// errRejected wraps the error of an event the stream refused, as opposed to one
// the sink could not persist.
var errRejected = errors.New("event rejected")

// append appends events in order, as one batch. Every event is checked against
// the stream and against the events before it in the batch before any is
// written, so a rejected batch leaves no trace. Each event is then written to
// the sink and applied: the sink is write-ahead, so an event the caller could
// not persist never reaches the stream. A streaming fragment is only checked
// and written to the sink.
func (s *session) append(events ...*Event) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	batch := map[string]struct{}{}
	for _, event := range events {
		if isDelta(event) {
			if err := event.Validate(); err != nil {
				return fmt.Errorf("%w: %w", errRejected, err)
			}
			continue
		}
		if err := s.admit(event); err != nil {
			return fmt.Errorf("%w: %w", errRejected, err)
		}
		// admit only checks these against the stream, not within the batch.
		keys := []string{"event id " + event.ID}
		switch {
		case event.HumanRequest != nil:
			keys = append(keys, "human request id "+event.HumanRequest.ID)
		case event.HumanResponse != nil:
			keys = append(keys, "human response for request "+event.HumanResponse.ID)
		}
		for _, key := range keys {
			if _, dup := batch[key]; dup {
				return fmt.Errorf("%w: duplicate %s in one batch", errRejected, key)
			}
			batch[key] = struct{}{}
		}
	}
	for _, event := range events {
		if s.sink != nil {
			if err := s.sink(s.sinkCtx, event); err != nil {
				return fmt.Errorf("event sink fail %w", err)
			}
		}
		if !isDelta(event) {
			s.apply(event)
		}
	}
	return nil
}

// admit checks that an event is valid and consistent with the stream, without
// changing anything.
func (s *session) admit(event *Event) error {
	if err := event.Validate(); err != nil {
		return fmt.Errorf("event validate fail %w", err)
	}
	if _, exists := s.ids[event.ID]; exists {
		return fmt.Errorf("duplicate event id %q", event.ID)
	}
	switch {
	case event.HumanRequest != nil:
		request := event.HumanRequest
		call, exists := s.toolCalls[request.ToolCall.ID]
		if !exists {
			return fmt.Errorf("human request references unknown tool call %q", request.ToolCall.ID)
		}
		if call.Name != request.ToolCall.Name {
			return fmt.Errorf("human request tool call %q name mismatch", request.ToolCall.ID)
		}
		if _, exists := s.humanRequests[request.ID]; exists {
			return fmt.Errorf("duplicate human request id %q", request.ID)
		}
	case event.HumanResponse != nil:
		response := event.HumanResponse
		request, exists := s.humanRequests[response.ID]
		if !exists {
			return fmt.Errorf("human response references unknown request %q", response.ID)
		}
		if _, resolved := s.toolResults[request.ToolCall.ID]; resolved {
			return fmt.Errorf("human response references resolved request %q", response.ID)
		}
		if _, exists := s.humanResponses[response.ID]; exists {
			return fmt.Errorf("duplicate human response for request %q", response.ID)
		}
	}
	return nil
}

// apply adds an admitted event to the stream and everything derived from it.
func (s *session) apply(event *Event) {
	switch {
	case event.Message != nil:
		for _, part := range event.Message.Parts {
			if part == nil {
				continue
			}
			if part.ToolCall != nil && part.ToolCall.ID != "" {
				s.toolCalls[part.ToolCall.ID] = part.ToolCall
			}
			if part.ToolResult != nil && part.ToolResult.ID != "" {
				s.toolResults[part.ToolResult.ID] = part.ToolResult
				s.resolve(part.ToolResult.ID)
			}
		}
	case event.HumanRequest != nil:
		s.humanRequests[event.HumanRequest.ID] = event.HumanRequest
	case event.HumanResponse != nil:
		s.humanResponses[event.HumanResponse.ID] = event.HumanResponse
		s.unresolved = append(s.unresolved, event.HumanResponse)
	case event.State != nil:
		if !event.State.Patch {
			s.state = map[string]string{}
		}
		for key, value := range event.State.State {
			s.state[key] = value
		}
	}
	s.events = append(s.events, event)
	s.ids[event.ID] = struct{}{}
}

func isDelta(event *Event) bool {
	return event != nil && event.Message != nil && event.Message.Delta
}

func (s *session) setSinkContext(ctx context.Context) {
	s.mu.Lock()
	s.sinkCtx = context.WithoutCancel(ctx)
	s.mu.Unlock()
}

func (s *session) len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.events)
}

// eventsFrom returns the events appended at or after index.
func (s *session) eventsFrom(index int) []*Event {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]*Event(nil), s.events[index:]...)
}

func (s *session) allEvents() []*Event {
	return s.eventsFrom(0)
}

// history returns the model history the stream describes: its messages, in
// order.
func (s *session) history() []*types.Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	var messages []*types.Message
	for _, event := range s.events {
		if event.Message != nil {
			messages = append(messages, event.Message)
		}
	}
	return messages
}

func (s *session) stateSnapshot() map[string]string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return cloneStringMap(s.state)
}

func (s *session) humanRequest(id string) *EventHumanRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.humanRequests[id]
}

func (s *session) resolved(callID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, resolved := s.toolResults[callID]
	return resolved
}

// resolve drops the unresolved response whose tool call just got a result.
func (s *session) resolve(callID string) {
	for index, response := range s.unresolved {
		if s.humanRequests[response.ID].ToolCall.ID == callID {
			s.unresolved = append(s.unresolved[:index:index], s.unresolved[index+1:]...)
			return
		}
	}
}

// unresolvedResponses returns, in order, the human responses whose tool call
// has no result yet.
func (s *session) unresolvedResponses() []*EventHumanResponse {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]*EventHumanResponse(nil), s.unresolved...)
}

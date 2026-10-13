package llmx

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// ErrConflict is returned by Session.Append when an event ID already exists,
// or when expectSeq is set and does not match: another driver appended the
// same outcome, or appended first. The caller reloads and plans again.
var ErrConflict = errors.New("session conflict")

// Session is the event log of one conversation: the single source of truth.
// Implementations decide where it lives (memory, a database, a remote
// service) and must be safe for concurrent use by several drivers.
type Session interface {
	// Load returns every event in order and the seq of the last one (zero
	// when empty).
	Load(ctx context.Context) ([]*Event, int64, error)
	// Append adds events in order as one batch, assigning Seq and At. A
	// negative expectSeq skips the seq check; otherwise the batch is rejected
	// with ErrConflict unless the session currently ends at expectSeq. A
	// batch holding an ID already in the session is rejected as a whole with
	// ErrConflict.
	Append(ctx context.Context, expectSeq int64, events ...*Event) error
	// Tail returns a channel that is closed once an event is appended after
	// this call, so a driver can wait for new events.
	Tail(ctx context.Context) <-chan struct{}
}

// MemorySession keeps the events in memory. It backs Local and tests.
type MemorySession struct {
	mu      sync.Mutex
	events  []*Event
	ids     map[string]struct{}
	changed chan struct{}
}

func NewMemorySession(events ...*Event) *MemorySession {
	s := &MemorySession{ids: map[string]struct{}{}, changed: make(chan struct{})}
	// Restored events without a seq are numbered in order.
	var seq int64
	for _, event := range events {
		clone := event.Clone()
		if clone.Seq <= seq {
			seq++
			clone.Seq = seq
		} else {
			seq = clone.Seq
		}
		s.events = append(s.events, clone)
		s.ids[clone.ID] = struct{}{}
	}
	return s
}

func (s *MemorySession) Load(context.Context) ([]*Event, int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	events := make([]*Event, len(s.events))
	copy(events, s.events)
	return events, s.seqLocked(), nil
}

func (s *MemorySession) seqLocked() int64 {
	if len(s.events) == 0 {
		return 0
	}
	return s.events[len(s.events)-1].Seq
}

func (s *MemorySession) Append(_ context.Context, expectSeq int64, events ...*Event) error {
	if len(events) == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if expectSeq >= 0 && s.seqLocked() != expectSeq {
		return fmt.Errorf("%w: expected seq %d, have %d", ErrConflict, expectSeq, s.seqLocked())
	}
	batch := map[string]struct{}{}
	for _, event := range events {
		if err := event.Validate(); err != nil {
			return err
		}
		if _, exists := s.ids[event.ID]; exists {
			return fmt.Errorf("%w: event %q exists", ErrConflict, event.ID)
		}
		if _, dup := batch[event.ID]; dup {
			return fmt.Errorf("%w: event %q twice in one batch", ErrConflict, event.ID)
		}
		batch[event.ID] = struct{}{}
	}
	seq := s.seqLocked()
	now := time.Now().UnixMilli()
	for _, event := range events {
		seq++
		clone := event.Clone()
		clone.Seq, clone.At = seq, now
		s.events = append(s.events, clone)
		s.ids[clone.ID] = struct{}{}
	}
	close(s.changed)
	s.changed = make(chan struct{})
	return nil
}

func (s *MemorySession) Tail(context.Context) <-chan struct{} {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.changed
}

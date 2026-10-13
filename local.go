package llmx

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// ErrClosed is returned by a closed Local.
var ErrClosed = errors.New("local closed")

// Local drives one in-memory session in this process: the simplest way to
// run an agent, and what the CLI example and tests use. All executors live
// here, so every effect is performed here.
type Local struct {
	session *MemorySession
	agent   *Agent

	cancel context.CancelFunc
	done   chan struct{}
	err    error

	mu     sync.Mutex
	closed bool
}

// NewLocal starts driving a new (or restored) session with the executors.
func NewLocal(agent *Agent, execs []Executor, events ...*Event) (*Local, error) {
	if err := agent.Validate(); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	l := &Local{
		session: NewMemorySession(events...),
		agent:   agent,
		cancel:  cancel,
		done:    make(chan struct{}),
	}
	execs = append(append([]Executor(nil), execs...), EmitExecutor())
	go func() {
		defer close(l.done)
		_, l.err = Drive(ctx, l.session, agent, DriveOptions{}, execs...)
	}()
	return l, nil
}

// Session exposes the session, for watching its events.
func (l *Local) Session() *MemorySession { return l.session }

// Append appends events, such as a tool answer produced outside the
// executors.
func (l *Local) Append(ctx context.Context, events ...*Event) error {
	if l.isClosed() {
		return ErrClosed
	}
	return l.session.Append(ctx, -1, events...)
}

// Send appends an input and returns its event ID.
func (l *Local) Send(ctx context.Context, input *Input) (string, error) {
	if l.isClosed() {
		return "", ErrClosed
	}
	event := NewInputEvent(input)
	if err := l.session.Append(ctx, -1, event); err != nil {
		return "", err
	}
	return event.ID, nil
}

// Enqueue queues a text input.
func (l *Local) Enqueue(ctx context.Context, text string) (string, error) {
	return l.Send(ctx, NewUserInput(InputQueue, text))
}

// Interject sends a text input that joins at the next turn boundary.
func (l *Local) Interject(ctx context.Context, text string) (string, error) {
	return l.Send(ctx, NewUserInput(InputInterject, text))
}

// Interrupt cancels what is running and discards waiting inputs; text, when
// not empty, is then asked.
func (l *Local) Interrupt(ctx context.Context, text string) (string, error) {
	input := &Input{Mode: InputInterrupt}
	if text != "" {
		input.Content = []Content{{Text: text}}
	}
	return l.Send(ctx, input)
}

// Run sends an input and waits until the conversation is idle again. It
// returns the events appended since the input, the input event included.
func (l *Local) Run(ctx context.Context, input *Input) ([]*Event, error) {
	if l.isClosed() {
		return nil, ErrClosed
	}
	_, seq, err := l.session.Load(ctx)
	if err != nil {
		return nil, err
	}
	id, err := l.Send(ctx, input)
	if err != nil {
		return nil, err
	}
	for {
		tail := l.session.Tail(ctx)
		events, _, err := l.session.Load(ctx)
		if err != nil {
			return nil, err
		}
		state := Fold(events)
		admitted := false
		for _, event := range state.Admitted {
			if event.ID == id {
				admitted = true
				break
			}
		}
		if admitted && len(Plan(state, l.agent)) == 0 {
			var since []*Event
			for _, event := range events {
				if event.Seq > seq {
					since = append(since, event)
				}
			}
			return since, nil
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-l.done:
			return nil, fmt.Errorf("%w: %v", ErrClosed, l.err)
		case <-tail:
		}
	}
}

// State folds the current session.
func (l *Local) State(ctx context.Context) (*State, error) {
	events, seq, err := l.session.Load(ctx)
	if err != nil {
		return nil, err
	}
	state := Fold(events)
	state.Seq = seq
	return state, nil
}

// Close stops driving. Events stay in the session.
func (l *Local) Close() error {
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return nil
	}
	l.closed = true
	l.mu.Unlock()
	l.cancel()
	<-l.done
	if errors.Is(l.err, context.Canceled) {
		return nil
	}
	return l.err
}

func (l *Local) isClosed() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.closed
}

package llmx

import (
	"context"
	"fmt"
	"sync"
)

// Runner is the asynchronous handle for a session started with Start.
type Runner struct {
	inbox chan *SessionInput
	done  chan struct{}

	mu     sync.RWMutex
	result *SessionResult
	err    error
}

func newRunner() *Runner {
	return &Runner{
		inbox: make(chan *SessionInput),
		done:  make(chan struct{}),
	}
}

// Send submits one input. It returns once the session accepts ownership of the
// input; processing continues under the context passed to Session.Start.
//
// A session only accepts input while it is waiting, so Send blocks for as long
// as the current run of autonomous turns lasts. Pass a ctx with a deadline to
// bound the wait; a Send the session never accepted fails with that ctx error
// or with ErrSessionClosed, never silently.
func (r *Runner) Send(ctx context.Context, input *SessionInput) error {
	if r == nil {
		return ErrSessionClosed
	}
	if ctx == nil {
		return fmt.Errorf("send context is nil")
	}
	if input.IsEmpty() {
		return fmt.Errorf("send input is empty")
	}
	if err := input.Validate(); err != nil {
		return err
	}
	input = input.Clone()

	select {
	case <-ctx.Done():
		return contextError(ctx)
	case <-r.done:
		return ErrSessionClosed
	case r.inbox <- input:
		return nil
	}
}

func (r *Runner) receive(ctx context.Context) (*SessionInput, error) {
	select {
	case <-ctx.Done():
		return nil, contextError(ctx)
	case input := <-r.inbox:
		if input == nil {
			return nil, fmt.Errorf("received nil session input")
		}
		return input, nil
	}
}

// Wait blocks until the started session finishes and returns its final snapshot.
func (r *Runner) Wait() (*SessionResult, error) {
	if r == nil {
		return nil, ErrSessionClosed
	}
	<-r.done
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.result.Clone(), r.err
}

// Done is closed when the started session finishes.
func (r *Runner) Done() <-chan struct{} {
	if r == nil {
		closed := make(chan struct{})
		close(closed)
		return closed
	}
	return r.done
}

func (r *Runner) finish(result *SessionResult, err error) {
	r.mu.Lock()
	r.result = result.Clone()
	r.err = err
	r.mu.Unlock()
	close(r.done)
}

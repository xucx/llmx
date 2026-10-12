package llmx

import (
	"context"
	"fmt"
	"slices"
	"sync"

	"github.com/google/uuid"
)

// Runner schedules the tasks of an agent started with Start. Inputs are
// submitted without blocking, and each one becomes part of a task:
//
//   - Enqueue queues an input; it runs as its own task once the current task
//     and every task queued before it are done.
//   - Interject ends the current task at its next turn boundary, then runs.
//     Interjections that arrive before that boundary run as one task.
//
// Interjections go before queued inputs. Until a task takes it, an input is
// pending: Withdraw takes one back by ID, and Interrupt takes them all back and
// cancels the current task, so the agent waits for the next input.
//
// A task takes its inputs right before it appends its running status, so a
// caller that refreshes Pending on each running status always sees what still
// waits. Inputs are only checked against the event stream once taken: one the
// stream refuses, such as a response to an unknown request, ends its task as
// rejected and the agent serves the next.
type Runner struct {
	// wake is signaled whenever an input arrives, so a waiting agent notices it.
	wake chan struct{}
	done chan struct{}

	mu         sync.Mutex
	closed     bool
	interjects []*Input
	queue      []*Input
	// cancel cancels the running task; nil while no task runs.
	cancel context.CancelCauseFunc
	result *Result
	err    error
}

func newRunner() *Runner {
	return &Runner{
		wake: make(chan struct{}, 1),
		done: make(chan struct{}),
	}
}

// Enqueue queues an input behind the current task and the queued ones. It
// returns the ID of the input, generated when the input has none.
func (r *Runner) Enqueue(input *Input) (string, error) {
	return r.send(input, func(input *Input) { r.queue = append(r.queue, input) })
}

// Interject ends the current task at its next turn boundary -- once the
// running turn has appended its model reply and tool results -- and runs the
// input as the next task. It never interrupts a model call or tool in flight;
// use Interrupt for that. It returns the ID of the input, generated when the
// input has none.
func (r *Runner) Interject(input *Input) (string, error) {
	return r.send(input, func(input *Input) { r.interjects = append(r.interjects, input) })
}

// Interrupt withdraws every pending input and cancels the current task at
// once, its model call and tools included. Tool calls cut short get an error
// result, so the event stream stays well formed. The agent then waits for the
// next input: to redirect it, Interrupt and then Enqueue the new input. It
// returns the withdrawn inputs in the order Pending reports.
func (r *Runner) Interrupt() []*Input {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	withdrawn := r.pendingLocked()
	r.interjects, r.queue = nil, nil
	if r.cancel != nil {
		r.cancel(errInterrupted)
	}
	return withdrawn
}

// Withdraw takes back the pending input with id. It returns nil when no
// pending input has id, as when a task has already taken it.
func (r *Runner) Withdraw(id string) *Input {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, inputs := range []*[]*Input{&r.interjects, &r.queue} {
		for index, input := range *inputs {
			if input.ID == id {
				*inputs = slices.Delete(*inputs, index, index+1)
				return input
			}
		}
	}
	return nil
}

// Pending returns the inputs no task has taken yet: interjections first, then
// queued inputs, each in arrival order. Once the agent is done, these are the
// inputs it will never process.
func (r *Runner) Pending() []*Input {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	pending := r.pendingLocked()
	for index, input := range pending {
		pending[index] = input.Clone()
	}
	return pending
}

func (r *Runner) pendingLocked() []*Input {
	return slices.Concat(r.interjects, r.queue)
}

// Wait blocks until the agent is done and returns its final snapshot.
func (r *Runner) Wait() (*Result, error) {
	if r == nil {
		return nil, ErrClosed
	}
	<-r.done
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.result.Clone(), r.err
}

// Done is closed when the agent is done.
func (r *Runner) Done() <-chan struct{} {
	if r == nil {
		closed := make(chan struct{})
		close(closed)
		return closed
	}
	return r.done
}

// send validates an input and stores it with store without blocking. It fails
// with ErrClosed once the agent is done, so an input is never silently lost: it
// is either rejected here or visible through Pending.
func (r *Runner) send(input *Input, store func(*Input)) (string, error) {
	if r == nil {
		return "", ErrClosed
	}
	if input.IsEmpty() {
		return "", fmt.Errorf("send input is empty")
	}
	if err := input.Validate(); err != nil {
		return "", err
	}
	input = input.Clone()
	if input.ID == "" {
		input.ID = uuid.NewString()
	}

	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return "", ErrClosed
	}
	if slices.ContainsFunc(r.pendingLocked(), func(pending *Input) bool {
		return pending.ID == input.ID
	}) {
		r.mu.Unlock()
		return "", fmt.Errorf("input %q is already pending", input.ID)
	}
	store(input)
	r.mu.Unlock()

	select {
	case r.wake <- struct{}{}:
	default:
	}
	return input.ID, nil
}

// next blocks until an input arrives, then takes the inputs of the next task
// by priority -- every interjection, else the oldest queued input -- and starts the task in the same step, so no interruption
// can slip between choosing a task and making it cancelable.
func (r *Runner) next(ctx context.Context) ([]*Input, context.Context, error) {
	for {
		r.mu.Lock()
		var inputs []*Input
		switch {
		case len(r.interjects) > 0:
			inputs, r.interjects = r.interjects, nil
		case len(r.queue) > 0:
			inputs, r.queue = r.queue[:1], r.queue[1:]
		}
		if len(inputs) > 0 {
			taskCtx := r.startTaskLocked(ctx)
			r.mu.Unlock()
			return inputs, taskCtx, nil
		}
		r.mu.Unlock()

		select {
		case <-ctx.Done():
			return nil, nil, contextError(ctx)
		case <-r.wake:
		}
	}
}

// startTask starts a task without inputs.
func (r *Runner) startTask(ctx context.Context) context.Context {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.startTaskLocked(ctx)
}

func (r *Runner) startTaskLocked(ctx context.Context) context.Context {
	taskCtx, cancel := context.WithCancelCause(ctx)
	r.cancel = cancel
	return taskCtx
}

func (r *Runner) endTask() {
	r.mu.Lock()
	if r.cancel != nil {
		r.cancel(nil)
		r.cancel = nil
	}
	r.mu.Unlock()
}

// interjected reports whether an interjection waits for a turn boundary.
func (r *Runner) interjected() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.interjects) > 0
}

func (r *Runner) finish(result *Result, err error) {
	r.mu.Lock()
	r.closed = true
	r.result = result.Clone()
	r.err = err
	r.mu.Unlock()
	close(r.done)
}

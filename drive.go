package llmx

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// DriveOptions tunes Drive.
type DriveOptions struct {
	// ExitWhenIdle returns from Drive as soon as no effect can be performed
	// here, instead of waiting for new events. A server driver uses it to
	// free the goroutine; the remaining effects tell it what to wait for.
	ExitWhenIdle bool
	// OnError is called when an executor fails; nil logs nothing. The failure
	// is also recorded in the session as the outcome of the effect.
	OnError func(effect *Effect, err error)
}

// Drive moves a conversation forward: it loads the session, folds the state,
// plans the effects, performs those the executors can, appends their
// outcomes and starts over. When nothing can be performed here it waits for
// new events, or returns when ExitWhenIdle is set. Several drivers may drive
// one session at once: outcomes have deterministic IDs, so the session keeps
// one copy whoever appends first.
//
// While effects run, Drive watches the session: an interrupt input appended
// meanwhile cancels them.
func Drive(ctx context.Context, sess Session, agent *Agent, opts DriveOptions, execs ...Executor) ([]Effect, error) {
	if err := agent.Validate(); err != nil {
		return nil, err
	}
	if sess == nil {
		return nil, fmt.Errorf("session is nil")
	}
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		tail := sess.Tail(ctx)
		events, seq, err := sess.Load(ctx)
		if err != nil {
			return nil, err
		}
		state := Fold(events)
		state.Seq = seq
		effects := Plan(state, agent)

		var mine []Effect
		for _, effect := range effects {
			if pick(execs, &effect) != nil {
				mine = append(mine, effect)
			}
		}
		if len(mine) == 0 {
			if opts.ExitWhenIdle {
				return effects, nil
			}
			select {
			case <-ctx.Done():
				return effects, ctx.Err()
			case <-tail:
			}
			continue
		}

		runCtx, cancel := context.WithCancelCause(ctx)
		stopWatch := watchInterrupt(runCtx, sess, seq, cancel)
		var wg sync.WaitGroup
		for _, effect := range mine {
			wg.Add(1)
			go func(effect Effect) {
				defer wg.Done()
				outcome := perform(runCtx, state, &effect, pick(execs, &effect), opts.OnError)
				if len(outcome) == 0 {
					return
				}
				if err := sess.Append(ctx, -1, outcome...); err != nil && !errors.Is(err, ErrConflict) && opts.OnError != nil {
					opts.OnError(&effect, err)
				}
			}(effect)
		}
		wg.Wait()
		cancel(nil)
		stopWatch()
	}
}

func pick(execs []Executor, effect *Effect) Executor {
	for _, exec := range execs {
		if exec != nil && exec.Can(effect) {
			return exec
		}
	}
	return nil
}

// perform runs one effect and returns its outcome events. A failed executor
// still yields events, so the conversation never waits on something that
// will not happen: a failed tool answers with its error, a failed model call
// stops the round. Events the executor returned alongside its error are kept
// ahead of the failure.
func perform(ctx context.Context, state *State, effect *Effect, exec Executor, onError func(*Effect, error)) (events []*Event) {
	var err error
	func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				err = fmt.Errorf("executor panic: %v", recovered)
			}
		}()
		events, err = exec.Exec(ctx, state, effect)
	}()
	if err == nil {
		return events
	}
	if onError != nil {
		onError(effect, err)
	}
	switch {
	case effect.CallTool != nil:
		if ctx.Err() != nil {
			err = ErrToolInterrupted
		}
		return append(events, NewToolResultEvent(&ToolResult{
			CallID: effect.CallTool.Call.ID,
			Name:   effect.CallTool.Call.Name,
			Error:  err.Error(),
		}))
	case effect.CallModel != nil:
		if ctx.Err() != nil {
			// Interrupted with nothing received: the interrupt input is
			// planned next and stops the round itself.
			return events
		}
		return append(events, NewStopEvent("stop:error:"+effect.ID, StopError, err.Error()))
	}
	return events
}

// watchInterrupt cancels runCtx with ErrToolInterrupted once an interrupt
// input appears in the session after seq. It returns a function that stops
// watching.
func watchInterrupt(runCtx context.Context, sess Session, seq int64, cancel context.CancelCauseFunc) func() {
	done := make(chan struct{})
	go func() {
		for {
			tail := sess.Tail(runCtx)
			select {
			case <-done:
				return
			case <-runCtx.Done():
				return
			case <-tail:
			}
			events, _, err := sess.Load(runCtx)
			if err != nil {
				continue
			}
			for _, event := range events {
				if event.Seq > seq && event.Input != nil && event.Input.Mode == InputInterrupt {
					cancel(ErrToolInterrupted)
					return
				}
			}
		}
	}()
	return func() { close(done) }
}

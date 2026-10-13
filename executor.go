package llmx

import "context"

// Executor performs effects. Where the executors live decides where effects
// happen: a server process may hold the model and its own tools, a client
// process the tools that touch its data and the human who answers questions.
// Every executor sees every effect and claims the ones it can perform.
type Executor interface {
	// Can reports whether this executor performs the effect.
	Can(*Effect) bool
	// Exec performs the effect and returns the events recording its outcome.
	// It must be safe for concurrent use: the effects of one plan run
	// concurrently. ctx is canceled when the conversation is interrupted.
	Exec(ctx context.Context, s *State, e *Effect) ([]*Event, error)
}

// EmitExecutor performs Emit effects by returning their events. Every driver
// should hold one.
func EmitExecutor() Executor { return emitExecutor{} }

type emitExecutor struct{}

func (emitExecutor) Can(e *Effect) bool { return e != nil && e.Emit != nil }

func (emitExecutor) Exec(_ context.Context, _ *State, e *Effect) ([]*Event, error) {
	return e.Emit.Events, nil
}

// ExecutorFunc adapts a pair of functions to an Executor.
type ExecutorFunc struct {
	CanFunc  func(*Effect) bool
	ExecFunc func(context.Context, *State, *Effect) ([]*Event, error)
}

func (f ExecutorFunc) Can(e *Effect) bool { return f.CanFunc != nil && f.CanFunc(e) }

func (f ExecutorFunc) Exec(ctx context.Context, s *State, e *Effect) ([]*Event, error) {
	return f.ExecFunc(ctx, s, e)
}

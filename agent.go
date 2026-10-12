package llmx

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"sync"

	"github.com/xucx/llmx/types"
)

// EventSink receives every event appended to an agent event stream, before
// the event joins the stream: an error rejects the event. It also receives the
// streaming fragments (delta messages) of a model reply, which never join the
// stream. Events are immutable; implementations must clone an Event before
// modifying it. The sink runs while the stream is locked, so it must not append
// to the agent itself.
type EventSink func(context.Context, *Event) error

type agentOptions struct {
	Model        string
	Instruction  string
	Loop         Loop
	ReasonLevel  int
	Tools        []*Tool
	MaxTurn      int
	Stream       bool
	EventSink    EventSink
	Interceptors []Interceptor
}

func defaultAgentOptions() *agentOptions {
	return &agentOptions{Loop: ReactLoop}
}

func (o *agentOptions) clone() *agentOptions {
	if o == nil {
		return defaultAgentOptions()
	}
	clone := *o
	clone.Tools = append([]*Tool(nil), o.Tools...)
	clone.Interceptors = append([]Interceptor(nil), o.Interceptors...)
	return &clone
}

func (o *agentOptions) validate() error {
	if o == nil {
		return fmt.Errorf("agent options are nil")
	}
	if o.Loop == nil {
		return fmt.Errorf("agent loop is nil")
	}
	if o.Model == "" {
		return fmt.Errorf("agent model is empty")
	}
	if o.MaxTurn < 0 {
		return fmt.Errorf("agent max turn must not be negative")
	}
	if o.ReasonLevel < 0 {
		return fmt.Errorf("agent reason level must not be negative")
	}
	toolNames := make(map[string]struct{}, len(o.Tools))
	for index, tool := range o.Tools {
		if tool == nil {
			return fmt.Errorf("agent tool %d is nil", index)
		}
		if tool.Name == "" {
			return fmt.Errorf("agent tool %d name is empty", index)
		}
		if tool.Invoker == nil {
			return fmt.Errorf("agent tool %q invoker is nil", tool.Name)
		}
		if _, exists := toolNames[tool.Name]; exists {
			return fmt.Errorf("duplicate agent tool %q", tool.Name)
		}
		toolNames[tool.Name] = struct{}{}
	}
	for index, interceptor := range o.Interceptors {
		if interceptor == nil {
			return fmt.Errorf("interceptor %d is nil", index)
		}
	}
	return nil
}

func applyAgentOptions(options *agentOptions, opts ...AgentOption) *agentOptions {
	current := options.clone()
	for _, opt := range opts {
		if opt != nil {
			opt(current)
		}
	}
	return current
}

type AgentOption func(*agentOptions)

func WithModel(model string) AgentOption {
	return func(opts *agentOptions) {
		opts.Model = model
	}
}

func WithInstruction(instruction string) AgentOption {
	return func(opts *agentOptions) {
		opts.Instruction = instruction
	}
}

func WithLoop(loop Loop) AgentOption {
	return func(opts *agentOptions) {
		opts.Loop = loop
	}
}

func WithReasonLevel(level int) AgentOption {
	return func(opts *agentOptions) {
		opts.ReasonLevel = level
	}
}

// WithTools replaces all tools set by earlier options.
func WithTools(tools ...*Tool) AgentOption {
	return func(opts *agentOptions) {
		opts.Tools = append([]*Tool(nil), tools...)
	}
}

// WithAdditionalTools appends tools to those set by earlier options.
func WithAdditionalTools(tools ...*Tool) AgentOption {
	return func(opts *agentOptions) {
		opts.Tools = append(opts.Tools, tools...)
	}
}

func WithMaxTurn(maxTurn int) AgentOption {
	return func(opts *agentOptions) {
		opts.MaxTurn = maxTurn
	}
}

func WithStream(stream bool) AgentOption {
	return func(opts *agentOptions) {
		opts.Stream = stream
	}
}

func WithEventSink(sink EventSink) AgentOption {
	return func(opts *agentOptions) {
		opts.EventSink = sink
	}
}

// WithInterceptor appends interceptors to those set by earlier options.
func WithInterceptor(interceptors ...Interceptor) AgentOption {
	return func(opts *agentOptions) {
		opts.Interceptors = append(opts.Interceptors, interceptors...)
	}
}

var (
	ErrMaxTurnExceeded = errors.New("max turn exceeded")
	ErrClosed          = errors.New("agent closed")
	ErrStarted         = errors.New("agent already started")
	ErrBusy            = errors.New("agent busy")
	// ErrInputRejected wraps the error of an input the event stream refused,
	// such as a human response to an unknown or already answered request. The
	// task ends as rejected without appending the input, and the agent stays
	// usable.
	ErrInputRejected = errors.New("input rejected")
)

// Result is the final snapshot of one Run, or of the lifetime of an agent
// started with Start.
type Result struct {
	// Events are the events appended during the execution.
	Events []*Event
	State  map[string]string

	Turns      int
	StopReason StopReason
	Cause      error
}

// Reply returns the last model reply among the events, or nil when the
// execution produced none.
func (r *Result) Reply() *types.Message {
	if r == nil {
		return nil
	}
	for index := len(r.Events) - 1; index >= 0; index-- {
		if message := r.Events[index].Message; message != nil &&
			message.Role == types.MessageRoleAssistant {
			return message
		}
	}
	return nil
}

func (r *Result) Clone() *Result {
	if r == nil {
		return nil
	}
	clone := *r
	clone.Events = append([]*Event(nil), r.Events...)
	clone.State = cloneStringMap(r.State)
	return &clone
}

type lifecycle uint8

const (
	idle lifecycle = iota
	running
	serving
	closed
)

// Agent runs tasks over one event stream. Every input becomes a task: its
// events are appended to the stream, then model turns run while Loop
// continues. Run executes one task synchronously; Start serves tasks from a
// Runner that queues, interjects, withdraws and interrupts them. Interrupting a
// task or rejecting its inputs never closes the agent; any other error does,
// and a new agent is rebuilt from the sinked events instead.
type Agent struct {
	Name        string
	Description string

	models  *Models
	options *agentOptions
	session *session
	// tools declares the options' tools to the model, built once.
	tools []*types.Tool

	// The stage handlers wrapped by the interceptors, built once.
	turnHandler     TurnHandler
	llmHandler      LLMHandler
	toolCallHandler ToolCallHandler

	mu        sync.Mutex
	lifecycle lifecycle

	// The execution state, only touched by the executing goroutine. runner is
	// nil for Run.
	runner *Runner
	from   int
	result Result
}

// NewAgent creates an agent over an event stream restored from events, which
// is nil for a new conversation. Restored events and their nested values are
// retained by reference and never written to the event sink again.
//
// The event sink is the way to keep the stream: an agent closed by an error is
// restored by passing every event its sink received to a new agent.
func NewAgent(
	name string,
	description string,
	models *Models,
	events []*Event,
	opts ...AgentOption,
) (*Agent, error) {
	if models == nil {
		return nil, fmt.Errorf("agent models is nil")
	}
	options := applyAgentOptions(defaultAgentOptions(), opts...)
	if err := options.validate(); err != nil {
		return nil, err
	}
	session, err := newSession(events, options.EventSink)
	if err != nil {
		return nil, err
	}
	agent := &Agent{
		Name:        name,
		Description: description,
		models:      models,
		options:     options,
		session:     session,
	}
	for _, tool := range options.Tools {
		agent.tools = append(agent.tools, types.NewFunctionTool(
			tool.Name,
			tool.Description,
			tool.Parameters,
		))
	}
	if err := agent.buildHandlers(); err != nil {
		return nil, err
	}
	return agent, nil
}

// buildHandlers wraps the agent stages with its interceptors once, so every
// turn and tool call reuses the same chains.
func (a *Agent) buildHandlers() error {
	a.turnHandler = a.runTurn
	a.llmHandler = a.callLLM
	a.toolCallHandler = a.callTool
	interceptors := a.options.Interceptors
	for index := len(interceptors) - 1; index >= 0; index-- {
		interceptor := interceptors[index]
		a.turnHandler = interceptor.InterceptTurn(a.turnHandler)
		a.llmHandler = interceptor.InterceptLLM(a.llmHandler)
		a.toolCallHandler = interceptor.InterceptToolCall(a.toolCallHandler)
		if a.turnHandler == nil || a.llmHandler == nil || a.toolCallHandler == nil {
			return fmt.Errorf("interceptor %d returned a nil handler", index)
		}
	}
	return nil
}

// Run runs one task for input and returns when it is done. The agent can Run
// again afterwards unless the task closed it. Canceling ctx interrupts the
// task like Runner.Interrupt: it ends as interrupted, Run returns the context
// error and the agent stays usable. An input the stream refuses ends the task
// as rejected: Run returns ErrInputRejected and the agent stays usable. Any
// other error closes the agent.
func (a *Agent) Run(ctx context.Context, input *Input) (*Result, error) {
	if ctx == nil {
		return nil, fmt.Errorf("run context is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := input.Validate(); err != nil {
		return nil, err
	}
	input = input.Clone()
	if err := a.acquire(running); err != nil {
		return nil, err
	}
	if input.IsEmpty() && len(a.session.unresolvedResponses()) == 0 {
		a.release(false)
		return nil, fmt.Errorf("run input is empty")
	}

	a.begin(ctx, nil)
	var inputs []*Input
	if !input.IsEmpty() {
		inputs = []*Input{input}
	}
	// The task outlives ctx on purpose: canceling ctx interrupts it, so it can
	// still close its turn and append its closing status.
	taskCtx, cancel := context.WithCancelCause(context.WithoutCancel(ctx))
	defer cancel(nil)
	defer context.AfterFunc(ctx, func() { cancel(errInterrupted) })()

	end, err := a.runTask(taskCtx, inputs)
	result, err := a.finalize(ctx, end, err)
	a.release(end == nil || end.Status == StatusStopped)
	return result, err
}

// Start serves tasks until ctx is canceled or a task closes the agent. The
// returned Runner feeds it: queued inputs run one task each after the current
// one, interjections end the current task at its next turn boundary, and an
// interruption withdraws the pending inputs and cancels the task at once.
func (a *Agent) Start(ctx context.Context) (*Runner, error) {
	if ctx == nil {
		return nil, fmt.Errorf("start context is nil")
	}
	if err := a.acquire(serving); err != nil {
		return nil, err
	}
	serveCtx, cancel := context.WithCancelCause(ctx)
	runner := newRunner()
	a.begin(serveCtx, runner)
	go func() {
		result, err := a.finalize(serveCtx, nil, a.serve(serveCtx))
		cancel(err)
		a.release(true)
		runner.finish(result, err)
	}()
	return runner, nil
}

func (a *Agent) serve(ctx context.Context) error {
	// A restored stream may hold answers whose tool never resumed: they are
	// the first task, which takes no input.
	resume := len(a.session.unresolvedResponses()) > 0
	for {
		var (
			inputs  []*Input
			taskCtx context.Context
		)
		if resume {
			resume = false
			taskCtx = a.runner.startTask(ctx)
		} else {
			var err error
			if inputs, taskCtx, err = a.runner.next(ctx); err != nil {
				return err
			}
		}
		end, err := a.runTask(taskCtx, inputs)
		a.runner.endTask()
		// A task that ended with an error but left the agent usable, such as
		// a rejected one, is over: serving goes on.
		if end == nil || end.Status == StatusStopped {
			return err
		}
	}
}

func (a *Agent) acquire(next lifecycle) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	switch a.lifecycle {
	case idle:
		a.lifecycle = next
		return nil
	case running:
		return ErrBusy
	case serving:
		if next == serving {
			return ErrStarted
		}
		return ErrBusy
	default:
		return ErrClosed
	}
}

func (a *Agent) release(close bool) {
	a.mu.Lock()
	if close {
		a.lifecycle = closed
	} else {
		a.lifecycle = idle
	}
	a.mu.Unlock()
}

func (a *Agent) begin(ctx context.Context, runner *Runner) {
	a.session.setSinkContext(ctx)
	a.runner = runner
	a.from = a.session.len()
	a.result = Result{}
}

// finalize builds the result of an execution from the closing status of its
// last task, which is nil for Start, and the error that ended it.
func (a *Agent) finalize(ctx context.Context, end *EventStatus, err error) (*Result, error) {
	result := a.result
	result.Events = a.session.eventsFrom(a.from)
	result.State = a.session.stateSnapshot()

	switch {
	case err == nil && end != nil && end.StopReason == StopInterrupted:
		result.StopReason = StopInterrupted
		result.Cause = contextError(ctx)
		err = result.Cause
	case err == nil:
		result.StopReason = StopDone
	case errors.Is(err, ErrMaxTurnExceeded):
		result.StopReason = StopMaxTurn
		result.Cause = err
	case errors.Is(err, ErrInputRejected):
		result.StopReason = StopRejected
		result.Cause = err
	case ctx.Err() != nil:
		result.StopReason = StopCanceled
		result.Cause = contextError(ctx)
		err = result.Cause
	case errors.Is(err, context.Canceled),
		errors.Is(err, context.DeadlineExceeded):
		result.StopReason = StopCanceled
		result.Cause = err
	default:
		result.StopReason = StopError
		result.Cause = err
	}
	return result.Clone(), err
}

func contextError(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("context is nil")
	}
	select {
	case <-ctx.Done():
		if cause := context.Cause(ctx); cause != nil {
			return cause
		}
		return ctx.Err()
	default:
		return nil
	}
}

func cloneStringMap(source map[string]string) map[string]string {
	if source == nil {
		return map[string]string{}
	}
	return maps.Clone(source)
}

func validateState(state map[string]string) error {
	for key := range state {
		if key == "" {
			return fmt.Errorf("state key is empty")
		}
	}
	return nil
}

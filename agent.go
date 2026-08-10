package llmx

import (
	"context"
	"fmt"

	"github.com/xucx/llmx/types"
)

type LLMHandler func(context.Context, *SessionTurn) (*types.Completion, error)
type LLMInterceptor func(LLMHandler) LLMHandler
type ToolCallHandler func(ToolContext, *types.MessageToolCall) (any, error)
type ToolCallInterceptor func(ToolCallHandler) ToolCallHandler

// EventSink receives the immutable Event instance committed by a Session.
// Implementations must clone an Event before modifying it.
type EventSink func(context.Context, *Event) error

type AgentOptions struct {
	Model                string
	Instruction          string
	Loop                 Loop
	ReasonLevel          int
	Tools                []*Tool
	MaxTurn              int
	Stream               bool
	Compress             *CompressOptions
	EventSink            EventSink
	LLMInterceptors      []LLMInterceptor
	ToolCallInterceptors []ToolCallInterceptor
}

func DefaultAgentOptions() *AgentOptions {
	return &AgentOptions{Loop: ReactLoop}
}

func (o *AgentOptions) Clone() *AgentOptions {
	if o == nil {
		return DefaultAgentOptions()
	}
	clone := *o
	clone.Tools = append([]*Tool(nil), o.Tools...)
	clone.LLMInterceptors = append([]LLMInterceptor(nil), o.LLMInterceptors...)
	clone.ToolCallInterceptors = append(
		[]ToolCallInterceptor(nil),
		o.ToolCallInterceptors...,
	)
	if o.Compress != nil {
		compress := *o.Compress
		clone.Compress = &compress
	}
	return &clone
}

func (o *AgentOptions) normalized() *AgentOptions {
	options := o.Clone()
	if options.Loop == nil {
		options.Loop = ReactLoop
	}
	if options.Compress != nil {
		options.Compress = options.Compress.normalized()
	}
	return options
}

func (o *AgentOptions) validate(requireModel bool) error {
	if o == nil {
		return fmt.Errorf("agent options are nil")
	}
	if o.Loop == nil {
		return fmt.Errorf("agent loop is nil")
	}
	if requireModel && o.Model == "" {
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
		if _, exists := toolNames[tool.Name]; exists {
			return fmt.Errorf("duplicate agent tool %q", tool.Name)
		}
		toolNames[tool.Name] = struct{}{}
	}
	for index, interceptor := range o.LLMInterceptors {
		if interceptor == nil {
			return fmt.Errorf("llm interceptor %d is nil", index)
		}
	}
	for index, interceptor := range o.ToolCallInterceptors {
		if interceptor == nil {
			return fmt.Errorf("tool call interceptor %d is nil", index)
		}
	}
	return nil
}

func ApplyAgentOptions(options *AgentOptions, opts ...AgentOption) *AgentOptions {
	current := options.Clone()
	for _, opt := range opts {
		if opt != nil {
			opt(current)
		}
	}
	return current.normalized()
}

type AgentOption func(*AgentOptions)

func WithModel(model string) AgentOption {
	return func(opts *AgentOptions) {
		opts.Model = model
	}
}

func WithInstruction(instruction string) AgentOption {
	return func(opts *AgentOptions) {
		opts.Instruction = instruction
	}
}

func WithLoop(loop Loop) AgentOption {
	return func(opts *AgentOptions) {
		opts.Loop = loop
	}
}

func WithReasonLevel(level int) AgentOption {
	return func(opts *AgentOptions) {
		opts.ReasonLevel = level
	}
}

// WithTools replaces all tools inherited from the Agent.
func WithTools(tools ...*Tool) AgentOption {
	return func(opts *AgentOptions) {
		opts.Tools = append([]*Tool(nil), tools...)
	}
}

// WithAdditionalTools appends tools to those inherited from the Agent.
func WithAdditionalTools(tools ...*Tool) AgentOption {
	return func(opts *AgentOptions) {
		opts.Tools = append(opts.Tools, tools...)
	}
}

func WithMaxTurn(maxTurn int) AgentOption {
	return func(opts *AgentOptions) {
		opts.MaxTurn = maxTurn
	}
}

func WithStream(stream bool) AgentOption {
	return func(opts *AgentOptions) {
		opts.Stream = stream
	}
}

func WithCompress(compress *CompressOptions) AgentOption {
	return func(opts *AgentOptions) {
		if compress == nil {
			opts.Compress = nil
			return
		}
		copy := *compress
		opts.Compress = &copy
	}
}

func WithEventSink(sink EventSink) AgentOption {
	return func(opts *AgentOptions) {
		opts.EventSink = sink
	}
}

func WithLLMInterceptor(interceptors ...LLMInterceptor) AgentOption {
	return func(opts *AgentOptions) {
		opts.LLMInterceptors = append(opts.LLMInterceptors, interceptors...)
	}
}

func WithToolCallInterceptor(interceptors ...ToolCallInterceptor) AgentOption {
	return func(opts *AgentOptions) {
		opts.ToolCallInterceptors = append(
			opts.ToolCallInterceptors,
			interceptors...,
		)
	}
}

type Agent struct {
	Name        string
	Description string

	models  *Models
	options *AgentOptions
}

func NewAgent(
	name string,
	description string,
	models *Models,
	opts ...AgentOption,
) (*Agent, error) {
	if name == "" {
		return nil, fmt.Errorf("agent name is empty")
	}
	if models == nil {
		return nil, fmt.Errorf("agent models is nil")
	}
	options := ApplyAgentOptions(DefaultAgentOptions(), opts...)
	if err := options.validate(false); err != nil {
		return nil, err
	}
	return &Agent{
		Name:        name,
		Description: description,
		models:      models,
		options:     options,
	}, nil
}

func (a *Agent) optionsForSession(opts ...AgentOption) (*AgentOptions, error) {
	if a == nil {
		return nil, fmt.Errorf("agent is nil")
	}
	options := ApplyAgentOptions(a.options.Clone(), opts...)
	if err := options.validate(true); err != nil {
		return nil, err
	}
	return options, nil
}

// NewSession restores one reusable session without starting an execution.
func (a *Agent) NewSession(
	history []*Event,
	opts ...AgentOption,
) (*Session, error) {
	return NewSession(a, history, opts...)
}

// Start creates a session in long-running mode. The returned runner initially
// waits for input and remains alive until ctx is canceled, LoopStop is returned,
// or an execution error occurs.
func (a *Agent) Start(
	ctx context.Context,
	history []*Event,
	opts ...AgentOption,
) (*Runner, error) {
	session, err := NewSession(a, history, opts...)
	if err != nil {
		return nil, err
	}
	return session.Start(ctx)
}

// Run creates a session and processes one fixed input synchronously.
func (a *Agent) Run(
	ctx context.Context,
	history []*Event,
	input *SessionInput,
	opts ...AgentOption,
) (*SessionResult, error) {
	session, err := NewSession(a, history, opts...)
	if err != nil {
		return nil, err
	}
	return session.Run(ctx, input)
}

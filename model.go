package llmx

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/google/uuid"
	"github.com/xucx/llmx/types"
)

// Delta is a snapshot of a reply being generated: Text and Reasoning are the
// whole text received so far. Deltas are not events: they are shown as they
// arrive, and the Reply event replaces them.
type Delta struct {
	// ReplyID is the ID the Reply event will have.
	ReplyID string
	// Attempt identifies one execution of the model call: a reply redone
	// after a failure streams again under another attempt.
	Attempt   string
	Text      string
	Reasoning string
}

// DeltaSink receives streaming fragments.
type DeltaSink func(context.Context, *Delta)

// ModelRequest is the model call a ModelExecutor makes.
type ModelRequest struct {
	Model    string
	Messages []*types.Message
	Options  *types.CompletionOptions
}

func (r ModelRequest) clone() ModelRequest {
	r.Messages = append([]*types.Message(nil), r.Messages...)
	r.Options = r.Options.Clone()
	return r
}

// ModelContext is the context of one model call, seen by interceptors.
type ModelContext interface {
	context.Context
	// Events is the conversation so far.
	Events() []*Event
	// State is the key-value state of the conversation.
	State() map[string]string
	// Request returns a copy of the request; set it back to change it.
	Request() ModelRequest
	SetRequest(ModelRequest)
	// Append records events alongside the reply, ahead of it: a state patch
	// an interceptor wants replayed next time, for instance.
	Append(events ...*Event) error
}

// ModelHandler performs the model call of its context. When interrupted, it
// may return the completion received so far along with the error.
type ModelHandler func(ModelContext) (*types.Completion, error)

// Interceptor wraps model calls; the first registered is the outermost.
type Interceptor interface {
	InterceptModel(ModelHandler) ModelHandler
}

// InterceptorFunc adapts a function to an Interceptor.
type InterceptorFunc func(ModelHandler) ModelHandler

func (f InterceptorFunc) InterceptModel(next ModelHandler) ModelHandler { return f(next) }

type modelOptions struct {
	stream       bool
	sink         DeltaSink
	interceptors []Interceptor
}

type ModelOption func(*modelOptions)

// WithDeltaSink streams fragments to sink.
func WithDeltaSink(sink DeltaSink) ModelOption {
	return func(o *modelOptions) { o.sink, o.stream = sink, sink != nil }
}

// WithInterceptors wraps the model call.
func WithInterceptors(interceptors ...Interceptor) ModelOption {
	return func(o *modelOptions) { o.interceptors = append(o.interceptors, interceptors...) }
}

// ModelExecutor performs CallModel effects with models.
func ModelExecutor(models *Models, opts ...ModelOption) Executor {
	options := &modelOptions{}
	for _, opt := range opts {
		opt(options)
	}
	x := &modelExecutor{models: models, options: options}
	x.handler = x.call
	for index := len(options.interceptors) - 1; index >= 0; index-- {
		x.handler = options.interceptors[index].InterceptModel(x.handler)
	}
	return x
}

type modelExecutor struct {
	models  *Models
	options *modelOptions
	handler ModelHandler
}

func (x *modelExecutor) Can(e *Effect) bool { return e != nil && e.CallModel != nil }

func (x *modelExecutor) Exec(ctx context.Context, s *State, e *Effect) ([]*Event, error) {
	call := e.CallModel
	level := call.ReasonLevel
	options := &types.CompletionOptions{
		Instructions:   call.Instruction,
		ReasoningLevel: &level,
		Tools:          ToolSpecsToTypes(call.Tools),
	}
	attempt := uuid.NewString()
	if x.options.stream {
		// 累积回调：Delta 携带至今全文，接收方按自己已展示的长度取增量。
		options.StreamingAccFunc = func(ctx context.Context, completion *types.Completion) error {
			if completion == nil || completion.Message == nil {
				return nil
			}
			x.options.sink(ctx, &Delta{
				ReplyID:   e.ID,
				Attempt:   attempt,
				Text:      completion.Message.Text(),
				Reasoning: completion.Message.Reasoning(),
			})
			return nil
		}
	}
	messages := call.Messages
	if !x.acceptsImages(call.Model) {
		messages = withoutImages(messages)
	}
	mctx := &modelContext{
		Context: ctx,
		state:   s,
		request: ModelRequest{Model: call.Model, Messages: messages, Options: options},
	}
	completion, err := x.handler(mctx)
	// Whatever happens, what interceptors appended (a compression, say) is
	// kept: it was decided on this conversation and replays next time.
	if err != nil {
		// An interrupted streaming call returns what it received so far:
		// keep it, so the conversation shows what the model already said.
		if ctx.Err() != nil || errors.Is(err, context.Canceled) {
			if reply := partialReply(call, completion); reply != nil {
				return append(mctx.appended, NewReplyEvent(reply)), nil
			}
		}
		return mctx.appended, err
	}
	if completion == nil || completion.Message == nil {
		return mctx.appended, fmt.Errorf("model returned no message")
	}
	reply := &Reply{
		BaseSeq:   call.BaseSeq,
		Model:     call.Model,
		Text:      completion.Message.Text(),
		Reasoning: completion.Message.Reasoning(),
	}
	for _, part := range completion.Message.ToolCalls() {
		if part == nil {
			continue
		}
		reply.ToolCalls = append(reply.ToolCalls, ToolCall{ID: part.ID, Name: part.Name, Arguments: part.Arguments})
	}
	events := append(mctx.appended, NewReplyEvent(reply), NewUsageEvent(call.BaseSeq, call.Model, completion.Usage))
	return events, nil
}

// partialReply keeps the text and reasoning of a completion cut short. Its
// tool calls are dropped, since their arguments may be truncated.
func partialReply(call *EffectCallModel, completion *types.Completion) *Reply {
	if completion == nil || completion.Message == nil {
		return nil
	}
	reply := &Reply{
		BaseSeq:   call.BaseSeq,
		Model:     call.Model,
		Text:      completion.Message.Text(),
		Reasoning: completion.Message.Reasoning(),
		Partial:   true,
	}
	if reply.Text == "" && reply.Reasoning == "" {
		return nil
	}
	return reply
}

func (x *modelExecutor) call(ctx ModelContext) (*types.Completion, error) {
	request := ctx.Request()
	return x.models.Completion(ctx, request.Model, request.Messages, types.CompletionWithOptions(request.Options))
}

type modelContext struct {
	context.Context
	state *State

	mu       sync.Mutex
	request  ModelRequest
	appended []*Event
}

func (c *modelContext) Events() []*Event         { return c.state.Events }
func (c *modelContext) State() map[string]string { return c.state.KV }

func (c *modelContext) Request() ModelRequest {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.request.clone()
}

func (c *modelContext) SetRequest(request ModelRequest) {
	request = request.clone()
	c.mu.Lock()
	c.request = request
	c.mu.Unlock()
}

func (c *modelContext) Append(events ...*Event) error {
	for _, event := range events {
		if err := event.Validate(); err != nil {
			return err
		}
	}
	c.mu.Lock()
	c.appended = append(c.appended, events...)
	c.mu.Unlock()
	return nil
}

// acceptsImages reports whether the model takes image input; unknown models
// are assumed not to, so images never break a text-only model.
func (x *modelExecutor) acceptsImages(name string) bool {
	if x.models == nil {
		return true
	}
	model, err := x.models.GetModel(name)
	return err == nil && model.Vision
}

// ImageOmittedNote replaces each image part for a model that cannot see images.
const ImageOmittedNote = "[image omitted: this model cannot view images]"

// withoutImages returns the messages with image parts replaced by a note.
// Messages without images are shared, not copied.
func withoutImages(messages []*types.Message) []*types.Message {
	out := make([]*types.Message, 0, len(messages))
	for _, message := range messages {
		if message == nil || !hasImage(message) {
			out = append(out, message)
			continue
		}
		clone := *message
		clone.Parts = make([]*types.MessagePart, 0, len(message.Parts))
		for _, part := range message.Parts {
			if part != nil && part.ImageURL != nil {
				clone.Parts = append(clone.Parts, &types.MessagePart{Text: &types.MessageText{Text: ImageOmittedNote}})
				continue
			}
			clone.Parts = append(clone.Parts, part)
		}
		out = append(out, &clone)
	}
	return out
}

func hasImage(message *types.Message) bool {
	for _, part := range message.Parts {
		if part != nil && part.ImageURL != nil {
			return true
		}
	}
	return false
}

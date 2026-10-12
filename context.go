package llmx

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/xucx/llmx/types"
)

// errTurnOver rejects appends through the context of a turn that has ended,
// such as those of a tool still running after its turn was interrupted.
var errTurnOver = errors.New("turn is over")

// The model call and the tool calls of a turn both belong to the turn, so
// LLMContext and ToolContext both extend TurnContext. Every method of every
// context is safe for concurrent use.

// TurnContext is the context of one model turn. It reads and appends to the
// agent event stream; appended events are part of the stream at once. It is
// canceled when the task running the turn is interrupted.
type TurnContext interface {
	context.Context
	// Events returns a snapshot of the whole event stream.
	Events() []*Event
	// State returns a snapshot of the state the event stream describes.
	State() map[string]string
	// Append appends events to the stream. Status and human response events
	// belong to the agent and are rejected.
	Append(events ...*Event) error
}

// LLMRequest is the model call a turn makes.
type LLMRequest struct {
	Model string
	// Messages are derived from the event stream. The messages themselves are
	// immutable; replace them to change what the model sees.
	Messages []*types.Message
	Options  *types.CompletionOptions
}

func (r LLMRequest) clone() LLMRequest {
	r.Messages = append([]*types.Message(nil), r.Messages...)
	r.Options = r.Options.Clone()
	return r
}

// LLMContext is the context of the model call of a turn.
type LLMContext interface {
	TurnContext
	// Request returns a copy of the request the model call sends. Changing the
	// copy changes nothing until it is set back with SetRequest.
	Request() LLMRequest
	// SetRequest replaces the request the model call sends.
	SetRequest(LLMRequest)
}

// ToolContext is the context of one tool call requested by the model call.
type ToolContext interface {
	TurnContext
	// ToolCall returns a copy of the call being run.
	ToolCall() types.MessageToolCall
	// SetToolCall replaces the call being run, such as to repair its arguments
	// or to route it to another tool. Its result still answers the call the
	// model made.
	SetToolCall(types.MessageToolCall)
	// HumanRequest and HumanResponse are set while a tool resumes an answered
	// human request. They are immutable; Clone before modifying them.
	HumanRequest() *EventHumanRequest
	HumanResponse() *EventHumanResponse
}

type turnContext struct {
	context.Context

	session *session

	mu     sync.Mutex
	closed bool
}

func newTurnContext(ctx context.Context, s *session) *turnContext {
	return &turnContext{Context: ctx, session: s}
}

func (c *turnContext) Events() []*Event { return c.session.allEvents() }

func (c *turnContext) State() map[string]string { return c.session.stateSnapshot() }

func (c *turnContext) Append(events ...*Event) error {
	for _, event := range events {
		if event != nil && (event.Status != nil || event.HumanResponse != nil) {
			return fmt.Errorf("turn cannot append agent event %q", event.ID)
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return errTurnOver
	}
	return c.session.append(events...)
}

// close ends the turn: later appends through its contexts are rejected.
func (c *turnContext) close() {
	c.mu.Lock()
	c.closed = true
	c.mu.Unlock()
}

type llmContext struct {
	TurnContext

	mu      sync.Mutex
	request LLMRequest
}

func newLLMContext(ctx TurnContext, request LLMRequest) *llmContext {
	return &llmContext{TurnContext: ctx, request: request.clone()}
}

func (c *llmContext) Request() LLMRequest {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.request.clone()
}

func (c *llmContext) SetRequest(request LLMRequest) {
	request = request.clone()
	c.mu.Lock()
	c.request = request
	c.mu.Unlock()
}

type toolContext struct {
	TurnContext

	// requested is the call the model made, which the result answers.
	requested *types.MessageToolCall
	request   *EventHumanRequest
	response  *EventHumanResponse

	mu   sync.Mutex
	call types.MessageToolCall
}

func newToolContext(
	ctx TurnContext,
	call *types.MessageToolCall,
	request *EventHumanRequest,
	response *EventHumanResponse,
) *toolContext {
	return &toolContext{
		TurnContext: ctx,
		requested:   call,
		request:     request,
		response:    response,
		call:        *call,
	}
}

func (c *toolContext) ToolCall() types.MessageToolCall {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.call
}

func (c *toolContext) SetToolCall(call types.MessageToolCall) {
	c.mu.Lock()
	c.call = call
	c.mu.Unlock()
}

func (c *toolContext) HumanRequest() *EventHumanRequest { return c.request }

func (c *toolContext) HumanResponse() *EventHumanResponse { return c.response }

package llmx

import (
	"github.com/xucx/llmx/types"
)

// TurnHandler runs one model turn: the model call and the tool calls it
// requests. It changes the event stream by appending to it through the
// context.
type TurnHandler func(TurnContext) error

// LLMHandler performs the model call described by the request of its context.
// When the call is interrupted, it may return the completion received so far
// along with the error; interceptors pass it on, and the turn keeps its text.
type LLMHandler func(LLMContext) (*types.Completion, error)

// ToolCallHandler performs the tool call of its context. Tool calls of one turn
// run concurrently, so it must be safe for concurrent use.
type ToolCallHandler func(ToolContext) (any, error)

// Interceptor wraps the three stages of an agent in onion order: a turn
// contains its model call, which is followed by its tool calls, and the
// contexts of both extend the turn context. The first registered interceptor
// is the outermost. Every Intercept method is called once per
// agent to build its handler chain. Embed NopInterceptor to intercept only
// some of the stages.
type Interceptor interface {
	InterceptTurn(TurnHandler) TurnHandler
	InterceptLLM(LLMHandler) LLMHandler
	InterceptToolCall(ToolCallHandler) ToolCallHandler
}

// NopInterceptor passes every stage through unchanged.
type NopInterceptor struct{}

func (NopInterceptor) InterceptTurn(next TurnHandler) TurnHandler { return next }

func (NopInterceptor) InterceptLLM(next LLMHandler) LLMHandler { return next }

func (NopInterceptor) InterceptToolCall(next ToolCallHandler) ToolCallHandler {
	return next
}

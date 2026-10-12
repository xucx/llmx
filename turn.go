package llmx

import (
	"context"
	"fmt"
	"slices"

	"github.com/xucx/llmx/types"
)

// runTurn is the innermost turn handler. It calls the model on the history of
// the event stream, appends the reply and its usage, and then runs the tools
// the reply asks for.
func (a *Agent) runTurn(ctx TurnContext) error {
	level := a.options.ReasonLevel
	options := &types.CompletionOptions{
		Instructions:   a.options.Instruction,
		ReasoningLevel: &level,
		Tools:          a.tools,
	}
	if a.options.Stream {
		// Streaming fragments go to the event sink as they arrive.
		options.StreamingFunc = func(_ context.Context, completion *types.Completion) error {
			if completion == nil || completion.Message == nil {
				return nil
			}
			return ctx.Append(NewEventMessage(completion.Message))
		}
	}
	llmCtx := newLLMContext(ctx, LLMRequest{
		Model:    a.options.Model,
		Messages: a.session.history(),
		Options:  options,
	})

	completion, err := a.llmHandler(llmCtx)
	if err != nil {
		// An interrupted streaming call returns what it received so far: keep
		// it, so the history shows what the model already said.
		if reply := interruptedReply(completion); reply != nil {
			if err := ctx.Append(NewEventMessage(reply)); err != nil {
				return err
			}
		}
		return err
	}
	if completion == nil || completion.Message == nil {
		return fmt.Errorf("llm returned no message")
	}
	if err := ctx.Append(
		NewEventMessage(completion.Message),
		NewEventUsage(completion.Usage),
	); err != nil {
		return err
	}

	toolCalls := completion.Message.ToolCalls()
	if len(toolCalls) == 0 {
		return nil
	}
	calls := make([]*toolContext, len(toolCalls))
	for index, call := range toolCalls {
		calls[index] = newToolContext(ctx, call, nil, nil)
	}
	return a.runTools(ctx, calls)
}

// interruptedReply keeps the text and reasoning of a completion cut short.
// Its tool calls are dropped, since their arguments may be truncated: an
// interrupted model call never calls tools. It returns nil when nothing is
// left.
func interruptedReply(completion *types.Completion) *types.Message {
	if completion == nil || completion.Message == nil {
		return nil
	}
	reply := completion.Message.Clone()
	reply.Delta = false
	reply.Parts = slices.DeleteFunc(reply.Parts, func(part *types.MessagePart) bool {
		return part == nil || part.ToolCall != nil
	})
	if len(reply.Parts) == 0 {
		return nil
	}
	return reply
}

// callLLM is the innermost llm handler: it sends the request to the model.
func (a *Agent) callLLM(ctx LLMContext) (*types.Completion, error) {
	request := ctx.Request()
	return a.models.Completion(
		ctx,
		request.Model,
		request.Messages,
		types.CompletionWithOptions(request.Options),
	)
}

package llmx

import (
	"context"
	"fmt"
)

// A tool asks a human by returning a ToolResult with AskHuman set: instead of a
// result, a human request is appended and the task goes on without it. An input
// carrying an EventHumanResponse answers it; the task that takes the input
// runs the tool call again with both in its ToolContext, and its result is
// appended like any other. The session checks that every request refers to a
// tool call and every response to an unanswered request.

// resume re-invokes the tool call that raised an answered human request. A
// request that already has a result is left alone, so replaying history never
// runs a tool again.
func (a *Agent) resume(ctx context.Context, response *EventHumanResponse) error {
	request := a.session.humanRequest(response.ID)
	if request == nil {
		return fmt.Errorf("human response references unknown request %q", response.ID)
	}
	if a.session.resolved(request.ToolCall.ID) {
		return nil
	}
	resumeCtx := newTurnContext(ctx, a.session)
	defer resumeCtx.close()
	return a.runTools(resumeCtx, []*toolContext{
		newToolContext(resumeCtx, request.ToolCall, request, response),
	})
}

package llmx

import (
	"context"

	"github.com/xucx/llmx/types"
)

// LoopAction tells a session what to do after one complete model turn.
type LoopAction uint8

const (
	// LoopWait yields control to the caller. Run returns, while Start waits for
	// the next input without closing the session.
	LoopWait LoopAction = iota

	// LoopContinue starts another model turn without waiting for new input.
	LoopContinue

	// LoopStop closes the session in both modes: a started session stops
	// serving, and a session that returned from Run cannot be Run again.
	LoopStop
)

// Loop decides what a session does after every complete model turn, once all
// events produced by that turn have been committed.
type Loop func(ctx context.Context, turn *SessionTurnResult) (LoopAction, error)

// ReactLoop continues while the last turn produced a tool result, which is the
// plain reason-act cycle: the model called tools, so it must see their results.
func ReactLoop(_ context.Context, turn *SessionTurnResult) (LoopAction, error) {
	if turn == nil {
		return LoopWait, nil
	}

	for eventIndex := len(turn.Events) - 1; eventIndex >= 0; eventIndex-- {
		event := turn.Events[eventIndex]
		if event.Message == nil {
			continue
		}
		if event.Message.Role != types.MessageRoleAssistant {
			return LoopContinue, nil
		}
		break
	}

	return LoopWait, nil
}

package llmx

import (
	"context"
	"errors"
	"fmt"

	"github.com/google/uuid"
	"github.com/xucx/llmx/types"
)

// errInterrupted is the cause a Runner cancels an interrupted task with.
var errInterrupted = errors.New("interrupted")

// LoopAction tells a task what to do after one complete model turn.
type LoopAction uint8

const (
	// LoopContinue starts another model turn of the same task.
	LoopContinue LoopAction = iota

	// LoopDone finishes the task. The agent waits for the next one.
	LoopDone

	// LoopStop finishes the task and closes the agent.
	LoopStop
)

// TurnResult is what a task has done after one complete model turn. It is the
// input a Loop decides on.
type TurnResult struct {
	// Index is the index of the turn within its task.
	Index int
	// Events are the events of the task so far, oldest first.
	Events []*Event
	// TurnEventIndex is the index in Events of the first event of this turn.
	TurnEventIndex int
}

// Loop decides what a task does after every complete model turn. It is the
// last step of the turn: ctx is the context of the turn, still open, so a Loop
// can read the state and append what the next turn needs -- feedback, the next
// step of a plan -- before it continues. Continuing after a plain reply without
// appending anything asks the model to answer its own reply.
//
// Events appended by a Loop join the stream at once, even when an interjection
// then ends the task at this boundary instead of continuing it.
type Loop func(ctx TurnContext, turn *TurnResult) (LoopAction, error)

// ReactLoop continues while the last turn produced a tool result, which is the
// plain reason-act cycle: the model called tools, so it must see their results.
func ReactLoop(_ TurnContext, turn *TurnResult) (LoopAction, error) {
	if turn == nil {
		return LoopDone, nil
	}
	for index := len(turn.Events) - 1; index >= 0; index-- {
		message := turn.Events[index].Message
		if message == nil {
			continue
		}
		if needsAnswer(message) {
			return LoopContinue, nil
		}
		break
	}
	return LoopDone, nil
}

// needsAnswer reports whether a message gives the model something to answer: a
// system or user message, or a tool result not marked final.
func needsAnswer(message *types.Message) bool {
	switch message.Role {
	case types.MessageRoleSystem, types.MessageRoleUser:
		return true
	case types.MessageRoleTool:
		for _, part := range message.Parts {
			if part.ToolResult != nil && !part.ToolResult.Final {
				return true
			}
		}
	}
	return false
}

// runTask runs one task: it appends its inputs, resumes answered human
// requests and runs model turns while Loop continues, bracketed by a running
// and a closing status. It returns the closing status, with the error the task
// ended with when it has one: a stopped status closes the agent, a waiting one
// leaves it usable. A nil status is a fatal error: it closes the agent, and no
// closing status is appended.
func (a *Agent) runTask(ctx context.Context, inputs []*Input) (*EventStatus, error) {
	from := a.session.len()
	if err := a.session.append(newStatusEvent(&EventStatus{Status: StatusRunning})); err != nil {
		return nil, err
	}
	end, err := a.runTurns(ctx, from, inputs)
	a.result.Turns += end.Turns
	if err != nil {
		switch {
		case errors.Is(context.Cause(ctx), errInterrupted):
			end.Status, end.StopReason = StatusWaiting, StopInterrupted
			err = nil
		case errors.Is(err, ErrInputRejected):
			end.Status, end.StopReason = StatusWaiting, StopRejected
		default:
			return nil, err
		}
	}
	if err := a.session.append(newStatusEvent(end)); err != nil {
		return nil, err
	}
	if end.StopReason == StopMaxTurn {
		return end, fmt.Errorf("%w: %d", ErrMaxTurnExceeded, a.options.MaxTurn)
	}
	return end, err
}

// runTurns is the body of a task whose events start at stream index from. It
// returns the closing status of the task, which counts the turns run even when
// an error ends the task.
func (a *Agent) runTurns(ctx context.Context, from int, inputs []*Input) (*EventStatus, error) {
	end := &EventStatus{Status: StatusWaiting, StopReason: StopDone}

	for _, response := range a.session.unresolvedResponses() {
		if err := a.resume(ctx, response); err != nil {
			return end, err
		}
	}
	for _, input := range inputs {
		events, err := input.events()
		if err != nil {
			return end, err
		}
		// The messages of an input follow the tool results its responses
		// resume, so they are the newest history: the rest goes first, as one
		// batch that the stream takes or rejects as a whole.
		var leading, messages []*Event
		for _, event := range events {
			if event.Message != nil {
				messages = append(messages, event)
			} else {
				leading = append(leading, event)
			}
		}
		if err := a.session.append(leading...); err != nil {
			if errors.Is(err, errRejected) {
				err = fmt.Errorf("%w: %w", ErrInputRejected, err)
			}
			return end, err
		}
		for _, event := range leading {
			if event.HumanResponse != nil {
				if err := a.resume(ctx, event.HumanResponse); err != nil {
					return end, err
				}
			}
		}
		if err := a.session.append(messages...); err != nil {
			return end, err
		}
	}

	// The task only runs turns when its input gives the model something to
	// answer. Pending human requests do not block a turn: an answer may never
	// come.
	needTurn := false
	for _, event := range a.session.eventsFrom(from) {
		if event.Message != nil && needsAnswer(event.Message) {
			needTurn = true
		}
	}
	if !needTurn {
		return end, nil
	}
	for {
		if err := contextError(ctx); err != nil {
			return end, err
		}
		if a.options.MaxTurn > 0 && end.Turns >= a.options.MaxTurn {
			end.Status, end.StopReason = StatusStopped, StopMaxTurn
			return end, nil
		}

		// Loop is the last step of the turn, so both share one turn context
		// that closes once Loop has decided.
		turnFrom := a.session.len()
		turnCtx := newTurnContext(ctx, a.session)
		end.Turns++
		err := a.turnHandler(turnCtx)
		var action LoopAction
		if err == nil {
			action, err = a.options.Loop(turnCtx, &TurnResult{
				Index:          end.Turns - 1,
				Events:         a.session.eventsFrom(from),
				TurnEventIndex: turnFrom - from,
			})
		}
		turnCtx.close()
		if err != nil {
			return end, err
		}
		switch action {
		case LoopContinue:
			// The turn boundary: an interjection ends the task here and runs
			// as the next task.
			if a.runner != nil && a.runner.interjected() {
				end.StopReason = StopInterjected
				return end, nil
			}
		case LoopDone:
			return end, nil
		case LoopStop:
			end.Status = StatusStopped
			return end, nil
		default:
			return end, fmt.Errorf("invalid loop action %d", action)
		}
	}
}

func newStatusEvent(status *EventStatus) *Event {
	value := *status
	return &Event{ID: uuid.NewString(), Status: &value}
}

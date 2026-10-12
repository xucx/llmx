package llmx

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/xucx/llmx/types"
)

// jsonLoop is a structured output loop: it asks the model to fix a reply that
// is not valid JSON, and is done once the reply is.
func jsonLoop(ctx TurnContext, turn *TurnResult) (LoopAction, error) {
	if action, err := ReactLoop(ctx, turn); err != nil || action == LoopContinue {
		return action, err // the model still works with its tools
	}
	var reply *types.Message
	for _, event := range turn.Events {
		if event.Message != nil && event.Message.Role == types.MessageRoleAssistant {
			reply = event.Message
		}
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(reply.Text()), &decoded); err != nil {
		return LoopContinue, ctx.Append(NewEventMessage(types.NewTextMessage(
			types.MessageRoleUser,
			"The reply is not valid JSON: "+err.Error()+". Output only the corrected JSON.",
		)))
	}
	return LoopDone, nil
}

func TestLoopAppendsFeedbackAndContinues(t *testing.T) {
	var seen [][]string
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			var texts []string
			for _, message := range ctx.Request().Messages {
				texts = append(texts, message.Text())
			}
			seen = append(seen, texts)
			reply := "not json"
			if len(seen) > 1 {
				reply = `{"ok":true}`
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, reply),
			}, nil
		},
	)
	session, err := agent.build(nil, WithLoop(jsonLoop))
	if err != nil {
		t.Fatal(err)
	}
	result, err := session.Run(context.Background(), NewUserInput("give me json"))
	if err != nil {
		t.Fatal(err)
	}

	if result.Turns != 2 || result.StopReason != StopDone {
		t.Fatalf("unexpected result: %+v", result)
	}
	// The second call answers the feedback the Loop appended after the first.
	if len(seen) != 2 || len(seen[1]) != 3 ||
		seen[1][1] != "not json" || seen[1][2] == "" {
		t.Fatalf("the model did not see the feedback: %v", seen)
	}
	if reply := result.Reply(); reply == nil || reply.Text() != `{"ok":true}` {
		t.Fatalf("unexpected reply: %+v", reply)
	}
}

func TestLoopContextClosesWithTheTurn(t *testing.T) {
	var loopCtx TurnContext
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "answer"),
			}, nil
		},
	)
	session, err := agent.build(nil, WithLoop(func(ctx TurnContext, turn *TurnResult) (LoopAction, error) {
		loopCtx = ctx
		return ReactLoop(ctx, turn)
	}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Run(context.Background(), NewUserInput("question")); err != nil {
		t.Fatal(err)
	}
	err = loopCtx.Append(NewEventState(map[string]string{"late": "yes"}, true))
	if !errors.Is(err, errTurnOver) {
		t.Fatalf("append after the loop decided = %v, want errTurnOver", err)
	}
}

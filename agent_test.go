package llmx

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/xucx/llmx/types"
)

type emptyToolParam struct{}

// agentFactory builds agents sharing one configuration, each over its own
// event stream; build adds options for that one agent.
type agentFactory struct {
	models *Models
	opts   []AgentOption
}

func newAgentFactory(models *Models, opts ...AgentOption) (*agentFactory, error) {
	return &agentFactory{models: models, opts: opts}, nil
}

func (f *agentFactory) build(events []*Event, opts ...AgentOption) (*Agent, error) {
	return NewAgent("test", "", f.models, events, append(append([]AgentOption(nil), f.opts...), opts...)...)
}

func newRunTestAgent(t *testing.T, handler LLMHandler, tools ...*Tool) *agentFactory {
	t.Helper()
	agent, err := newAgentFactory(
		NewModels(),
		WithModel("fake"),
		WithTools(tools...),
		WithInterceptor(llmInterceptorFunc(func(LLMHandler) LLMHandler { return handler })),
	)
	if err != nil {
		t.Fatal(err)
	}
	return agent
}

// stopAfterTurns is a ReactLoop that closes the agent after n turns of the
// whole run: turn indexes restart with every task, so it counts them itself.
func stopAfterTurns(n int) Loop {
	turns := 0
	return func(ctx TurnContext, turn *TurnResult) (LoopAction, error) {
		turns++
		if turns >= n {
			return LoopStop, nil
		}
		return ReactLoop(ctx, turn)
	}
}

func initialUserEvent(text string) *Event {
	return NewEventMessage(types.NewTextMessage(types.MessageRoleUser, text))
}

func TestRunReturnsEventsAndRestoredHistory(t *testing.T) {
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "answer"),
				Usage: types.CompletionUsage{
					PromptTokens:     2,
					CompletionTokens: 3,
					TotalTokens:      5,
				},
			}, nil
		},
	)

	resultSession, err := agent.build(nil)
	if err != nil {
		t.Fatal(err)
	}
	result, err := resultSession.Run(context.Background(), NewUserInput("question"))
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != StopDone || result.Turns != 1 {
		t.Fatalf("unexpected result: %+v", result)
	}
	// A turn reports what it produced before what it cost: the usage carries the
	// completion tokens of the message right before it. A running status opens
	// the run and a waiting status closes it once every event is committed.
	if len(result.Events) != 5 ||
		result.Events[0].Status == nil ||
		result.Events[0].Status.Status != StatusRunning ||
		result.Events[1].Message == nil ||
		result.Events[2].Message == nil ||
		result.Events[3].Usage == nil ||
		result.Events[4].Status == nil ||
		result.Events[4].Status.Status != StatusWaiting ||
		result.Events[4].Status.StopReason != StopDone ||
		result.Events[4].Status.Turns != 1 {
		t.Fatalf("unexpected output events: %+v", result.Events)
	}
	// Usage is context size, not a session total, so it is reported as an event.
	if result.Events[3].Usage.Usage.TotalTokens != 5 {
		t.Fatalf("unexpected usage: %+v", result.Events[3].Usage)
	}
}

func TestAgentSharesImmutableEventsAcrossObservers(t *testing.T) {
	var sinkEvents []*Event
	var loopEvents []*Event
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "answer"),
			}, nil
		},
	)
	resultSession, err := agent.build(
		nil,
		WithEventSink(func(_ context.Context, event *Event) error {
			sinkEvents = append(sinkEvents, event)
			return nil
		}),
		WithLoop(func(_ TurnContext, turn *TurnResult) (LoopAction, error) {
			loopEvents = append([]*Event(nil), turn.Events...)
			return LoopDone, nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := resultSession.Run(context.Background(), NewUserInput("question"))
	if err != nil {
		t.Fatal(err)
	}
	// Loop decides before the closing status is committed, so it sees every
	// event but that last one.
	if len(sinkEvents) != len(result.Events) ||
		len(loopEvents) != len(result.Events)-1 {
		t.Fatalf("observers saw different events: %d %d %d", len(sinkEvents), len(loopEvents), len(result.Events))
	}
	for index := range result.Events {
		if sinkEvents[index] != result.Events[index] ||
			(index < len(loopEvents) && loopEvents[index] != result.Events[index]) {
			t.Fatalf("event %d was cloned between observers", index)
		}
	}
}

func TestAgentRunCanBeRepeatedAndRetainsHistory(t *testing.T) {
	var histories [][]string
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			texts := make([]string, 0, len(messages))
			for _, message := range messages {
				texts = append(texts, message.Text())
			}
			histories = append(histories, texts)
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "answer"),
			}, nil
		},
	)
	session, err := agent.build(nil)
	if err != nil {
		t.Fatal(err)
	}

	first, err := session.Run(context.Background(), NewUserInput("first"))
	if err != nil {
		t.Fatal(err)
	}
	second, err := session.Run(context.Background(), NewUserInput("second"))
	if err != nil {
		t.Fatal(err)
	}

	if first.Turns != 1 || second.Turns != 1 {
		t.Fatalf("runs did not keep independent turn counts: %+v %+v", first, second)
	}
	if len(histories) != 2 ||
		len(histories[1]) != 3 ||
		histories[1][0] != "first" ||
		histories[1][1] != "answer" ||
		histories[1][2] != "second" {
		t.Fatalf("session history was not retained: %+v", histories)
	}
}

func TestAgentRejectsConcurrentRunAndStart(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			close(started)
			<-release
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)
	session, err := agent.build(nil)
	if err != nil {
		t.Fatal(err)
	}

	firstDone := make(chan error, 1)
	go func() {
		_, runErr := session.Run(context.Background(), NewUserInput("first"))
		firstDone <- runErr
	}()
	<-started

	if _, err := session.Run(
		context.Background(),
		NewUserInput("second"),
	); !errors.Is(err, ErrBusy) {
		t.Fatalf("expected concurrent Run to be busy, got %v", err)
	}
	if _, err := session.Start(context.Background()); !errors.Is(err, ErrBusy) {
		t.Fatalf("expected Start during Run to be busy, got %v", err)
	}

	close(release)
	if err := <-firstDone; err != nil {
		t.Fatal(err)
	}
}

func TestAgentStartWaitsForInput(t *testing.T) {
	called := make(chan struct{}, 1)
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			called <- struct{}{}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)
	runnerSession, err := agent.build(
		nil,
		WithLoop(func(TurnContext, *TurnResult) (LoopAction, error) {
			return LoopStop, nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := runnerSession.Start(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	select {
	case <-called:
		t.Fatal("model was called before the first input")
	case <-time.After(20 * time.Millisecond):
	}
	if _, err := runner.Enqueue(NewUserInput("run")); err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Wait(); err != nil {
		t.Fatal(err)
	}
}

func TestStartAcceptsDynamicEventsAndResetsMaxTurn(t *testing.T) {
	firstEntered := make(chan struct{})
	releaseFirst := make(chan struct{})
	var mu sync.Mutex
	var histories [][]string
	calls := 0
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			mu.Lock()
			calls++
			call := calls
			texts := make([]string, 0, len(messages))
			for _, message := range messages {
				texts = append(texts, message.Text())
			}
			histories = append(histories, texts)
			mu.Unlock()
			if call == 1 {
				close(firstEntered)
				<-releaseFirst
			}
			return &types.Completion{
				Message: types.NewTextMessage(
					types.MessageRoleAssistant,
					"answer",
				),
			}, nil
		},
	)

	runnerSession, err := agent.build(
		nil,
		WithMaxTurn(1),
		WithLoop(stopAfterTurns(2)),
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := runnerSession.Start(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("initial")); err != nil {
		t.Fatal(err)
	}
	<-firstEntered

	// Enqueued while the first turn runs: accepted only once it yields.
	if _, err := runner.Enqueue(NewUserInput("follow-up")); err != nil {
		t.Fatal(err)
	}
	close(releaseFirst)

	result, err := runner.Wait()
	if err != nil {
		t.Fatal(err)
	}
	if result.Turns != 2 || result.StopReason != StopDone {
		t.Fatalf("unexpected result: %+v", result)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(histories) != 2 ||
		histories[1][len(histories[1])-1] != "follow-up" {
		t.Fatalf("dynamic input missing from next turn: %+v", histories)
	}
}

func TestRunReturnsMaxTurnStopReason(t *testing.T) {
	tool := MustNewTool(
		"continue",
		"",
		func(ToolContext, *emptyToolParam) (any, error) {
			return "next", nil
		},
	)
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			message := types.NewMessage(types.MessageRoleAssistant)
			message.Parts = append(message.Parts, &types.MessagePart{
				ToolCall: &types.MessageToolCall{
					ID:        "continue",
					Name:      "continue",
					Arguments: "{}",
				},
			})
			return &types.Completion{Message: message}, nil
		},
		tool,
	)

	resultSession, err := agent.build(
		nil,
		WithMaxTurn(2),
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := resultSession.Run(context.Background(), NewUserInput("run"))
	if !errors.Is(err, ErrMaxTurnExceeded) {
		t.Fatalf("expected max turn error, got %v", err)
	}
	if result.StopReason != StopMaxTurn || result.Turns != 2 {
		t.Fatalf("unexpected result: %+v", result)
	}
	last := result.Events[len(result.Events)-1]
	if last.Status == nil || last.Status.Status != StatusStopped ||
		last.Status.StopReason != StopMaxTurn || last.Status.Turns != 2 {
		t.Fatalf("expected a maxTurn stopped status last, got %+v", last)
	}
}

func TestInputRejectedBySinkDoesNotReachModel(t *testing.T) {
	sinkErr := errors.New("fenced")
	called := atomic.Bool{}
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			called.Store(true)
			return nil, errors.New("must not be called")
		},
	)

	resultSession, err := agent.build(
		nil,
		WithEventSink(func(_ context.Context, event *Event) error {
			if event.Message != nil &&
				event.Message.Role == types.MessageRoleUser {
				return sinkErr
			}
			return nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := resultSession.Run(context.Background(), NewUserInput("must persist first"))
	if !errors.Is(err, sinkErr) {
		t.Fatalf("expected sink error, got %v", err)
	}
	if called.Load() {
		t.Fatal("model must not see an input rejected by EventSink")
	}
	// Only the running status that opened the run was persisted.
	if len(result.Events) != 1 || result.Events[0].Status == nil {
		t.Fatalf("rejected input leaked into result: %+v", result)
	}
}

func TestAgentSendAfterStopReturnsClosed(t *testing.T) {
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)
	runnerSession, err := agent.build(
		nil,
		WithLoop(func(TurnContext, *TurnResult) (LoopAction, error) {
			return LoopStop, nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := runnerSession.Start(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("run")); err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Wait(); err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("late")); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected session closed, got %v", err)
	}
}

func TestAgentStartsOnlyOnce(t *testing.T) {
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)
	session, err := agent.build(nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	runner, err := session.Start(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Start(context.Background()); !errors.Is(err, ErrStarted) {
		t.Fatalf("expected session already started, got %v", err)
	}
	cancel()
	if _, err := runner.Wait(); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected cancellation, got %v", err)
	}
}

func TestToolStatePatchesAreVisibleNextTurn(t *testing.T) {
	var observed map[string]string
	writerA := MustNewTool(
		"writer_a",
		"",
		func(ctx ToolContext, _ *emptyToolParam) (any, error) {
			return "a", ctx.Append(NewEventState(map[string]string{"onlyA": "1"}, true))
		},
	)
	writerB := MustNewTool(
		"writer_b",
		"",
		func(ctx ToolContext, _ *emptyToolParam) (any, error) {
			return "b", ctx.Append(NewEventState(map[string]string{"onlyB": "1"}, true))
		},
	)
	reader := MustNewTool(
		"reader",
		"",
		func(ctx ToolContext, _ *emptyToolParam) (any, error) {
			observed = ctx.State()
			return "read", nil
		},
	)

	turn := 0
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			turn++
			message := types.NewMessage(types.MessageRoleAssistant)
			switch turn {
			case 1:
				message.Parts = append(
					message.Parts,
					&types.MessagePart{ToolCall: &types.MessageToolCall{
						ID: "a", Name: "writer_a", Arguments: "{}",
					}},
					&types.MessagePart{ToolCall: &types.MessageToolCall{
						ID: "b", Name: "writer_b", Arguments: "{}",
					}},
				)
			case 2:
				message.Parts = append(message.Parts, &types.MessagePart{
					ToolCall: &types.MessageToolCall{
						ID: "read", Name: "reader", Arguments: "{}",
					},
				})
			default:
				message = types.NewTextMessage(types.MessageRoleAssistant, "done")
			}
			return &types.Completion{Message: message}, nil
		},
		writerA,
		writerB,
		reader,
	)

	resultSession, err := agent.build(nil)
	if err != nil {
		t.Fatal(err)
	}
	result, err := resultSession.Run(context.Background(), NewUserInput("run"))
	if err != nil {
		t.Fatal(err)
	}
	// Parallel tools append in the order they run; both patches are in the
	// stream before the next turn.
	if observed["onlyA"] != "1" || observed["onlyB"] != "1" {
		t.Fatalf("reader did not see prior state: %+v", observed)
	}
	if result.State["onlyA"] != "1" || result.State["onlyB"] != "1" {
		t.Fatalf("unexpected final state: %+v", result.State)
	}
}

func TestHumanRequestCanResumeWithinSession(t *testing.T) {
	humanTool := MustNewTool(
		"approval",
		"",
		func(ctx ToolContext, _ *emptyToolParam) (any, error) {
			if ctx.HumanResponse() == nil {
				return &ToolResult{AskHuman: &ToolAskHuman{
					Arguments: map[string]any{"question": "approve?"},
				}}, nil
			}
			return ctx.HumanResponse().Result["answer"], nil
		},
	)

	turn := 0
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			turn++
			if turn == 1 {
				message := types.NewMessage(types.MessageRoleAssistant)
				message.Parts = append(message.Parts, &types.MessagePart{
					ToolCall: &types.MessageToolCall{
						ID: "approval-call", Name: "approval", Arguments: "{}",
					},
				})
				return &types.Completion{Message: message}, nil
			}
			foundResult := false
			for _, message := range messages {
				for _, part := range message.Parts {
					if part.ToolResult != nil &&
						part.ToolResult.ID == "approval-call" &&
						part.ToolResult.Result == "yes" {
						foundResult = true
					}
				}
			}
			if !foundResult {
				t.Fatalf("human tool result missing: %+v", messages)
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
		humanTool,
	)

	requests := make(chan *EventHumanRequest, 1)
	runnerSession, err := agent.build(
		nil,
		WithEventSink(func(_ context.Context, event *Event) error {
			if event.HumanRequest != nil {
				requests <- event.HumanRequest.Clone()
			}
			return nil
		}),
		WithLoop(stopAfterTurns(2)),
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := runnerSession.Start(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("run")); err != nil {
		t.Fatal(err)
	}

	request := <-requests
	if _, err := runner.Enqueue(NewHumanResponseInput(
		request.ID,
		map[string]any{"answer": "yes"},
	),
	); err != nil {
		t.Fatal(err)
	}

	result, err := runner.Wait()
	if err != nil {
		t.Fatal(err)
	}
	if result.Turns != 2 || result.StopReason != StopDone {
		t.Fatalf("unexpected result: %+v", result)
	}
	foundResponse := false
	for _, event := range result.Events {
		if event.HumanResponse != nil {
			foundResponse = true
		}
	}
	if !foundResponse {
		t.Fatal("human response was not included in output events")
	}
}

func TestInitialHumanResponseResumesPendingTool(t *testing.T) {
	tool := MustNewTool(
		"approval",
		"",
		func(ctx ToolContext, _ *emptyToolParam) (any, error) {
			if ctx.HumanRequest() == nil || ctx.HumanResponse() == nil {
				return nil, errors.New("missing restored human context")
			}
			return ctx.HumanResponse().Result["answer"], nil
		},
	)
	call := &types.MessageToolCall{
		ID: "approval-call", Name: "approval", Arguments: "{}",
	}
	assistant := types.NewMessage(types.MessageRoleAssistant)
	assistant.Parts = append(
		assistant.Parts,
		&types.MessagePart{ToolCall: cloneToolCall(call)},
	)
	history := []*Event{
		initialUserEvent("run"),
		NewEventMessage(assistant),
		NewEventHumanRequest(&EventHumanRequest{
			ID:       "approval-request",
			ToolCall: cloneToolCall(call),
		}),
		NewEventHumanResponse(&EventHumanResponse{
			ID:     "approval-request",
			Result: map[string]any{"answer": "restored"},
		}),
	}
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			found := false
			for _, message := range messages {
				for _, part := range message.Parts {
					if part.ToolResult != nil &&
						part.ToolResult.ID == call.ID &&
						part.ToolResult.Result == "restored" {
						found = true
					}
				}
			}
			if !found {
				t.Fatalf("restored tool result missing: %+v", messages)
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
		tool,
	)

	resultSession, err := agent.build(history)
	if err != nil {
		t.Fatal(err)
	}
	result, err := resultSession.Run(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Turns != 1 || result.StopReason != StopDone {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestRunContextCancellationInterruptsTheTask(t *testing.T) {
	started := make(chan struct{})
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			if messages[len(messages)-1].Text() == "run" {
				close(started)
				<-ctx.Done()
				return nil, context.Cause(ctx)
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "again"),
			}, nil
		},
	)
	ctx, cancel := context.WithCancelCause(context.Background())
	type runOutput struct {
		result *Result
		err    error
	}
	session, err := agent.build(nil)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan runOutput, 1)
	go func() {
		result, err := session.Run(ctx, NewUserInput("run"))
		done <- runOutput{result: result, err: err}
	}()
	<-started
	cause := errors.New("stop now")
	cancel(cause)

	output := <-done
	if !errors.Is(output.err, cause) {
		t.Fatalf("expected cancellation cause, got %v", output.err)
	}
	if output.result.StopReason != StopInterrupted ||
		!errors.Is(output.result.Cause, cause) {
		t.Fatalf("unexpected canceled result: %+v", output.result)
	}
	last := output.result.Events[len(output.result.Events)-1].Status
	if last == nil || last.StopReason != StopInterrupted {
		t.Fatalf("the interrupted task has no closing status: %+v", output.result.Events)
	}

	// Canceling a Run interrupts its task, not the agent.
	result, err := session.Run(context.Background(), NewUserInput("next"))
	if err != nil {
		t.Fatalf("the agent did not survive the canceled run: %v", err)
	}
	if reply := result.Reply(); reply == nil || reply.Text() != "again" {
		t.Fatalf("unexpected reply: %+v", reply)
	}
}

func TestContextCancellationStopsHumanWait(t *testing.T) {
	humanTool := MustNewTool(
		"approval",
		"",
		func(ToolContext, *emptyToolParam) (any, error) {
			return &ToolResult{AskHuman: &ToolAskHuman{}}, nil
		},
	)
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			message := types.NewMessage(types.MessageRoleAssistant)
			message.Parts = append(message.Parts, &types.MessagePart{
				ToolCall: &types.MessageToolCall{
					ID: "approval-call", Name: "approval", Arguments: "{}",
				},
			})
			return &types.Completion{Message: message}, nil
		},
		humanTool,
	)

	requested := make(chan struct{}, 1)
	ctx, cancel := context.WithCancelCause(context.Background())
	runnerSession, err := agent.build(
		nil,
		WithEventSink(func(_ context.Context, event *Event) error {
			if event.HumanRequest != nil {
				requested <- struct{}{}
			}
			return nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := runnerSession.Start(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("run")); err != nil {
		t.Fatal(err)
	}
	<-requested
	cause := errors.New("human wait canceled")
	cancel(cause)

	result, err := runner.Wait()
	if !errors.Is(err, cause) {
		t.Fatalf("expected cancellation cause, got %v", err)
	}
	if result.StopReason != StopCanceled {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestContextCancellationDoesNotWaitForIgnoringTool(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	tool := MustNewTool(
		"blocking",
		"",
		func(ToolContext, *emptyToolParam) (any, error) {
			close(started)
			<-release
			return "late", nil
		},
	)
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			message := types.NewMessage(types.MessageRoleAssistant)
			message.Parts = append(message.Parts, &types.MessagePart{
				ToolCall: &types.MessageToolCall{
					ID: "blocking-call", Name: "blocking", Arguments: "{}",
				},
			})
			return &types.Completion{Message: message}, nil
		},
		tool,
	)

	ctx, cancel := context.WithCancelCause(context.Background())
	runnerSession, err := agent.build(nil)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := runnerSession.Start(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("run")); err != nil {
		t.Fatal(err)
	}
	<-started
	cause := errors.New("tool canceled")
	cancel(cause)

	waited := make(chan struct{})
	var result *Result
	var waitErr error
	go func() {
		result, waitErr = runner.Wait()
		close(waited)
	}()
	select {
	case <-waited:
	case <-time.After(time.Second):
		t.Fatal("session waited for a tool that ignored context")
	}
	close(release)
	if !errors.Is(waitErr, cause) {
		t.Fatalf("expected cancellation cause, got %v", waitErr)
	}
	if result.StopReason != StopCanceled {
		t.Fatalf("unexpected result: %+v", result)
	}
}

type streamingTestProvider struct {
	types.ProviderNop
}

func (streamingTestProvider) Completion(
	ctx context.Context,
	_ string,
	_ []*types.Message,
	options ...types.CompletionOption,
) (*types.Completion, error) {
	opts := types.ApplyCompletionOptions(&types.CompletionOptions{}, options...)
	if opts.StreamingFunc != nil {
		delta := types.NewTextMessage(types.MessageRoleAssistant, "hel")
		delta.Delta = true
		if err := opts.StreamingFunc(ctx, &types.Completion{
			Delta:   true,
			Message: delta,
		}); err != nil {
			return nil, err
		}
	}
	return &types.Completion{
		Message: types.NewTextMessage(types.MessageRoleAssistant, "hello"),
	}, nil
}

type captureCompletionProvider struct {
	types.ProviderNop
	options *types.CompletionOptions
}

func (p *captureCompletionProvider) Completion(
	_ context.Context,
	_ string,
	_ []*types.Message,
	options ...types.CompletionOption,
) (*types.Completion, error) {
	p.options = types.ApplyCompletionOptions(&types.CompletionOptions{}, options...)
	return &types.Completion{
		Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
	}, nil
}

func TestLLMInterceptorAdaptsInstructionAndTools(t *testing.T) {
	provider := new(captureCompletionProvider)
	models := NewModels()
	models.models["fake"] = &Model{
		Name:     "fake",
		Model:    "fake",
		Provider: provider,
	}
	keep := MustNewTool(
		"keep",
		"kept tool",
		func(ToolContext, *emptyToolParam) (any, error) { return nil, nil },
	)
	drop := MustNewTool(
		"drop",
		"dropped tool",
		func(ToolContext, *emptyToolParam) (any, error) { return nil, nil },
	)

	agent, err := newAgentFactory(
		models,
		WithModel("fake"),
		WithInstruction("original"),
		WithTools(keep, drop),
		WithInterceptor(llmInterceptorFunc(func(next LLMHandler) LLMHandler {
			return func(ctx LLMContext) (*types.Completion, error) {
				// The context reads the whole event stream, restored history included.
				if events := ctx.Events(); len(events) == 0 || events[0].State == nil ||
					ctx.State()["locale"] != "zh" {
					return nil, fmt.Errorf("context misses the restored events: %+v", events)
				}
				// The request is adapted for this call without touching the agent.
				request := ctx.Request()
				request.Options.Instructions = "adapted"
				request.Options.Tools = []*types.Tool{
					types.NewFunctionTool(keep.Name, keep.Description, keep.Parameters),
				}
				ctx.SetRequest(request)
				return next(ctx)
			}
		})),
	)
	if err != nil {
		t.Fatal(err)
	}

	resultSession, err := agent.build(
		[]*Event{
			NewEventState(map[string]string{"locale": "zh"}, false),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := resultSession.Run(context.Background(), NewUserInput("run")); err != nil {
		t.Fatal(err)
	}
	if provider.options == nil || provider.options.Instructions != "adapted" {
		t.Fatalf("unexpected completion options: %+v", provider.options)
	}
	if len(provider.options.Tools) != 1 ||
		provider.options.Tools[0].Function.Name != "keep" {
		t.Fatalf("unexpected exposed tools: %+v", provider.options.Tools)
	}
}

func TestToolCallInterceptorReceivesToolContext(t *testing.T) {
	var interceptorCallID string
	var invokerCallID string
	tool := MustNewTool(
		"inspect",
		"",
		func(ctx ToolContext, _ *emptyToolParam) (any, error) {
			invokerCallID = ctx.ToolCall().ID
			if ctx.State()["intercepted"] != "yes" {
				return nil, fmt.Errorf("interceptor state is not visible to the tool")
			}
			return "ok", nil
		},
	)

	turn := 0
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			turn++
			if turn > 1 {
				return &types.Completion{
					Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
				}, nil
			}
			message := types.NewMessage(types.MessageRoleAssistant)
			message.Parts = append(message.Parts, &types.MessagePart{
				ToolCall: &types.MessageToolCall{
					ID: "call-1", Name: "inspect", Arguments: "{}",
				},
			})
			return &types.Completion{Message: message}, nil
		},
		tool,
	)

	resultSession, err := agent.build(
		[]*Event{
			NewEventState(map[string]string{"seed": "value"}, false),
		},
		WithInterceptor(toolCallInterceptorFunc(func(next ToolCallHandler) ToolCallHandler {
			return func(ctx ToolContext) (any, error) {
				call := ctx.ToolCall()
				interceptorCallID = ctx.ToolCall().ID
				if interceptorCallID != call.ID || ctx.State()["seed"] != "value" {
					return nil, fmt.Errorf("unexpected tool context")
				}
				if err := ctx.Append(NewEventState(map[string]string{"intercepted": "yes"}, true)); err != nil {
					return nil, err
				}
				return next(ctx)
			}
		})),
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := resultSession.Run(context.Background(), NewUserInput("run"))
	if err != nil {
		t.Fatal(err)
	}
	if interceptorCallID != "call-1" || invokerCallID != "call-1" {
		t.Fatalf(
			"tool call id was not propagated: interceptor=%q invoker=%q",
			interceptorCallID,
			invokerCallID,
		)
	}
	if result.State["intercepted"] != "yes" {
		t.Fatalf("interceptor state patch was not committed: %+v", result.State)
	}
}

func TestStreamingMessagesUseEventSinkAndDoNotEnterHistory(t *testing.T) {
	models := NewModels()
	models.models["fake"] = &Model{
		Name:     "fake",
		Model:    "fake",
		MaxToken: 100,
		Provider: streamingTestProvider{},
	}
	agent, err := newAgentFactory(
		models,
		WithModel("fake"),
		WithStream(true),
	)
	if err != nil {
		t.Fatal(err)
	}

	var sinkMessages []*types.Message
	runSession, err := agent.build(
		nil,
		WithEventSink(func(_ context.Context, event *Event) error {
			if event.Message != nil {
				sinkMessages = append(
					sinkMessages,
					event.Message.Clone(),
				)
			}
			return nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	_, err = runSession.Run(context.Background(), NewUserInput("run"))
	if err != nil {
		t.Fatal(err)
	}
	if len(sinkMessages) != 3 ||
		sinkMessages[0].Delta ||
		!sinkMessages[1].Delta ||
		sinkMessages[2].Delta {
		t.Fatalf("unexpected streamed messages: %+v", sinkMessages)
	}
}

func TestRunOptionsDoNotMutateAgentOptions(t *testing.T) {
	baseTool := MustNewTool(
		"base",
		"",
		func(ToolContext, *emptyToolParam) (any, error) { return nil, nil },
	)
	runTool := MustNewTool(
		"run",
		"",
		func(ToolContext, *emptyToolParam) (any, error) { return nil, nil },
	)
	agent, err := newAgentFactory(
		NewModels(),
		WithModel("base-model"),
		WithInstruction("base"),
		WithTools(baseTool),
		WithInterceptor(NopInterceptor{}),
	)
	if err != nil {
		t.Fatal(err)
	}

	base := applyAgentOptions(defaultAgentOptions(), agent.opts...)
	derived := applyAgentOptions(base,
		WithModel("run-model"),
		WithInstruction("run"),
		WithTools(runTool),
		WithInterceptor(NopInterceptor{}),
	)
	if derived.Model != "run-model" ||
		derived.Instruction != "run" ||
		len(derived.Interceptors) != 2 ||
		len(derived.Tools) != 1 ||
		derived.Tools[0].Name != "run" {
		t.Fatalf("unexpected derived options: %+v", derived)
	}
	if base.Model != "base-model" ||
		base.Instruction != "base" ||
		len(base.Interceptors) != 1 ||
		len(base.Tools) != 1 ||
		base.Tools[0].Name != "base" {
		t.Fatalf("base options were mutated: %+v", base)
	}
}

func approvalCall() *types.MessageToolCall {
	return &types.MessageToolCall{
		ID: "approval-call", Name: "approval", Arguments: "{}",
	}
}

func pendingApprovalHistory() []*Event {
	assistant := types.NewMessage(types.MessageRoleAssistant)
	assistant.Parts = append(
		assistant.Parts,
		&types.MessagePart{ToolCall: approvalCall()},
	)
	return []*Event{
		initialUserEvent("run"),
		NewEventMessage(assistant),
		NewEventHumanRequest(&EventHumanRequest{
			ID:       "approval-request",
			ToolCall: approvalCall(),
		}),
	}
}

func TestHistoryIsRestoredOnlyAndInputIsEmitted(t *testing.T) {
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			if len(messages) != 2 ||
				messages[0].Text() != "old" ||
				messages[1].Text() != "new" {
				t.Fatalf("unexpected history: %+v", messages)
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)

	var sinkTexts []string
	resultSession, err := agent.build(
		[]*Event{initialUserEvent("old")},
		WithEventSink(func(_ context.Context, event *Event) error {
			if event.Message != nil {
				sinkTexts = append(sinkTexts, event.Message.Text())
			}
			return nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := resultSession.Run(context.Background(), NewUserInput("new"))
	if err != nil {
		t.Fatal(err)
	}
	if len(sinkTexts) != 2 || sinkTexts[0] != "new" || sinkTexts[1] != "done" {
		t.Fatalf("history was replayed to the sink: %+v", sinkTexts)
	}
	if len(result.Events) != 5 || result.Events[1].Message == nil {
		t.Fatalf("input was not part of the session output: %+v", result.Events)
	}
}

func TestInputHumanResponseResumesPendingRequest(t *testing.T) {
	tool := MustNewTool(
		"approval",
		"",
		func(ctx ToolContext, _ *emptyToolParam) (any, error) {
			if ctx.HumanRequest() == nil || ctx.HumanResponse() == nil {
				return nil, errors.New("missing restored human context")
			}
			return ctx.HumanResponse().Result["answer"], nil
		},
	)
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			found := false
			for _, message := range messages {
				for _, part := range message.Parts {
					if part.ToolResult != nil &&
						part.ToolResult.ID == "approval-call" &&
						part.ToolResult.Result == "approved" {
						found = true
					}
				}
			}
			if !found {
				t.Fatalf("input response did not resume the tool: %+v", messages)
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
		tool,
	)

	resultSession, err := agent.build(pendingApprovalHistory())
	if err != nil {
		t.Fatal(err)
	}
	result, err := resultSession.Run(context.Background(), NewHumanResponseInput(
		"approval-request",
		map[string]any{"answer": "approved"},
	))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Events) < 2 || result.Events[1].HumanResponse == nil {
		t.Fatalf("human response was not part of the session output: %+v", result.Events)
	}
	if result.Turns != 1 || result.StopReason != StopDone {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestInputMessagesFollowResumedToolResults(t *testing.T) {
	tool := MustNewTool(
		"approval",
		"",
		func(ctx ToolContext, _ *emptyToolParam) (any, error) {
			return ctx.HumanResponse().Result["answer"], nil
		},
	)
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			roles := make([]types.MessageRole, 0, len(messages))
			for _, message := range messages {
				roles = append(roles, message.Role)
			}
			if len(roles) != 4 ||
				roles[2] != types.MessageRoleTool ||
				roles[3] != types.MessageRoleUser ||
				messages[3].Text() != "and then this" {
				t.Fatalf("unexpected history order: %+v", roles)
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
		tool,
	)

	input := NewHumanResponseInput(
		"approval-request",
		map[string]any{"answer": "approved"},
	).WithMessages(types.NewTextMessage(types.MessageRoleUser, "and then this"))

	resultSession, err := agent.build(pendingApprovalHistory())
	if err != nil {
		t.Fatal(err)
	}
	result, err := resultSession.Run(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if result.Turns != 1 || result.StopReason != StopDone {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestInputValidation(t *testing.T) {
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			return nil, errors.New("must not be called")
		},
	)

	delta := types.NewTextMessage(types.MessageRoleUser, "partial")
	delta.Delta = true
	session, err := agent.build(nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Run(
		context.Background(),
		NewInput(delta),
	); err == nil || !strings.Contains(err.Error(), "delta") {
		t.Fatalf("expected delta input error, got %v", err)
	}

	if _, err := session.Run(
		context.Background(),
		NewHumanResponseInput("", nil),
	); err == nil || !strings.Contains(err.Error(), "request id is empty") {
		t.Fatalf("expected empty request id error, got %v", err)
	}

	for _, input := range []*Input{
		{State: map[string]string{"": "value"}},
		{StateDelta: map[string]string{"": "value"}},
	} {
		if err := input.Validate(); err == nil ||
			!strings.Contains(err.Error(), "state key is empty") {
			t.Fatalf("expected invalid state key error, got %v", err)
		}
	}

	s, err := newSession(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	turnCtx := newTurnContext(context.Background(), s)
	if err := turnCtx.Append(NewEventState(map[string]string{"": "value"}, true)); err == nil ||
		!strings.Contains(err.Error(), "state key is empty") {
		t.Fatalf("expected tool state key error, got %v", err)
	}
	if _, err := agent.build(
		[]*Event{
			NewEventState(map[string]string{"": "value"}, false),
		},
	); err == nil || !strings.Contains(err.Error(), "state key is empty") {
		t.Fatalf("expected history state key error, got %v", err)
	}
}

// A duplicate response is kept as its own event: rejecting it is the job of
// session.append, which sees the whole event stream.
func TestInputKeepsDuplicateHumanResponse(t *testing.T) {
	input := NewHumanResponseInput("request", map[string]any{"answer": "first"}).
		WithHumanResponse("request", map[string]any{"answer": "second"}).
		WithHumanResponse("other", nil)

	events, err := input.events()
	if err != nil {
		t.Fatalf("duplicate response must not fail the input: %v", err)
	}
	if len(events) != 3 {
		t.Fatalf("unexpected events: %+v", events)
	}
	if events[0].HumanResponse.ID != "request" ||
		events[0].HumanResponse.Result["answer"] != "first" {
		t.Fatalf("unexpected first response: %+v", events[0].HumanResponse)
	}
	if events[1].HumanResponse.ID != "request" ||
		events[1].HumanResponse.Result["answer"] != "second" {
		t.Fatalf("unexpected second response: %+v", events[1].HumanResponse)
	}
	if events[2].HumanResponse.ID != "other" {
		t.Fatalf("unexpected third response: %+v", events[2].HumanResponse)
	}
}

func TestLoopErrorEndsSessionRun(t *testing.T) {
	loopErr := errors.New("loop rejected the turn")
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)

	resultSession, err := agent.build(
		nil,
		WithLoop(func(TurnContext, *TurnResult) (LoopAction, error) {
			return LoopStop, loopErr
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := resultSession.Run(context.Background(), NewUserInput("run"))
	if !errors.Is(err, loopErr) {
		t.Fatalf("expected the loop error, got %v", err)
	}
	if result.StopReason != StopError {
		t.Fatalf("unexpected result: %+v", result)
	}
}

// An empty input is rejected before it reaches the inbox, so this holds
// whether or not the session is still running.
func TestSendRejectsEmptyInput(t *testing.T) {
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)
	ctx, cancel := context.WithCancel(context.Background())
	runnerSession, err := agent.build(nil)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := runnerSession.Start(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(nil); err == nil {
		t.Fatal("expected empty input error")
	}
	if _, err := runner.Enqueue(new(Input)); err == nil {
		t.Fatal("expected empty input error")
	}
	cancel()
	if _, err := runner.Wait(); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected cancellation, got %v", err)
	}
}

func TestInputEventsOrderAndOwnership(t *testing.T) {
	message := types.NewTextMessage(types.MessageRoleUser, "before")
	input := new(Input).
		WithState(map[string]string{"key": "value"}, true).
		WithHumanResponse("request", map[string]any{"answer": "yes"}).
		WithMessages(message)
	message.Parts[0].Text.Text = "after"

	events, err := input.events()
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 3 ||
		events[0].State == nil ||
		events[1].HumanResponse == nil ||
		events[2].Message == nil {
		t.Fatalf("unexpected input events: %+v", events)
	}
	if events[2].Message.Text() != "before" {
		t.Fatalf("input retained the caller message: %+v", events[2])
	}
	if events[0].ID == "" || events[1].ID == "" || events[2].ID == "" {
		t.Fatalf("input events are missing ids: %+v", events)
	}
}

func TestEmptyStateReplacementClearsSessionState(t *testing.T) {
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)

	resultSession, err := agent.build(
		[]*Event{
			NewEventState(map[string]string{"old": "value"}, false),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := resultSession.Run(context.Background(), &Input{State: map[string]string{}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.State) != 0 {
		t.Fatalf("empty replacement did not clear state: %+v", result.State)
	}
	if len(result.Events) < 2 || result.Events[1].State == nil ||
		result.Events[1].State.Patch {
		t.Fatalf("missing full state replacement event: %+v", result.Events)
	}
}

func TestEventsRoundTripAcrossRuns(t *testing.T) {
	var approvals atomic.Int32
	tool := MustNewTool(
		"approval",
		"",
		func(ctx ToolContext, _ *emptyToolParam) (any, error) {
			approvals.Add(1)
			if ctx.HumanResponse() == nil {
				return &ToolResult{AskHuman: &ToolAskHuman{}}, nil
			}
			return ctx.HumanResponse().Result["answer"], nil
		},
	)

	var mu sync.Mutex
	var histories [][]string
	turn := 0
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			mu.Lock()
			texts := make([]string, 0, len(messages))
			for _, message := range messages {
				texts = append(texts, string(message.Role)+":"+message.Text())
			}
			histories = append(histories, texts)
			turn++
			current := turn
			mu.Unlock()
			if current == 1 {
				message := types.NewMessage(types.MessageRoleAssistant)
				message.Parts = append(message.Parts, &types.MessagePart{
					ToolCall: &types.MessageToolCall{
						ID: "approval-call", Name: "approval", Arguments: "{}",
					},
				})
				return &types.Completion{Message: message}, nil
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
		tool,
	)

	requests := make(chan *EventHumanRequest, 1)
	runnerSession, err := agent.build(
		nil,
		WithEventSink(func(_ context.Context, event *Event) error {
			if event.HumanRequest != nil {
				requests <- event.HumanRequest.Clone()
			}
			return nil
		}),
		WithLoop(stopAfterTurns(2)),
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := runnerSession.Start(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("start")); err != nil {
		t.Fatal(err)
	}
	request := <-requests
	if _, err := runner.Enqueue(NewHumanResponseInput(request.ID, map[string]any{"answer": "ok"})); err != nil {
		t.Fatal(err)
	}
	first, err := runner.Wait()
	if err != nil {
		t.Fatal(err)
	}

	// The caller stores nothing but the events the session produced.
	stored := first.Events

	secondSession, err := agent.build(stored)
	if err != nil {
		t.Fatal(err)
	}
	second, err := secondSession.Run(context.Background(), NewUserInput("more"))
	if err != nil {
		t.Fatal(err)
	}
	stored = append(stored, second.Events...)

	if approvals.Load() != 2 {
		t.Fatalf("human tool ran %d times, want 2 in the first run only", approvals.Load())
	}
	mu.Lock()
	defer mu.Unlock()
	if len(histories) != 3 {
		t.Fatalf("unexpected turns: %+v", histories)
	}
	replayed := histories[2]
	if len(replayed) != 5 ||
		replayed[0] != "user:start" ||
		replayed[2] != "tool:" ||
		replayed[3] != "assistant:done" ||
		replayed[4] != "user:more" {
		t.Fatalf("replayed history is not equal to the first run: %+v", replayed)
	}

	if len(stored) != len(first.Events)+len(second.Events) {
		t.Fatalf("stored events are not append only: %d", len(stored))
	}
}

func TestToolResultAcceptsValueAndPointer(t *testing.T) {
	call := &types.MessageToolCall{ID: "call-1", Name: "tool"}
	instructed := ToolResult{Result: &ToolOutput{
		Value:       "ok",
		Instruction: map[string]string{"format": "short"},
	}}
	human := ToolResult{AskHuman: &ToolAskHuman{Arguments: map[string]any{"a": 1}}}
	cases := []struct {
		name   string
		result any
		want   string
	}{
		{"human value", human, "human"},
		{"human pointer", &human, "human"},
		{"instruction value", instructed, "instructed"},
		{"instruction pointer", &instructed, "instructed"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			events, err := toolRunEvents([]*toolRun{{call: call, result: tc.result}})
			if err != nil {
				t.Fatal(err)
			}
			switch tc.want {
			case "human":
				if len(events) != 1 || events[0].HumanRequest == nil ||
					events[0].HumanRequest.Arguments["a"] != 1 {
					t.Fatalf("expected one human request, got %+v", events)
				}
			case "instructed":
				if len(events) != 1 || events[0].Message == nil {
					t.Fatalf("expected one tool message, got %+v", events)
				}
				result := events[0].Message.Parts[0].ToolResult
				if result == nil || result.Result != "ok" ||
					len(result.Instruction["format"]) != 1 ||
					result.Instruction["format"][0] != "short" {
					t.Fatalf("unexpected tool result: %+v", result)
				}
			}
		})
	}
}

func TestNilToolResultIsRejected(t *testing.T) {
	call := &types.MessageToolCall{ID: "call-1", Name: "tool"}
	if _, err := toolRunEvents([]*toolRun{{call: call, result: (*ToolResult)(nil)}}); err == nil {
		t.Fatal("nil *ToolResult was accepted")
	}
}

func TestToolResultMustSetExactlyOne(t *testing.T) {
	call := &types.MessageToolCall{ID: "call-1", Name: "tool"}
	for name, result := range map[string]ToolResult{
		"neither": {},
		"both":    {Result: &ToolOutput{Value: "ok"}, AskHuman: &ToolAskHuman{}},
	} {
		if _, err := toolRunEvents([]*toolRun{{call: call, result: result}}); err == nil {
			t.Fatalf("ToolResult with %s of Result and AskHuman was accepted", name)
		}
	}
}

func TestResumedToolMustAnswer(t *testing.T) {
	call := &types.MessageToolCall{ID: "call-1", Name: "tool"}
	result := ToolResult{AskHuman: &ToolAskHuman{}}
	if _, err := toolRunEvents([]*toolRun{{call: call, result: result, resumed: true}}); err == nil {
		t.Fatal("a resumed tool asked a human again")
	}
}

func TestToolWithoutInvokerIsRejected(t *testing.T) {
	// A call without a result would leave the history unanswerable, so a
	// tool must always be able to answer.
	_, err := NewAgent("test", "", NewModels(), nil,
		WithModel("fake"),
		WithTools(&Tool{Name: "silent"}),
	)
	if err == nil || !strings.Contains(err.Error(), "invoker is nil") {
		t.Fatalf("expected an invoker error, got %v", err)
	}
}

func TestRunnerInterjectsAtTurnBoundaryAndQueuesPerTask(t *testing.T) {
	firstEntered := make(chan struct{})
	releaseFirst := make(chan struct{})
	var turns atomic.Int32
	echo := MustNewTool(
		"echo",
		"",
		func(ToolContext, *emptyToolParam) (any, error) { return "echoed", nil },
	)
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			if turns.Add(1) == 1 {
				close(firstEntered)
				<-releaseFirst
				// The first task wants another turn: it called a tool.
				message := types.NewMessage(types.MessageRoleAssistant)
				message.Parts = append(message.Parts, &types.MessagePart{
					ToolCall: &types.MessageToolCall{ID: "call", Name: "echo", Arguments: "{}"},
				})
				return &types.Completion{Message: message}, nil
			}
			input := messages[len(messages)-1].Text()
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "re:"+input),
			}, nil
		},
		echo,
	)

	// trace records the sink stream: assistant replies and status changes.
	var mu sync.Mutex
	var trace []string
	session, err := agent.build(
		nil,
		WithEventSink(func(_ context.Context, event *Event) error {
			mu.Lock()
			defer mu.Unlock()
			switch {
			case event.Status != nil:
				trace = append(trace, string(event.Status.Status)+string(event.Status.StopReason))
			case event.Message != nil && event.Message.Role == types.MessageRoleAssistant &&
				event.Message.Text() != "":
				trace = append(trace, event.Message.Text())
			}
			return nil
		}),
		WithLoop(stopAfterTurns(4)),
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := session.Start(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("first")); err != nil {
		t.Fatal(err)
	}
	<-firstEntered

	// While the first turn runs: two queued inputs, then one interjection that
	// arrives last but runs first.
	for _, input := range []*Input{
		NewUserInput("q1"),
		NewUserInput("q2"),
	} {
		if _, err := runner.Enqueue(input); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := runner.Interject(NewUserInput("steer")); err != nil {
		t.Fatal(err)
	}
	if pending := runner.Pending(); len(pending) != 3 ||
		pending[0].Messages[0].Text() != "steer" {
		t.Fatalf("unexpected pending inputs: %+v", pending)
	}
	close(releaseFirst)

	result, err := runner.Wait()
	if err != nil {
		t.Fatal(err)
	}
	if result.Turns != 4 {
		t.Fatalf("unexpected turns: %+v", result)
	}

	mu.Lock()
	defer mu.Unlock()
	// The interjection ends the first task at its turn boundary although Loop
	// would continue it; then it runs as a task of its own, and each queued
	// input after it.
	want := []string{
		"running", "waitinginterjected",
		"running", "re:steer", "waitingdone",
		"running", "re:q1", "waitingdone",
		"running", "re:q2", "stoppeddone",
	}
	if strings.Join(trace, ",") != strings.Join(want, ",") {
		t.Fatalf("unexpected trace:\n got %v\nwant %v", trace, want)
	}
}

func TestRunnerLoopStopLeavesInterjectionsPending(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			close(entered)
			<-release
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)
	session, err := agent.build(
		nil,
		WithLoop(func(TurnContext, *TurnResult) (LoopAction, error) {
			return LoopStop, nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := session.Start(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("run")); err != nil {
		t.Fatal(err)
	}
	<-entered
	if _, err := runner.Interject(NewUserInput("too late")); err != nil {
		t.Fatal(err)
	}
	close(release)

	result, err := runner.Wait()
	if err != nil {
		t.Fatal(err)
	}
	last := result.Events[len(result.Events)-1]
	if last.Status == nil || last.Status.Status != StatusStopped ||
		last.Status.StopReason != StopDone {
		t.Fatalf("expected a done stopped status last, got %+v", last)
	}
	pending := runner.Pending()
	if len(pending) != 1 || pending[0].Messages[0].Text() != "too late" {
		t.Fatalf("unprocessed interjection was lost: %+v", pending)
	}
	if _, err := runner.Interject(NewUserInput("closed")); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected session closed, got %v", err)
	}
}

func TestOnlyAnswerableInputStartsModelTurn(t *testing.T) {
	toolMessage := func(final bool) *types.Message {
		message := types.NewMessage(types.MessageRoleTool)
		message.Parts = append(message.Parts, &types.MessagePart{
			ToolResult: &types.MessageToolResult{
				ID: "call", Name: "tool", Result: "ok", Final: final,
			},
		})
		return message
	}
	cases := []struct {
		name  string
		input *Input
		calls int32
	}{
		{"state only", new(Input).WithState(map[string]string{"k": "v"}, true), 0},
		{"assistant message", NewInput(types.NewTextMessage(types.MessageRoleAssistant, "prefill")), 0},
		{"final tool result", NewInput(toolMessage(true)), 0},
		{"tool result", NewInput(toolMessage(false)), 1},
		{"system message", NewInput(types.NewTextMessage(types.MessageRoleSystem, "note")), 1},
		{"user message", NewUserInput("hi"), 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			agent := newRunTestAgent(
				t,
				func(LLMContext) (*types.Completion, error) {
					calls.Add(1)
					return &types.Completion{
						Message: types.NewTextMessage(types.MessageRoleAssistant, "answer"),
					}, nil
				},
			)
			session, err := agent.build(nil)
			if err != nil {
				t.Fatal(err)
			}
			result, err := session.Run(context.Background(), tc.input)
			if err != nil {
				t.Fatal(err)
			}
			if calls.Load() != tc.calls || result.Turns != int(tc.calls) {
				t.Fatalf("model called %d times over %d turns, want %d",
					calls.Load(), result.Turns, tc.calls)
			}
			last := result.Events[len(result.Events)-1]
			if last.Status == nil || last.Status.Status != StatusWaiting ||
				last.Status.Turns != int(tc.calls) {
				t.Fatalf("expected a waiting status last, got %+v", last)
			}
		})
	}
}

func TestFinalResumedHumanToolDoesNotCallModel(t *testing.T) {
	humanTool := MustNewTool(
		"approval",
		"",
		func(ctx ToolContext, _ *emptyToolParam) (any, error) {
			if ctx.HumanResponse() == nil {
				return &ToolResult{AskHuman: &ToolAskHuman{}}, nil
			}
			return &ToolResult{Result: &ToolOutput{Value: "approved", Final: true}}, nil
		},
	)
	var calls atomic.Int32
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			calls.Add(1)
			message := types.NewMessage(types.MessageRoleAssistant)
			message.Parts = append(message.Parts, &types.MessagePart{
				ToolCall: &types.MessageToolCall{
					ID: "approval-call", Name: "approval", Arguments: "{}",
				},
			})
			return &types.Completion{Message: message}, nil
		},
		humanTool,
	)
	session, err := agent.build(nil)
	if err != nil {
		t.Fatal(err)
	}
	first, err := session.Run(context.Background(), NewUserInput("delete it"))
	if err != nil {
		t.Fatal(err)
	}
	var requestID string
	for _, event := range first.Events {
		if event.HumanRequest != nil {
			requestID = event.HumanRequest.ID
		}
	}
	if requestID == "" || calls.Load() != 1 {
		t.Fatalf("expected one model call raising a human request: %+v", first.Events)
	}

	second, err := session.Run(
		context.Background(),
		NewHumanResponseInput(requestID, map[string]any{"approved": true}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 || second.Turns != 0 {
		t.Fatalf("a final resumed result must not call the model: calls=%d %+v",
			calls.Load(), second)
	}
	var resumed *types.MessageToolResult
	for _, event := range second.Events {
		if event.Message != nil && event.Message.Role == types.MessageRoleTool {
			resumed = event.Message.Parts[0].ToolResult
		}
	}
	if resumed == nil || resumed.Result != "approved" || !resumed.Final {
		t.Fatalf("resumed tool result missing: %+v", second.Events)
	}
}

func TestRejectedInputEndsTheTaskAndKeepsTheAgent(t *testing.T) {
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)
	resultSession, err := agent.build(nil)
	if err != nil {
		t.Fatal(err)
	}

	// The input is one batch: its state patch must not land either.
	input := NewHumanResponseInput("unknown", nil).
		WithState(map[string]string{"key": "value"}, true)
	result, err := resultSession.Run(context.Background(), input)
	if !errors.Is(err, ErrInputRejected) {
		t.Fatalf("expected ErrInputRejected, got %v", err)
	}
	if result.StopReason != StopRejected || len(result.State) != 0 {
		t.Fatalf("unexpected result: %+v", result)
	}
	if len(result.Events) != 2 ||
		result.Events[0].Status == nil || result.Events[0].Status.Status != StatusRunning ||
		result.Events[1].Status == nil || result.Events[1].Status.StopReason != StopRejected {
		t.Fatalf("a rejected task must append only its status events: %+v", result.Events)
	}

	second, err := resultSession.Run(context.Background(), NewUserInput("run"))
	if err != nil {
		t.Fatalf("the agent must stay usable after a rejected input: %v", err)
	}
	if reply := second.Reply(); reply == nil || reply.Text() != "done" {
		t.Fatalf("unexpected reply: %+v", reply)
	}
}

func TestRunnerServesAfterARejectedInput(t *testing.T) {
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			return &types.Completion{
				Message: types.NewTextMessage(
					types.MessageRoleAssistant,
					"re:"+messages[len(messages)-1].Text(),
				),
			}, nil
		},
	)
	var mu sync.Mutex
	var reasons []StopReason
	runnerSession, err := agent.build(
		nil,
		WithLoop(stopAfterTurns(1)),
		WithEventSink(func(_ context.Context, event *Event) error {
			if event.Status != nil && event.Status.Status != StatusRunning {
				mu.Lock()
				reasons = append(reasons, event.Status.StopReason)
				mu.Unlock()
			}
			return nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := runnerSession.Start(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewHumanResponseInput("unknown", nil)); err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("run")); err != nil {
		t.Fatal(err)
	}

	result, err := runner.Wait()
	if err != nil {
		t.Fatal(err)
	}
	if reply := result.Reply(); reply == nil || reply.Text() != "re:run" {
		t.Fatalf("the queued input was not served after the rejected one: %+v", result.Events)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(reasons) != 2 || reasons[0] != StopRejected || reasons[1] != StopDone {
		t.Fatalf("unexpected stop reasons: %v", reasons)
	}
}

func TestToolErrorWinsOverAskHuman(t *testing.T) {
	call := &types.MessageToolCall{ID: "call-1", Name: "tool"}
	events, err := toolRunEvents([]*toolRun{{
		call:   call,
		result: ToolResult{AskHuman: &ToolAskHuman{}},
		err:    errors.New("boom"),
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].Message == nil || len(events[0].Message.Parts) != 1 {
		t.Fatalf("expected one tool message, got %+v", events)
	}
	result := events[0].Message.Parts[0].ToolResult
	if result == nil || result.ID != call.ID || result.Error != "boom" {
		t.Fatalf("the error was not recorded: %+v", result)
	}
}

func TestToolCallInterceptorPanicAnswersTheCall(t *testing.T) {
	tool := MustNewTool(
		"boom",
		"",
		func(ToolContext, *emptyToolParam) (any, error) { return "ok", nil },
	)
	turn := 0
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			turn++
			if turn > 1 {
				return &types.Completion{
					Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
				}, nil
			}
			message := types.NewMessage(types.MessageRoleAssistant)
			message.Parts = append(message.Parts, &types.MessagePart{
				ToolCall: &types.MessageToolCall{ID: "call-1", Name: "boom", Arguments: "{}"},
			})
			return &types.Completion{Message: message}, nil
		},
		tool,
	)
	resultSession, err := agent.build(
		nil,
		WithInterceptor(toolCallInterceptorFunc(func(ToolCallHandler) ToolCallHandler {
			return func(ToolContext) (any, error) { panic("interceptor exploded") }
		})),
	)
	if err != nil {
		t.Fatal(err)
	}

	result, err := resultSession.Run(context.Background(), NewUserInput("run"))
	if err != nil {
		t.Fatalf("a panic in the handler chain must answer the call, not fail the run: %v", err)
	}
	var toolResult *types.MessageToolResult
	for _, event := range result.Events {
		if event.Message == nil {
			continue
		}
		for _, part := range event.Message.Parts {
			if part.ToolResult != nil && part.ToolResult.ID == "call-1" {
				toolResult = part.ToolResult
			}
		}
	}
	if toolResult == nil || !strings.Contains(toolResult.Error, "interceptor exploded") {
		t.Fatalf("the panic was not recorded as the tool result: %+v", toolResult)
	}
	if result.StopReason != StopDone || result.Turns != 2 {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestWithStateNilClearsTheState(t *testing.T) {
	events, err := NewUserInput("reset").WithState(nil, false).events()
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 2 || events[0].State == nil ||
		events[0].State.Patch || len(events[0].State.State) != 0 {
		t.Fatalf("a nil state must become an empty replacement: %+v", events)
	}
}

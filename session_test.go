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

func newRunTestAgent(t *testing.T, handler LLMHandler, tools ...*Tool) *Agent {
	t.Helper()
	agent, err := NewAgent(
		"test",
		"",
		NewModels(),
		WithModel("fake"),
		WithTools(tools...),
		WithLLMInterceptor(func(LLMHandler) LLMHandler { return handler }),
	)
	if err != nil {
		t.Fatal(err)
	}
	return agent
}

func initialUserEvent(text string) *Event {
	return NewEventMessage(types.NewTextMessage(types.MessageRoleUser, text))
}

func TestRunReturnsEventsAndRestoredHistory(t *testing.T) {
	agent := newRunTestAgent(
		t,
		func(context.Context, *SessionTurn) (*types.Completion, error) {
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

	result, err := agent.Run(
		context.Background(),
		nil,
		NewUserInput("question"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != StopDone || result.Turns != 1 {
		t.Fatalf("unexpected result: %+v", result)
	}
	// A turn reports what it produced before what it cost: the usage carries the
	// completion tokens of the message right before it.
	if len(result.Events) != 3 ||
		result.Events[0].Message == nil ||
		result.Events[1].Message == nil ||
		result.Events[2].Usage == nil {
		t.Fatalf("unexpected output events: %+v", result.Events)
	}
	// Usage is context size, not a session total, so it is reported as an event.
	if result.Events[2].Usage.Usage.TotalTokens != 5 {
		t.Fatalf("unexpected usage: %+v", result.Events[2].Usage)
	}
}

func TestSessionSharesImmutableEventsAcrossObservers(t *testing.T) {
	var sinkEvents []*Event
	var loopEvents []*Event
	agent := newRunTestAgent(
		t,
		func(context.Context, *SessionTurn) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "answer"),
			}, nil
		},
	)
	result, err := agent.Run(
		context.Background(),
		nil,
		NewUserInput("question"),
		WithEventSink(func(_ context.Context, event *Event) error {
			sinkEvents = append(sinkEvents, event)
			return nil
		}),
		WithLoop(func(_ context.Context, turn *SessionTurnResult) (LoopAction, error) {
			loopEvents = append([]*Event(nil), turn.Events...)
			return LoopWait, nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(sinkEvents) != len(result.Events) ||
		len(loopEvents) != len(result.Events) {
		t.Fatalf("observers saw different events: %d %d %d", len(sinkEvents), len(loopEvents), len(result.Events))
	}
	for index := range result.Events {
		if sinkEvents[index] != result.Events[index] ||
			loopEvents[index] != result.Events[index] {
			t.Fatalf("event %d was cloned between observers", index)
		}
	}
}

func TestSessionRunCanBeRepeatedAndRetainsHistory(t *testing.T) {
	var histories [][]string
	agent := newRunTestAgent(
		t,
		func(_ context.Context, turn *SessionTurn) (*types.Completion, error) {
			texts := make([]string, 0, len(turn.History))
			for _, message := range turn.History {
				texts = append(texts, message.Text())
			}
			histories = append(histories, texts)
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "answer"),
			}, nil
		},
	)
	session, err := NewSession(agent, nil)
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

func TestSessionRejectsConcurrentRunAndStart(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	agent := newRunTestAgent(
		t,
		func(context.Context, *SessionTurn) (*types.Completion, error) {
			close(started)
			<-release
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)
	session, err := NewSession(agent, nil)
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
	); !errors.Is(err, ErrSessionBusy) {
		t.Fatalf("expected concurrent Run to be busy, got %v", err)
	}
	if _, err := session.Start(context.Background()); !errors.Is(err, ErrSessionBusy) {
		t.Fatalf("expected Start during Run to be busy, got %v", err)
	}

	close(release)
	if err := <-firstDone; err != nil {
		t.Fatal(err)
	}
}

func TestSessionStartWaitsForInput(t *testing.T) {
	called := make(chan struct{}, 1)
	agent := newRunTestAgent(
		t,
		func(context.Context, *SessionTurn) (*types.Completion, error) {
			called <- struct{}{}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)
	runner, err := agent.Start(
		context.Background(),
		nil,
		WithLoop(func(context.Context, *SessionTurnResult) (LoopAction, error) {
			return LoopStop, nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	select {
	case <-called:
		t.Fatal("model was called before the first input")
	case <-time.After(20 * time.Millisecond):
	}
	if err := runner.Send(context.Background(), NewUserInput("run")); err != nil {
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
		func(_ context.Context, turn *SessionTurn) (*types.Completion, error) {
			mu.Lock()
			calls++
			call := calls
			texts := make([]string, 0, len(turn.History))
			for _, message := range turn.History {
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

	runner, err := agent.Start(
		context.Background(),
		nil,
		WithMaxTurn(1),
		WithLoop(func(ctx context.Context, turn *SessionTurnResult) (LoopAction, error) {
			if turn.Index >= 1 {
				return LoopStop, nil
			}
			return ReactLoop(ctx, turn)
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := runner.Send(context.Background(), NewUserInput("initial")); err != nil {
		t.Fatal(err)
	}
	<-firstEntered

	sendDone := make(chan error, 1)
	go func() {
		sendDone <- runner.Send(
			context.Background(),
			NewUserInput("follow-up"),
		)
	}()
	time.Sleep(10 * time.Millisecond)
	close(releaseFirst)

	if err := <-sendDone; err != nil {
		t.Fatal(err)
	}
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
		func(context.Context, *SessionTurn) (*types.Completion, error) {
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

	result, err := agent.Run(
		context.Background(),
		nil,
		NewUserInput("run"),
		WithMaxTurn(2),
	)
	if !errors.Is(err, ErrMaxTurnExceeded) {
		t.Fatalf("expected max turn error, got %v", err)
	}
	if result.StopReason != StopMaxTurn || result.Turns != 2 {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestInputRejectedBySinkDoesNotReachModel(t *testing.T) {
	sinkErr := errors.New("fenced")
	called := atomic.Bool{}
	agent := newRunTestAgent(
		t,
		func(context.Context, *SessionTurn) (*types.Completion, error) {
			called.Store(true)
			return nil, errors.New("must not be called")
		},
	)

	result, err := agent.Run(
		context.Background(),
		nil,
		NewUserInput("must persist first"),
		WithEventSink(func(_ context.Context, event *Event) error {
			if event.Message != nil &&
				event.Message.Role == types.MessageRoleUser {
				return sinkErr
			}
			return nil
		}),
	)
	if !errors.Is(err, sinkErr) {
		t.Fatalf("expected sink error, got %v", err)
	}
	if called.Load() {
		t.Fatal("model must not see an input rejected by EventSink")
	}
	if len(result.Events) != 0 {
		t.Fatalf("rejected input leaked into result: %+v", result)
	}
}

func TestSessionSendAfterStopReturnsClosed(t *testing.T) {
	agent := newRunTestAgent(
		t,
		func(context.Context, *SessionTurn) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)
	runner, err := agent.Start(
		context.Background(),
		nil,
		WithLoop(func(context.Context, *SessionTurnResult) (LoopAction, error) {
			return LoopStop, nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := runner.Send(context.Background(), NewUserInput("run")); err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Wait(); err != nil {
		t.Fatal(err)
	}
	if err := runner.Send(
		context.Background(),
		NewUserInput("late"),
	); !errors.Is(err, ErrSessionClosed) {
		t.Fatalf("expected session closed, got %v", err)
	}
}

func TestSessionStartsOnlyOnce(t *testing.T) {
	agent := newRunTestAgent(
		t,
		func(context.Context, *SessionTurn) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)
	session, err := NewSession(agent, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	runner, err := session.Start(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Start(context.Background()); !errors.Is(err, ErrSessionStarted) {
		t.Fatalf("expected session already started, got %v", err)
	}
	cancel()
	if _, err := runner.Wait(); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected cancellation, got %v", err)
	}
}

func TestToolStatePatchesAreDeterministicAndVisibleNextTurn(t *testing.T) {
	var observed map[string]string
	writerA := MustNewTool(
		"writer_a",
		"",
		func(ctx ToolContext, _ *emptyToolParam) (any, error) {
			return "a", ctx.SetState(map[string]string{"shared": "a", "onlyA": "1"})
		},
	)
	writerB := MustNewTool(
		"writer_b",
		"",
		func(ctx ToolContext, _ *emptyToolParam) (any, error) {
			return "b", ctx.SetState(map[string]string{"shared": "b", "onlyB": "1"})
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
		func(context.Context, *SessionTurn) (*types.Completion, error) {
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

	var patchOrder []string
	result, err := agent.Run(
		context.Background(),
		nil,
		NewUserInput("run"),
		WithEventSink(func(_ context.Context, event *Event) error {
			if event.State != nil {
				patchOrder = append(
					patchOrder,
					event.State.State["shared"],
				)
			}
			return nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(patchOrder) != 2 ||
		patchOrder[0] != "a" ||
		patchOrder[1] != "b" {
		t.Fatalf("unexpected patch order: %+v", patchOrder)
	}
	if observed["onlyA"] != "1" || observed["onlyB"] != "1" {
		t.Fatalf("reader did not see prior state: %+v", observed)
	}
	if result.State["shared"] != "b" {
		t.Fatalf("unexpected final state: %+v", result.State)
	}
}

func TestRunEmitsCompressionEventBeforeCompletion(t *testing.T) {
	models := NewModels()
	models.models["fake"] = &Model{Name: "fake", MaxToken: 100}

	toolCallMessage := types.NewMessage(types.MessageRoleAssistant)
	toolCallMessage.Parts = append(toolCallMessage.Parts, &types.MessagePart{
		ToolCall: &types.MessageToolCall{
			ID: "call", Name: "long_tool", Arguments: "{}",
		},
	})
	toolResultMessage := types.NewToolResultMessage(
		"call",
		"long_tool",
		"a very long tool result that should be omitted",
		"",
	)
	history := []*Event{
		NewEventMessage(toolCallMessage),
		NewEventMessage(toolResultMessage),
		NewEventUsage(types.CompletionUsage{
			PromptTokens: 100,
			TotalTokens:  100,
		}),
	}

	agent, err := NewAgent(
		"test",
		"",
		models,
		WithModel("fake"),
		WithCompress(&CompressOptions{
			KeepLastTurn:      1,
			ToolResultLevel:   1,
			ToolResultReplace: "omitted",
		}),
		WithLLMInterceptor(func(LLMHandler) LLMHandler {
			return func(
				_ context.Context,
				turn *SessionTurn,
			) (*types.Completion, error) {
				found := false
				for _, message := range turn.History {
					for _, part := range message.Parts {
						if part.ToolResult != nil &&
							part.ToolResult.Result == "omitted" {
							found = true
						}
					}
				}
				if !found {
					t.Fatalf("compressed history was not used: %+v", turn.History)
				}
				return &types.Completion{
					Message: types.NewTextMessage(
						types.MessageRoleAssistant,
						"done",
					),
				}, nil
			}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	result, err := agent.Run(context.Background(), history, NewUserInput("latest"))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Events) < 3 ||
		result.Events[0].Message == nil ||
		result.Events[1].Compression == nil {
		t.Fatalf("compression did not follow the new input: %+v", result.Events)
	}
	compression := result.Events[1].Compression
	if compression.ToolResult == nil ||
		compression.ToolResult.Result != "omitted" {
		t.Fatalf("unexpected compression event: %+v", compression)
	}
	// Compressing shrinks the context, so the usage that decides the next
	// compression has to follow it down. The reduction is not stored in the
	// event: applying the compression to the history it covered derives it.
	compressedAt := []*types.Message{
		history[0].Message,
		history[1].Message,
		result.Events[0].Message,
	}
	if _, reduced, err := ApplyCompression(compressedAt, compression); err != nil {
		t.Fatal(err)
	} else if reduced <= 0 {
		t.Fatalf(
			"compression did not report how much it removed: %+v",
			compression,
		)
	}
	// The event after a compression is the completion of this turn, not a
	// separate usage event.
	if result.Events[2].Message == nil {
		t.Fatalf(
			"compression emitted a second event: %+v",
			result.Events[2],
		)
	}
}

func TestUsageIsTheLatestContextSizeNotATotal(t *testing.T) {
	tool := MustNewTool(
		"echo",
		"",
		func(ToolContext, *emptyToolParam) (any, error) { return "ok", nil },
	)
	turn := 0
	agent := newRunTestAgent(
		t,
		func(context.Context, *SessionTurn) (*types.Completion, error) {
			turn++
			if turn == 1 {
				message := types.NewMessage(types.MessageRoleAssistant)
				message.Parts = append(message.Parts, &types.MessagePart{
					ToolCall: &types.MessageToolCall{
						ID: "call", Name: "echo", Arguments: "{}",
					},
				})
				return &types.Completion{
					Message: message,
					Usage:   types.CompletionUsage{PromptTokens: 100, TotalTokens: 100},
				}, nil
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
				Usage:   types.CompletionUsage{PromptTokens: 150, TotalTokens: 150},
			}, nil
		},
		tool,
	)

	session, err := NewSession(agent, nil)
	if err != nil {
		t.Fatal(err)
	}
	result, err := session.Run(context.Background(), NewUserInput("run"))
	if err != nil {
		t.Fatal(err)
	}
	if result.Turns != 2 {
		t.Fatalf("unexpected result: %+v", result)
	}
	// A completion reports the whole context it consumed: 150, not 100 + 150.
	if session.contextTokens != 150 {
		t.Fatalf("usage was accumulated: %d", session.contextTokens)
	}
}

func TestHumanRequestCanResumeWithinSession(t *testing.T) {
	humanTool := MustNewTool(
		"approval",
		"",
		func(ctx ToolContext, _ *emptyToolParam) (any, error) {
			if ctx.HumanResponse() == nil {
				return &ToolResultHuman{
					Arguments: map[string]any{"question": "approve?"},
				}, nil
			}
			return ctx.HumanResponse().Result["answer"], nil
		},
	)

	turn := 0
	agent := newRunTestAgent(
		t,
		func(_ context.Context, turnInput *SessionTurn) (*types.Completion, error) {
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
			for _, message := range turnInput.History {
				for _, part := range message.Parts {
					if part.ToolResult != nil &&
						part.ToolResult.ID == "approval-call" &&
						part.ToolResult.Result == "yes" {
						foundResult = true
					}
				}
			}
			if !foundResult {
				t.Fatalf("human tool result missing: %+v", turnInput.History)
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
		humanTool,
	)

	requests := make(chan *EventHumanRequest, 1)
	runner, err := agent.Start(
		context.Background(),
		nil,
		WithEventSink(func(_ context.Context, event *Event) error {
			if event.HumanRequest != nil {
				requests <- event.HumanRequest.Clone()
			}
			return nil
		}),
		WithLoop(func(ctx context.Context, turn *SessionTurnResult) (LoopAction, error) {
			if turn.Index >= 1 {
				return LoopStop, nil
			}
			return ReactLoop(ctx, turn)
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := runner.Send(context.Background(), NewUserInput("run")); err != nil {
		t.Fatal(err)
	}

	request := <-requests
	if err := runner.Send(
		context.Background(),
		NewHumanResponseInput(
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
		func(_ context.Context, turn *SessionTurn) (*types.Completion, error) {
			found := false
			for _, message := range turn.History {
				for _, part := range message.Parts {
					if part.ToolResult != nil &&
						part.ToolResult.ID == call.ID &&
						part.ToolResult.Result == "restored" {
						found = true
					}
				}
			}
			if !found {
				t.Fatalf("restored tool result missing: %+v", turn.History)
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
		tool,
	)

	result, err := agent.Run(context.Background(), history, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Turns != 1 || result.StopReason != StopDone {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestRunContextCancellationStopsCompletion(t *testing.T) {
	started := make(chan struct{})
	agent := newRunTestAgent(
		t,
		func(ctx context.Context, _ *SessionTurn) (*types.Completion, error) {
			close(started)
			<-ctx.Done()
			return nil, context.Cause(ctx)
		},
	)
	ctx, cancel := context.WithCancelCause(context.Background())
	type runOutput struct {
		result *SessionResult
		err    error
	}
	done := make(chan runOutput, 1)
	go func() {
		result, err := agent.Run(ctx, nil, NewUserInput("run"))
		done <- runOutput{result: result, err: err}
	}()
	<-started
	cause := errors.New("stop now")
	cancel(cause)

	output := <-done
	if !errors.Is(output.err, cause) {
		t.Fatalf("expected cancellation cause, got %v", output.err)
	}
	if output.result.StopReason != StopCanceled ||
		!errors.Is(output.result.Cause, cause) {
		t.Fatalf("unexpected canceled result: %+v", output.result)
	}
}

func TestContextCancellationStopsHumanWait(t *testing.T) {
	humanTool := MustNewTool(
		"approval",
		"",
		func(ToolContext, *emptyToolParam) (any, error) {
			return &ToolResultHuman{}, nil
		},
	)
	agent := newRunTestAgent(
		t,
		func(context.Context, *SessionTurn) (*types.Completion, error) {
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
	runner, err := agent.Start(
		ctx,
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
	if err := runner.Send(context.Background(), NewUserInput("run")); err != nil {
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
		func(context.Context, *SessionTurn) (*types.Completion, error) {
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
	runner, err := agent.Start(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := runner.Send(context.Background(), NewUserInput("run")); err != nil {
		t.Fatal(err)
	}
	<-started
	cause := errors.New("tool canceled")
	cancel(cause)

	waited := make(chan struct{})
	var result *SessionResult
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

func TestSessionTurnExposesSafeStateAndAdaptableTools(t *testing.T) {
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

	agent, err := NewAgent(
		"test",
		"",
		models,
		WithModel("fake"),
		WithInstruction("original"),
		WithTools(keep, drop),
		WithLLMInterceptor(func(next LLMHandler) LLMHandler {
			return func(
				ctx context.Context,
				turn *SessionTurn,
			) (*types.Completion, error) {
				if value, ok := turn.GetState("locale"); !ok || value != "zh" {
					return nil, fmt.Errorf("unexpected turn state %q, %v", value, ok)
				}
				snapshot := turn.StateSnapshot()
				snapshot["locale"] = "mutated"
				if value, _ := turn.GetState("locale"); value != "zh" {
					return nil, fmt.Errorf("state snapshot mutated the turn")
				}

				adapted := turn.Clone()
				adapted.Instruction = "adapted"
				adapted.Tools = adapted.Tools[:1]
				return next(ctx, adapted)
			}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	result, err := agent.Run(
		context.Background(),
		[]*Event{
			NewEventState(map[string]string{"locale": "zh"}, false),
		},
		NewUserInput("run"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.State["locale"] != "zh" {
		t.Fatalf("turn state escaped into session state: %+v", result.State)
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
			invokerCallID = ctx.ToolCallID()
			if ctx.State()["intercepted"] != "yes" {
				return nil, fmt.Errorf("interceptor state is not visible to the tool")
			}
			return "ok", nil
		},
	)

	turn := 0
	agent := newRunTestAgent(
		t,
		func(context.Context, *SessionTurn) (*types.Completion, error) {
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

	result, err := agent.Run(
		context.Background(),
		[]*Event{
			NewEventState(map[string]string{"seed": "value"}, false),
		},
		NewUserInput("run"),
		WithToolCallInterceptor(func(next ToolCallHandler) ToolCallHandler {
			return func(ctx ToolContext, call *types.MessageToolCall) (any, error) {
				interceptorCallID = ctx.ToolCallID()
				if interceptorCallID != call.ID || ctx.State()["seed"] != "value" {
					return nil, fmt.Errorf("unexpected tool context")
				}
				if err := ctx.SetState(map[string]string{"intercepted": "yes"}); err != nil {
					return nil, err
				}
				return next(ctx, call)
			}
		}),
	)
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
	agent, err := NewAgent(
		"test",
		"",
		models,
		WithModel("fake"),
		WithStream(true),
	)
	if err != nil {
		t.Fatal(err)
	}

	var sinkMessages []*types.Message
	_, err = agent.Run(
		context.Background(),
		nil,
		NewUserInput("run"),
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
	agent, err := NewAgent(
		"test",
		"",
		NewModels(),
		WithModel("base-model"),
		WithInstruction("base"),
		WithTools(baseTool),
		WithCompress(&CompressOptions{SummaryLevel: 0.5}),
	)
	if err != nil {
		t.Fatal(err)
	}

	sessionOptions, err := agent.optionsForSession(
		WithModel("run-model"),
		WithInstruction("run"),
		WithTools(runTool),
		WithCompress(nil),
	)
	if err != nil {
		t.Fatal(err)
	}
	if sessionOptions.Model != "run-model" ||
		sessionOptions.Instruction != "run" ||
		sessionOptions.Compress != nil ||
		len(sessionOptions.Tools) != 1 ||
		sessionOptions.Tools[0].Name != "run" {
		t.Fatalf("unexpected session options: %+v", sessionOptions)
	}
	if agent.options.Model != "base-model" ||
		agent.options.Instruction != "base" ||
		agent.options.Compress == nil ||
		len(agent.options.Tools) != 1 ||
		agent.options.Tools[0].Name != "base" {
		t.Fatalf("agent options were mutated: %+v", agent.options)
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
		func(_ context.Context, turn *SessionTurn) (*types.Completion, error) {
			if len(turn.History) != 2 ||
				turn.History[0].Text() != "old" ||
				turn.History[1].Text() != "new" {
				t.Fatalf("unexpected history: %+v", turn.History)
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)

	var sinkTexts []string
	result, err := agent.Run(
		context.Background(),
		[]*Event{initialUserEvent("old")},
		NewUserInput("new"),
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
	if len(sinkTexts) != 2 || sinkTexts[0] != "new" || sinkTexts[1] != "done" {
		t.Fatalf("history was replayed to the sink: %+v", sinkTexts)
	}
	if len(result.Events) != 3 || result.Events[0].Message == nil {
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
		func(_ context.Context, turn *SessionTurn) (*types.Completion, error) {
			found := false
			for _, message := range turn.History {
				for _, part := range message.Parts {
					if part.ToolResult != nil &&
						part.ToolResult.ID == "approval-call" &&
						part.ToolResult.Result == "approved" {
						found = true
					}
				}
			}
			if !found {
				t.Fatalf("input response did not resume the tool: %+v", turn.History)
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
		tool,
	)

	result, err := agent.Run(
		context.Background(),
		pendingApprovalHistory(),
		NewHumanResponseInput(
			"approval-request",
			map[string]any{"answer": "approved"},
		),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Events) == 0 || result.Events[0].HumanResponse == nil {
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
		func(_ context.Context, turn *SessionTurn) (*types.Completion, error) {
			roles := make([]types.MessageRole, 0, len(turn.History))
			for _, message := range turn.History {
				roles = append(roles, message.Role)
			}
			if len(roles) != 4 ||
				roles[2] != types.MessageRoleTool ||
				roles[3] != types.MessageRoleUser ||
				turn.History[3].Text() != "and then this" {
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

	result, err := agent.Run(context.Background(), pendingApprovalHistory(), input)
	if err != nil {
		t.Fatal(err)
	}
	if result.Turns != 1 || result.StopReason != StopDone {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestSessionInputValidation(t *testing.T) {
	agent := newRunTestAgent(
		t,
		func(context.Context, *SessionTurn) (*types.Completion, error) {
			return nil, errors.New("must not be called")
		},
	)

	delta := types.NewTextMessage(types.MessageRoleUser, "partial")
	delta.Delta = true
	session, err := NewSession(agent, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Run(
		context.Background(),
		NewSessionInput(delta),
	); err == nil || !strings.Contains(err.Error(), "delta") {
		t.Fatalf("expected delta input error, got %v", err)
	}

	if _, err := session.Run(
		context.Background(),
		NewHumanResponseInput("", nil),
	); err == nil || !strings.Contains(err.Error(), "request id is empty") {
		t.Fatalf("expected empty request id error, got %v", err)
	}

	for _, input := range []*SessionInput{
		{State: map[string]string{"": "value"}},
		{StateDelta: map[string]string{"": "value"}},
	} {
		if err := input.Validate(); err == nil ||
			!strings.Contains(err.Error(), "state key is empty") {
			t.Fatalf("expected invalid state key error, got %v", err)
		}
	}

	toolCtx := &sessionToolContext{
		Context: context.Background(),
		state:   map[string]string{},
	}
	if err := toolCtx.SetState(map[string]string{"": "value"}); err == nil ||
		!strings.Contains(err.Error(), "state key is empty") {
		t.Fatalf("expected tool state key error, got %v", err)
	}
	if _, err := NewSession(
		agent,
		[]*Event{
			NewEventState(map[string]string{"": "value"}, false),
		},
	); err == nil || !strings.Contains(err.Error(), "state key is empty") {
		t.Fatalf("expected history state key error, got %v", err)
	}
}

// A duplicate response is kept as its own event: rejecting it is the job of
// addEvent, which sees the whole event log.
func TestSessionInputKeepsDuplicateHumanResponse(t *testing.T) {
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
		func(context.Context, *SessionTurn) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)

	result, err := agent.Run(
		context.Background(),
		nil,
		NewUserInput("run"),
		WithLoop(func(context.Context, *SessionTurnResult) (LoopAction, error) {
			return LoopStop, loopErr
		}),
	)
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
		func(context.Context, *SessionTurn) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)
	ctx, cancel := context.WithCancel(context.Background())
	runner, err := agent.Start(
		ctx,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := runner.Send(context.Background(), nil); err == nil {
		t.Fatal("expected empty input error")
	}
	if err := runner.Send(context.Background(), new(SessionInput)); err == nil {
		t.Fatal("expected empty input error")
	}
	cancel()
	if _, err := runner.Wait(); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected cancellation, got %v", err)
	}
}

func TestSessionInputEventsOrderAndOwnership(t *testing.T) {
	message := types.NewTextMessage(types.MessageRoleUser, "before")
	input := new(SessionInput).
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
		func(context.Context, *SessionTurn) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		},
	)

	result, err := agent.Run(
		context.Background(),
		[]*Event{
			NewEventState(map[string]string{"old": "value"}, false),
		},
		&SessionInput{State: map[string]string{}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.State) != 0 {
		t.Fatalf("empty replacement did not clear state: %+v", result.State)
	}
	if len(result.Events) == 0 || result.Events[0].State == nil ||
		result.Events[0].State.Patch {
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
				return &ToolResultHuman{}, nil
			}
			return ctx.HumanResponse().Result["answer"], nil
		},
	)

	var mu sync.Mutex
	var histories [][]string
	turn := 0
	agent := newRunTestAgent(
		t,
		func(_ context.Context, input *SessionTurn) (*types.Completion, error) {
			mu.Lock()
			texts := make([]string, 0, len(input.History))
			for _, message := range input.History {
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
	runner, err := agent.Start(
		context.Background(),
		nil,
		WithEventSink(func(_ context.Context, event *Event) error {
			if event.HumanRequest != nil {
				requests <- event.HumanRequest.Clone()
			}
			return nil
		}),
		WithLoop(func(ctx context.Context, turn *SessionTurnResult) (LoopAction, error) {
			if turn.Index >= 1 {
				return LoopStop, nil
			}
			return ReactLoop(ctx, turn)
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := runner.Send(context.Background(), NewUserInput("start")); err != nil {
		t.Fatal(err)
	}
	request := <-requests
	if err := runner.Send(
		context.Background(),
		NewHumanResponseInput(request.ID, map[string]any{"answer": "ok"}),
	); err != nil {
		t.Fatal(err)
	}
	first, err := runner.Wait()
	if err != nil {
		t.Fatal(err)
	}

	// The caller stores nothing but the events the session produced.
	stored := first.Events

	second, err := agent.Run(context.Background(), stored, NewUserInput("more"))
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

func TestCompressionEventRoundTripAcrossRuns(t *testing.T) {
	models := NewModels()
	models.models["fake"] = &Model{Name: "fake", MaxToken: 100}

	toolCall := types.NewMessage(types.MessageRoleAssistant)
	toolCall.Parts = append(toolCall.Parts, &types.MessagePart{
		ToolCall: &types.MessageToolCall{
			ID: "call", Name: "long_tool", Arguments: "{}",
		},
	})
	toolResult := types.NewToolResultMessage(
		"call",
		"long_tool",
		"a very long tool result that should be omitted",
		"",
	)

	var mu sync.Mutex
	var seen [][]string
	agent, err := NewAgent(
		"test",
		"",
		models,
		WithModel("fake"),
		WithCompress(&CompressOptions{
			KeepLastTurn:      1,
			ToolResultLevel:   1,
			ToolResultReplace: "omitted",
		}),
		WithLLMInterceptor(func(LLMHandler) LLMHandler {
			return func(_ context.Context, input *SessionTurn) (*types.Completion, error) {
				results := make([]string, 0)
				for _, message := range input.History {
					for _, part := range message.Parts {
						if part.ToolResult != nil {
							results = append(results, part.ToolResult.Result)
						}
					}
				}
				mu.Lock()
				seen = append(seen, results)
				mu.Unlock()
				return &types.Completion{
					Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
				}, nil
			}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	initialHistory := []*Event{
		NewEventMessage(toolCall),
		NewEventMessage(toolResult),
		NewEventUsage(types.CompletionUsage{
			PromptTokens: 100,
			TotalTokens:  100,
		}),
	}
	first, err := agent.Run(
		context.Background(),
		initialHistory,
		NewUserInput("first"),
	)
	if err != nil {
		t.Fatal(err)
	}
	compressions := 0
	for _, event := range first.Events {
		if event.Compression != nil {
			compressions++
		}
	}
	if compressions != 1 {
		t.Fatalf("first run emitted %d compression events, want 1", compressions)
	}

	stored := append([]*Event(nil), initialHistory...)
	stored = append(stored, first.Events...)
	second, err := agent.Run(context.Background(), stored, NewUserInput("second"))
	if err != nil {
		t.Fatal(err)
	}
	for _, event := range second.Events {
		if event.Compression != nil {
			t.Fatalf("already compressed history was compressed again: %+v", event)
		}
	}

	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 2 ||
		len(seen[0]) != 1 || seen[0][0] != "omitted" ||
		len(seen[1]) != 1 || seen[1][0] != "omitted" {
		t.Fatalf("compressed snapshot was not restored: %+v", seen)
	}

}

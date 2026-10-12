package llmx

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/xucx/llmx/types"
)

// statusTrace records the status changes and the non-empty assistant replies
// an agent appends.
type statusTrace struct {
	mu    sync.Mutex
	trace []string
}

func (s *statusTrace) sink(_ context.Context, event *Event) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch {
	case event.Status != nil:
		s.trace = append(s.trace, string(event.Status.Status)+string(event.Status.StopReason))
	case event.Message != nil && !event.Message.Delta &&
		event.Message.Role == types.MessageRoleAssistant && event.Message.Text() != "":
		s.trace = append(s.trace, event.Message.Text())
	}
	return nil
}

func (s *statusTrace) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return strings.Join(s.trace, ",")
}

// An interrupted turn never reaches Loop, so stopAfterTurns counts only the
// turns after the interruption.
func TestInterruptCancelsTheModelCallThenTheAgentServesTheNextInput(t *testing.T) {
	entered := make(chan struct{})
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			input := messages[len(messages)-1].Text()
			if input == "slow" {
				close(entered)
				<-ctx.Done() // the model call honors cancellation
				return nil, ctx.Err()
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "re:"+input),
			}, nil
		},
	)
	trace := &statusTrace{}
	session, err := agent.build(nil, WithEventSink(trace.sink), WithLoop(stopAfterTurns(1)))
	if err != nil {
		t.Fatal(err)
	}
	runner, err := session.Start(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("slow")); err != nil {
		t.Fatal(err)
	}
	<-entered
	runner.Interrupt()
	if _, err := runner.Enqueue(NewUserInput("fast")); err != nil {
		t.Fatal(err)
	}

	result, err := runner.Wait()
	if err != nil {
		t.Fatal(err)
	}
	// The canceled call appends no reply; the interruption is not an error, so
	// the agent goes on with the input enqueued after it as the next task.
	if got, want := trace.String(), "running,waitinginterrupted,running,re:fast,stoppeddone"; got != want {
		t.Fatalf("unexpected trace:\n got %s\nwant %s", got, want)
	}
	if result.StopReason != StopDone {
		t.Fatalf("unexpected result: %+v", result)
	}
}

func TestInterruptClosesToolCallsCutShort(t *testing.T) {
	entered := make(chan struct{})
	slow := MustNewTool(
		"slow",
		"",
		func(ctx ToolContext, _ *emptyToolParam) (any, error) {
			close(entered)
			<-ctx.Done()
			return nil, ctx.Err()
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
					ToolCall: &types.MessageToolCall{ID: "call", Name: "slow", Arguments: "{}"},
				})
				return &types.Completion{Message: message}, nil
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "re:"+messages[len(messages)-1].Text()),
			}, nil
		},
		slow,
	)
	session, err := agent.build(nil, WithLoop(stopAfterTurns(1)))
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
	runner.Interrupt()
	if _, err := runner.Enqueue(NewUserInput("stop that")); err != nil {
		t.Fatal(err)
	}
	result, err := runner.Wait()
	if err != nil {
		t.Fatal(err)
	}

	// The tool call cut short has an error result, so the next task sends the
	// model a well formed history.
	var toolResult *types.MessageToolResult
	for _, event := range result.Events {
		if event.Message == nil {
			continue
		}
		for _, part := range event.Message.Parts {
			if part.ToolResult != nil && part.ToolResult.ID == "call" {
				toolResult = part.ToolResult
			}
		}
	}
	if toolResult == nil || toolResult.Error != errToolInterrupted.Error() {
		t.Fatalf("the interrupted tool call has no fixed error result: %+v", toolResult)
	}
	var last *types.Message
	for _, event := range result.Events {
		if event.Message != nil && event.Message.Role == types.MessageRoleAssistant {
			last = event.Message
		}
	}
	if last == nil || last.Text() != "re:stop that" {
		t.Fatalf("the input after the interruption did not run next: %+v", result.Events)
	}
}

func TestInterjectionsBeforeOneBoundaryRunAsOneTask(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	var mu sync.Mutex
	var seen [][]string
	echo := MustNewTool("echo", "", func(ToolContext, *emptyToolParam) (any, error) { return "ok", nil })
	turn := 0
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			mu.Lock()
			turn++
			current := turn
			texts := make([]string, 0, len(messages))
			for _, message := range messages {
				texts = append(texts, message.Text())
			}
			seen = append(seen, texts)
			mu.Unlock()
			if current == 1 {
				close(entered)
				<-release
				message := types.NewMessage(types.MessageRoleAssistant)
				message.Parts = append(message.Parts, &types.MessagePart{
					ToolCall: &types.MessageToolCall{ID: "call", Name: "echo", Arguments: "{}"},
				})
				return &types.Completion{Message: message}, nil
			}
			return &types.Completion{Message: types.NewTextMessage(types.MessageRoleAssistant, "ok")}, nil
		},
		echo,
	)
	session, err := agent.build(nil, WithLoop(stopAfterTurns(2)))
	if err != nil {
		t.Fatal(err)
	}
	runner, err := session.Start(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("task")); err != nil {
		t.Fatal(err)
	}
	<-entered
	for _, text := range []string{"one", "two"} {
		if _, err := runner.Interject(NewUserInput(text)); err != nil {
			t.Fatal(err)
		}
	}
	close(release)
	if _, err := runner.Wait(); err != nil {
		t.Fatal(err)
	}

	mu.Lock()
	defer mu.Unlock()
	// The second task sees both interjections, in arrival order.
	if len(seen) != 2 {
		t.Fatalf("unexpected model calls: %v", seen)
	}
	history := seen[1]
	if len(history) < 2 || history[len(history)-2] != "one" || history[len(history)-1] != "two" {
		t.Fatalf("interjections did not run as one task: %v", history)
	}
}

func TestInterruptWithdrawsPendingInputsAndCancels(t *testing.T) {
	entered := make(chan struct{})
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			input := messages[len(messages)-1].Text()
			if input == "slow" {
				close(entered)
				<-ctx.Done()
				return nil, ctx.Err()
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "re:"+input),
			}, nil
		},
	)
	trace := &statusTrace{}
	session, err := agent.build(nil, WithEventSink(trace.sink), WithLoop(stopAfterTurns(1)))
	if err != nil {
		t.Fatal(err)
	}
	runner, err := session.Start(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("slow")); err != nil {
		t.Fatal(err)
	}
	<-entered
	if _, err := runner.Interject(NewUserInput("steer")); err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("queued")); err != nil {
		t.Fatal(err)
	}

	// Stop everything: what waits is withdrawn, what runs is canceled.
	withdrawn := runner.Interrupt()
	if len(withdrawn) != 2 ||
		withdrawn[0].Messages[0].Text() != "steer" ||
		withdrawn[1].Messages[0].Text() != "queued" {
		t.Fatalf("unexpected withdrawn inputs: %+v", withdrawn)
	}
	if pending := runner.Pending(); len(pending) != 0 {
		t.Fatalf("inputs left after interrupt: %+v", pending)
	}

	// Nothing withdrawn runs; the agent serves what comes next.
	if _, err := runner.Enqueue(NewUserInput("next")); err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Wait(); err != nil {
		t.Fatal(err)
	}
	if got, want := trace.String(), "running,waitinginterrupted,running,re:next,stoppeddone"; got != want {
		t.Fatalf("unexpected trace:\n got %s\nwant %s", got, want)
	}
}

func TestWithdrawTakesBackOnePendingInput(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			input := messages[len(messages)-1].Text()
			if input == "slow" {
				close(entered)
				<-release
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "re:"+input),
			}, nil
		},
	)
	// Each running status refreshes the pending view, as a UI would.
	var runner *Runner
	var mu sync.Mutex
	var views []string
	session, err := agent.build(
		nil,
		WithEventSink(func(_ context.Context, event *Event) error {
			if event.Status == nil || event.Status.Status != StatusRunning {
				return nil
			}
			var texts []string
			for _, input := range runner.Pending() {
				texts = append(texts, input.Messages[0].Text())
			}
			mu.Lock()
			views = append(views, strings.Join(texts, "+"))
			mu.Unlock()
			return nil
		}),
		WithLoop(stopAfterTurns(3)),
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err = session.Start(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("slow")); err != nil {
		t.Fatal(err)
	}
	<-entered

	// A generated ID, a caller ID, and an interjection.
	dropID, err := runner.Enqueue(NewUserInput("drop"))
	if err != nil || dropID == "" {
		t.Fatalf("unexpected enqueue: %q, %v", dropID, err)
	}
	keep := NewUserInput("keep")
	keep.ID = "keep"
	if id, err := runner.Enqueue(keep); err != nil || id != "keep" {
		t.Fatalf("unexpected enqueue: %q, %v", id, err)
	}
	if _, err := runner.Enqueue(keep); err == nil {
		t.Fatal("expected a duplicate pending ID to be rejected")
	}
	if _, err := runner.Interject(NewUserInput("steer")); err != nil {
		t.Fatal(err)
	}

	withdrawn := runner.Withdraw(dropID)
	if withdrawn == nil || withdrawn.ID != dropID || withdrawn.Messages[0].Text() != "drop" {
		t.Fatalf("unexpected withdrawn input: %+v", withdrawn)
	}
	if runner.Withdraw(dropID) != nil || runner.Withdraw("unknown") != nil {
		t.Fatal("withdrew an input that is not pending")
	}
	pending := runner.Pending()
	if len(pending) != 2 || pending[0].Messages[0].Text() != "steer" || pending[1].ID != "keep" {
		t.Fatalf("unexpected pending inputs: %+v", pending)
	}
	close(release)

	if _, err := runner.Wait(); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	// The withdrawn input never runs, and each running status sees the inputs
	// its task took already gone.
	if got, want := strings.Join(views, ","), ",keep,"; got != want {
		t.Fatalf("unexpected pending views:\n got %q\nwant %q", got, want)
	}
}

func TestInterruptKeepsTheStreamedTextWithoutToolCalls(t *testing.T) {
	entered := make(chan struct{})
	var mu sync.Mutex
	var histories [][]string
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			texts := make([]string, 0, len(messages))
			for _, message := range messages {
				texts = append(texts, message.Text())
			}
			mu.Lock()
			histories = append(histories, texts)
			mu.Unlock()
			if messages[len(messages)-1].Text() != "slow" {
				return &types.Completion{
					Message: types.NewTextMessage(types.MessageRoleAssistant, "ok"),
				}, nil
			}
			close(entered)
			<-ctx.Done()
			// A streaming call cut short: the text so far and a truncated tool
			// call come back along with the cancellation.
			partial := types.NewTextMessage(types.MessageRoleAssistant, "half an ans")
			partial.Parts = append(partial.Parts, &types.MessagePart{
				ToolCall: &types.MessageToolCall{ID: "call", Name: "echo", Arguments: `{"a`},
			})
			return &types.Completion{Message: partial}, context.Cause(ctx)
		},
	)
	trace := &statusTrace{}
	session, err := agent.build(nil, WithEventSink(trace.sink), WithLoop(stopAfterTurns(1)))
	if err != nil {
		t.Fatal(err)
	}
	runner, err := session.Start(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("slow")); err != nil {
		t.Fatal(err)
	}
	<-entered
	runner.Interrupt()
	if _, err := runner.Enqueue(NewUserInput("go on")); err != nil {
		t.Fatal(err)
	}
	result, err := runner.Wait()
	if err != nil {
		t.Fatal(err)
	}

	if got, want := trace.String(), "running,half an ans,waitinginterrupted,running,ok,stoppeddone"; got != want {
		t.Fatalf("unexpected trace:\n got %s\nwant %s", got, want)
	}
	for _, event := range result.Events {
		if event.Message != nil && len(event.Message.ToolCalls()) > 0 {
			t.Fatalf("an interrupted reply kept its tool calls: %+v", event.Message)
		}
	}
	// The next task sees what the model already said.
	mu.Lock()
	defer mu.Unlock()
	if want := []string{"slow", "half an ans", "go on"}; strings.Join(histories[1], ",") != strings.Join(want, ",") {
		t.Fatalf("unexpected history:\n got %v\nwant %v", histories[1], want)
	}
}

func TestInterruptBeforeToolsNeverRunsThem(t *testing.T) {
	var calls atomic.Int32
	side := MustNewTool("side", "", func(ToolContext, *emptyToolParam) (any, error) {
		calls.Add(1) // a side effect that ignores cancellation
		return "done", nil
	})
	var runner *Runner
	turn := 0
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			turn++
			if turn > 1 {
				return &types.Completion{Message: types.NewTextMessage(types.MessageRoleAssistant, "ok")}, nil
			}
			// The interruption lands after the model replied with a tool call.
			runner.Interrupt()
			message := types.NewMessage(types.MessageRoleAssistant)
			message.Parts = append(message.Parts, &types.MessagePart{
				ToolCall: &types.MessageToolCall{ID: "call", Name: "side", Arguments: "{}"},
			})
			return &types.Completion{Message: message}, nil
		},
		side,
	)
	interrupted := make(chan struct{})
	session, err := agent.build(
		nil,
		WithEventSink(func(_ context.Context, event *Event) error {
			if event.Status != nil && event.Status.StopReason == StopInterrupted {
				close(interrupted)
			}
			return nil
		}),
		WithLoop(stopAfterTurns(1)),
	)
	if err != nil {
		t.Fatal(err)
	}
	runner, err = session.Start(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Enqueue(NewUserInput("run")); err != nil {
		t.Fatal(err)
	}
	<-interrupted
	if _, err := runner.Enqueue(NewUserInput("next")); err != nil {
		t.Fatal(err)
	}
	result, err := runner.Wait()
	if err != nil {
		t.Fatal(err)
	}

	if n := calls.Load(); n != 0 {
		t.Fatalf("the tool ran %d times after the interruption", n)
	}
	var toolResult *types.MessageToolResult
	for _, event := range result.Events {
		if event.Message == nil {
			continue
		}
		for _, part := range event.Message.Parts {
			if part.ToolResult != nil {
				toolResult = part.ToolResult
			}
		}
	}
	if toolResult == nil || toolResult.ID != "call" || toolResult.Error != errToolInterrupted.Error() {
		t.Fatalf("expected the fixed interrupted answer, got %+v", toolResult)
	}
}

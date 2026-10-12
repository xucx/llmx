package llmx

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/xucx/llmx/types"
)

// The adapters below turn one stage function into an Interceptor, so a test
// only writes the stage it intercepts.

type turnInterceptorFunc func(TurnHandler) TurnHandler

func (f turnInterceptorFunc) InterceptTurn(next TurnHandler) TurnHandler { return f(next) }
func (turnInterceptorFunc) InterceptLLM(next LLMHandler) LLMHandler      { return next }
func (turnInterceptorFunc) InterceptToolCall(next ToolCallHandler) ToolCallHandler {
	return next
}

type llmInterceptorFunc func(LLMHandler) LLMHandler

func (llmInterceptorFunc) InterceptTurn(next TurnHandler) TurnHandler { return next }
func (f llmInterceptorFunc) InterceptLLM(next LLMHandler) LLMHandler  { return f(next) }
func (llmInterceptorFunc) InterceptToolCall(next ToolCallHandler) ToolCallHandler {
	return next
}

type toolCallInterceptorFunc func(ToolCallHandler) ToolCallHandler

func (toolCallInterceptorFunc) InterceptTurn(next TurnHandler) TurnHandler { return next }
func (toolCallInterceptorFunc) InterceptLLM(next LLMHandler) LLMHandler    { return next }
func (f toolCallInterceptorFunc) InterceptToolCall(next ToolCallHandler) ToolCallHandler {
	return f(next)
}

func TestTurnInterceptorAppendsAroundTheTurn(t *testing.T) {
	var seen []string
	agent := newRunTestAgent(
		t,
		func(ctx LLMContext) (*types.Completion, error) {
			messages := ctx.Request().Messages
			for _, message := range messages {
				seen = append(seen, message.Text())
			}
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "answer"),
			}, nil
		},
	)
	session, err := agent.build(
		nil,
		WithInterceptor(turnInterceptorFunc(func(next TurnHandler) TurnHandler {
			return func(ctx TurnContext) error {
				if err := ctx.Append(NewEventMessage(
					types.NewTextMessage(types.MessageRoleUser, "reminder"),
				)); err != nil {
					return err
				}
				if err := next(ctx); err != nil {
					return err
				}
				return ctx.Append(NewEventState(map[string]string{"answered": "yes"}, true))
			}
		})),
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := session.Run(context.Background(), NewUserInput("question"))
	if err != nil {
		t.Fatal(err)
	}

	if len(seen) != 2 || seen[0] != "question" || seen[1] != "reminder" {
		t.Fatalf("the model did not see the appended reminder: %v", seen)
	}
	// running, user input, reminder, answer, usage, state, waiting: every
	// append joins the one event stream in order.
	if len(result.Events) != 7 ||
		result.Events[2].Message == nil ||
		result.Events[2].Message.Text() != "reminder" ||
		result.Events[3].Message == nil ||
		result.Events[3].Message.Text() != "answer" ||
		result.Events[4].Usage == nil ||
		result.Events[5].State == nil ||
		result.Events[6].Status == nil {
		t.Fatalf("unexpected events: %+v", result.Events)
	}
	if result.State["answered"] != "yes" {
		t.Fatalf("appended state was not applied: %+v", result.State)
	}
}

func TestTurnContextRejectsAgentEvents(t *testing.T) {
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "answer"),
			}, nil
		},
	)
	session, err := agent.build(
		nil,
		WithInterceptor(turnInterceptorFunc(func(next TurnHandler) TurnHandler {
			return func(ctx TurnContext) error {
				if err := ctx.Append(&Event{
					ID:     "status",
					Status: &EventStatus{Status: StatusWaiting, StopReason: StopDone},
				}); err != nil {
					return err
				}
				return next(ctx)
			}
		})),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Run(context.Background(), NewUserInput("question")); err == nil {
		t.Fatal("a turn appended a status event")
	}
}

func TestTurnContextRejectsAppendsAfterTheTurn(t *testing.T) {
	var turnCtx TurnContext
	agent := newRunTestAgent(
		t,
		func(LLMContext) (*types.Completion, error) {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "answer"),
			}, nil
		},
	)
	session, err := agent.build(
		nil,
		WithInterceptor(turnInterceptorFunc(func(next TurnHandler) TurnHandler {
			return func(ctx TurnContext) error {
				turnCtx = ctx
				return next(ctx)
			}
		})),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Run(context.Background(), NewUserInput("question")); err != nil {
		t.Fatal(err)
	}
	// A tool still running after its turn ended must not write into the stream.
	err = turnCtx.Append(NewEventState(map[string]string{"late": "yes"}, true))
	if !errors.Is(err, errTurnOver) {
		t.Fatalf("append after the turn = %v, want errTurnOver", err)
	}
}

// recordingInterceptor records entering and leaving every stage.
type recordingInterceptor struct {
	name  string
	mu    *sync.Mutex
	trace *[]string
	built *int
}

func (r recordingInterceptor) record(entry string) {
	r.mu.Lock()
	*r.trace = append(*r.trace, r.name+":"+entry)
	r.mu.Unlock()
}

func (r recordingInterceptor) InterceptTurn(next TurnHandler) TurnHandler {
	return func(ctx TurnContext) error {
		r.record("turn")
		defer r.record("/turn")
		return next(ctx)
	}
}

func (r recordingInterceptor) InterceptLLM(next LLMHandler) LLMHandler {
	*r.built++
	return func(ctx LLMContext) (*types.Completion, error) {
		r.record("llm")
		defer r.record("/llm")
		return next(ctx)
	}
}

func (r recordingInterceptor) InterceptToolCall(next ToolCallHandler) ToolCallHandler {
	return func(ctx ToolContext) (any, error) {
		r.record("tool")
		defer r.record("/tool")
		return next(ctx)
	}
}

func TestInterceptorsWrapStagesInOnionOrder(t *testing.T) {
	tool := MustNewTool(
		"echo",
		"",
		func(ToolContext, *emptyToolParam) (any, error) { return "ok", nil },
	)
	turn := 0
	model := func(LLMContext) (*types.Completion, error) {
		turn++
		if turn > 1 {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
			}, nil
		}
		message := types.NewMessage(types.MessageRoleAssistant)
		message.Parts = append(message.Parts, &types.MessagePart{
			ToolCall: &types.MessageToolCall{ID: "call", Name: "echo", Arguments: "{}"},
		})
		return &types.Completion{Message: message}, nil
	}

	var mu sync.Mutex
	var trace []string
	built := 0
	outer := recordingInterceptor{name: "a", mu: &mu, trace: &trace, built: &built}
	inner := recordingInterceptor{name: "b", mu: &mu, trace: &trace, built: &built}
	// The fake model is the innermost interceptor, so both recorders wrap it.
	agent, err := newAgentFactory(
		NewModels(),
		WithModel("fake"),
		WithTools(tool),
		WithInterceptor(outer, inner),
	)
	if err != nil {
		t.Fatal(err)
	}
	session, err := agent.build(
		nil,
		WithInterceptor(llmInterceptorFunc(func(LLMHandler) LLMHandler { return model })),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := session.Run(context.Background(), NewUserInput("run")); err != nil {
		t.Fatal(err)
	}

	// The first turn calls the model and the tool, the second only the model.
	want := []string{
		"a:turn", "b:turn",
		"a:llm", "b:llm", "b:/llm", "a:/llm",
		"a:tool", "b:tool", "b:/tool", "a:/tool",
		"b:/turn", "a:/turn",
		"a:turn", "b:turn",
		"a:llm", "b:llm", "b:/llm", "a:/llm",
		"b:/turn", "a:/turn",
	}
	if len(trace) != len(want) {
		t.Fatalf("unexpected trace: %v", trace)
	}
	for index := range want {
		if trace[index] != want[index] {
			t.Fatalf("unexpected trace: %v", trace)
		}
	}
	// Two interceptors, each built once although the agent ran two turns.
	if built != 2 {
		t.Fatalf("llm chain was built %d times, want once per interceptor", built)
	}
}

func TestLLMContextRequestIsACopyAndSafeForConcurrentUse(t *testing.T) {
	s, err := newSession(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	level := 3
	ctx := newLLMContext(newTurnContext(context.Background(), s), LLMRequest{
		Model:    "model",
		Messages: []*types.Message{types.NewTextMessage(types.MessageRoleUser, "hi")},
		Options:  &types.CompletionOptions{Instructions: "base", ReasoningLevel: &level},
	})

	// Changing a copy changes nothing until it is set back.
	request := ctx.Request()
	request.Messages[0] = types.NewTextMessage(types.MessageRoleUser, "changed")
	request.Options.Instructions = "changed"
	*request.Options.ReasoningLevel = 9
	if current := ctx.Request(); current.Messages[0].Text() != "hi" ||
		current.Options.Instructions != "base" ||
		*current.Options.ReasoningLevel != 3 {
		t.Fatalf("a copy of the request changed the context: %+v", current)
	}

	// Concurrent tools read while an interceptor writes; run with -race.
	var wg sync.WaitGroup
	for index := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 100 {
				request := ctx.Request()
				if index%2 == 0 {
					request.Options.Instructions = "set"
					ctx.SetRequest(request)
				}
			}
		}()
	}
	wg.Wait()
	if ctx.Request().Options.Instructions != "set" {
		t.Fatalf("SetRequest did not replace the request")
	}
}

func TestSetToolCallRewritesTheCallButKeepsItsAnswer(t *testing.T) {
	type echoArgs struct {
		Text string `json:"text"`
	}
	var got string
	echo := MustNewTool("echo", "", func(_ ToolContext, args *echoArgs) (any, error) {
		got = args.Text
		return "echoed " + args.Text, nil
	})
	legacy := MustNewTool("legacy_echo", "", func(ToolContext, *echoArgs) (any, error) {
		return nil, errors.New("the legacy tool must not run")
	})
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
				ToolCall: &types.MessageToolCall{ID: "call", Name: "legacy_echo", Arguments: "not json"},
			})
			return &types.Completion{Message: message}, nil
		},
		echo,
		legacy,
	)
	session, err := agent.build(
		nil,
		WithInterceptor(toolCallInterceptorFunc(func(next ToolCallHandler) ToolCallHandler {
			return func(ctx ToolContext) (any, error) {
				call := ctx.ToolCall()
				call.Name = "echo"                     // route to the current tool
				call.Arguments = `{"text":"repaired"}` // repair the arguments
				ctx.SetToolCall(call)
				return next(ctx)
			}
		})),
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := session.Run(context.Background(), NewUserInput("run"))
	if err != nil {
		t.Fatal(err)
	}
	if got != "repaired" {
		t.Fatalf("the tool did not run the rewritten call: %q", got)
	}
	var answer *types.MessageToolResult
	for _, event := range result.Events {
		if event.Message == nil {
			continue
		}
		for _, part := range event.Message.Parts {
			if part.ToolResult != nil {
				answer = part.ToolResult
			}
		}
	}
	// The result answers the call the model made, so the history stays paired.
	if answer == nil || answer.ID != "call" || answer.Name != "legacy_echo" ||
		answer.Result != "echoed repaired" {
		t.Fatalf("unexpected tool result: %+v", answer)
	}
}

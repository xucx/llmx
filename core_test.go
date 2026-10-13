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

// ---- helpers ----

var testAgent = &Agent{Model: "fake", MaxTurn: 8}

func input(id string, mode InputMode, text string) *Event {
	return &Event{ID: id, Input: NewUserInput(mode, text)}
}

func reply(baseSeq int64, text string, calls ...ToolCall) *Event {
	return NewReplyEvent(&Reply{BaseSeq: baseSeq, Text: text, ToolCalls: calls})
}

func call(id, name string) ToolCall { return ToolCall{ID: id, Name: name, Arguments: "{}"} }

func result(callID, value string) *Event {
	return NewToolResultEvent(&ToolResult{CallID: callID, Value: value})
}

// numbered assigns seqs in order, as a Session would.
func numbered(events ...*Event) []*Event {
	for index, event := range events {
		event.Seq = int64(index + 1)
	}
	return events
}

func ids(effects []Effect) []string {
	out := make([]string, 0, len(effects))
	for _, effect := range effects {
		out = append(out, effect.ID)
	}
	return out
}

// fakeModel answers CallModel effects with scripted replies.
type fakeModel struct {
	mu      sync.Mutex
	replies []*Reply
	calls   int32
	// block, when set, holds every call until released or canceled.
	block chan struct{}
	seen  [][]*types.Message
}

func (f *fakeModel) Can(e *Effect) bool { return e != nil && e.CallModel != nil }

func (f *fakeModel) Exec(ctx context.Context, _ *State, e *Effect) ([]*Event, error) {
	atomic.AddInt32(&f.calls, 1)
	f.mu.Lock()
	f.seen = append(f.seen, e.CallModel.Messages)
	block := f.block
	f.mu.Unlock()
	if block != nil {
		select {
		case <-block:
		case <-ctx.Done():
			return []*Event{NewReplyEvent(&Reply{BaseSeq: e.CallModel.BaseSeq, Text: "partial", Partial: true})}, nil
		}
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	var r *Reply
	if len(f.replies) > 0 {
		r, f.replies = f.replies[0], f.replies[1:]
	} else {
		r = &Reply{Text: "ok"}
	}
	r = r.Clone()
	r.BaseSeq = e.CallModel.BaseSeq
	return []*Event{NewReplyEvent(r)}, nil
}

func waitIdle(t *testing.T, l *Local) *State {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		state, err := l.State(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if len(state.Inputs) == 0 && len(Plan(state, l.agent)) == 0 {
			return state
		}
		if time.Now().After(deadline) {
			t.Fatalf("not idle: pending=%d inputs=%d", len(state.Pending), len(state.Inputs))
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// ---- Fold / Plan ----

func TestPlanCallsModelAfterAnInput(t *testing.T) {
	s := Fold(numbered(input("q1", InputQueue, "hi")))
	effects := Plan(s, testAgent)
	if len(effects) != 1 || effects[0].CallModel == nil || effects[0].ID != ReplyID(1) {
		t.Fatalf("effects = %v", ids(effects))
	}
	if got := effects[0].CallModel.Messages; len(got) != 1 || got[0].Role != types.MessageRoleUser || got[0].ID != "q1" {
		t.Fatalf("messages = %+v", got)
	}
}

func TestPlanRunsEveryPendingToolCallAtOnce(t *testing.T) {
	s := Fold(numbered(
		input("q1", InputQueue, "hi"),
		reply(1, "", call("a", "x"), call("b", "y")),
	))
	effects := Plan(s, testAgent)
	if got := ids(effects); len(got) != 2 || got[0] != "tool:a" || got[1] != "tool:b" {
		t.Fatalf("effects = %v", got)
	}
	s = Fold(numbered(
		input("q1", InputQueue, "hi"),
		reply(1, "", call("a", "x"), call("b", "y")),
		result("b", "B"),
	))
	if got := ids(Plan(s, testAgent)); len(got) != 1 || got[0] != "tool:a" {
		t.Fatalf("effects after one result = %v", got)
	}
	s = Fold(numbered(
		input("q1", InputQueue, "hi"),
		reply(1, "", call("a", "x"), call("b", "y")),
		result("b", "B"),
		result("a", "A"),
	))
	if got := ids(Plan(s, testAgent)); len(got) != 1 || got[0] != ReplyID(4) {
		t.Fatalf("effects after both results = %v", got)
	}
}

func TestInterjectWaitsForPendingToolsThenJoins(t *testing.T) {
	events := numbered(
		input("q1", InputQueue, "weather"),
		reply(1, "", call("a", "get_weather")),
		input("q2", InputInterject, "tomorrow instead"),
	)
	s := Fold(events)
	if got := ids(Plan(s, testAgent)); len(got) != 1 || got[0] != "tool:a" {
		t.Fatalf("pending tool must finish first: %v", got)
	}
	if len(s.Inputs) != 1 {
		t.Fatalf("interjection admitted early")
	}
	s = Fold(numbered(append(events, result("a", "sunny"))...))
	last := s.Admitted[len(s.Admitted)-1]
	if last.ID != "q2" {
		t.Fatalf("interjection not admitted at the turn boundary: %s", last.ID)
	}
	if got := ids(Plan(s, testAgent)); len(got) != 1 || got[0] != ReplyID(4) {
		t.Fatalf("effects = %v", got)
	}
}

func TestQueuedInputWaitsUntilIdleAndInterjectionGoesFirst(t *testing.T) {
	events := numbered(
		input("q1", InputQueue, "first"),
		input("q2", InputQueue, "second"),
		input("q3", InputInterject, "now"),
	)
	s := Fold(events)
	// q1 joined when idle; q3 joins ahead of q2 since nothing is pending.
	if len(s.Admitted) != 2 || s.Admitted[0].ID != "q1" || s.Admitted[1].ID != "q3" {
		t.Fatalf("admitted = %+v", s.Admitted)
	}
	if len(s.Inputs) != 1 || s.Inputs[0].ID != "q2" {
		t.Fatalf("queued input must wait: %+v", s.Inputs)
	}
	s = Fold(numbered(append(events, reply(3, "answer"))...))
	if last := s.Admitted[len(s.Admitted)-1]; last.ID != "q2" {
		t.Fatalf("queued input not admitted once idle: %s", last.ID)
	}
}

func TestInputDuringModelCallWaitsForTheReply(t *testing.T) {
	// The reply at seq 3 was made from seq 1: the input at seq 2 arrived
	// while the model was running and joins after the reply.
	s := Fold(numbered(
		input("q1", InputQueue, "first"),
		input("q2", InputInterject, "while running"),
		reply(1, "answer one"),
	))
	if len(s.Admitted) != 3 || s.Admitted[1].ID != ReplyID(1) || s.Admitted[2].ID != "q2" {
		t.Fatalf("admitted = %v", eventIDs(s.Admitted))
	}
	if got := ids(Plan(s, testAgent)); len(got) != 1 || got[0] != ReplyID(3) {
		t.Fatalf("effects = %v", got)
	}
}

func TestInterruptCancelsPendingToolsThenJoins(t *testing.T) {
	events := numbered(
		input("q1", InputQueue, "first"),
		input("q2", InputQueue, "queued"),
		reply(2, "", call("a", "slow")),
		input("q3", InputInterrupt, "stop and do this"),
	)
	s := Fold(events)
	effects := Plan(s, testAgent)
	if len(effects) != 1 || effects[0].Emit == nil {
		t.Fatalf("expected an emit, got %v", ids(effects))
	}
	emitted := effects[0].Emit.Events
	if len(emitted) != 2 || emitted[0].ToolResult == nil || emitted[0].ToolResult.Error == "" || emitted[1].Stop == nil {
		t.Fatalf("emitted = %+v", emitted)
	}
	s = Fold(numbered(append(events, emitted...)...))
	last := s.Admitted[len(s.Admitted)-1]
	if last.ID != "q3" {
		t.Fatalf("interrupt not admitted after the stop: %s", last.ID)
	}
	if len(s.Inputs) != 0 {
		t.Fatalf("queued input must be discarded by the interrupt: %v", eventIDs(s.Inputs))
	}
	if got := ids(Plan(s, testAgent)); len(got) != 1 || got[0] != ReplyID(6) {
		t.Fatalf("effects = %v", got)
	}
}

func TestInterruptWithoutContentOnlyStops(t *testing.T) {
	events := numbered(
		input("q1", InputQueue, "first"),
		&Event{ID: "q2", Input: &Input{Mode: InputInterrupt}},
	)
	s := Fold(events)
	effects := Plan(s, testAgent)
	if len(effects) != 1 || effects[0].Emit == nil {
		t.Fatalf("effects = %v", ids(effects))
	}
	s = Fold(numbered(append(events, effects[0].Emit.Events...)...))
	if len(Plan(s, testAgent)) != 0 {
		t.Fatalf("an empty interrupt must leave the conversation idle")
	}
}

func TestMaxTurnStopsTheRound(t *testing.T) {
	agent := &Agent{Model: "fake", MaxTurn: 2}
	events := numbered(
		input("q1", InputQueue, "go"),
		reply(1, "", call("a", "t")),
		result("a", "1"),
		reply(3, "", call("b", "t")),
		result("b", "2"),
	)
	s := Fold(events)
	effects := Plan(s, agent)
	if len(effects) != 1 || effects[0].Emit == nil || effects[0].Emit.Events[0].Stop == nil ||
		effects[0].Emit.Events[0].Stop.Reason != StopMaxTurn {
		t.Fatalf("effects = %v", ids(effects))
	}
	s = Fold(numbered(append(events, effects[0].Emit.Events...)...))
	if len(Plan(s, agent)) != 0 {
		t.Fatalf("stopped round must be idle")
	}
	s = Fold(numbered(append(events, effects[0].Emit.Events[0], input("q2", InputQueue, "again"))...))
	if got := ids(Plan(s, agent)); len(got) != 1 || got[0] != ReplyID(7) {
		t.Fatalf("a new input must reopen: %v", got)
	}
}

func TestAskedToolWaitsForAnswerThenRunsAgain(t *testing.T) {
	events := numbered(
		input("q1", InputQueue, "delete"),
		reply(1, "", call("a", "delete_file")),
		NewToolAskEvent(&ToolAsk{CallID: "a", Args: map[string]any{"prompt": "sure?"}}),
	)
	s := Fold(events)
	effects := Plan(s, testAgent)
	if len(effects) != 1 || effects[0].CallTool == nil || effects[0].CallTool.Ask == nil || effects[0].CallTool.Answer != nil {
		t.Fatalf("expected a call carrying the question: %+v", effects)
	}
	tools := ToolExecutor(&Tool{Name: "delete_file", Invoker: func(ToolContext, string) (any, error) { return "x", nil }})
	if tools.Can(&effects[0]) {
		t.Fatalf("tool executor must not run an unanswered question")
	}
	human := HumanExecutor(func(context.Context, ToolCall, *ToolAsk) (map[string]any, error) { return nil, nil })
	if !human.Can(&effects[0]) {
		t.Fatalf("human executor must take the question")
	}
	s = Fold(numbered(append(events, NewToolAnswerEvent(&ToolAnswer{CallID: "a", Result: map[string]any{"ok": true}}))...))
	effects = Plan(s, testAgent)
	if len(effects) != 1 || effects[0].CallTool.Answer == nil || !tools.Can(&effects[0]) || human.Can(&effects[0]) {
		t.Fatalf("answered call must go back to the tool: %+v", effects)
	}
}

func TestDuplicateAndUnknownToolResultsAreIgnored(t *testing.T) {
	s := Fold(numbered(
		input("q1", InputQueue, "go"),
		reply(1, "", call("a", "t")),
		result("zzz", "unknown"),
		result("a", "1"),
		&Event{ID: "dup", ToolResult: &ToolResult{CallID: "a", Value: "2"}},
	))
	if len(s.Admitted) != 3 || s.Admitted[2].ToolResult.Value != "1" {
		t.Fatalf("admitted = %v", eventIDs(s.Admitted))
	}
}

func eventIDs(events []*Event) []string {
	out := make([]string, 0, len(events))
	for _, event := range events {
		out = append(out, event.ID)
	}
	return out
}

// ---- Drive / Local ----

func TestLocalRunsToolsAndAnswers(t *testing.T) {
	model := &fakeModel{replies: []*Reply{
		{ToolCalls: []ToolCall{{ID: "c1", Name: "echo", Arguments: `{"text":"hi"}`}}},
		{Text: "done"},
	}}
	type echoArgs struct {
		Text string `json:"text"`
	}
	echo := MustNewTool("echo", "", func(_ ToolContext, args *echoArgs) (any, error) {
		return "echo:" + args.Text, nil
	})
	local, err := NewLocal(testAgent, []Executor{model, ToolExecutor(echo)})
	if err != nil {
		t.Fatal(err)
	}
	defer local.Close()
	since, err := local.Run(t.Context(), NewUserInput(InputQueue, "say hi"))
	if err != nil {
		t.Fatal(err)
	}
	var kinds []string
	for _, event := range since {
		switch {
		case event.Input != nil:
			kinds = append(kinds, "input")
		case event.Reply != nil:
			kinds = append(kinds, "reply")
		case event.ToolResult != nil:
			kinds = append(kinds, "tool:"+event.ToolResult.Value)
		}
	}
	if got := strings.Join(kinds, ","); got != "input,reply,tool:echo:hi,reply" {
		t.Fatalf("events = %s", got)
	}
	if model.seen[1][2].Role != types.MessageRoleTool {
		t.Fatalf("second call did not see the tool result: %+v", model.seen[1])
	}
}

func TestLocalInterruptKeepsPartialReply(t *testing.T) {
	model := &fakeModel{block: make(chan struct{})}
	local, err := NewLocal(testAgent, []Executor{model})
	if err != nil {
		t.Fatal(err)
	}
	defer local.Close()
	if _, err := local.Enqueue(t.Context(), "think hard"); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for atomic.LoadInt32(&model.calls) == 0 {
		if time.Now().After(deadline) {
			t.Fatal("model never called")
		}
		time.Sleep(time.Millisecond)
	}
	if _, err := local.Interrupt(t.Context(), ""); err != nil {
		t.Fatal(err)
	}
	state := waitIdle(t, local)
	last := state.LastReply()
	if last == nil || !last.Reply.Partial || last.Reply.Text != "partial" {
		t.Fatalf("partial reply not kept: %+v", last)
	}
	if atomic.LoadInt32(&model.calls) != 1 {
		t.Fatalf("model called %d times after an empty interrupt", model.calls)
	}
}

func TestLocalAskFlow(t *testing.T) {
	model := &fakeModel{replies: []*Reply{
		{ToolCalls: []ToolCall{{ID: "d1", Name: "delete", Arguments: `{}`}}},
		{Text: "deleted"},
	}}
	var asked int32
	del := &Tool{Name: "delete", Invoker: func(ctx ToolContext, _ string) (any, error) {
		if ctx.Answer() == nil {
			return &Ask{Args: map[string]any{"prompt": "sure?"}}, nil
		}
		return fmt.Sprintf("approved=%v", ctx.Answer().Result["approved"]), nil
	}}
	human := HumanExecutor(func(_ context.Context, _ ToolCall, ask *ToolAsk) (map[string]any, error) {
		atomic.AddInt32(&asked, 1)
		return map[string]any{"approved": true}, nil
	})
	local, err := NewLocal(testAgent, []Executor{model, ToolExecutor(del), human})
	if err != nil {
		t.Fatal(err)
	}
	defer local.Close()
	since, err := local.Run(t.Context(), NewUserInput(InputQueue, "delete it"))
	if err != nil {
		t.Fatal(err)
	}
	var sawAsk, sawAnswer bool
	var value string
	for _, event := range since {
		switch {
		case event.ToolAsk != nil:
			sawAsk = true
		case event.ToolAnswer != nil:
			sawAnswer = true
		case event.ToolResult != nil:
			value = event.ToolResult.Value
		}
	}
	if !sawAsk || !sawAnswer || value != "approved=true" || atomic.LoadInt32(&asked) != 1 {
		t.Fatalf("ask flow broken: ask=%v answer=%v value=%q asked=%d", sawAsk, sawAnswer, value, asked)
	}
}

func TestTwoDriversSplitTheWorkAndNeverDuplicate(t *testing.T) {
	// Driver A holds the model and tool "server", driver B holds tool
	// "client": one turn needs both, and only one reply per base seq lands.
	model := &fakeModel{replies: []*Reply{
		{ToolCalls: []ToolCall{{ID: "s", Name: "server", Arguments: `{}`}, {ID: "c", Name: "client", Arguments: `{}`}}},
		{Text: "done"},
	}}
	session := NewMemorySession()
	serverTool := &Tool{Name: "server", Invoker: func(ToolContext, string) (any, error) { return "S", nil }}
	clientTool := &Tool{Name: "client", Invoker: func(ToolContext, string) (any, error) { return "C", nil }}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	var wg sync.WaitGroup
	for _, execs := range [][]Executor{
		{model, ToolExecutor(serverTool), EmitExecutor()},
		{ToolExecutor(clientTool), EmitExecutor()},
	} {
		wg.Add(1)
		go func(execs []Executor) {
			defer wg.Done()
			_, _ = Drive(ctx, session, testAgent, DriveOptions{}, execs...)
		}(execs)
	}
	if err := session.Append(ctx, -1, NewInputEvent(NewUserInput(InputQueue, "go"))); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		events, _, _ := session.Load(ctx)
		state := Fold(events)
		if len(Plan(state, testAgent)) == 0 && len(state.Admitted) == 5 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("not done: %v", eventIDs(events))
		}
		time.Sleep(5 * time.Millisecond)
	}
	cancel()
	wg.Wait()
	events, _, _ := session.Load(context.Background())
	replies := 0
	results := map[string]string{}
	for _, event := range events {
		if event.Reply != nil {
			replies++
		}
		if event.ToolResult != nil {
			results[event.ToolResult.CallID] = event.ToolResult.Value
		}
	}
	if replies != 2 || results["s"] != "S" || results["c"] != "C" {
		t.Fatalf("replies=%d results=%v", replies, results)
	}
}

func TestConcurrentDriversWithTheSameExecutorsKeepOneCopy(t *testing.T) {
	model := &fakeModel{replies: []*Reply{{Text: "one"}, {Text: "two"}, {Text: "three"}}}
	session := NewMemorySession()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	var wg sync.WaitGroup
	for range 3 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = Drive(ctx, session, testAgent, DriveOptions{}, model, EmitExecutor())
		}()
	}
	if err := session.Append(ctx, -1, NewInputEvent(NewUserInput(InputQueue, "go"))); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		events, _, _ := session.Load(ctx)
		if state := Fold(events); len(Plan(state, testAgent)) == 0 && state.LastReply() != nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("never answered")
		}
		time.Sleep(5 * time.Millisecond)
	}
	cancel()
	wg.Wait()
	events, _, _ := session.Load(context.Background())
	replies := 0
	for _, event := range events {
		if event.Reply != nil {
			replies++
		}
	}
	if replies != 1 {
		t.Fatalf("%d replies for one input", replies)
	}
}

func TestFailedModelCallStopsTheRound(t *testing.T) {
	failing := ExecutorFunc{
		CanFunc: func(e *Effect) bool { return e.CallModel != nil },
		ExecFunc: func(context.Context, *State, *Effect) ([]*Event, error) {
			return nil, errors.New("boom")
		},
	}
	local, err := NewLocal(testAgent, []Executor{failing})
	if err != nil {
		t.Fatal(err)
	}
	defer local.Close()
	since, err := local.Run(t.Context(), NewUserInput(InputQueue, "go"))
	if err != nil {
		t.Fatal(err)
	}
	if last := since[len(since)-1]; last.Stop == nil || last.Stop.Reason != StopError || last.Stop.Error != "boom" {
		t.Fatalf("expected an error stop: %+v", last)
	}
}

func TestDriveExitWhenIdleReportsRemainingEffects(t *testing.T) {
	session := NewMemorySession(
		input("q1", InputQueue, "go"),
		reply(1, "", call("c", "client_only")),
	)
	remaining, err := Drive(t.Context(), session, testAgent, DriveOptions{ExitWhenIdle: true}, EmitExecutor())
	if err != nil {
		t.Fatal(err)
	}
	if len(remaining) != 1 || remaining[0].CallTool == nil || remaining[0].CallTool.Call.Name != "client_only" {
		t.Fatalf("remaining = %+v", remaining)
	}
}

func TestBuildMessagesRendersTheConversation(t *testing.T) {
	s := Fold(numbered(
		input("q1", InputQueue, "hi"),
		NewReplyEvent(&Reply{BaseSeq: 1, Reasoning: "think", Text: "", ToolCalls: []ToolCall{call("a", "t")}}),
		NewToolResultEvent(&ToolResult{CallID: "a", Name: "t", Value: "v", Instruction: map[string]string{"note": "n"}}),
		reply(3, "bye"),
	))
	messages := BuildMessages(s)
	if len(messages) != 4 {
		t.Fatalf("messages = %d", len(messages))
	}
	if err := types.ValidateMessages(messages); err != nil {
		t.Fatal(err)
	}
	if messages[1].Parts[0].Reasoning == nil || messages[1].Parts[1].ToolCall == nil {
		t.Fatalf("assistant parts = %+v", messages[1].Parts)
	}
	if got := messages[2].Parts[0].ToolResult; got.Result != "v" || got.Instruction["note"][0] != "n" {
		t.Fatalf("tool result = %+v", got)
	}
}

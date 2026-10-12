package compress

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/xucx/llmx"
	"github.com/xucx/llmx/types"
)

// testProvider answers every completion with complete, which sees the history
// the model was sent.
type testProvider struct {
	types.ProviderNop
	complete func([]*types.Message) *types.Completion
}

func (p testProvider) Completion(
	_ context.Context,
	_ string,
	messages []*types.Message,
	_ ...types.CompletionOption,
) (*types.Completion, error) {
	return p.complete(messages), nil
}

func fixedCompletion(completion *types.Completion) func([]*types.Message) *types.Completion {
	return func([]*types.Message) *types.Completion { return completion }
}

// agentFactory builds agents sharing one configuration, each over its own
// event stream.
type agentFactory func(events []*llmx.Event) *llmx.Agent

func newAgentFactory(t *testing.T, models *llmx.Models, opts ...llmx.AgentOption) agentFactory {
	return func(events []*llmx.Event) *llmx.Agent {
		t.Helper()
		agent, err := llmx.NewAgent("test", "", models, events,
			append([]llmx.AgentOption{llmx.WithModel("fake")}, opts...)...)
		if err != nil {
			t.Fatal(err)
		}
		return agent
	}
}

// newTestModels registers one model "fake" with a 100 token window.
func newTestModels(
	t *testing.T,
	complete func([]*types.Message) *types.Completion,
) *llmx.Models {
	t.Helper()
	models := llmx.NewModels()
	if err := models.RegisterProvider("fake", testProvider{complete: complete}); err != nil {
		t.Fatal(err)
	}
	if _, err := models.AddModel(llmx.ModelConfig{
		Name:     "fake",
		Model:    "fake-provider-model",
		Provider: "fake",
		MaxToken: 100,
	}); err != nil {
		t.Fatal(err)
	}
	return models
}

// longToolHistory is a tool call and its long result, followed by a usage that
// fills the whole context window.
func longToolHistory() []*llmx.Event {
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
	return []*llmx.Event{
		llmx.NewEventMessage(toolCall),
		llmx.NewEventMessage(toolResult),
		llmx.NewEventUsage(types.CompletionUsage{
			PromptTokens: 100,
			TotalTokens:  100,
		}),
	}
}

// filler makes a message long enough that a summary bridge is shorter than it.
var filler = strings.Repeat("filler ", 400)

// latestInput is the newest user message, which a compression always keeps.
func latestInput() *types.Message {
	return types.NewTextMessage(types.MessageRoleUser, "latest")
}

func toolResultsOf(messages []*types.Message) []string {
	results := make([]string, 0)
	for _, message := range messages {
		for _, part := range message.Parts {
			if part.ToolResult != nil {
				results = append(results, part.ToolResult.Result)
			}
		}
	}
	return results
}

// omitOptions keeps only the newest message and omits every older tool result
// as soon as any context size is reported.
var omitOptions = Options{
	KeepRecent:        1,
	ToolResultLevel:   1,
	ToolResultReplace: "omitted",
}

// recordedCompressions returns the compressions recorded by state events.
func recordedCompressions(t *testing.T, events []*llmx.Event) []*compression {
	t.Helper()
	var compressions []*compression
	for _, event := range events {
		if event.State == nil {
			continue
		}
		for key, value := range event.State.State {
			if !strings.HasPrefix(key, compressionStatePrefix) {
				continue
			}
			compression := &compression{}
			if err := json.Unmarshal([]byte(value), compression); err != nil {
				t.Fatal(err)
			}
			compressions = append(compressions, compression)
		}
	}
	return compressions
}

// rawHistory records the tool results of the messages it is handed.
type rawHistory struct {
	llmx.NopInterceptor
	seen *[]string
}

func (r rawHistory) InterceptLLM(next llmx.LLMHandler) llmx.LLMHandler {
	return func(ctx llmx.LLMContext) (*types.Completion, error) {
		*r.seen = toolResultsOf(ctx.Request().Messages)
		return next(ctx)
	}
}

func TestCompressRecordsBeforeTheModelCall(t *testing.T) {
	var seen, raw []string
	models := newTestModels(t, func(messages []*types.Message) *types.Completion {
		seen = toolResultsOf(messages)
		return &types.Completion{
			Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
		}
	})
	agent := newAgentFactory(t, models,
		llmx.WithInterceptor(rawHistory{seen: &raw}, New(models, omitOptions)),
	)

	session := agent(longToolHistory())
	result, err := session.Run(context.Background(), llmx.NewUserInput("latest"))
	if err != nil {
		t.Fatal(err)
	}
	if len(seen) != 1 || seen[0] != "omitted" {
		t.Fatalf("compressed history was not sent to the model: %+v", seen)
	}
	// running, user input, the recorded compression, then the completion.
	if len(result.Events) < 4 ||
		result.Events[1].Message == nil ||
		result.Events[2].State == nil ||
		result.Events[3].Message == nil {
		t.Fatalf("compression was not recorded before the completion: %+v", result.Events)
	}
	compressions := recordedCompressions(t, result.Events[2:3])
	if len(compressions) != 1 ||
		compressions[0].ToolResult == nil ||
		compressions[0].ToolResult.Result != "omitted" {
		t.Fatalf("unexpected recorded compression: %+v", compressions)
	}
	// The event stream itself stays whole: an LLM interceptor registered
	// before compress still sees the original tool result.
	if len(raw) != 1 || raw[0] == "omitted" {
		t.Fatalf("session history was compressed: %+v", raw)
	}
}

func TestCompressionReplaysFromTheEventLog(t *testing.T) {
	var mu sync.Mutex
	var seen [][]string
	models := newTestModels(t, func(messages []*types.Message) *types.Completion {
		mu.Lock()
		seen = append(seen, toolResultsOf(messages))
		mu.Unlock()
		return &types.Completion{
			Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
		}
	})
	agent := newAgentFactory(t, models,
		llmx.WithInterceptor(New(models, omitOptions)),
	)

	initialHistory := longToolHistory()
	firstSession := agent(initialHistory)
	first, err := firstSession.Run(context.Background(), llmx.NewUserInput("first"))
	if err != nil {
		t.Fatal(err)
	}
	if got := len(recordedCompressions(t, first.Events)); got != 1 {
		t.Fatalf("first run recorded %d compressions, want 1", got)
	}

	stored := append([]*llmx.Event(nil), initialHistory...)
	stored = append(stored, first.Events...)
	secondSession := agent(stored)
	second, err := secondSession.Run(context.Background(), llmx.NewUserInput("second"))
	if err != nil {
		t.Fatal(err)
	}
	if got := len(recordedCompressions(t, second.Events)); got != 0 {
		t.Fatalf("restored history was compressed again: %d", got)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 2 ||
		len(seen[0]) != 1 || seen[0][0] != "omitted" ||
		len(seen[1]) != 1 || seen[1][0] != "omitted" {
		t.Fatalf("recorded compression was not replayed: %+v", seen)
	}
}

// A tool-result compression committed before a summary may reach into the tail
// the summary kept: replay must apply it, not skip to the summary.
func TestReplayAppliesToolResultCompressionsReachingPastTheSummary(t *testing.T) {
	var mu sync.Mutex
	var seen []string
	models := newTestModels(t, func(messages []*types.Message) *types.Completion {
		mu.Lock()
		seen = toolResultsOf(messages)
		mu.Unlock()
		return &types.Completion{
			Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
		}
	})

	old := types.NewTextMessage(types.MessageRoleUser, "old")
	toolCall := types.NewMessage(types.MessageRoleAssistant)
	toolCall.Parts = append(toolCall.Parts, &types.MessagePart{
		ToolCall: &types.MessageToolCall{ID: "call", Name: "long_tool", Arguments: "{}"},
	})
	toolResult := types.NewToolResultMessage(
		"call",
		"long_tool",
		"a very long tool result that should be omitted",
		"",
	)
	record := func(index int, recorded *compression) *llmx.Event {
		data, err := json.Marshal(recorded)
		if err != nil {
			t.Fatal(err)
		}
		return llmx.NewEventState(map[string]string{
			compressionStatePrefix + strconv.Itoa(index): string(data),
		}, true)
	}
	// Both compressions were decided on the reported size, so none is
	// generated: the model sees the replay alone.
	history := []*llmx.Event{
		llmx.NewEventMessage(old),
		llmx.NewEventMessage(toolCall),
		llmx.NewEventMessage(toolResult),
		record(0, &compression{
			ToolResult:     &toolResultCompression{Result: "omitted"},
			UntilMessageID: toolResult.ID,
			Tokens:         100,
		}),
		record(1, &compression{
			Summary:        types.NewTextMessage(types.MessageRoleUser, "bridge"),
			UntilMessageID: old.ID,
			Tokens:         100,
		}),
		llmx.NewEventUsage(types.CompletionUsage{PromptTokens: 100, TotalTokens: 100}),
	}

	agent := newAgentFactory(t, models, llmx.WithInterceptor(New(models, omitOptions)))
	result, err := agent(history).Run(context.Background(), llmx.NewUserInput("latest"))
	if err != nil {
		t.Fatal(err)
	}
	if got := len(recordedCompressions(t, result.Events)); got != 0 {
		t.Fatalf("replay generated %d compressions, want 0", got)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 1 || seen[0] != "omitted" {
		t.Fatalf("tool-result compression before the summary was not replayed: %+v", seen)
	}
}

func TestCompressionSurvivesAFailedModelCall(t *testing.T) {
	fail := true
	calls := 0
	var seen []string
	models := newTestModels(t, func(messages []*types.Message) *types.Completion {
		calls++
		seen = toolResultsOf(messages)
		if fail {
			return nil // the model call fails: no message
		}
		return &types.Completion{
			Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
		}
	})
	withSummary := omitOptions
	withSummary.SummaryLevel = 0.5
	agent := newAgentFactory(t, models, llmx.WithInterceptor(New(models, omitOptions)))
	rebuilt := newAgentFactory(t, models, llmx.WithInterceptor(New(models, withSummary)))

	history := longToolHistory()
	first, err := agent(history).Run(context.Background(), llmx.NewUserInput("latest"))
	if err == nil {
		t.Fatal("the failed model call was not reported")
	}
	if got := len(recordedCompressions(t, first.Events)); got != 1 {
		t.Fatalf("the compression was not recorded before the model call: %d", got)
	}

	// Rebuilt from what was appended, the agent replays the compression. No
	// model call reported a size since, so it does not compress again -- the
	// summary level would otherwise trigger on the stale size.
	fail = false
	second, err := rebuilt(append(history, first.Events...)).
		Run(context.Background(), llmx.NewUserInput("again"))
	if err != nil {
		t.Fatal(err)
	}
	if got := len(recordedCompressions(t, second.Events)); got != 0 || calls != 2 {
		t.Fatalf("compressed again on a stale size: compressions=%d calls=%d", got, calls)
	}
	if len(seen) != 1 || seen[0] != "omitted" {
		t.Fatalf("the recorded compression was not replayed: %v", seen)
	}
}

func TestContextTokensIsTheNewestUsage(t *testing.T) {
	events := []*llmx.Event{
		llmx.NewEventUsage(types.CompletionUsage{TotalTokens: 100}),
		llmx.NewEventMessage(latestInput()),
		llmx.NewEventUsage(types.CompletionUsage{TotalTokens: 150}),
	}
	// A completion reports the whole context it consumed: 150, not 100 + 150.
	if got := contextTokens(events); got != 150 {
		t.Fatalf("context tokens = %d, want 150", got)
	}
}
func TestCompressReportsOneToolResultCompression(t *testing.T) {
	models := newTestModels(t, nil)
	toolResult := types.NewToolResultMessage(
		"call",
		"tool",
		"a long historical tool result",
		"",
	)

	compression, err := compress(t.Context(), models, &compressInput{
		Model:    "fake",
		Messages: []*types.Message{toolResult, latestInput()},
		Tokens:   100,
		Options:  &omitOptions,
	})
	if err != nil {
		t.Fatal(err)
	}
	if compression == nil || compression.ToolResult == nil ||
		compression.ToolResult.Result != "omitted" ||
		compression.UntilMessageID != toolResult.ID {
		t.Fatalf("unexpected compression: %+v", compression)
	}

	compressed, changed, err := apply([]*types.Message{toolResult}, compression)
	if err != nil {
		t.Fatal(err)
	}
	if !changed || compressed[0].Parts[0].ToolResult.Result != "omitted" {
		t.Fatalf("unexpected compressed result: %+v", compressed)
	}
}

func TestCompressSkipsAToolResultCompressionThatChangesNothing(t *testing.T) {
	models := newTestModels(t, nil)
	compression, err := compress(t.Context(), models, &compressInput{
		Model: "fake",
		Messages: []*types.Message{
			types.NewToolResultMessage("call", "tool", "short", ""),
			latestInput(),
		},
		Tokens:  100,
		Options: &omitOptions,
	})
	if err != nil {
		t.Fatal(err)
	}
	if compression != nil {
		t.Fatalf("a compression that changes nothing was reported: %+v", compression)
	}
}

func TestCompressDoesNotChangeTheHistoryItIsGiven(t *testing.T) {
	models := newTestModels(t, nil)
	toolResult := types.NewToolResultMessage(
		"call",
		"tool",
		"a long historical tool result",
		"",
	)
	history := []*types.Message{toolResult, latestInput()}

	if _, err := compress(t.Context(), models, &compressInput{
		Model:    "fake",
		Messages: history,
		Tokens:   100,
		Options:  &omitOptions,
	}); err != nil {
		t.Fatal(err)
	}
	if history[0] != toolResult ||
		toolResult.Parts[0].ToolResult.Result != "a long historical tool result" {
		t.Fatalf("compress changed the input history: %+v", history)
	}
}

func TestCompressUsesOnlyInputTokensToDecide(t *testing.T) {
	models := newTestModels(t, nil)

	compression, err := compress(t.Context(), models, &compressInput{
		Model: "fake",
		Messages: []*types.Message{types.NewToolResultMessage(
			"call",
			"tool",
			"a very long historical tool result that would otherwise be estimated",
			"",
		), latestInput()},
		Options: &omitOptions,
	})
	if err != nil {
		t.Fatal(err)
	}
	if compression != nil {
		t.Fatalf("compression ignored zero input tokens: %+v", compression)
	}
}

func TestCompressSummaryReplacesTheCoveredRange(t *testing.T) {
	models := newTestModels(t, fixedCompletion(&types.Completion{
		Message: types.NewTextMessage(types.MessageRoleAssistant, "summary"),
	}))

	old := types.NewTextMessage(types.MessageRoleAssistant, "old "+filler)
	compression, err := compress(t.Context(), models, &compressInput{
		Model:    "fake",
		Messages: []*types.Message{old, latestInput()},
		Tokens:   90,
		Options:  &Options{SummaryLevel: 0.5, KeepRecent: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if compression == nil || compression.Summary == nil ||
		compression.UntilMessageID != old.ID {
		t.Fatalf("unexpected compression: %+v", compression)
	}
	// The bridge is new history, so it must not reuse the id of a message the
	// summary replaced.
	if compression.Summary.ID == old.ID ||
		!strings.Contains(compression.Summary.Text(), "summary") {
		t.Fatalf("unexpected summary bridge: %+v", compression.Summary)
	}
}

func TestCompressSummaryKeepsUserPromptsAcrossRepeatedSummaries(t *testing.T) {
	models := newTestModels(t, fixedCompletion(&types.Completion{
		Message: types.NewTextMessage(types.MessageRoleAssistant, "summary"),
	}))
	summarize := func(history []*types.Message) *types.Message {
		t.Helper()
		compression, err := compress(t.Context(), models, &compressInput{
			Model:    "fake",
			Messages: history,
			Tokens:   90,
			Options: &Options{
				SummaryLevel:     0.5,
				SummaryBridgeTag: "summary",
				KeepRecent:       1,
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		if compression == nil || compression.Summary == nil {
			t.Fatalf("unexpected compression: %+v", compression)
		}
		compressed, _, err := apply(history, compression)
		if err != nil {
			t.Fatal(err)
		}
		return compressed[0]
	}

	first := summarize([]*types.Message{
		types.NewTextMessage(types.MessageRoleUser, "first question"),
		types.NewTextMessage(types.MessageRoleAssistant, "first answer "+filler),
		latestInput(),
	})
	second := summarize([]*types.Message{
		first,
		types.NewTextMessage(types.MessageRoleUser, "second question"),
		types.NewTextMessage(types.MessageRoleAssistant, "second answer "+filler),
		latestInput(),
	})

	// The second bridge must list the prompts the first bridge replaced, and
	// must not list the first bridge itself as a prompt.
	if got, want := second.GetState(summaryUserPromptsState),
		`["first question","second question"]`; got != want {
		t.Fatalf("unexpected carried prompts: %s, want %s", got, want)
	}
	text := second.Parts[0].Text.TextWithInstruction()
	if !strings.Contains(text, "- first question") ||
		!strings.Contains(text, "- second question") {
		t.Fatalf("bridge lost user prompts: %s", text)
	}
}

// toolStep is one assistant tool call and its result.
func toolStep(id string, result string) []*types.Message {
	call := types.NewMessage(types.MessageRoleAssistant)
	call.Parts = append(call.Parts, &types.MessagePart{
		ToolCall: &types.MessageToolCall{ID: id, Name: "tool", Arguments: "{}"},
	})
	return []*types.Message{call, types.NewToolResultMessage(id, "tool", result, "")}
}

func estimate(messages ...*types.Message) int64 {
	var total int64
	for _, message := range messages {
		tokens, _ := message.EstimateTokens()
		total += tokens
	}
	return total
}

func TestSplitHistoryRecentCutsInsideOneUserTurn(t *testing.T) {
	// One request followed by three tool steps: there is no later user
	// message to cut at, the cut has to fall between steps.
	history := []*types.Message{types.NewTextMessage(types.MessageRoleUser, "task")}
	for _, id := range []string{"1", "2", "3"} {
		history = append(history, toolStep(id, "a tool result of step "+id)...)
	}
	lastStep := history[5:]

	compressible, kept := splitHistoryRecent(history, estimate(lastStep...))
	if len(compressible) != 5 || len(kept) != 2 || kept[0] != history[5] {
		t.Fatalf("cut is not before the last step: kept %d messages", len(kept))
	}

	// A budget the last tool result alone already fills must not cut between
	// the result and its tool call.
	_, kept = splitHistoryRecent(history, estimate(history[6]))
	if kept[0] != history[5] {
		t.Fatalf("cut separated a tool result from its call: %+v", kept[0])
	}

	// A budget slightly above the last step moves the cut back a whole step.
	_, kept = splitHistoryRecent(history, estimate(lastStep...)+1)
	if len(kept) != 4 || kept[0] != history[3] {
		t.Fatalf("cut is not before the second last step: kept %d messages", len(kept))
	}

	// A history that fits in the budget has nothing to compress.
	compressible, kept = splitHistoryRecent(history, estimate(history...)+1)
	if len(compressible) != 0 || len(kept) != len(history) {
		t.Fatalf("a history within budget was split: %d", len(compressible))
	}
}

func TestSummaryBridgeMarksAnInProgressRequest(t *testing.T) {
	models := newTestModels(t, fixedCompletion(&types.Completion{
		Message: types.NewTextMessage(types.MessageRoleAssistant, "snapshot"),
	}))
	bridge := func(history []*types.Message) string {
		t.Helper()
		compression, err := compress(t.Context(), models, &compressInput{
			Model:    "fake",
			Messages: history,
			Tokens:   90,
			Options:  &Options{SummaryLevel: 0.5, KeepRecent: 1},
		})
		if err != nil {
			t.Fatal(err)
		}
		if compression == nil || compression.Summary == nil {
			t.Fatalf("unexpected compression: %+v", compression)
		}
		return compression.Summary.Text()
	}

	task := append(
		[]*types.Message{types.NewTextMessage(types.MessageRoleUser, "task")},
		toolStep("1", "result "+filler)...,
	)
	task = append(task, types.NewTextMessage(types.MessageRoleAssistant, "working"))
	if text := bridge(task); !strings.Contains(text, "still in progress") ||
		!strings.Contains(text, "- task") {
		t.Fatalf("bridge of a cut inside a request does not mark it in progress: %s", text)
	}

	chat := []*types.Message{
		types.NewTextMessage(types.MessageRoleUser, "question"),
		types.NewTextMessage(types.MessageRoleAssistant, "answer "+filler),
		latestInput(),
	}
	if text := bridge(chat); strings.Contains(text, "still in progress") {
		t.Fatalf("bridge of a finished request marks it in progress: %s", text)
	}
}

type emptyParam struct{}

func TestCompressShrinksALongRunStartedByOneRequest(t *testing.T) {
	const steps = 4
	var calls int
	var lastSeen []string
	models := newTestModels(t, func(messages []*types.Message) *types.Completion {
		calls++
		lastSeen = toolResultsOf(messages)
		// Every call reports a context size above the level, a new one each
		// time as a growing history would.
		usage := types.CompletionUsage{TotalTokens: int64(90 + calls)}
		if calls > steps {
			return &types.Completion{
				Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
				Usage:   usage,
			}
		}
		return &types.Completion{
			Message: toolStep("call-"+strconv.Itoa(calls), "")[0],
			Usage:   usage,
		}
	})
	tool := llmx.MustNewTool(
		"tool",
		"",
		func(llmx.ToolContext, *emptyParam) (any, error) {
			return "a long tool result the run no longer needs", nil
		},
	)
	agent := newAgentFactory(t, models,
		llmx.WithTools(tool),
		llmx.WithInterceptor(New(models, omitOptions)),
	)
	session := agent(nil)
	if _, err := session.Run(context.Background(), llmx.NewUserInput("long task")); err != nil {
		t.Fatal(err)
	}

	// The whole run is one user turn. Only the newest step is kept, every
	// older tool result of that same turn is omitted.
	if calls != steps+1 || len(lastSeen) != steps {
		t.Fatalf("unexpected run: calls=%d results=%v", calls, lastSeen)
	}
	for index, result := range lastSeen[:steps-1] {
		if result != "omitted" {
			t.Fatalf("tool result %d of the running request was kept: %v", index, lastSeen)
		}
	}
	if lastSeen[steps-1] == "omitted" {
		t.Fatalf("the newest tool result was omitted: %v", lastSeen)
	}
}

func TestCompressFallsBackWhenAHigherCompressionShrinksNothing(t *testing.T) {
	models := newTestModels(t, fixedCompletion(&types.Completion{
		Message: types.NewTextMessage(types.MessageRoleAssistant, "summary"),
	}))
	old := types.NewTextMessage(types.MessageRoleAssistant, "old "+filler)
	// No tool result is left to omit: the tool-result compression, tried first
	// on equal thresholds, has nothing to shrink and the summary takes over.
	compression, err := compress(t.Context(), models, &compressInput{
		Model:    "fake",
		Messages: []*types.Message{old, latestInput()},
		Tokens:   90,
		Options:  &Options{ToolResultLevel: 0.5, SummaryLevel: 0.5, KeepRecent: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if compression == nil || compression.Summary == nil || compression.UntilMessageID != old.ID {
		t.Fatalf("summary did not take over: %+v", compression)
	}
}

func TestFailedSummaryFallsBackAndDoesNotFailTheModelCall(t *testing.T) {
	var seen []string
	models := newTestModels(t, func(messages []*types.Message) *types.Completion {
		if messages[len(messages)-1].Text() == summaryRequest {
			return &types.Completion{} // the summary call fails: no message
		}
		seen = toolResultsOf(messages)
		return &types.Completion{
			Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
		}
	})
	options := omitOptions
	options.SummaryLevel = 2
	agent := newAgentFactory(t, models, llmx.WithInterceptor(New(models, options)))

	result, err := agent(longToolHistory()).Run(context.Background(), llmx.NewUserInput("latest"))
	if err != nil {
		t.Fatalf("a failed summary failed the model call: %v", err)
	}
	compressions := recordedCompressions(t, result.Events)
	if len(compressions) != 1 || compressions[0].ToolResult == nil {
		t.Fatalf("did not fall back to the tool-result compression: %+v", compressions)
	}
	if len(seen) != 1 || seen[0] != "omitted" {
		t.Fatalf("the fallback compression was not sent: %v", seen)
	}
}

func TestBrokenCompressionDoesNotFailTheModelCall(t *testing.T) {
	var seen []string
	models := newTestModels(t, func(messages []*types.Message) *types.Completion {
		seen = toolResultsOf(messages)
		return &types.Completion{
			Message: types.NewTextMessage(types.MessageRoleAssistant, "done"),
		}
	})
	agent := newAgentFactory(t, models, llmx.WithInterceptor(New(models, omitOptions)))

	broken, err := json.Marshal(&compression{
		ToolResult:     &toolResultCompression{Result: "omitted"},
		UntilMessageID: "missing",
	})
	if err != nil {
		t.Fatal(err)
	}
	history := append(longToolHistory(), llmx.NewEventState(map[string]string{
		compressionStatePrefix + "0": string(broken),
	}, true))
	result, err := agent(history).Run(context.Background(), llmx.NewUserInput("latest"))
	if err != nil {
		t.Fatalf("a broken compression failed the model call: %v", err)
	}
	// Nothing is recorded after a compression that never replays, and the
	// model gets the history uncompressed.
	if got := len(recordedCompressions(t, result.Events)); got != 0 {
		t.Fatalf("compressed after a broken compression: %d", got)
	}
	if len(seen) != 1 || seen[0] == "omitted" {
		t.Fatalf("unexpected history sent: %v", seen)
	}
}

func TestSummaryBridgeCarriesTheNewestUserPromptsOnly(t *testing.T) {
	var sent []*types.Message
	models := newTestModels(t, func(messages []*types.Message) *types.Completion {
		sent = messages
		return &types.Completion{Message: types.NewTextMessage(types.MessageRoleAssistant,
			"<scratchpad>private notes</scratchpad>\n<state_snapshot>snapshot</state_snapshot>")}
	})
	var history []*types.Message
	for index := range summaryUserPromptsLimit + 3 {
		history = append(history,
			types.NewTextMessage(types.MessageRoleUser, "question "+strconv.Itoa(index)),
			types.NewTextMessage(types.MessageRoleAssistant, filler),
		)
	}
	history = append(history, latestInput())

	compression, err := compress(t.Context(), models, &compressInput{
		Model:    "fake",
		Messages: history,
		Tokens:   90,
		Options:  &Options{SummaryLevel: 0.5, KeepRecent: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if compression == nil || compression.Summary == nil {
		t.Fatalf("unexpected compression: %+v", compression)
	}
	if got, want := compression.Summary.GetState(summaryUserPromptsState),
		`["question 3","question 4","question 5","question 6","question 7"]`; got != want {
		t.Fatalf("carried prompts = %s, want %s", got, want)
	}
	text := compression.Summary.Text()
	if strings.Contains(text, "question 2") || strings.Contains(text, "private notes") ||
		!strings.Contains(text, "<state_snapshot>snapshot</state_snapshot>") {
		t.Fatalf("unexpected bridge: %s", text)
	}
	// The summary model is asked for the snapshot after the history.
	if last := sent[len(sent)-1]; last.Role != types.MessageRoleUser || last.Text() != summaryRequest {
		t.Fatalf("summary request does not end with the snapshot request: %+v", last)
	}
}

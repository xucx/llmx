package llmx

import (
	"context"
	"strings"
	"testing"

	"github.com/xucx/llmx/types"
)

type compressTestProvider struct {
	types.ProviderNop
	completion *types.Completion
}

func (p compressTestProvider) Completion(
	context.Context,
	string,
	[]*types.Message,
	...types.CompletionOption,
) (*types.Completion, error) {
	return p.completion, nil
}

func TestApplyCompressionReplacesToolResultsAndDoesNotMutateInput(t *testing.T) {
	oldResult := &types.MessageToolResult{
		ID:     "call-old",
		Result: "a long historical tool result",
	}
	oldTool := types.NewMessage(types.MessageRoleTool)
	oldTool.Parts = append(
		oldTool.Parts,
		&types.MessagePart{ToolResult: oldResult},
	)
	failedResult := &types.MessageToolResult{
		ID:     "call-failed",
		Result: "a long failed tool result",
		Error:  "boom",
	}
	failedTool := types.NewMessage(types.MessageRoleTool)
	failedTool.Parts = append(
		failedTool.Parts,
		&types.MessagePart{ToolResult: failedResult},
	)
	history := []*types.Message{
		types.NewTextMessage(types.MessageRoleUser, "old"),
		oldTool,
		types.NewTextMessage(types.MessageRoleUser, "recent"),
		failedTool,
	}

	compressed, reduced, err := ApplyCompression(history, &EventCompression{
		ToolResult:     &CompressResultToolResult{Result: "omitted"},
		UntilMessageID: history[len(history)-1].ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	if reduced <= 0 {
		t.Fatalf("replacing a long tool result must reduce tokens")
	}
	if len(compressed) != len(history) {
		t.Fatalf("compressed history dropped messages: %+v", compressed)
	}
	if compressed[0].Text() != "old" || compressed[2].Text() != "recent" {
		t.Fatalf("non tool messages were not kept: %+v", compressed)
	}
	if got := compressed[1].Parts[0].ToolResult.Result; got != "omitted" {
		t.Fatalf("tool result = %q, want omitted", got)
	}
	if got := compressed[3].Parts[0].ToolResult.Result; got != failedResult.Result {
		t.Fatalf("failed tool result changed to %q", got)
	}
	if oldResult.Result != "a long historical tool result" {
		t.Fatal("input history was mutated")
	}
	if history[1].Parts[0].ToolResult.Result != oldResult.Result {
		t.Fatal("input history slice was mutated")
	}
}

func TestApplyCompressionLeavesMessagesAfterTheBoundaryAlone(t *testing.T) {
	covered := types.NewToolResultMessage(
		"call-covered",
		"tool",
		"a long historical tool result",
		"",
	)
	uncovered := types.NewToolResultMessage(
		"call-uncovered",
		"tool",
		"a long recent tool result",
		"",
	)
	history := []*types.Message{covered, uncovered}

	compressed, _, err := ApplyCompression(history, &EventCompression{
		ToolResult:     &CompressResultToolResult{Result: "omitted"},
		UntilMessageID: covered.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(compressed) != 2 {
		t.Fatalf("compressed history dropped messages: %+v", compressed)
	}
	if compressed[0].Parts[0].ToolResult.Result != "omitted" {
		t.Fatalf("covered tool result was kept: %+v", compressed[0])
	}
	if compressed[1].Parts[0].ToolResult.Result != "a long recent tool result" {
		t.Fatalf("uncovered tool result was replaced: %+v", compressed[1])
	}
}

func TestApplyCompressionSummaryReplacesTheCoveredRange(t *testing.T) {
	old := types.NewTextMessage(types.MessageRoleUser, "old")
	boundary := types.NewTextMessage(types.MessageRoleAssistant, "boundary")
	recent := types.NewTextMessage(types.MessageRoleUser, "recent")
	bridge := types.NewTextMessage(types.MessageRoleUser, "summary")

	compressed, _, err := ApplyCompression(
		[]*types.Message{old, boundary, recent},
		&EventCompression{
			Summary:        &CompressResultSummary{Message: bridge},
			UntilMessageID: boundary.ID,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(compressed) != 2 ||
		compressed[0] != bridge ||
		compressed[1] != recent {
		t.Fatalf("unexpected compressed history: %+v", compressed)
	}
}

func TestApplyCompressionRejectsAnUnknownBoundary(t *testing.T) {
	history := []*types.Message{types.NewTextMessage(types.MessageRoleUser, "hi")}

	if _, _, err := ApplyCompression(history, &EventCompression{
		ToolResult:     &CompressResultToolResult{Result: "omitted"},
		UntilMessageID: "missing",
	}); err == nil {
		t.Fatal("an unknown boundary was accepted")
	}
}

func TestCompressReportsToolResultTokensInOneEvent(t *testing.T) {
	models := NewModels()
	models.models["fake"] = &Model{Name: "fake", MaxToken: 100}
	toolResult := types.NewToolResultMessage(
		"call",
		"tool",
		"a long historical tool result",
		"",
	)

	events, err := Compress(
		t.Context(),
		models,
		&CompressInput{
			Model:    "fake",
			Messages: []*types.Message{toolResult},
			Tokens:   100,
			Options: &CompressOptions{
				ToolResultLevel:   1,
				ToolResultReplace: "omitted",
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 {
		t.Fatalf("got %d compression events, want 1: %+v", len(events), events)
	}
	event := events[0]
	if event == nil || event.Compression == nil ||
		event.Compression.ToolResult == nil {
		t.Fatalf("unexpected compression event: %+v", event)
	}
	if event.Compression.ToolResult.Result != "omitted" {
		t.Fatalf("unexpected compression event: %+v", event.Compression)
	}
	if event.Compression.UntilMessageID != toolResult.ID {
		t.Fatalf("unexpected compression boundary: %+v", event.Compression)
	}

	compressed, reduced, err := ApplyCompression(
		[]*types.Message{toolResult},
		event.Compression,
	)
	if err != nil {
		t.Fatal(err)
	}
	if compressed[0].Parts[0].ToolResult.Result != "omitted" {
		t.Fatalf("unexpected compressed result: %+v", compressed)
	}

	// The reduction is not stored in the event: applying the compression
	// derives it from the tool results it actually replaced.
	before, _ := toolResult.EstimateTokens()
	after, _ := compressed[0].EstimateTokens()
	wantReduce := subTokens(before, after)
	if wantReduce <= 0 {
		t.Fatalf("replacing a long tool result must reduce tokens")
	}
	if reduced != wantReduce {
		t.Fatalf("unexpected compression tokens: %d, want %d", reduced, wantReduce)
	}
}

func TestCompressDoesNotChangeTheHistoryItIsGiven(t *testing.T) {
	models := NewModels()
	models.models["fake"] = &Model{Name: "fake", MaxToken: 100}
	toolResult := types.NewToolResultMessage(
		"call",
		"tool",
		"a long historical tool result",
		"",
	)
	history := []*types.Message{toolResult}

	if _, err := Compress(t.Context(), models, &CompressInput{
		Model:    "fake",
		Messages: history,
		Tokens:   100,
		Options: &CompressOptions{
			ToolResultLevel:   1,
			ToolResultReplace: "omitted",
		},
	}); err != nil {
		t.Fatal(err)
	}
	if history[0] != toolResult ||
		toolResult.Parts[0].ToolResult.Result != "a long historical tool result" {
		t.Fatalf("compress changed the input history: %+v", history)
	}
}

func TestCompressReturnsNilWithoutOptions(t *testing.T) {
	models := NewModels()
	models.models["fake"] = &Model{Name: "fake", MaxToken: 100}

	events, err := Compress(t.Context(), models, &CompressInput{
		Model:    "fake",
		Messages: []*types.Message{types.NewTextMessage(types.MessageRoleUser, "hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 0 {
		t.Fatalf("unexpected compression events: %+v", events)
	}
}

func TestCompressUsesOnlyInputTokensToDecide(t *testing.T) {
	models := NewModels()
	models.models["fake"] = &Model{Name: "fake", MaxToken: 100}

	events, err := Compress(t.Context(), models, &CompressInput{
		Model: "fake",
		Messages: []*types.Message{types.NewToolResultMessage(
			"call",
			"tool",
			"a very long historical tool result that would otherwise be estimated",
			"",
		)},
		Options: &CompressOptions{
			ToolResultLevel:   1,
			ToolResultReplace: "omitted",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 0 {
		t.Fatalf("compression ignored zero input tokens: %+v", events)
	}
}

func TestCompressSummaryTokensUseInputMinusSummaryReduction(t *testing.T) {
	models := NewModels()
	completion := &types.Completion{
		Message: types.NewTextMessage(types.MessageRoleAssistant, "summary"),
		Usage: types.CompletionUsage{
			PromptTokens:     80,
			CompletionTokens: 20,
			TotalTokens:      100,
		},
	}
	models.models["fake"] = &Model{
		Name:     "fake",
		Model:    "fake-provider-model",
		MaxToken: 100,
		Provider: compressTestProvider{completion: completion},
	}

	old := types.NewTextMessage(types.MessageRoleUser, "old")
	events, err := Compress(t.Context(), models, &CompressInput{
		Model:    "fake",
		Messages: []*types.Message{old},
		Tokens:   90,
		Options:  &CompressOptions{SummaryLevel: 0.5},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].Compression == nil ||
		events[0].Compression.Summary == nil {
		t.Fatalf("unexpected compression events: %+v", events)
	}

	compression := events[0].Compression
	// The reduction is derived from the recorded summary usage when the event
	// is applied.
	_, reduced, err := ApplyCompression([]*types.Message{old}, compression)
	if err != nil {
		t.Fatal(err)
	}
	const wantReduce = 60 // 80 input - 20 output
	if reduced != wantReduce {
		t.Fatalf("unexpected summary token reduction: %d, want %d", reduced, wantReduce)
	}
	if compression.UntilMessageID != old.ID {
		t.Fatalf("unexpected summary boundary: %+v", compression)
	}
	// The bridge is new history, so it must not reuse the id of a message the
	// summary replaced.
	if compression.Summary.Message.ID == old.ID ||
		!strings.Contains(compression.Summary.Message.Text(), "summary") {
		t.Fatalf("unexpected summary bridge: %+v", compression.Summary.Message)
	}
}

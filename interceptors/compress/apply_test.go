package compress

import (
	"testing"

	"github.com/xucx/llmx/types"
)

func TestApplyReplacesToolResultsAndDoesNotMutateInput(t *testing.T) {
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

	compressed, changed, err := apply(history, &compression{
		ToolResult:     &toolResultCompression{Result: "omitted"},
		UntilMessageID: history[len(history)-1].ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !changed {
		t.Fatalf("replacing a long tool result must change the history")
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

func TestApplyLeavesMessagesAfterTheBoundaryAlone(t *testing.T) {
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

	compressed, _, err := apply(history, &compression{
		ToolResult:     &toolResultCompression{Result: "omitted"},
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

func TestApplySummaryReplacesTheCoveredRange(t *testing.T) {
	old := types.NewTextMessage(types.MessageRoleUser, "old")
	boundary := types.NewTextMessage(types.MessageRoleAssistant, "boundary")
	recent := types.NewTextMessage(types.MessageRoleUser, "recent")
	bridge := types.NewTextMessage(types.MessageRoleUser, "summary")

	compressed, _, err := apply(
		[]*types.Message{old, boundary, recent},
		&compression{
			Summary:        bridge,
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

func TestApplyRejectsAnUnknownBoundary(t *testing.T) {
	history := []*types.Message{types.NewTextMessage(types.MessageRoleUser, "hi")}

	if _, _, err := apply(history, &compression{
		ToolResult:     &toolResultCompression{Result: "omitted"},
		UntilMessageID: "missing",
	}); err == nil {
		t.Fatal("an unknown boundary was accepted")
	}
}

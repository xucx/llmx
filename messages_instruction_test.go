package llmx

import "testing"

func TestBuildMessagesKeepsInputInstruction(t *testing.T) {
	input := &Input{Mode: InputQueue, Content: []Content{{
		Text:        "hi",
		Instruction: map[string]string{"system-reminder": "now"},
	}}}
	event := NewInputEvent(input)
	// The event owns a copy: changing the input afterwards must not leak in.
	input.Content[0].Instruction["system-reminder"] = "changed"

	messages := BuildMessages(Fold([]*Event{event}))
	if len(messages) != 1 || len(messages[0].Parts) != 1 || messages[0].Parts[0].Text == nil {
		t.Fatalf("unexpected messages %+v", messages)
	}
	text := messages[0].Parts[0].Text
	if text.Text != "hi" || len(text.Instruction["system-reminder"]) != 1 || text.Instruction["system-reminder"][0] != "now" {
		t.Fatalf("unexpected text part %+v", text)
	}
}

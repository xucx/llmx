package llmx

import (
	"encoding/json"
	"testing"
)

func TestInputToolsDeclarationSurvivesJSON(t *testing.T) {
	for _, tc := range []struct {
		name  string
		tools []ToolSpec
		want  int
	}{{"keep", nil, 1}, {"clear", []ToolSpec{}, 0}} {
		t.Run(tc.name, func(t *testing.T) {
			first := NewInputEvent(&Input{Mode: InputQueue, Content: []Content{{Text: "first"}}, Tools: []ToolSpec{{Name: "device"}}})
			next := NewInputEvent(&Input{Mode: InputQueue, Content: []Content{{Text: "next"}}, Tools: tc.tools})
			data, err := json.Marshal(next)
			if err != nil {
				t.Fatal(err)
			}
			var restored Event
			if err := json.Unmarshal(data, &restored); err != nil {
				t.Fatal(err)
			}
			state := Fold(numbered(first, reply(1, "done"), &restored))
			if len(state.Tools) != tc.want {
				t.Fatalf("tools after replay: %+v", state.Tools)
			}
		})
	}
}

func TestBuildMessagesKeepsInstructionOnlyContent(t *testing.T) {
	in := &Input{Mode: InputQueue, Content: []Content{{Instruction: map[string]string{"system-reminder": "context"}}}}
	messages := BuildMessages(Fold(numbered(NewInputEvent(in))))
	if in.Text() != "" {
		t.Fatalf("instruction leaked into user text: %q", in.Text())
	}
	if len(messages) != 1 || len(messages[0].Parts) != 1 || messages[0].Parts[0].Text == nil {
		t.Fatalf("instruction-only content was lost: %+v", messages)
	}
	instruction := messages[0].Parts[0].Text.Instruction["system-reminder"]
	if len(instruction) != 1 || instruction[0] != "context" {
		t.Fatalf("instruction = %v", instruction)
	}
}

func TestPlannedToolStateIsAnIndependentSnapshot(t *testing.T) {
	state := Fold(numbered(
		NewInputEvent(&Input{Mode: InputQueue, Content: []Content{{Text: "run"}}, State: map[string]string{"device": "original"}}),
		reply(1, "", call("a", "device_tool"), call("b", "device_tool")),
	))
	effects := Plan(state, testAgent)
	if len(effects) != 2 || effects[0].CallTool == nil || effects[1].CallTool == nil {
		t.Fatalf("effects = %+v", effects)
	}
	state.KV["device"] = "changed"
	if effects[0].CallTool.State["device"] != "original" {
		t.Fatal("effect aliases conversation state")
	}
	effects[0].CallTool.State["device"] = "first executor"
	if effects[1].CallTool.State["device"] != "original" || state.KV["device"] != "changed" {
		t.Fatal("effect state is shared")
	}
}

func TestAnsweredToolDeadlineStartsAtAnswer(t *testing.T) {
	for _, answerAt := range []int64{0, 900} {
		events := numbered(
			input("q", InputQueue, "run"),
			reply(1, "", call("a", "device_tool")),
			NewToolAskEvent(&ToolAsk{CallID: "a"}),
			NewToolAnswerEvent(&ToolAnswer{CallID: "a", Result: map[string]any{"ok": true}}),
		)
		events[1].At, events[3].At = 100, answerAt
		effects := Plan(Fold(events), testAgent)
		want := answerAt
		if want == 0 {
			want = 100
		}
		if len(effects) != 1 || effects[0].CallTool == nil || effects[0].CallTool.Since != want {
			t.Fatalf("answer at %d: effects = %+v", answerAt, effects)
		}
	}
}

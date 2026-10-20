package llmx

import (
	"testing"

	"github.com/xucx/llmx/types"
)

func TestBuildMessagesShowsToolImagesInUserMessage(t *testing.T) {
	events := []*Event{
		NewInputEvent(NewUserInput(InputQueue, "look")),
		{ID: "r1", Reply: &Reply{BaseSeq: 1, ToolCalls: []ToolCall{{ID: "c1", Name: "shot"}}}},
		NewToolResultEvent(&ToolResult{CallID: "c1", Name: "shot", Value: "ok", Images: []string{"data:image/jpeg;base64,AAAA"}}),
	}
	messages := BuildMessages(Fold(events))
	if len(messages) != 4 {
		t.Fatalf("want 4 messages, got %d", len(messages))
	}
	images := messages[3]
	if images.Role != types.MessageRoleUser || images.ID != ToolResultID("c1")+ImagesMessageSuffix {
		t.Fatalf("unexpected images message %+v", images)
	}
	if len(images.Parts) != 2 || images.Parts[1].ImageURL == nil || images.Parts[1].ImageURL.URL != "data:image/jpeg;base64,AAAA" {
		t.Fatalf("unexpected parts %+v", images.Parts)
	}
	// A failed call shows no images.
	failed := BuildMessages(Fold([]*Event{events[0], events[1],
		NewToolResultEvent(&ToolResult{CallID: "c1", Error: "boom", Images: []string{"data:image/png;base64,BB"}})}))
	if len(failed) != 3 {
		t.Fatalf("failed call must not add an images message, got %d", len(failed))
	}
}

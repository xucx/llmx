package llmx

import (
	"testing"

	"github.com/xucx/llmx/types"
)

func TestWithoutImagesReplacesImagePartsOnly(t *testing.T) {
	text := &types.Message{ID: "a", Role: types.MessageRoleUser, Parts: []*types.MessagePart{{Text: &types.MessageText{Text: "hi"}}}}
	image := &types.Message{ID: "b", Role: types.MessageRoleUser, Parts: []*types.MessagePart{
		{Text: &types.MessageText{Text: "shot:"}},
		{ImageURL: &types.MessageImageURL{URL: "data:image/png;base64,AA"}},
	}}
	out := withoutImages([]*types.Message{text, image})
	if out[0] != text {
		t.Fatalf("text-only message must be shared")
	}
	if len(out[1].Parts) != 2 || out[1].Parts[1].ImageURL != nil || out[1].Parts[1].Text.Text != ImageOmittedNote {
		t.Fatalf("image part not replaced: %+v", out[1].Parts)
	}
	if image.Parts[1].ImageURL == nil {
		t.Fatalf("original message must not be mutated")
	}
}

func TestVisionDefaultsAndOverride(t *testing.T) {
	if !defaultModelInfo("gpt-5.4").Vision || defaultModelInfo("glm-5.1").Vision || defaultModelInfo("unknown-x").Vision {
		t.Fatalf("unexpected vision defaults")
	}
}

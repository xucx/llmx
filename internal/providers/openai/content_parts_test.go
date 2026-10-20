package openai

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestCompletionMessageParamMarshalsContentParts(t *testing.T) {
	plain, _ := json.Marshal(&CompletionMessageParam{Role: "user", Content: "hi"})
	if string(plain) != `{"role":"user","content":"hi"}` {
		t.Fatalf("plain content changed: %s", plain)
	}
	withImage, _ := json.Marshal(&CompletionMessageParam{Role: "user", ContentParts: []*CompletionContentPart{
		{Type: "text", Text: "look"},
		{Type: "image_url", ImageURL: &CompletionImageURL{URL: "data:image/jpeg;base64,AA"}},
	}})
	if !strings.Contains(string(withImage), `"content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,AA"}}]`) {
		t.Fatalf("array content missing: %s", withImage)
	}
	if strings.Count(string(withImage), `"content"`) != 1 {
		t.Fatalf("content emitted twice: %s", withImage)
	}
}

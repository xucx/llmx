package compress

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/xucx/llmx"
	"github.com/xucx/llmx/types"
)

// compression is one recorded compression: the range of the history it covers
// and the transform it applies to that range.
type compression struct {
	// one of
	ToolResult *toolResultCompression `json:"toolResult,omitempty"`
	// Summary is the bridge message replacing the whole covered range. The
	// messages it replaces stay in the session history, so only the bridge
	// itself has to be recorded.
	Summary *types.Message `json:"summary,omitempty"`

	// UntilMessageID is the last history message this compression covers,
	// inclusive. Messages after it are left unchanged.
	UntilMessageID string `json:"untilMessageId"`
	// Tokens is the context size this compression was decided on.
	Tokens int64 `json:"tokens,omitempty"`
}

type toolResultCompression struct {
	Result       string            `json:"result,omitempty"`
	Instructions map[string]string `json:"instructions,omitempty"`
}

// loadCompressions reads the compressions recorded by patch events, in commit
// order.
func loadCompressions(events []*llmx.Event) ([]*compression, error) {
	recorded := map[int]string{}
	for _, event := range events {
		if event == nil || event.Patch == nil {
			continue
		}
		for key, value := range event.Patch.KV {
			suffix, found := strings.CutPrefix(key, compressionStatePrefix)
			index, err := strconv.Atoi(suffix)
			if !found || err != nil || index < 0 {
				continue
			}
			recorded[index] = value
		}
	}
	compressions := make([]*compression, len(recorded))
	for index := range compressions {
		value, ok := recorded[index]
		if !ok {
			return nil, fmt.Errorf("compression %d is not recorded", index)
		}
		compressions[index] = &compression{}
		if err := json.Unmarshal([]byte(value), compressions[index]); err != nil {
			return nil, fmt.Errorf("compression %d: %w", index, err)
		}
	}
	return compressions, nil
}

// apply returns the history with one compression applied, and whether the
// compression changed anything. It is the single definition of what a
// compression means, used both to generate one and to replay it. The given
// history is never modified.
func apply(
	history []*types.Message,
	compression *compression,
) ([]*types.Message, bool, error) {
	bound, err := compressionBound(history, compression.UntilMessageID)
	if err != nil {
		return nil, false, err
	}

	var (
		covered []*types.Message
		changed = true
	)
	switch {
	case compression.ToolResult != nil:
		covered, changed = applyToolResultCompress(history[:bound], compression.ToolResult)
	case compression.Summary != nil:
		covered = []*types.Message{compression.Summary}
	default:
		return nil, false, fmt.Errorf("compression has no result")
	}
	return append(covered, history[bound:]...), changed, nil
}

func applyToolResultCompress(
	history []*types.Message,
	compress *toolResultCompression,
) ([]*types.Message, bool) {
	// A replacement only pays off when it is shorter than what it replaces.
	replaced := compress.Result
	for _, value := range compress.Instructions {
		replaced += value
	}
	replaceLength := len(replaced)

	changed := false
	compressed := make([]*types.Message, 0, len(history))
	for _, msg := range history {
		if msg.Role != types.MessageRoleTool {
			compressed = append(compressed, msg)
			continue
		}

		newParts := map[int]*types.MessagePart{}
		for partIndex, part := range msg.Parts {
			if part == nil ||
				part.ToolResult == nil ||
				part.ToolResult.Error != "" ||
				len(part.ToolResult.Content()) <= replaceLength {
				continue
			}
			newToolResult := &types.MessageToolResult{
				ID:          part.ToolResult.ID,
				Name:        part.ToolResult.Name,
				Result:      compress.Result,
				Instruction: types.CloneInstructionMap(part.ToolResult.Instruction),
			}
			if len(compress.Instructions) > 0 &&
				newToolResult.Instruction == nil {
				newToolResult.Instruction = map[string][]string{}
			}
			for k, v := range compress.Instructions {
				newToolResult.Instruction[k] = append(newToolResult.Instruction[k], v)
			}
			newParts[partIndex] = &types.MessagePart{ToolResult: newToolResult}
		}

		if len(newParts) == 0 {
			compressed = append(compressed, msg)
			continue
		}
		newMsg := msg.Clone()
		for partIndex := range newMsg.Parts {
			if newPart, ok := newParts[partIndex]; ok {
				newMsg.Parts[partIndex] = newPart
			}
		}
		changed = true
		compressed = append(compressed, newMsg)
	}

	return compressed, changed
}

// compressionBound resolves the covered range into the index just past the last
// message a compression covers. A boundary that is not in the history means the
// recorded compressions and the history disagree, which is a corruption rather
// than a state to recover from.
func compressionBound(
	history []*types.Message,
	untilMessageID string,
) (int, error) {
	if untilMessageID == "" {
		return 0, fmt.Errorf("compression until message id is empty")
	}
	for index := len(history) - 1; index >= 0; index-- {
		if history[index] != nil && history[index].ID == untilMessageID {
			return index + 1, nil
		}
	}
	return 0, fmt.Errorf(
		"compression until message %q is not in the history",
		untilMessageID,
	)
}

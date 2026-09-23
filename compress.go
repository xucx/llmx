package llmx

import (
	"bytes"
	"cmp"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"text/template"
	"time"

	"github.com/Masterminds/sprig/v3"
	"github.com/xucx/gox/logx"
	"github.com/xucx/llmx/types"
)

//go:embed compress_instruction.md
var defaultCompressInstruction string

//go:embed compress_bridge.md
var defaultCompressBridge string

const defaultToolResultReplace = "The tool call results have been omitted to save context. You may call the tool again if needed"

// summaryUserPromptsState is the bridge state key holding, as a JSON string
// list, every user prompt the bridge replaced. A later summary expands it
// instead of reading the bridge text, so prompts survive repeated summaries.
const summaryUserPromptsState = "llmx.compress.userPrompts"

type CompressOptions struct {
	// KeepLastTurn is the number of recent user turns left unchanged.
	KeepLastTurn int
	// ToolResultLevel uses a model-token ratio when between zero and one, an
	// absolute token threshold when at least one, and is disabled when zero.

	ToolResultLevel      float64
	ToolResultReplace    string
	ToolResultReplaceTag string

	SummaryLevel       float64
	SummaryModel       string
	SummaryInstruction string
	SummaryBridge      string
	SummaryBridgeTag   string
	SummaryTimeout     time.Duration
}

func (o *CompressOptions) normalized() *CompressOptions {
	copy := *o
	if copy.KeepLastTurn < 1 {
		copy.KeepLastTurn = 1
	}
	if copy.ToolResultReplace == "" {
		copy.ToolResultReplace = defaultToolResultReplace
	}
	if copy.SummaryInstruction == "" {
		copy.SummaryInstruction = defaultCompressInstruction
	}
	if copy.SummaryBridge == "" {
		copy.SummaryBridge = defaultCompressBridge
	}
	if copy.SummaryTimeout <= 0 {
		copy.SummaryTimeout = 5 * time.Minute
	}
	return &copy
}

func shouldCompress(tokens int64, threshold int64) bool {
	if tokens <= 0 || threshold <= 0 {
		return false
	}
	return tokens > threshold
}

type CompressInput struct {
	Model    string
	Messages []*types.Message
	Options  *CompressOptions

	// Tokens is the context usage the model last reported. Compression is
	// decided exclusively from this value; zero means there is no reported
	// context size to compress yet.
	Tokens int64
}

// compressFunc generates one compression event against the given history, or
// returns nil when its compression does not apply -- because its level is not
// reached, or because compressing would not shrink anything. tokens is the
// context size the decision is made from.
type compressFunc func(
	ctx context.Context,
	messages []*types.Message,
) (*EventCompression, error)

// compressor pairs a compression with the absolute token count it triggers
// at, so compressions can be ordered from the lowest threshold to the highest.
type compressor struct {
	threshold int64
	run       compressFunc
}

// Compress decides whether a history must shrink and describes the transforms
// that shrink it. It is stateless: it never reads or writes session storage and
// never changes the history it is given, it only returns the compression events
// describing what to do. Callers apply those events with ApplyCompression, so
// generating and replaying a compression always agree.
//
// At most one compression runs per call: the highest threshold the reported
// context size exceeds wins, and a history that is still too large after it
// triggers the next compression on a later call. When both thresholds are
// equal the tool-result compression wins, being the cheaper of the two. It
// returns no events when nothing has to be compressed.
func Compress(
	ctx context.Context,
	models *Models,
	in *CompressInput,
) ([]*Event, error) {
	if models == nil {
		return nil, fmt.Errorf("models is nil")
	}
	if in == nil {
		return nil, fmt.Errorf("compress input is nil")
	}
	if in.Options == nil {
		return nil, nil
	}

	chatModel, err := models.GetModel(in.Model)
	if err != nil {
		return nil, err
	}

	options := in.Options.normalized()
	compressors := []compressor{
		{
			threshold: compressThreshold(chatModel.MaxToken, options.ToolResultLevel),
			run:       toolResultCompressor(options),
		},
		{
			threshold: compressThreshold(chatModel.MaxToken, options.SummaryLevel),
			run:       summaryCompressor(models, chatModel, options),
		},
	}
	// A stable sort keeps equal thresholds in registration order, so the
	// cheaper tool-result compression stays first.
	slices.SortStableFunc(compressors, func(a, b compressor) int {
		return cmp.Compare(b.threshold, a.threshold)
	})

	events := []*Event{}

	for _, compress := range compressors {
		// shouldCompress also skips disabled compressions: their threshold
		// resolves to zero, which any positive context size would exceed.
		if shouldCompress(in.Tokens, compress.threshold) {

			// KeepLastTurn is applied once here with the caller's raw value;
			// compressors receive only the compressible range and never
			// re-split it themselves.
			history, _ := splitHistoryLastTurn(in.Messages, in.Options.KeepLastTurn)
			if len(history) == 0 {
				break
			}

			compression, err := compress.run(ctx, history)
			if err != nil {
				return nil, err
			}

			if compression != nil {
				events = append(events, NewEventCompression(compression))
			}

			break
		}
	}

	return events, nil
}

// compressThreshold resolves a compression level into the absolute token count
// it triggers at, so levels expressed as ratios and as absolute counts order
// correctly against each other.
func compressThreshold(maxTokens int64, level float64) int64 {
	if maxTokens <= 0 || level < 0.000001 {
		return 0
	}

	if level < 1 {
		return int64(level * float64(maxTokens))
	}

	return int64(level)
}

func toolResultCompressor(options *CompressOptions) compressFunc {
	return func(
		ctx context.Context,
		messages []*types.Message,
	) (*EventCompression, error) {
		if len(messages) == 0 {
			return nil, nil
		}

		toolResult := &CompressResultToolResult{}
		if options.ToolResultReplaceTag == "" {
			toolResult.Result = options.ToolResultReplace
		} else {
			toolResult.Instructions = map[string]string{
				options.ToolResultReplaceTag: options.ToolResultReplace,
			}
		}

		// Every message in the current history is fair game, so the range ends
		// at the newest one.
		compression := &EventCompression{
			ToolResult:     toolResult,
			UntilMessageID: messages[len(messages)-1].ID,
		}
		// A compression that removes nothing must not be committed: the context
		// size it fails to shrink would trigger it again on every later call.
		_, reduced, err := ApplyCompression(messages, compression)
		if err != nil {
			return nil, err
		}
		if reduced <= 0 {
			return nil, nil
		}
		return compression, nil
	}
}

func summaryCompressor(models *Models, chatModel *Model, options *CompressOptions) compressFunc {
	return func(ctx context.Context, messages []*types.Message) (*EventCompression, error) {

		if len(messages) == 0 {
			return nil, nil
		}

		summaryModel := chatModel
		if options.SummaryModel != "" {
			var err error
			summaryModel, err = models.GetModel(options.SummaryModel)
			if err != nil {
				return nil, err
			}
		}

		summaryCtx := ctx
		cancel := func() {}
		if options.SummaryTimeout > 0 {
			summaryCtx, cancel = context.WithTimeout(ctx, options.SummaryTimeout)
		}
		summary, err := compressSummary(
			summaryCtx,
			messages,
			summaryModel,
			options.SummaryInstruction,
			options.SummaryBridge,
		)
		cancel()
		if err != nil {
			return nil, err
		}
		if summary.UntilMessageID == "" {
			return nil, nil
		}

		bridge := types.NewTextMessage(
			types.MessageRoleUser,
			summary.Summary,
		)
		if options.SummaryBridgeTag != "" {
			bridge = types.NewTextMessage(
				types.MessageRoleUser,
				"",
				options.SummaryBridgeTag,
				summary.Summary,
			)
		}
		userPrompts, err := json.Marshal(summary.UserPrompts)
		if err != nil {
			return nil, err
		}
		bridge.SetState(summaryUserPromptsState, string(userPrompts))

		compression := &EventCompression{
			Summary:        &CompressResultSummary{Message: bridge},
			UntilMessageID: summary.UntilMessageID,
		}
		if summary.Completion != nil {
			compression.Summary.Usage = summary.Completion.Usage
		}
		return compression, nil
	}
}

// ApplyCompression applies one compression event to a history and returns the
// compressed history along with an estimate of how much context the
// compression removed. It is the single definition of what a compression
// means: Compress uses it to report the history a compression leaves behind,
// and a session uses it to restore that same history from the event log. The
// reduction is derived here instead of being stored in the event, so
// generating and replaying a compression always report the same value.
//
// The given history is never modified.
func ApplyCompression(history []*types.Message, compress *EventCompression) ([]*types.Message, int64, error) {
	if len(history) == 0 {
		return nil, 0, fmt.Errorf("no history")
	}

	if compress == nil {
		return nil, 0, fmt.Errorf("compression is nil")
	}

	bound, err := compressionBound(history, compress.UntilMessageID)
	if err != nil {
		return nil, 0, err
	}

	var (
		covered           = history[:bound]
		compressedHistory []*types.Message
		reduceToken       int64
	)
	switch {
	case compress.ToolResult != nil:
		compressedHistory, reduceToken, err = applyToolResultCompress(covered, compress.ToolResult)
	case compress.Summary != nil:
		compressedHistory, reduceToken, err = applySummaryCompress(covered, compress.Summary)
	default:
		err = fmt.Errorf("compression has no result")
	}

	if err != nil {
		return nil, 0, err
	}

	compressedHistory = append(compressedHistory, history[bound:]...)

	return compressedHistory, reduceToken, nil
}

func applyToolResultCompress(history []*types.Message, compress *CompressResultToolResult) ([]*types.Message, int64, error) {
	// A replacement only pays off when it is shorter than what it replaces.
	replaced := compress.Result
	for _, value := range compress.Instructions {
		replaced += value
	}
	replaceLength := len(replaced)

	var reduced int64
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
		before, _ := msg.EstimateTokens()
		after, _ := newMsg.EstimateTokens()
		reduced += subTokens(before, after)
		compressed = append(compressed, newMsg)
	}

	return compressed, reduced, nil
}

func applySummaryCompress(history []*types.Message, compress *CompressResultSummary) ([]*types.Message, int64, error) {
	if compress.Message == nil {
		return nil, 0, fmt.Errorf("compression summary message is nil")
	}
	compressed := []*types.Message{}
	compressed = append(compressed, compress.Message)
	// The summary call read the range the bridge replaces and wrote the
	// bridge itself, so its usage is the closest estimate of what the
	// replacement removed.
	reduced := subTokens(
		compress.Usage.PromptTokens,
		compress.Usage.CompletionTokens,
	)
	return compressed, reduced, nil
}

// compressionBound resolves the covered range into the index just past the last
// message a compression covers. A boundary that is not in the history means the
// event log and the history disagree, which is a corruption rather than a state
// to recover from.
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

// subTokens subtracts one token count from another, in both directions a
// compression needs it: deriving how much a transform removed, and applying
// that removal to a context size. Token counts never go below zero, and a
// reduction that is not positive is no reduction at all -- a compression
// replayed from an event log must never grow the context it is applied to.
func subTokens(tokens, reduction int64) int64 {
	if reduction <= 0 {
		return tokens
	}
	if reduction >= tokens {
		return 0
	}
	return tokens - reduction
}

type bridgeParams struct {
	ComposedUserPrompts []string
	Summary             string
}

type compressSummaryResult struct {
	// UntilMessageID is the last summarized message, inclusive. It is empty
	// when there was nothing to summarize.
	UntilMessageID string            `json:"untilMessageId"`
	Completion     *types.Completion `json:"completion"`
	Summary        string            `json:"summary"`
	// UserPrompts is every user prompt the summary replaced, including those
	// carried by bridges of earlier summaries.
	UserPrompts []string `json:"userPrompts"`
}

func compressSummary(
	ctx context.Context,
	composed []*types.Message,
	model *Model,
	instruction string,
	bridge string,
) (*compressSummaryResult, error) {
	result := &compressSummaryResult{}
	if len(composed) == 0 {
		return result, nil
	}

	logx.Debug("start compress summary")

	// The summary is addressed by the message it ends at, so an unidentified
	// boundary has to fail before the summary is paid for.
	until := composed[len(composed)-1].ID
	if until == "" {
		return nil, errors.New("last summarized message has no id")
	}

	completion, err := model.Completion(
		ctx,
		composed,
		types.CompletionWithInstructions(instruction),
		types.CompletionWithReasoningLevel(10),
	)
	if err != nil {
		return nil, err
	}
	result.Completion = completion
	if completion.Message == nil {
		return nil, errors.New("llm return no message")
	}

	var summaryBuilder strings.Builder
	for _, part := range completion.Message.Parts {
		if part.Text != nil {
			summaryBuilder.WriteString(part.Text.Text)
		}
	}
	summary := summaryBuilder.String()
	if summary == "" {
		return nil, errors.New("llm return no text")
	}

	userPrompts := make([]string, 0)
	for _, message := range composed {
		// A bridge of an earlier summary stands for the prompts it replaced,
		// its own text is not a user prompt.
		if carried := message.GetState(summaryUserPromptsState); carried != "" {
			var prompts []string
			if err := json.Unmarshal([]byte(carried), &prompts); err != nil {
				return nil, err
			}
			userPrompts = append(userPrompts, prompts...)
			continue
		}
		if message.Role == types.MessageRoleUser && message.Text() != "" {
			userPrompts = append(userPrompts, message.Text())
		}
	}

	bridgeSummary := summary
	if bridge != "" {
		bridgeTemplate, err := template.New("bridge").
			Funcs(sprig.FuncMap()).
			Parse(bridge)
		if err != nil {
			return nil, err
		}
		var buffer bytes.Buffer
		if err := bridgeTemplate.Execute(&buffer, &bridgeParams{
			ComposedUserPrompts: userPrompts,
			Summary:             summary,
		}); err != nil {
			return nil, err
		}
		bridgeSummary = buffer.String()
	}

	result.UntilMessageID = until
	result.Summary = bridgeSummary
	result.UserPrompts = userPrompts
	return result, nil
}

func splitHistoryLastTurn(
	history []*types.Message,
	last int,
) ([]*types.Message, []*types.Message) {
	if last <= 0 {
		return history, nil
	}
	count := 0
	index := -1
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == types.MessageRoleUser {
			count++
		}
		if count >= last {
			index = i
			break
		}
	}
	if index >= 0 {
		return history[:index], history[index:]
	}
	return []*types.Message{}, history
}

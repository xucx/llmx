package compress

import (
	"bytes"
	"cmp"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"text/template"
	"time"

	"github.com/Masterminds/sprig/v3"
	"github.com/xucx/gox/logx"
	"github.com/xucx/llmx"
	"github.com/xucx/llmx/types"
)

//go:embed compress_instruction.md
var defaultCompressInstruction string

//go:embed compress_bridge.md
var defaultCompressBridge string

const defaultToolResultReplace = "The tool call results have been omitted to save context. You may call the tool again if needed"

// defaultKeepRecent is the recent context, in tokens, a compression keeps when
// Options.KeepRecent is not set.
const defaultKeepRecent = 20000

// summaryUserPromptsState is the bridge state key holding, as a JSON string
// list, the newest user prompts the bridge replaced. A later summary expands it
// instead of reading the bridge text, so prompts survive repeated summaries.
const summaryUserPromptsState = "llmx.compress.userPrompts"

// summaryUserPromptsLimit is how many of the newest replaced user prompts a
// bridge carries verbatim. Older ones only live on in the summary, so the
// bridge stays bounded however many times it is summarized again.
const summaryUserPromptsLimit = 5

// summaryRequest closes the history sent to the summary model, so the model is
// asked for the snapshot instead of continuing the conversation it reads.
const summaryRequest = "Generate the state snapshot of the conversation above now, following your instructions."

// summarySnapshotTag starts the snapshot in the summary model reply; the
// scratchpad before it is dropped from the bridge.
const summarySnapshotTag = "<state_snapshot>"

// compressionStatePrefix prefixes the state patch keys recording the
// compressions, numbered from zero in commit order.
const compressionStatePrefix = "llmx.compress."

type Options struct {
	// KeepRecent is how much recent history, in estimated tokens, is left
	// unchanged. Like the levels below it is a model-token ratio when between
	// zero and one and an absolute token count when at least one; zero keeps
	// defaultKeepRecent tokens. The kept history is measured in tokens rather
	// than user turns, so a long autonomous run started by one request can
	// still be compressed.
	KeepRecent float64
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

func (o *Options) normalized() *Options {
	copy := *o
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

// New returns an interceptor that shrinks the history the model sees once the
// context size the model last reported crosses a configured level.
//
// The event stream itself is never rewritten. Before every model call, every
// compression recorded in the stream is applied to the messages sent to the
// model, and a newly needed compression is recorded as a state patch appended
// to the stream. A session replayed with this interceptor therefore keeps its
// compressions. Interceptors registered before it see the uncompressed
// messages, those after it the compressed ones.
func New(models *llmx.Models, options Options) llmx.Interceptor {
	return &interceptor{models: models, options: options}
}

type interceptor struct {
	models  *llmx.Models
	options Options
}

// Compression only saves context, so it never fails the model call: whatever
// goes wrong is logged and the call goes on with the history compressed as far
// as it could be. A context that really is too large is left to the model call
// to report.
func (i *interceptor) InterceptModel(next llmx.ModelHandler) llmx.ModelHandler {
	return func(ctx llmx.ModelContext) (*types.Completion, error) {
		events := ctx.Events()
		request := ctx.Request()
		messages := request.Messages

		compressions, err := loadCompressions(events)
		if err != nil {
			logx.Warnw("load compressions fail", "error", err)
		}
		// Every compression is replayed in commit order: a summary covers the
		// history before its boundary, but a tool-result compression committed
		// earlier may reach past it, into the tail the summary kept.
		for index := 0; err == nil && index < len(compressions); index++ {
			var compressed []*types.Message
			if compressed, _, err = apply(messages, compressions[index]); err != nil {
				logx.Warnw("replay compression fail", "index", index, "error", err)
				break
			}
			messages = compressed
		}

		// A failed replay leaves the recorded compressions at odds with the
		// history: a new one would be recorded after one that never replays, so
		// none is generated. The last compression was decided on this very
		// context size: no model call has reported the size it left behind yet,
		// so compressing again would act on a stale size.
		tokens := contextTokens(events)
		if count := len(compressions); err == nil &&
			(count == 0 || compressions[count-1].Tokens != tokens) {
			compression, err := compress(ctx, i.models, &compressInput{
				Model:    request.Model,
				Messages: messages,
				Options:  &i.options,
				Tokens:   tokens,
			})
			var compressed []*types.Message
			if err == nil && compression != nil {
				compressed, _, err = apply(messages, compression)
			}
			if err == nil && compression != nil {
				var data []byte
				if data, err = json.Marshal(compression); err == nil {
					err = ctx.Append(llmx.NewPatchEvent(map[string]string{
						compressionStatePrefix + strconv.Itoa(count): string(data),
					}, false))
				}
			}
			switch {
			case err != nil:
				// Only a recorded compression is sent: later calls replay it.
				logx.Warnw("compress fail", "error", err)
			case compression != nil:
				messages = compressed
			}
		}

		request.Messages = messages
		ctx.SetRequest(request)
		return next(ctx)
	}
}

// contextTokens returns the context size the model last reported. A completion
// reports the whole context it consumed, so the newest usage is the size.
func contextTokens(events []*llmx.Event) int64 {
	for index := len(events) - 1; index >= 0; index-- {
		if events[index] != nil && events[index].Usage != nil {
			return events[index].Usage.Usage.TotalTokens
		}
	}
	return 0
}

func shouldCompress(tokens int64, threshold int64) bool {
	if tokens <= 0 || threshold <= 0 {
		return false
	}
	return tokens > threshold
}

type compressInput struct {
	Model    string
	Messages []*types.Message
	Options  *Options

	// Tokens is the context usage the model last reported. Compression is
	// decided exclusively from this value; zero means there is no reported
	// context size to compress yet.
	Tokens int64
}

// compressFunc generates one compression against the given history, or returns
// nil when compressing would not shrink anything. inProgress reports that the
// history ends inside a user turn whose work continues after it.
type compressFunc func(
	ctx context.Context,
	messages []*types.Message,
	inProgress bool,
) (*compression, error)

// compressor pairs a compression with the absolute token count it triggers
// at, so compressions can be ordered from the lowest threshold to the highest.
type compressor struct {
	threshold int64
	run       compressFunc
}

// compress decides whether a history must shrink and describes the transform
// that shrinks it. It never changes the history it is given; apply performs
// the transform, so generating and replaying a compression always agree.
//
// At most one compression is generated per call: the highest threshold the
// reported context size exceeds is tried first, and when it fails or has
// nothing to shrink the next lower one is tried. A history that is still too
// large after it triggers the next compression on a later call. When both
// thresholds are equal the tool-result compression is tried first, being the
// cheaper of the two. It returns nil when nothing has to be compressed.
func compress(
	ctx context.Context,
	models *llmx.Models,
	in *compressInput,
) (*compression, error) {
	if models == nil {
		return nil, fmt.Errorf("models is nil")
	}
	if in == nil {
		return nil, fmt.Errorf("compress input is nil")
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

	for _, compressor := range compressors {
		// shouldCompress also skips disabled compressions: their threshold
		// resolves to zero, which any positive context size would exceed.
		if !shouldCompress(in.Tokens, compressor.threshold) {
			continue
		}

		// The kept history is split off here; compressors receive only the
		// compressible range and never re-split it themselves. The kept
		// history alone must fit well under the threshold, or no compression
		// could bring the context below it and every call would compress
		// again.
		keep := compressThreshold(chatModel.MaxToken, options.KeepRecent)
		if keep <= 0 {
			keep = defaultKeepRecent
		}
		keep = min(keep, compressor.threshold/2)
		history, kept := splitHistoryRecent(in.Messages, keep)
		if len(history) == 0 {
			continue
		}
		// A kept history starting at an assistant message means the cut fell
		// inside a user turn: its request is still being worked on.
		inProgress := kept[0].Role == types.MessageRoleAssistant

		compression, err := compressor.run(ctx, history, inProgress)
		if err != nil {
			logx.Warnw("compression fail", "threshold", compressor.threshold, "error", err)
			continue
		}
		if compression == nil {
			continue
		}
		compression.Tokens = in.Tokens
		return compression, nil
	}

	return nil, nil
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

func toolResultCompressor(options *Options) compressFunc {
	return func(
		ctx context.Context,
		messages []*types.Message,
		_ bool,
	) (*compression, error) {
		if len(messages) == 0 {
			return nil, nil
		}

		toolResult := &toolResultCompression{}
		if options.ToolResultReplaceTag == "" {
			toolResult.Result = options.ToolResultReplace
		} else {
			toolResult.Instructions = map[string]string{
				options.ToolResultReplaceTag: options.ToolResultReplace,
			}
		}

		// Every message in the current history is fair game, so the range ends
		// at the newest one.
		compression := &compression{
			ToolResult:     toolResult,
			UntilMessageID: messages[len(messages)-1].ID,
		}
		// A compression that removes nothing must not be recorded: the context
		// size it fails to shrink would trigger it again on every later call.
		_, changed, err := apply(messages, compression)
		if err != nil || !changed {
			return nil, err
		}
		return compression, nil
	}
}

func summaryCompressor(models *llmx.Models, chatModel *llmx.Model, options *Options) compressFunc {
	return func(
		ctx context.Context,
		messages []*types.Message,
		inProgress bool,
	) (*compression, error) {

		// Summarizing nothing but the bridge of the previous summary would only
		// rewrite it.
		if len(messages) == 0 ||
			len(messages) == 1 && messages[0].GetState(summaryUserPromptsState) != "" {
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
			inProgress,
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

		// A bridge no shorter than what it replaces saves nothing.
		var covered int64
		for _, message := range messages {
			tokens, _ := message.EstimateTokens()
			covered += tokens
		}
		if tokens, _ := bridge.EstimateTokens(); tokens >= covered {
			return nil, nil
		}

		return &compression{
			Summary:        bridge,
			UntilMessageID: summary.UntilMessageID,
		}, nil
	}
}

type bridgeParams struct {
	ComposedUserPrompts []string
	Summary             string
	// InProgress reports that the last composed user prompt is still being
	// worked on in the messages after the bridge.
	InProgress bool
}

type compressSummaryResult struct {
	// UntilMessageID is the last summarized message, inclusive. It is empty
	// when there was nothing to summarize.
	UntilMessageID string `json:"untilMessageId"`
	Summary        string `json:"summary"`
	// UserPrompts is the newest summaryUserPromptsLimit user prompts the
	// summary replaced, including those carried by bridges of earlier
	// summaries.
	UserPrompts []string `json:"userPrompts"`
}

func compressSummary(
	ctx context.Context,
	composed []*types.Message,
	model *llmx.Model,
	instruction string,
	bridge string,
	inProgress bool,
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

	// The reasoning level is left to the model: the instruction reasons in a
	// scratchpad, and a thinking budget would compete with the snapshot for
	// the output tokens.
	completion, err := model.Completion(
		ctx,
		append(slices.Clip(composed), types.NewTextMessage(types.MessageRoleUser, summaryRequest)),
		types.CompletionWithInstructions(instruction),
	)
	if err != nil {
		return nil, err
	}
	if completion == nil || completion.Message == nil {
		return nil, errors.New("llm return no message")
	}

	var summaryBuilder strings.Builder
	for _, part := range completion.Message.Parts {
		if part.Text != nil {
			summaryBuilder.WriteString(part.Text.Text)
		}
	}
	summary := summaryBuilder.String()
	if index := strings.Index(summary, summarySnapshotTag); index >= 0 {
		summary = summary[index:]
	}
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
	userPrompts = userPrompts[max(0, len(userPrompts)-summaryUserPromptsLimit):]

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
			InProgress:          inProgress,
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

// splitHistoryRecent splits a history into the range a compression may cover
// and the recent range it keeps. The kept range is the shortest tail holding at
// least keepTokens estimated tokens that starts at a message other than a tool
// result, so tool results are never cut off from the tool call they answer.
// The first message never starts the kept range: when the whole history fits
// in keepTokens, nothing is compressible.
func splitHistoryRecent(
	history []*types.Message,
	keepTokens int64,
) ([]*types.Message, []*types.Message) {
	var tokens int64
	for index := len(history) - 1; index > 0; index-- {
		messageTokens, _ := history[index].EstimateTokens()
		tokens += messageTokens
		if tokens >= keepTokens && history[index].Role != types.MessageRoleTool {
			return history[:index], history[index:]
		}
	}
	return nil, history
}

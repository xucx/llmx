package types

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/xucx/llmx/internal/utils"
)

type Tool struct {
	Type     ToolType
	Function *ToolFunction
}

type ToolType string

const (
	ToolTypeFunction ToolType = "function"
)

type ToolFunction struct {
	Name        string
	Description string
	Parameters  map[string]any
}

func NewFunctionTool(name, desc string, params map[string]any) *Tool {
	return &Tool{
		Type: ToolTypeFunction,
		Function: &ToolFunction{
			Name:        name,
			Description: desc,
			Parameters:  params,
		},
	}
}

type MessageRole string

const (
	MessageRoleSystem    MessageRole = "system"
	MessageRoleAssistant MessageRole = "assistant"
	MessageRoleUser      MessageRole = "user"
	MessageRoleTool      MessageRole = "tool"
)

type Message struct {
	ID          string            `json:"id,omitempty" yaml:"id,omitempty"`
	Role        MessageRole       `json:"role,omitempty" yaml:"role,omitempty"`
	Delta       bool              `json:"delta" yaml:"delta"`
	Parts       []*MessagePart    `json:"parts,omitempty" yaml:"parts,omitempty"`
	Token       int64             `json:"token,omitempty" yaml:"token,omitempty"`
	ReasonToken int64             `json:"reasonToken,omitempty" yaml:"reasonToken,omitempty"`
	State       map[string]string `json:"state,omitempty" yaml:"state,omitempty"`
}

func NewMessage(role MessageRole) *Message {
	return &Message{
		ID:   uuid.NewString(),
		Role: role,
	}
}

func (m *Message) Clone() *Message {
	if m == nil {
		return nil
	}

	cp := *m
	if m.State != nil {
		cp.State = maps.Clone(m.State)
	}

	cp.Parts = make([]*MessagePart, len(m.Parts))
	for i, p := range m.Parts {
		if p == nil {
			continue
		}
		pc := *p
		if p.Text != nil {
			pc.Text = pc.Text.Clone()
		}
		if p.Reasoning != nil {
			v := *p.Reasoning
			pc.Reasoning = &v
		}
		if p.Refusal != nil {
			v := *p.Refusal
			pc.Refusal = &v
		}
		if p.ImageURL != nil {
			v := *p.ImageURL
			pc.ImageURL = &v
		}
		if p.Audio != nil {
			v := *p.Audio
			pc.Audio = &v
		}
		if p.File != nil {
			v := *p.File
			pc.File = &v
		}
		if p.ToolCall != nil {
			c := *p.ToolCall
			pc.ToolCall = &c
		}
		if p.ToolResult != nil {
			pc.ToolResult = p.ToolResult.Clone()
		}
		if p.RealtimeResponse != nil {
			c := *p.RealtimeResponse
			pc.RealtimeResponse = &c
		}

		cp.Parts[i] = &pc
	}

	return &cp
}

func (m *Message) SetState(k, v string) {
	if m.State == nil {
		m.State = map[string]string{}
	}
	m.State[k] = v
}

// messagePartKind names the single content field a MessagePart carries.
type messagePartKind string

const (
	partKindText             messagePartKind = "text"
	partKindReasoning        messagePartKind = "reasoning"
	partKindRefusal          messagePartKind = "refusal"
	partKindImageURL         messagePartKind = "imageUrl"
	partKindAudio            messagePartKind = "audio"
	partKindFile             messagePartKind = "file"
	partKindToolCall         messagePartKind = "toolCall"
	partKindToolResult       messagePartKind = "toolResult"
	partKindRealtimeResponse messagePartKind = "realtimeResponse"
)

// rolePartKinds lists the part kinds each role accepts. A message may repeat
// any allowed kind: one tool message can carry several tool results, and one
// assistant message can carry several tool calls.
var rolePartKinds = map[MessageRole]map[messagePartKind]struct{}{
	MessageRoleSystem: {
		partKindText: {},
	},
	MessageRoleUser: {
		partKindText:     {},
		partKindImageURL: {},
		partKindAudio:    {},
		partKindFile:     {},
	},
	MessageRoleAssistant: {
		partKindText:             {},
		partKindReasoning:        {},
		partKindRefusal:          {},
		partKindToolCall:         {},
		partKindAudio:            {},
		partKindRealtimeResponse: {},
	},
	MessageRoleTool: {
		partKindToolResult: {},
	},
}

// partKind reports the single content field set on this part. Exactly one
// field must be set, otherwise the part is not a valid oneof.
func (p *MessagePart) partKind() (messagePartKind, error) {
	if p == nil {
		return "", fmt.Errorf("%w: part is nil", ErrInvalidMessage)
	}

	kind := messagePartKind("")
	fields := 0
	for candidate, present := range map[messagePartKind]bool{
		partKindText:             p.Text != nil,
		partKindReasoning:        p.Reasoning != nil,
		partKindRefusal:          p.Refusal != nil,
		partKindImageURL:         p.ImageURL != nil,
		partKindAudio:            p.Audio != nil,
		partKindFile:             p.File != nil,
		partKindToolCall:         p.ToolCall != nil,
		partKindToolResult:       p.ToolResult != nil,
		partKindRealtimeResponse: p.RealtimeResponse != nil,
	} {
		if present {
			kind = candidate
			fields++
		}
	}
	if fields != 1 {
		return "", fmt.Errorf(
			"%w: part has %d content fields, want 1",
			ErrInvalidMessage,
			fields,
		)
	}
	return kind, nil
}

func (m *Message) Validate() error {
	if m == nil {
		return fmt.Errorf("%w: message is nil", ErrInvalidMessage)
	}
	allowed, ok := rolePartKinds[m.Role]
	if !ok {
		return fmt.Errorf("%w: unsupported role %q", ErrInvalidMessage, m.Role)
	}
	if len(m.Parts) == 0 {
		return fmt.Errorf("%w: message has no parts", ErrInvalidMessage)
	}
	for i, part := range m.Parts {
		kind, err := part.partKind()
		if err != nil {
			return fmt.Errorf("part %d: %w", i, err)
		}
		if _, ok := allowed[kind]; !ok {
			return fmt.Errorf(
				"%w: part %d kind %q is not allowed for role %q",
				ErrInvalidMessage,
				i,
				kind,
				m.Role,
			)
		}
	}
	return nil
}

func ValidateMessages(messages []*Message) error {
	for i, message := range messages {
		if err := message.Validate(); err != nil {
			return fmt.Errorf("message %d: %w", i, err)
		}
	}
	return nil
}

func (m *Message) GetState(k string) string {
	if m.State == nil {
		return ""
	}
	if v, ok := m.State[k]; ok {
		return v
	}

	return ""
}

func (m *Message) Tokens() (int64, int64) {
	if m == nil {
		return 0, 0
	}

	if m.Token > 0 || m.ReasonToken > 0 {
		return m.Token, m.ReasonToken
	}

	return m.EstimateTokens()
}

func (m *Message) EstimateTokens() (int64, int64) {
	if m == nil {
		return 0, 0
	}

	total := int64(0)
	reason := int64(0)
	for _, part := range m.Parts {
		if part == nil {
			continue
		}
		switch {
		case part.Text != nil:
			total += utils.EstimateTokens(part.Text.TextWithInstruction())
		case part.Reasoning != nil:
			partReason := utils.EstimateTokens(part.Reasoning.Text)
			reason += partReason
			total += partReason
		case part.ToolCall != nil:
			total += utils.EstimateTokens(part.ToolCall.Name) + utils.EstimateTokens(part.ToolCall.Arguments)
		case part.ToolResult != nil:
			total += utils.EstimateTokens(part.ToolResult.ContentWithInstruction())
		}
	}
	return total, reason
}

func (m *Message) String() string {
	sb := strings.Builder{}
	sb.WriteString(string(m.Role))
	sb.WriteString(":\n")
	for _, part := range m.Parts {
		if part == nil {
			continue
		}
		switch {
		case part.Text != nil:
			sb.WriteString(fmt.Sprintf("%s\n", part.Text.Text))
			for k, v := range part.Text.Instruction {
				sb.WriteString(fmt.Sprintf("<%s>%s</%s>\n", k, strings.Join(v, "\n"), k))
			}
		case part.ToolCall != nil:
			sb.WriteString(fmt.Sprintf("[toolCall] %s %s\n", part.ToolCall.Name, part.ToolCall.Arguments))
		case part.ToolResult != nil:
			sb.WriteString(fmt.Sprintf("[toolResult] %s %s %s\n", part.ToolResult.Name, part.ToolResult.Result, part.ToolResult.Error))
			for k, v := range part.ToolResult.Instruction {
				sb.WriteString(fmt.Sprintf("<%s>%s</%s>\n", k, strings.Join(v, "\n"), k))
			}
		case part.Reasoning != nil:
			sb.WriteString(fmt.Sprintf("[reasoning] %s\n", part.Reasoning.Text))
		default:
			if b, err := json.Marshal(part); err == nil {
				sb.Write(b)
				sb.WriteString("\n")
			}
		}
	}

	return sb.String()
}

func (c *Message) Text() string {
	sb := strings.Builder{}
	for _, part := range c.Parts {
		if part == nil {
			continue
		}
		if part.Text != nil {
			sb.WriteString(part.Text.Text)
		}
	}
	return sb.String()
}

func (c *Message) ToolCalls() []*MessageToolCall {
	toolCalls := []*MessageToolCall{}
	for _, part := range c.Parts {
		if part == nil {
			continue
		}
		if part.ToolCall != nil {
			toolCalls = append(toolCalls, part.ToolCall)
		}
	}
	return toolCalls
}

func (c *Message) IsReasoning() bool {
	if c.Role != MessageRoleAssistant {
		return false
	}

	hasReasonPart := false
	hasOtherPart := false
	for _, part := range c.Parts {
		if part == nil {
			continue
		}
		if part.Reasoning != nil {
			hasReasonPart = true
		} else {
			hasOtherPart = true
		}
	}

	return hasReasonPart && !hasOtherPart
}

type MessagePart struct {
	// oneof
	Text             *MessageText             `json:"text,omitempty" yaml:"text,omitempty"`
	Reasoning        *MessageReasoning        `json:"reasoning,omitempty" yaml:"reasoning,omitempty"`
	Refusal          *MessageRefusal          `json:"refusal,omitempty" yaml:"refusal,omitempty"`
	ImageURL         *MessageImageURL         `json:"imageUrl,omitempty" yaml:"imageUrl,omitempty"`
	Audio            *MessageAudio            `json:"audio,omitempty" yaml:"audio,omitempty"`
	File             *MessageFile             `json:"file,omitempty" yaml:"file,omitempty"`
	ToolCall         *MessageToolCall         `json:"toolCall,omitempty" yaml:"toolCall,omitempty"`
	ToolResult       *MessageToolResult       `json:"toolResult,omitempty" yaml:"toolResult,omitempty"`
	RealtimeResponse *MessageRealtimeResponse `json:"realtimeResponse,omitempty" yaml:"realtimeResponse,omitempty"`
}

type MessageText struct {
	Text        string              `json:"text,omitempty" yaml:"text,omitempty"`
	Instruction map[string][]string `json:"instruction,omitempty" yaml:"instruction,omitempty"`
}

func SliceToInstructionMap(items []string) map[string][]string {
	if items == nil {
		return nil
	}

	instructionMap := map[string][]string{}
	for i := 0; i < len(items); i += 2 {
		key := items[i]
		value := ""
		if i+1 < len(items) {
			value = items[i+1]
		}
		instructionMap[key] = append(instructionMap[key], value)
	}

	return instructionMap
}

func CloneInstructionMap(m map[string][]string) map[string][]string {
	if m == nil {
		return nil
	}

	clone := map[string][]string{}
	for k, v := range m {
		vv := append([]string(nil), v...)
		clone[k] = vv
	}
	return clone
}

func NewTextMessage(role MessageRole, text string, instructions ...string) *Message {
	return NewTextMessageWithInstruction(role, text, SliceToInstructionMap(instructions))
}

func NewTextMessageWithInstruction(role MessageRole, text string, instructions map[string][]string) *Message {
	part := &MessageText{Text: text, Instruction: instructions}
	msg := NewMessage(role)
	msg.Parts = append(msg.Parts, &MessagePart{Text: part})
	return msg
}
func (m *MessageText) Clone() *MessageText {
	return &MessageText{
		Text:        m.Text,
		Instruction: CloneInstructionMap(m.Instruction),
	}
}

func (m *MessageText) AddInstruction(k, v string) {
	if m.Instruction == nil {
		m.Instruction = map[string][]string{}
	}
	m.Instruction[k] = append(m.Instruction[k], v)
}

func (m *MessageText) TextWithInstruction() string {
	sb := strings.Builder{}
	sb.WriteString(m.Text)
	if len(m.Instruction) > 0 {
		sb.WriteString("\n")
		keys := make([]string, 0, len(m.Instruction))
		for key := range m.Instruction {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, k := range keys {
			v := m.Instruction[k]
			sb.WriteString("<")
			sb.WriteString(k)
			sb.WriteString(">")
			sb.WriteString(strings.Join(v, "\n"))
			sb.WriteString("</")
			sb.WriteString(k)
			sb.WriteString(">")
			sb.WriteString("\n")
		}
	}

	return sb.String()
}

type MessageReasoning struct {
	Text             string `json:"text,omitempty" yaml:"text,omitempty"`
	ThoughtSignature string `json:"thoughtSignature,omitempty" yaml:"thoughtSignature,omitempty"` //for gemini
}

func (c *Message) Reasoning() string {
	sb := strings.Builder{}
	for _, part := range c.Parts {
		if part.Reasoning != nil {
			sb.WriteString(part.Reasoning.Text)
		}
	}
	return sb.String()
}

type MessageRefusal struct {
	Text string `json:"text,omitempty" yaml:"text,omitempty"`
}

type MessageImageURL struct {
	URL string `json:"url,omitempty" yaml:"url,omitempty"`
	// Any of "auto", "low", "high".
	Detail string `json:"detail,omitempty" yaml:"detail,omitempty"`
	Format string `json:"format,omitempty" yaml:"format,omitempty"`
}

type MessageAudio struct {
	ID         string `json:"id,omitempty" yaml:"id,omitempty"`
	Data       string `json:"data,omitempty" yaml:"data,omitempty"`     // base64 encoded audio data.
	Format     string `json:"format,omitempty" yaml:"format,omitempty"` // openai: any of "wav", "mp3", "pcm16".
	Transcript string `json:"transcript,omitempty" yaml:"transcript,omitempty"`
	Delta      bool   `json:"delta,omitempty" yaml:"delta,omitempty"`
}

type MessageFile struct {
	MIMEType string `json:"mimeType,omitempty" yaml:"mimeType,omitempty"`
	Name     string `json:"name,omitempty" yaml:"name,omitempty"`
	Data     string `json:"data,omitempty" yaml:"data,omitempty"` //base64
}

type MessageToolCall struct {
	ID        string   `json:"id,omitempty" yaml:"id,omitempty"`
	Type      ToolType `json:"type,omitempty" yaml:"type,omitempty"`
	Name      string   `json:"name,omitempty" yaml:"name,omitempty"`
	Arguments string   `json:"arguments,omitempty" yaml:"arguments,omitempty"`
}

type MessageToolResult struct {
	ID          string              `json:"id,omitempty" yaml:"id,omitempty"`
	Name        string              `json:"name,omitempty" yaml:"name,omitempty"`
	Result      string              `json:"result,omitempty" yaml:"result,omitempty"`
	Error       string              `json:"error,omitempty" yaml:"error,omitempty"`
	Instruction map[string][]string `json:"instruction,omitempty" yaml:"instruction,omitempty"`
}

func NewToolResultMessage(id, name, result, error string, instructions ...string) *Message {
	part := &MessageToolResult{
		ID:          id,
		Name:        name,
		Result:      result,
		Error:       error,
		Instruction: SliceToInstructionMap(instructions),
	}

	msg := NewMessage(MessageRoleTool)
	msg.Parts = append(msg.Parts, &MessagePart{ToolResult: part})

	return msg
}

func (m *MessageToolResult) Clone() *MessageToolResult {
	cp := *m
	cp.Instruction = CloneInstructionMap(m.Instruction)
	return &cp
}

func (m *MessageToolResult) Content() string {
	result := m.Result
	if m.Error != "" {
		result += fmt.Sprintf("\nError: %s", m.Error)
	}
	return result
}

func (m *MessageToolResult) ContentWithInstruction() string {
	sb := strings.Builder{}
	sb.WriteString(m.Content())
	if len(m.Instruction) > 0 {
		sb.WriteString("\n")
		keys := make([]string, 0, len(m.Instruction))
		for key := range m.Instruction {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, k := range keys {
			v := m.Instruction[k]
			sb.WriteString("<")
			sb.WriteString(k)
			sb.WriteString(">")
			sb.WriteString(strings.Join(v, "\n"))
			sb.WriteString("</")
			sb.WriteString(k)
			sb.WriteString(">")
			sb.WriteString("\n")
		}
	}
	return sb.String()
}

type MessageRealtimeResponse struct {
}

type Retry struct {
	InitInterval time.Duration
	MaxInterval  time.Duration
	Multiplier   float64
	Max          time.Duration
}

type Embedding struct {
	Model        string
	Data         []*EmbeddingData
	PromptTokens int64
	TotalTokens  int64
}

type EmbeddingData struct {
	Index     int
	Embedding []float64
}

type Completion struct {
	ID      string
	Model   string
	Delta   bool
	Message *Message
	Usage   CompletionUsage
}

type CompletionUsage struct {
	PromptTokens     int64
	CompletionTokens int64
	TotalTokens      int64
	ReasoningTokens  int64
}

type EmbeddingOption func(*EmbeddingOptions) *EmbeddingOptions
type EmbeddingOptions struct {
	Dimensions int
}

func EmbeddingWithOptions(options *EmbeddingOptions) EmbeddingOption {
	return func(opts *EmbeddingOptions) *EmbeddingOptions {
		return options.Clone()
	}
}

func ApplyEmbeddingOptions(def *EmbeddingOptions, opts ...EmbeddingOption) *EmbeddingOptions {
	if def == nil {
		def = &EmbeddingOptions{}
	} else {
		def = def.Clone()
	}
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		def = opt(def)
	}
	return def
}

func (o *EmbeddingOptions) Clone() *EmbeddingOptions {
	if o == nil {
		return &EmbeddingOptions{}
	}
	clone := *o
	return &clone
}

func EmbeddingWithDimensions(dim int) EmbeddingOption {
	return func(opt *EmbeddingOptions) *EmbeddingOptions {
		opt.Dimensions = dim
		return opt
	}
}

type CompletionOption func(*CompletionOptions) *CompletionOptions
type CompletionOptions struct {
	Retry            *Retry
	Instructions     string
	Tools            []*Tool
	StreamingFunc    ChatStreamingFunc
	StreamingAccFunc ChatStreamingFunc
	Temperature      *float32
	TopP             *float32
	TopK             *int
	ReasoningLevel   *int //0-10
	MaxTokens        *int64
	StopSequences    []string
	// Modalities    []Modality
	AudioVoice AudioVoiceType
}

func CompletionWithOptions(options *CompletionOptions) CompletionOption {
	return func(opts *CompletionOptions) *CompletionOptions {
		return options.Clone()
	}
}

func CompletionWithRetry(initInterval, maxInterval, max time.Duration, multiplier float64) CompletionOption {
	return func(opts *CompletionOptions) *CompletionOptions {
		opts.Retry = &Retry{
			InitInterval: initInterval,
			MaxInterval:  maxInterval,
			Multiplier:   multiplier,
			Max:          max,
		}
		return opts
	}
}

func CompletionWithInstructions(Instructions string) CompletionOption {
	return func(opts *CompletionOptions) *CompletionOptions {
		opts.Instructions = Instructions
		return opts
	}
}

func CompletionWithAudioVoice(voice AudioVoiceType) CompletionOption {
	return func(opts *CompletionOptions) *CompletionOptions {
		opts.AudioVoice = voice
		return opts
	}
}

func CompletionWithTools(tools []*Tool) CompletionOption {
	return func(opts *CompletionOptions) *CompletionOptions {
		opts.Tools = append([]*Tool(nil), tools...)
		return opts
	}
}

func CompletionWithTemperature(temperature float32) CompletionOption {
	return func(opts *CompletionOptions) *CompletionOptions {
		opts.Temperature = &temperature
		return opts
	}
}

func CompletionWithTopP(topP float32) CompletionOption {
	return func(opts *CompletionOptions) *CompletionOptions {
		opts.TopP = &topP
		return opts
	}
}

func CompletionWithTopK(topK int) CompletionOption {
	return func(opts *CompletionOptions) *CompletionOptions {
		opts.TopK = &topK
		return opts
	}
}

func CompletionWithMaxTokens(maxTokens int64) CompletionOption {
	return func(opts *CompletionOptions) *CompletionOptions {
		opts.MaxTokens = &maxTokens
		return opts
	}
}

func CompletionWithStopSequences(stopSequences ...string) CompletionOption {
	return func(opts *CompletionOptions) *CompletionOptions {
		opts.StopSequences = append([]string(nil), stopSequences...)
		return opts
	}
}

func ApplyCompletionOptions(def *CompletionOptions, opts ...CompletionOption) *CompletionOptions {
	if def == nil {
		def = &CompletionOptions{}
	} else {
		def = def.Clone()
	}
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		def = opt(def)
	}
	return def
}

func (o *CompletionOptions) Clone() *CompletionOptions {
	if o == nil {
		return &CompletionOptions{}
	}
	clone := *o
	if o.Retry != nil {
		retry := *o.Retry
		clone.Retry = &retry
	}
	clone.Tools = append([]*Tool(nil), o.Tools...)
	clone.StopSequences = append([]string(nil), o.StopSequences...)
	if o.Temperature != nil {
		value := *o.Temperature
		clone.Temperature = &value
	}
	if o.TopP != nil {
		value := *o.TopP
		clone.TopP = &value
	}
	if o.TopK != nil {
		value := *o.TopK
		clone.TopK = &value
	}
	if o.ReasoningLevel != nil {
		value := *o.ReasoningLevel
		clone.ReasoningLevel = &value
	}
	if o.MaxTokens != nil {
		value := *o.MaxTokens
		clone.MaxTokens = &value
	}
	return &clone
}

type ChatStreamingFunc func(ctx context.Context, completion *Completion) error

func CompletionWithStreamingFunc(stream ChatStreamingFunc) CompletionOption {
	return func(opts *CompletionOptions) *CompletionOptions {
		opts.StreamingFunc = stream
		return opts
	}
}

func CompletionWithStreamingAccFunc(stream ChatStreamingFunc) CompletionOption {
	return func(opts *CompletionOptions) *CompletionOptions {
		opts.StreamingAccFunc = stream
		return opts
	}
}

func CompletionWithReasoningLevel(level int) CompletionOption {
	return func(opts *CompletionOptions) *CompletionOptions {
		opts.ReasoningLevel = &level
		return opts
	}
}

type AudioVoiceType string

var (
	AudioVoiceWomen AudioVoiceType = "women"
	AudioVoiceMen   AudioVoiceType = "men"
)

type Modality string

const (
	ModalityText  Modality = "text"
	ModalityAudio Modality = "audio"
)

type TurnDetectionType string

const (
	TurnDetectionServer   TurnDetectionType = "server"
	TurnDetectionSemantic TurnDetectionType = "semantic"
)

type RealTimeOptions struct {
	Instructions string
	Tools        []*Tool
	AudioVoice   *AudioVoiceType
}

type RealTimeOption func(*RealTimeOptions) *RealTimeOptions

func RealTimeWith(opts *RealTimeOptions) RealTimeOption {
	return func(current *RealTimeOptions) *RealTimeOptions {
		return opts.Clone()
	}
}

func ApplyRealTimeOptions(def *RealTimeOptions, opts ...RealTimeOption) *RealTimeOptions {
	if def == nil {
		def = &RealTimeOptions{}
	} else {
		def = def.Clone()
	}
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		def = opt(def)
	}
	return def
}

func (o *RealTimeOptions) Clone() *RealTimeOptions {
	if o == nil {
		return &RealTimeOptions{}
	}
	clone := *o
	clone.Tools = append([]*Tool(nil), o.Tools...)
	if o.AudioVoice != nil {
		voice := *o.AudioVoice
		clone.AudioVoice = &voice
	}
	return &clone
}

func RealTimeWithInstructions(instructions string) RealTimeOption {
	return func(opts *RealTimeOptions) *RealTimeOptions {
		opts.Instructions = instructions
		return opts
	}
}

func RealTimeWithTools(tools []*Tool) RealTimeOption {
	return func(opts *RealTimeOptions) *RealTimeOptions {
		opts.Tools = append([]*Tool(nil), tools...)
		return opts
	}
}

func RealTimeWithAudioVoice(voice AudioVoiceType) RealTimeOption {
	return func(opts *RealTimeOptions) *RealTimeOptions {
		opts.AudioVoice = &voice
		return opts
	}
}

type RealTimeSession interface {
	Send(context.Context, *Message) error
	Recv(context.Context) (*Completion, error)
	Close()
}

type RerankOptions struct {
	Instruction string
	OutputDoc   bool
}

type RerankOption func(*RerankOptions) *RerankOptions

func ApplyRerankOptions(def *RerankOptions, opts ...RerankOption) *RerankOptions {
	if def == nil {
		def = &RerankOptions{}
	} else {
		clone := *def
		def = &clone
	}
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		def = opt(def)
	}
	return def
}

func RerankWithInstruction(ins string) RerankOption {
	return func(o *RerankOptions) *RerankOptions {
		o.Instruction = ins
		return o
	}
}

func RerankWithOutputDoc(t bool) RerankOption {
	return func(o *RerankOptions) *RerankOptions {
		o.OutputDoc = t
		return o
	}
}

type RerankInput struct {
	Query     string
	Documents []string
	TopN      int
}

type RerankOutput struct {
	Result []*RerankResultItem
}

type RerankResultItem struct {
	Index    int
	Document string
	Score    float64
}

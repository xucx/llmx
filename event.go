package llmx

import (
	"fmt"
	"maps"
	"strconv"

	"github.com/google/uuid"
	"github.com/xucx/llmx/types"
)

// Event is one fact appended to a Session: what happened in a conversation,
// independent of any model provider. Exactly one payload field is set. Once
// appended an event is immutable; Clone before modifying.
//
// Events are the only state of an agent. Fold derives the State from them,
// Plan derives the Effects that must happen next, and executors append the
// outcome of those effects as new events.
type Event struct {
	// ID identifies the event. Outcome events use deterministic IDs (see
	// ReplyID, ToolResultID, ...) so two executors producing the same outcome
	// collide in the Session instead of duplicating it.
	ID string `json:"id"`
	// Seq is the position in the Session, assigned by Session.Append.
	Seq int64 `json:"seq,omitempty"`
	// At is the append time in unix milliseconds, assigned by Session.Append.
	At int64 `json:"at,omitempty"`

	Input      *Input      `json:"input,omitempty"`
	Reply      *Reply      `json:"reply,omitempty"`
	ToolResult *ToolResult `json:"toolResult,omitempty"`
	ToolAsk    *ToolAsk    `json:"toolAsk,omitempty"`
	ToolAnswer *ToolAnswer `json:"toolAnswer,omitempty"`
	Patch      *Patch      `json:"patch,omitempty"`
	Usage      *Usage      `json:"usage,omitempty"`
	Stop       *Stop       `json:"stop,omitempty"`
}

// InputMode tells when an input joins the conversation.
type InputMode string

const (
	// InputQueue waits until the conversation is idle: no tool call pending
	// and nothing left for the model to answer.
	InputQueue InputMode = "queue"
	// InputInterject joins at the next turn boundary: once every tool call of
	// the current turn has its result, before the next model call.
	InputInterject InputMode = "interject"
	// InputInterrupt cancels what is running: pending tool calls get an error
	// result, an in-flight model call is canceled and keeps its partial text,
	// inputs still waiting are discarded. The input then joins at once.
	InputInterrupt InputMode = "interrupt"
)

// Content is one piece of user input, independent of any provider.
type Content struct {
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"imageUrl,omitempty"`
}

// ToolSpec declares a tool to the model: its name, what it does and the JSON
// schema of its arguments. It says nothing about where the tool runs.
type ToolSpec struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// Input is what a caller feeds the conversation.
type Input struct {
	Mode    InputMode `json:"mode,omitempty"`
	Content []Content `json:"content,omitempty"`
	// State patches the key-value state before the content is seen; Reset
	// replaces the state instead of merging.
	State map[string]string `json:"state,omitempty"`
	Reset bool              `json:"reset,omitempty"`
	// Tools declares the tools the sender can run, visible to the model from
	// this input on until another input declares tools.
	Tools []ToolSpec `json:"tools,omitempty"`
}

// ToolCall is one call the model made.
type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments,omitempty"`
}

// Reply is everything one model call produced.
type Reply struct {
	// BaseSeq is the Session seq the call was made from: the messages sent
	// to the model were built from the events up to it. Inputs appended after
	// it and before the reply arrived during the turn (see Fold).
	BaseSeq   int64      `json:"baseSeq"`
	Model     string     `json:"model,omitempty"`
	Text      string     `json:"text,omitempty"`
	Reasoning string     `json:"reasoning,omitempty"`
	ToolCalls []ToolCall `json:"toolCalls,omitempty"`
	// Partial marks a reply cut short by an interruption: its text is what
	// arrived, and it never carries tool calls.
	Partial bool `json:"partial,omitempty"`
}

// ToolResult answers one tool call.
type ToolResult struct {
	CallID string `json:"callId"`
	Name   string `json:"name,omitempty"`
	// Value is shown to the model: a string as is, anything else as JSON.
	Value string `json:"value,omitempty"`
	Error string `json:"error,omitempty"`
	// Instruction adds tagged instructions to the result the model sees.
	Instruction map[string]string `json:"instruction,omitempty"`
	// Final marks a result the model need not answer.
	Final bool `json:"final,omitempty"`
}

// ToolAsk is a question a tool raises before answering its call, typically
// a confirmation. The call stays pending until a ToolAnswer arrives, then the
// tool runs again with the answer.
type ToolAsk struct {
	CallID string         `json:"callId"`
	Args   map[string]any `json:"args,omitempty"`
}

// ToolAnswer answers a ToolAsk.
type ToolAnswer struct {
	CallID string         `json:"callId"`
	Result map[string]any `json:"result,omitempty"`
}

// Patch changes the key-value state; Reset replaces it instead of merging.
type Patch struct {
	KV    map[string]string `json:"kv,omitempty"`
	Reset bool              `json:"reset,omitempty"`
}

// Usage records the tokens one model call consumed.
type Usage struct {
	BaseSeq int64                 `json:"baseSeq,omitempty"`
	Model   string                `json:"model,omitempty"`
	Usage   types.CompletionUsage `json:"usage"`
}

// StopReason tells why a round stopped before answering.
type StopReason string

const (
	StopInterrupted StopReason = "interrupted"
	StopMaxTurn     StopReason = "maxTurn"
	StopError       StopReason = "error"
)

// Stop closes the current round: the model is not asked again until a new
// input is admitted.
type Stop struct {
	Reason StopReason `json:"reason"`
	Error  string     `json:"error,omitempty"`
}

// ---- deterministic IDs ----

func ReplyID(baseSeq int64) string      { return "reply:" + strconv.FormatInt(baseSeq, 10) }
func UsageID(baseSeq int64) string      { return "usage:" + strconv.FormatInt(baseSeq, 10) }
func ToolResultID(callID string) string { return "tool:" + callID }
func ToolAskID(callID string) string    { return "ask:" + callID }
func ToolAnswerID(callID string) string { return "answer:" + callID }

// ---- constructors ----

// NewInputEvent wraps an input into an event with a fresh ID.
func NewInputEvent(input *Input) *Event {
	return &Event{ID: uuid.NewString(), Input: input.Clone()}
}

// NewUserInput creates a text input in the given mode.
func NewUserInput(mode InputMode, text string) *Input {
	return &Input{Mode: mode, Content: []Content{{Text: text}}}
}

func NewReplyEvent(reply *Reply) *Event {
	return &Event{ID: ReplyID(reply.BaseSeq), Reply: reply.Clone()}
}

func NewToolResultEvent(result *ToolResult) *Event {
	return &Event{ID: ToolResultID(result.CallID), ToolResult: result.Clone()}
}

func NewToolAskEvent(ask *ToolAsk) *Event {
	return &Event{ID: ToolAskID(ask.CallID), ToolAsk: ask.Clone()}
}

func NewToolAnswerEvent(answer *ToolAnswer) *Event {
	return &Event{ID: ToolAnswerID(answer.CallID), ToolAnswer: answer.Clone()}
}

func NewPatchEvent(kv map[string]string, reset bool) *Event {
	return &Event{ID: uuid.NewString(), Patch: &Patch{KV: maps.Clone(kv), Reset: reset}}
}

func NewUsageEvent(baseSeq int64, model string, usage types.CompletionUsage) *Event {
	return &Event{ID: UsageID(baseSeq), Usage: &Usage{BaseSeq: baseSeq, Model: model, Usage: usage}}
}

func NewStopEvent(id string, reason StopReason, err string) *Event {
	return &Event{ID: id, Stop: &Stop{Reason: reason, Error: err}}
}

// ---- validation ----

func (e *Event) Validate() error {
	if e == nil {
		return fmt.Errorf("event is nil")
	}
	if e.ID == "" {
		return fmt.Errorf("event id is empty")
	}
	fields := 0
	for _, present := range []bool{
		e.Input != nil, e.Reply != nil, e.ToolResult != nil, e.ToolAsk != nil,
		e.ToolAnswer != nil, e.Patch != nil, e.Usage != nil, e.Stop != nil,
	} {
		if present {
			fields++
		}
	}
	if fields != 1 {
		return fmt.Errorf("event %q has %d payloads, want 1", e.ID, fields)
	}
	switch {
	case e.Input != nil:
		return e.Input.Validate()
	case e.Reply != nil:
		for index, call := range e.Reply.ToolCalls {
			if call.ID == "" || call.Name == "" {
				return fmt.Errorf("event %q tool call %d has empty id or name", e.ID, index)
			}
		}
		if e.Reply.Partial && len(e.Reply.ToolCalls) > 0 {
			return fmt.Errorf("event %q partial reply carries tool calls", e.ID)
		}
	case e.ToolResult != nil:
		if e.ToolResult.CallID == "" {
			return fmt.Errorf("event %q tool result call id is empty", e.ID)
		}
	case e.ToolAsk != nil:
		if e.ToolAsk.CallID == "" {
			return fmt.Errorf("event %q tool ask call id is empty", e.ID)
		}
	case e.ToolAnswer != nil:
		if e.ToolAnswer.CallID == "" {
			return fmt.Errorf("event %q tool answer call id is empty", e.ID)
		}
	case e.Patch != nil:
		for key := range e.Patch.KV {
			if key == "" {
				return fmt.Errorf("event %q patch key is empty", e.ID)
			}
		}
	case e.Stop != nil:
		if e.Stop.Reason == "" {
			return fmt.Errorf("event %q stop reason is empty", e.ID)
		}
	}
	return nil
}

func (i *Input) Validate() error {
	if i == nil {
		return fmt.Errorf("input is nil")
	}
	switch i.Mode {
	case InputQueue, InputInterject, InputInterrupt:
	case "":
		return fmt.Errorf("input mode is empty")
	default:
		return fmt.Errorf("invalid input mode %q", i.Mode)
	}
	for key := range i.State {
		if key == "" {
			return fmt.Errorf("input state key is empty")
		}
	}
	names := map[string]struct{}{}
	for index, tool := range i.Tools {
		if tool.Name == "" {
			return fmt.Errorf("input tool %d name is empty", index)
		}
		if _, dup := names[tool.Name]; dup {
			return fmt.Errorf("input declares tool %q twice", tool.Name)
		}
		names[tool.Name] = struct{}{}
	}
	return nil
}

// HasContent reports whether the input gives the model something to answer.
func (i *Input) HasContent() bool {
	if i == nil {
		return false
	}
	for _, content := range i.Content {
		if content.Text != "" || content.ImageURL != "" {
			return true
		}
	}
	return false
}

// Text joins the text contents of the input.
func (i *Input) Text() string {
	text := ""
	for _, content := range i.Content {
		text += content.Text
	}
	return text
}

// ---- clones ----

func (e *Event) Clone() *Event {
	if e == nil {
		return nil
	}
	clone := &Event{ID: e.ID, Seq: e.Seq, At: e.At}
	clone.Input = e.Input.Clone()
	clone.Reply = e.Reply.Clone()
	clone.ToolResult = e.ToolResult.Clone()
	clone.ToolAsk = e.ToolAsk.Clone()
	clone.ToolAnswer = e.ToolAnswer.Clone()
	if e.Patch != nil {
		clone.Patch = &Patch{KV: maps.Clone(e.Patch.KV), Reset: e.Patch.Reset}
	}
	if e.Usage != nil {
		value := *e.Usage
		clone.Usage = &value
	}
	if e.Stop != nil {
		value := *e.Stop
		clone.Stop = &value
	}
	return clone
}

func (i *Input) Clone() *Input {
	if i == nil {
		return nil
	}
	clone := *i
	clone.Content = append([]Content(nil), i.Content...)
	clone.State = maps.Clone(i.State)
	clone.Tools = cloneToolSpecs(i.Tools)
	return &clone
}

func cloneToolSpecs(specs []ToolSpec) []ToolSpec {
	if specs == nil {
		return nil
	}
	clones := make([]ToolSpec, len(specs))
	for index, spec := range specs {
		clones[index] = ToolSpec{
			Name:        spec.Name,
			Description: spec.Description,
			Parameters:  cloneAnyMap(spec.Parameters),
		}
	}
	return clones
}

func (r *Reply) Clone() *Reply {
	if r == nil {
		return nil
	}
	clone := *r
	clone.ToolCalls = append([]ToolCall(nil), r.ToolCalls...)
	return &clone
}

func (r *ToolResult) Clone() *ToolResult {
	if r == nil {
		return nil
	}
	clone := *r
	clone.Instruction = maps.Clone(r.Instruction)
	return &clone
}

func (a *ToolAsk) Clone() *ToolAsk {
	if a == nil {
		return nil
	}
	return &ToolAsk{CallID: a.CallID, Args: cloneAnyMap(a.Args)}
}

func (a *ToolAnswer) Clone() *ToolAnswer {
	if a == nil {
		return nil
	}
	return &ToolAnswer{CallID: a.CallID, Result: cloneAnyMap(a.Result)}
}

func cloneAnyMap(source map[string]any) map[string]any {
	if source == nil {
		return nil
	}
	clone := make(map[string]any, len(source))
	for key, value := range source {
		clone[key] = cloneAny(value)
	}
	return clone
}

func cloneAny(value any) any {
	switch value := value.(type) {
	case map[string]any:
		return cloneAnyMap(value)
	case []any:
		clone := make([]any, len(value))
		for index, item := range value {
			clone[index] = cloneAny(item)
		}
		return clone
	default:
		return value
	}
}

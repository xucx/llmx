package llmx

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"sync"

	"github.com/xucx/gox/logx"
	"github.com/xucx/llmx/types"
)

var (
	ErrMaxTurnExceeded = errors.New("max turn exceeded")
	ErrSessionClosed   = errors.New("session closed")
	ErrSessionStarted  = errors.New("session already started")
	ErrSessionBusy     = errors.New("session busy")
)

// SessionTurn is the immutable input for one model turn in a session. History,
// tools, and their nested values are read-only; clone the turn before changing
// its slice containers. State is exposed only through read-only snapshots.
type SessionTurn struct {
	Index       int
	Model       string
	Instruction string
	History     []*types.Message
	Tools       []*Tool

	state map[string]string
}

// Clone returns a turn with independently owned slice containers and state.
// Messages and tools remain shared immutable values.
func (t *SessionTurn) Clone() *SessionTurn {
	if t == nil {
		return nil
	}
	clone := *t
	clone.History = append([]*types.Message(nil), t.History...)
	clone.Tools = append([]*Tool(nil), t.Tools...)
	clone.state = cloneStringMap(t.state)
	return &clone
}

// GetState reads one value from the state snapshot captured for this turn.
func (t *SessionTurn) GetState(key string) (string, bool) {
	if t == nil {
		return "", false
	}
	value, ok := t.state[key]
	return value, ok
}

// StateSnapshot returns an independently owned copy of this turn's state.
// Changing the returned map never changes the session.
func (t *SessionTurn) StateSnapshot() map[string]string {
	if t == nil {
		return map[string]string{}
	}
	return cloneStringMap(t.state)
}

// SessionTurnResult is what one model turn produced, after every event of that
// turn is committed. It is the input a Loop decides on.
type SessionTurnResult struct {
	Index          int
	Events         []*Event
	TurnEventIndex int
}

type StopReason string

const (
	StopDone     StopReason = "done"
	StopMaxTurn  StopReason = "maxTurn"
	StopCanceled StopReason = "canceled"
	StopError    StopReason = "error"
)

// SessionResult is the final snapshot produced by one Run, or by the lifetime
// of a session started with Start.
type SessionResult struct {
	Events []*Event
	State  map[string]string

	Turns      int
	StopReason StopReason
	Cause      error
}

func (r *SessionResult) Clone() *SessionResult {
	if r == nil {
		return nil
	}
	clone := *r
	clone.Events = append([]*Event(nil), r.Events...)
	clone.State = cloneStringMap(r.State)
	return &clone
}

// SessionHumanResponse answers one human request raised by a session. ID must
// match the ID of an EventHumanRequest emitted by the session.
type SessionHumanResponse struct {
	ID     string         `json:"id,omitempty" yaml:"id,omitempty"`
	Result map[string]any `json:"result,omitempty" yaml:"result,omitempty"`
}

func (r *SessionHumanResponse) clone() *SessionHumanResponse {
	if r == nil {
		return nil
	}
	clone := *r
	clone.Result = cloneAnyMap(r.Result)
	return &clone
}

// SessionInput is the only way a caller feeds new input into a session. It is
// used both by Session.Run and as the payload of Runner.Send.
//
// Callers never build input events themselves: the session turns one input
// into events, so every event a caller observes comes from session output.
type SessionInput struct {
	// Messages are appended to the model history, one message event each.
	Messages []*types.Message `json:"messages,omitempty" yaml:"messages,omitempty"`

	// HumanResponses resolve human requests that are still pending.
	HumanResponses []*SessionHumanResponse `json:"humanResponses,omitempty" yaml:"humanResponses,omitempty"`

	// State replaces the session state before the rest of this input.
	// StateDelta merges into the current session state instead.
	State      map[string]string `json:"state,omitempty" yaml:"state,omitempty"`
	StateDelta map[string]string `json:"stateDelta,omitempty" yaml:"stateDelta,omitempty"`
}

// NewSessionInput creates one input carrying messages.
func NewSessionInput(messages ...*types.Message) *SessionInput {
	return &SessionInput{Messages: cloneMessages(messages)}
}

// NewUserInput creates one input carrying a single user text message.
func NewUserInput(text string) *SessionInput {
	return NewSessionInput(types.NewTextMessage(types.MessageRoleUser, text))
}

// NewHumanResponseInput creates one input answering a pending human request.
func NewHumanResponseInput(
	requestID string,
	result map[string]any,
) *SessionInput {
	return new(SessionInput).WithHumanResponse(requestID, result)
}

// WithMessages appends messages to this input.
func (i *SessionInput) WithMessages(messages ...*types.Message) *SessionInput {
	if i == nil {
		i = &SessionInput{}
	}
	i.Messages = append(i.Messages, cloneMessages(messages)...)
	return i
}

// WithHumanResponse appends one human response to this input.
func (i *SessionInput) WithHumanResponse(
	id string,
	result map[string]any,
) *SessionInput {
	if i == nil {
		i = &SessionInput{}
	}
	i.HumanResponses = append(i.HumanResponses, &SessionHumanResponse{
		ID:     id,
		Result: cloneAnyMap(result),
	})
	return i
}

// WithState sets either a full state replacement or a state patch on this
// input. When delta is true, state is merged; otherwise it replaces the state.
func (i *SessionInput) WithState(state map[string]string, delta bool) *SessionInput {
	if i == nil {
		i = &SessionInput{}
	}

	clone := maps.Clone(state)
	if delta {
		i.StateDelta = clone
	} else {
		i.State = clone
	}

	return i
}

// IsEmpty reports whether this input carries nothing for the session to accept.
func (i *SessionInput) IsEmpty() bool {
	return i == nil ||
		(i.State == nil &&
			len(i.StateDelta) == 0 &&
			len(i.HumanResponses) == 0 &&
			len(i.Messages) == 0)
}

func (i *SessionInput) Clone() *SessionInput {
	if i == nil {
		return nil
	}

	clone := &SessionInput{
		State:      maps.Clone(i.State),
		StateDelta: maps.Clone(i.StateDelta),
		Messages:   cloneMessages(i.Messages),
	}
	if len(i.HumanResponses) == 0 {
		return clone
	}
	clone.HumanResponses = make([]*SessionHumanResponse, len(i.HumanResponses))
	for index, response := range i.HumanResponses {
		clone.HumanResponses[index] = response.clone()
	}
	return clone
}

func (i *SessionInput) Validate() error {
	if i.IsEmpty() {
		return nil
	}
	if i.State != nil && i.StateDelta != nil {
		return fmt.Errorf("state and stateDelta both set")
	}

	if err := validateState(i.State); err != nil {
		return fmt.Errorf("state: %w", err)
	}
	if err := validateState(i.StateDelta); err != nil {
		return fmt.Errorf("stateDelta: %w", err)
	}
	for index, response := range i.HumanResponses {
		if response == nil {
			return fmt.Errorf("human response %d is nil", index)
		}
		if response.ID == "" {
			return fmt.Errorf("human response %d request id is empty", index)
		}
	}
	for index, message := range i.Messages {
		if message == nil {
			return fmt.Errorf("message %d is nil", index)
		}
		if message.Delta {
			return fmt.Errorf("message %d is delta", index)
		}
	}
	return types.ValidateMessages(i.Messages)
}

// events converts this input into the ordered events the session commits. State
// lands first so tools observe it, human responses next so pending tools can
// resume, and messages last so they are the newest history.
//
// Responses are not deduplicated here: a second response for the same request
// is rejected when the session commits it, so the contradiction fails loudly
// instead of being silently dropped.
func (i *SessionInput) events() ([]*Event, error) {
	if err := i.Validate(); err != nil {
		return nil, err
	}
	if i.IsEmpty() {
		return nil, nil
	}

	events := make([]*Event, 0, len(i.HumanResponses)+len(i.Messages)+2)
	if i.State != nil {
		events = append(events, NewEventState(i.State, false))
	}
	if len(i.StateDelta) > 0 {
		events = append(events, NewEventState(i.StateDelta, true))
	}
	for _, response := range i.HumanResponses {
		events = append(events, NewEventHumanResponse(&EventHumanResponse{
			ID:     response.ID,
			Result: response.Result,
		}))
	}
	for _, message := range i.Messages {
		events = append(events, NewEventMessage(message))
	}
	return events, nil
}

type sessionLifecycle uint8

const (
	sessionIdle sessionLifecycle = iota
	sessionRunning
	sessionServing
	sessionClosed
)

// Session owns conversation history, state, and tool correlation across runs.
// Run processes one fixed input synchronously. Start switches the session into
// a long-running mode that waits for inputs until its context is canceled.
type Session struct {
	ctx context.Context

	lifecycleMu sync.Mutex
	lifecycle   sessionLifecycle

	models  *Models
	options *AgentOptions

	events    []*Event
	eventsMap map[string]*Event
	history   []*types.Message
	// contextTokens is the size of the context the next turn will send. A usage
	// event reports it, a compression estimates it down; nothing else reads the
	// rest of a completion's usage, so only this number is kept.
	contextTokens    int64
	state            map[string]string
	humanRequests    map[string]*EventHumanRequest
	humanResponses   map[string]*EventHumanResponse
	pendingResponses []*EventHumanResponse
	toolCalls        map[string]*types.MessageToolCall
	toolResults      map[string]*types.MessageToolResult

	// turnsSinceInput counts model turns since the last accepted input. It is
	// the counter maxTurn bounds, so fresh input grants a new turn budget
	// instead of sharing one budget with the whole session.
	turnsSinceInput int
	// result is only touched by the executing goroutine: streaming callbacks
	// run synchronously inside complete, so no lock guards it.
	result SessionResult
}

// NewSession restores one reusable in-memory session. Historical events and
// their nested values are treated as immutable and retained by reference; clone
// them before making changes. They restore state but are neither re-emitted to
// EventSink nor included in future results.
func NewSession(
	agent *Agent,
	history []*Event,
	opts ...AgentOption,
) (*Session, error) {
	if agent == nil {
		return nil, fmt.Errorf("agent is nil")
	}

	options, err := agent.optionsForSession(opts...)
	if err != nil {
		return nil, err
	}

	session := &Session{
		models:         agent.models,
		options:        options,
		events:         []*Event{},
		eventsMap:      map[string]*Event{},
		state:          map[string]string{},
		humanRequests:  map[string]*EventHumanRequest{},
		humanResponses: map[string]*EventHumanResponse{},
		toolCalls:      map[string]*types.MessageToolCall{},
		toolResults:    map[string]*types.MessageToolResult{},
	}

	for _, evt := range history {
		if err := session.addEvent(evt, true); err != nil {
			return nil, err
		}
	}
	for _, evt := range session.events {
		if evt == nil || evt.HumanResponse == nil {
			continue
		}
		request := session.humanRequests[evt.HumanResponse.ID]
		if request == nil {
			continue
		}
		if _, resolved := session.toolResults[request.ToolCall.ID]; !resolved {
			session.pendingResponses = append(
				session.pendingResponses,
				evt.HumanResponse,
			)
		}
	}

	return session, nil
}

// Start switches this session into long-running mode. It initially waits for
// input, runs autonomous turns while Loop returns LoopContinue, and returns to
// waiting whenever Loop returns LoopWait. The context owns the whole lifetime.
func (s *Session) Start(ctx context.Context) (*Runner, error) {
	if s == nil || s.models == nil {
		return nil, fmt.Errorf("session is not initialized")
	}
	if ctx == nil {
		return nil, fmt.Errorf("start context is nil")
	}
	if err := s.acquireStart(); err != nil {
		return nil, err
	}

	sessionCtx, cancel := context.WithCancelCause(ctx)
	runner := newRunner()
	s.beginExecution(sessionCtx)

	go func() {
		result, finalErr := s.finalize(s.serve(runner))
		cancel(finalErr)
		s.finishStart()
		runner.finish(result, finalErr)
	}()

	return runner, nil
}

// Run processes one fixed input and every autonomous turn it causes. It returns
// when Loop yields or stops. After LoopWait the session can be Run again with a
// later input. Any error -- including cancellation of this Run's context --
// closes the session, because a partial turn may leave the history in a shape
// the model must not resume; rebuild a session from the sinked events instead.
func (s *Session) Run(
	ctx context.Context,
	input *SessionInput,
) (*SessionResult, error) {
	if s == nil || s.models == nil {
		return nil, fmt.Errorf("session is not initialized")
	}
	if ctx == nil {
		return nil, fmt.Errorf("run context is nil")
	}
	if err := input.Validate(); err != nil {
		return nil, err
	}
	input = input.Clone()

	if err := s.acquireRun(); err != nil {
		return nil, err
	}
	if input.IsEmpty() && len(s.pendingResponses) == 0 {
		s.finishRun(false)
		return nil, fmt.Errorf("run input is empty")
	}
	s.beginExecution(ctx)

	err := s.resumePendingHumanTools()
	if err == nil && !input.IsEmpty() {
		err = s.commitInput(input)
	}
	action := LoopWait
	if err == nil {
		action, err = s.executeTurns()
	}
	result, finalErr := s.finalize(err)
	s.finishRun(action == LoopStop || err != nil)
	return result, finalErr
}

func (s *Session) acquireStart() error {
	s.lifecycleMu.Lock()
	defer s.lifecycleMu.Unlock()
	switch s.lifecycle {
	case sessionIdle:
		s.lifecycle = sessionServing
		return nil
	case sessionRunning:
		return ErrSessionBusy
	case sessionServing:
		return ErrSessionStarted
	default:
		return ErrSessionClosed
	}
}

func (s *Session) acquireRun() error {
	s.lifecycleMu.Lock()
	defer s.lifecycleMu.Unlock()
	switch s.lifecycle {
	case sessionIdle:
		s.lifecycle = sessionRunning
		return nil
	case sessionRunning, sessionServing:
		return ErrSessionBusy
	default:
		return ErrSessionClosed
	}
}

func (s *Session) finishStart() {
	s.lifecycleMu.Lock()
	s.lifecycle = sessionClosed
	s.lifecycleMu.Unlock()
}

func (s *Session) finishRun(closeSession bool) {
	s.lifecycleMu.Lock()
	if closeSession {
		s.lifecycle = sessionClosed
	} else {
		s.lifecycle = sessionIdle
	}
	s.lifecycleMu.Unlock()
}

func (s *Session) beginExecution(ctx context.Context) {
	s.ctx = ctx
	s.turnsSinceInput = 0
	s.result = SessionResult{}
}

func (s *Session) serve(runner *Runner) error {
	if len(s.pendingResponses) > 0 {
		if err := s.resumePendingHumanTools(); err != nil {
			return err
		}
		action, err := s.executeTurns()
		if err != nil {
			return err
		}
		if action == LoopStop {
			return nil
		}
	}

	for {
		input, err := runner.receive(s.ctx)
		if err != nil {
			return err
		}
		if err := s.commitInput(input); err != nil {
			return err
		}
		s.turnsSinceInput = 0

		action, err := s.executeTurns()
		if err != nil {
			return err
		}
		if action == LoopStop {
			return nil
		}
	}
}

func (s *Session) resumePendingHumanTools() error {
	for len(s.pendingResponses) > 0 {
		response := s.pendingResponses[0]
		if err := s.resumeHumanTool(response); err != nil {
			return err
		}
		s.pendingResponses = s.pendingResponses[1:]
	}
	return nil
}

func (s *Session) executeTurns() (LoopAction, error) {
	for {
		if err := contextError(s.ctx); err != nil {
			return LoopStop, err
		}

		if s.options.MaxTurn > 0 && s.turnsSinceInput >= s.options.MaxTurn {
			return LoopStop, fmt.Errorf("%w: %d", ErrMaxTurnExceeded, s.options.MaxTurn)
		}
		turnIndex := s.result.Turns
		s.result.Turns++
		s.turnsSinceInput++
		turnEventIndex := len(s.result.Events)

		if s.options.Compress != nil {
			events, err := Compress(s.ctx, s.models, &CompressInput{
				Model:    s.options.Model,
				Messages: s.history,
				Options:  s.options.Compress,
				Tokens:   s.contextTokens,
			})
			if err != nil {
				return LoopStop, err
			}
			if len(events) > 0 {
				for _, event := range events {
					if err := s.addEvent(event, false); err != nil {
						return LoopStop, err
					}
				}
			}
		}

		logx.Debug("session call complete")
		completion, err := s.complete(turnIndex)
		if err != nil {
			return LoopStop, err
		}
		if completion == nil || completion.Message == nil {
			return LoopStop, fmt.Errorf("llm returned no message")
		}

		if err := s.addEvent(NewEventMessage(completion.Message), false); err != nil {
			return LoopStop, err
		}

		if err := s.addEvent(NewEventUsage(completion.Usage), false); err != nil {
			return LoopStop, err
		}

		toolCalls := completion.Message.ToolCalls()
		if len(toolCalls) > 0 {
			toolRuns, err := s.invokeToolsParallel(s.ctx, toolCalls, nil, nil)
			if err != nil {
				return LoopStop, err
			}
			toolEvents, err := s.toolRunEvents(toolRuns, false)
			if err != nil {
				return LoopStop, err
			}
			for _, event := range toolEvents {
				if err := s.addEvent(event, false); err != nil {
					return LoopStop, err
				}
			}
		}

		// This turn is complete: every event it produced is committed, so the
		// loop decides on the same facts the caller sees.
		turnEvents := append([]*Event(nil), s.result.Events...)
		action, err := s.options.Loop(s.ctx, &SessionTurnResult{
			Index:          turnIndex,
			Events:         turnEvents,
			TurnEventIndex: turnEventIndex,
		})
		if err != nil {
			return LoopStop, err
		}
		switch action {
		case LoopContinue:
			continue
		case LoopWait, LoopStop:
			return action, nil
		default:
			return LoopStop, fmt.Errorf("invalid loop action %d", action)
		}
	}
}

func (s *Session) addEvent(event *Event, isHistory bool) error {
	if err := event.Validate(); err != nil {
		return fmt.Errorf("event validate fail %w", err)
	}

	if _, exists := s.eventsMap[event.ID]; exists {
		err := fmt.Errorf("duplicate event id %q", event.ID)
		return err
	}

	// The sink is write-ahead: an event the caller could not persist must not
	// reach the history, the result or the model.
	if !isHistory {
		if err := s.commitEvent(event); err != nil {
			return err
		}
	}

	switch {
	case event.Message != nil:
		message := event.Message
		if message.Delta {
			break
		}

		for _, part := range message.Parts {
			if part != nil {
				if part.ToolCall != nil && part.ToolCall.ID != "" {
					s.toolCalls[part.ToolCall.ID] = part.ToolCall
				}
				if part.ToolResult != nil && part.ToolResult.ID != "" {
					s.toolResults[part.ToolResult.ID] = part.ToolResult
				}
			}
		}

		logx.Debugw("session add history", "message", message)
		s.history = append(s.history, message)
	case event.HumanRequest != nil:
		request := event.HumanRequest
		call, exists := s.toolCalls[request.ToolCall.ID]
		if !exists {
			return fmt.Errorf(
				"human request references unknown tool call %q",
				request.ToolCall.ID,
			)
		}
		if call.Name != request.ToolCall.Name {
			return fmt.Errorf(
				"human request tool call %q name mismatch",
				request.ToolCall.ID,
			)
		}
		if _, exists := s.humanRequests[request.ID]; exists {
			return fmt.Errorf(
				"duplicate human request id %q",
				request.ID,
			)
		}
		s.humanRequests[request.ID] = request
	case event.HumanResponse != nil:
		response := event.HumanResponse
		request, exists := s.humanRequests[response.ID]
		if !exists {
			return fmt.Errorf(
				"human response references unknown request %q",
				response.ID,
			)
		}
		if _, resolved := s.toolResults[request.ToolCall.ID]; resolved {
			return fmt.Errorf(
				"human response references resolved request %q",
				response.ID,
			)
		}
		if _, exists := s.humanResponses[response.ID]; exists {
			return fmt.Errorf(
				"duplicate human response for request %q",
				response.ID,
			)
		}
		s.humanResponses[response.ID] = response
	case event.Compression != nil:
		compressedHistory, reduced, err := ApplyCompression(s.history, event.Compression)
		if err != nil {
			return err
		}
		s.history = compressedHistory
		// Only a usage event is a context size the model reported. A compression
		// is an estimate of what it removed, so it adjusts the last reported
		// size down instead of replacing it, and the history and the context size
		// that decides the next compression still move together.
		s.contextTokens = subTokens(s.contextTokens, reduced)
		logx.Debugw("session apply compress", "compress", event.Compression, "tokens", s.contextTokens)
	case event.State != nil:
		if event.State.Patch {
			for key, value := range event.State.State {
				s.state[key] = value
			}
		} else {
			if event.State.State != nil {
				s.state = cloneStringMap(event.State.State)
			} else {
				s.state = map[string]string{}
			}
		}
		logx.Debugw("session change state", "state", s.state)
	case event.Usage != nil:
		// A completion reports the whole context it consumed, so the newest
		// report replaces the previous one instead of adding to it.
		s.contextTokens = event.Usage.Usage.TotalTokens
		logx.Debugw("session change context by usage", "tokens", s.contextTokens)
	}

	s.events = append(s.events, event)
	s.eventsMap[event.ID] = event

	return nil
}

// commitEvent publishes one immutable event this session accepted or produced.
// EventSink and SessionResult share the same pointer; consumers must clone an
// event before modifying it. Historical events never reach this method.
func (s *Session) commitEvent(event *Event) error {
	if s.options.EventSink != nil {
		if err := s.options.EventSink(s.ctx, event); err != nil {
			return fmt.Errorf("event sink fail %w", err)
		}
	}

	s.result.Events = append(s.result.Events, event)

	return nil
}

// resumeHumanTool re-invokes the tool call that raised request, now that a
// human answered it. The tool result is committed like any other tool result,
// so the model history keeps the assistant tool-call / tool-result order.
//
// Only responses this session accepts are resumed: replaying history never
// runs a tool again, and a request that already has a tool result is left alone.
func (s *Session) resumeHumanTool(response *EventHumanResponse) error {
	request, exists := s.humanRequests[response.ID]
	if !exists {
		return fmt.Errorf(
			"human response references unknown request %q",
			response.ID,
		)
	}
	if _, resolved := s.toolResults[request.ToolCall.ID]; resolved {
		return nil
	}

	call := request.ToolCall
	toolRuns, err := s.invokeToolsParallel(
		s.ctx,
		[]*types.MessageToolCall{call},
		map[string]*EventHumanRequest{call.ID: request},
		map[string]*EventHumanResponse{call.ID: response},
	)
	if err != nil {
		return err
	}
	toolEvents, err := s.toolRunEvents(toolRuns, true)
	if err != nil {
		return err
	}
	for _, event := range toolEvents {
		if err := s.addEvent(event, false); err != nil {
			return err
		}
	}
	return nil
}

// commitInput turns one accepted input into events before starting its first
// model turn. Human responses resume their original tool call immediately.
func (s *Session) commitInput(input *SessionInput) error {
	events, err := input.events()
	if err != nil {
		return err
	}

	for _, event := range events {
		if err := s.addEvent(event, false); err != nil {
			return err
		}
		if event.HumanResponse != nil {
			if err := s.resumeHumanTool(event.HumanResponse); err != nil {
				return err
			}
		}
	}

	return nil
}

func (s *Session) complete(turnIndex int) (*types.Completion, error) {
	turn := &SessionTurn{
		Index:       turnIndex,
		Model:       s.options.Model,
		Instruction: s.options.Instruction,
		History:     append([]*types.Message(nil), s.history...),
		Tools:       append([]*Tool(nil), s.options.Tools...),
		state:       s.stateSnapshot(),
	}

	handler := func(ctx context.Context, turn *SessionTurn) (*types.Completion, error) {
		completionOptions := []types.CompletionOption{
			types.CompletionWithInstructions(turn.Instruction),
			types.CompletionWithReasoningLevel(s.options.ReasonLevel),
		}
		if len(turn.Tools) > 0 {
			tools := make([]*types.Tool, 0, len(turn.Tools))
			seen := make(map[string]struct{}, len(turn.Tools))
			for index, tool := range turn.Tools {
				if tool == nil {
					return nil, fmt.Errorf("turn tool %d is nil", index)
				}
				if _, exists := seen[tool.Name]; exists {
					return nil, fmt.Errorf("turn tool %q is duplicated", tool.Name)
				}
				seen[tool.Name] = struct{}{}
				registered, err := s.getTool(tool.Name)
				if err != nil {
					return nil, fmt.Errorf("turn tool %d: %w", index, err)
				}
				if registered != tool {
					return nil, fmt.Errorf(
						"turn tool %q is not the registered tool instance",
						tool.Name,
					)
				}
				tools = append(tools, types.NewFunctionTool(
					tool.Name,
					tool.Description,
					tool.Parameters,
				))
			}
			completionOptions = append(
				completionOptions,
				types.CompletionWithTools(tools),
			)
		}
		if s.options.Stream {
			completionOptions = append(
				completionOptions,
				types.CompletionWithStreamingFunc(
					func(streamCtx context.Context, completion *types.Completion) error {
						if completion == nil || completion.Message == nil {
							return nil
						}
						return s.commitEvent(NewEventMessage(completion.Message))
					},
				),
			)
		}
		return s.models.Completion(
			ctx,
			turn.Model,
			turn.History,
			completionOptions...,
		)
	}
	for index := len(s.options.LLMInterceptors) - 1; index >= 0; index-- {
		handler = s.options.LLMInterceptors[index](handler)
	}

	return handler(s.ctx, turn)
}

func (s *Session) finalize(runErr error) (*SessionResult, error) {
	result := s.result
	result.Events = append([]*Event(nil), s.result.Events...)
	result.State = cloneStringMap(s.state)

	switch {
	case runErr == nil:
		result.StopReason = StopDone
	case errors.Is(runErr, ErrMaxTurnExceeded):
		result.StopReason = StopMaxTurn
		result.Cause = runErr
	case s.ctx != nil && s.ctx.Err() != nil:
		result.StopReason = StopCanceled
		result.Cause = contextError(s.ctx)
		runErr = result.Cause
	case errors.Is(runErr, context.Canceled),
		errors.Is(runErr, context.DeadlineExceeded):
		result.StopReason = StopCanceled
		result.Cause = runErr
	default:
		result.StopReason = StopError
		result.Cause = runErr
	}
	return result.Clone(), runErr
}

func contextError(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("context is nil")
	}
	select {
	case <-ctx.Done():
		if cause := context.Cause(ctx); cause != nil {
			return cause
		}
		return ctx.Err()
	default:
		return nil
	}
}

func cloneStringMap(source map[string]string) map[string]string {
	if source == nil {
		return map[string]string{}
	}
	return maps.Clone(source)
}

func validateState(state map[string]string) error {
	for key := range state {
		if key == "" {
			return fmt.Errorf("state key is empty")
		}
	}
	return nil
}

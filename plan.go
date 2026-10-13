package llmx

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/xucx/llmx/types"
)

// Effect is something that must happen now for the conversation to move on.
// Exactly one field is set. Effects have deterministic IDs: two drivers
// planning from the same events produce the same effects, and their outcomes
// collide in the Session instead of duplicating.
type Effect struct {
	ID        string
	CallModel *EffectCallModel
	CallTool  *EffectCallTool
	Emit      *EffectEmit
}

// EffectCallModel asks a model for the next reply.
type EffectCallModel struct {
	BaseSeq     int64
	Model       string
	Instruction string
	ReasonLevel int
	Messages    []*types.Message
	Tools       []ToolSpec
}

// EffectCallTool runs one tool call. Ask is set while the tool waits for an
// answer, Answer once it has one and the tool must run again with it.
type EffectCallTool struct {
	Call   ToolCall
	Ask    *ToolAsk
	Answer *ToolAnswer
	// Since is the append time of the reply that made the call.
	Since int64
}

// EffectEmit appends bookkeeping events: cancellations, stops.
type EffectEmit struct {
	Events []*Event
}

// State keys the conversation may set (through Input.State or Patch) to
// override the agent definition for the model calls that follow.
const (
	// StateModel overrides Agent.Model.
	StateModel = "llmx.model"
	// StateReasonLevel overrides Agent.ReasonLevel (an integer, 0-10).
	StateReasonLevel = "llmx.reasonLevel"
)

// ErrToolInterrupted answers every tool call an interruption cut short or
// never started.
var ErrToolInterrupted = errors.New(
	"tool call interrupted: it may not have run, or may have partly taken effect",
)

// Plan derives from the state the effects that must happen now:
//
//  1. an interrupt waits while a round is active: cancel the pending tool
//     calls and stop the round, so Fold admits it next;
//  2. pending tool calls run, each as its own effect: with its answer when
//     answered, as a question for whoever answers while asked;
//  3. the model is called while the newest admitted event needs an answer
//     and MaxTurn is not used up;
//  4. otherwise nothing: the conversation is idle or waits for events.
func Plan(s *State, a *Agent) []Effect {
	if s == nil || a == nil {
		return nil
	}
	if interrupt := s.FirstInput(InputInterrupt); interrupt != nil && s.Active() {
		var events []*Event
		for _, pending := range s.Pending {
			events = append(events, NewToolResultEvent(&ToolResult{
				CallID: pending.Call.ID,
				Name:   pending.Call.Name,
				Error:  ErrToolInterrupted.Error(),
			}))
		}
		events = append(events, NewStopEvent("stop:interrupt:"+interrupt.ID, StopInterrupted, ""))
		return []Effect{{ID: "emit:interrupt:" + interrupt.ID, Emit: &EffectEmit{Events: events}}}
	}
	if len(s.Pending) > 0 {
		effects := make([]Effect, 0, len(s.Pending))
		for _, pending := range s.Pending {
			effects = append(effects, Effect{
				ID: ToolResultID(pending.Call.ID),
				CallTool: &EffectCallTool{
					Call:   pending.Call,
					Ask:    pending.Ask,
					Answer: pending.Answer,
					Since:  pending.Since,
				},
			})
		}
		return effects
	}
	if s.NeedsAnswer() && a.continues(s) {
		if a.MaxTurn > 0 && s.Turns >= a.MaxTurn {
			id := "stop:maxturn:" + strconv.FormatInt(s.Seq, 10)
			return []Effect{{ID: "emit:" + id, Emit: &EffectEmit{Events: []*Event{
				NewStopEvent(id, StopMaxTurn, fmt.Sprintf("max turn %d exceeded", a.MaxTurn)),
			}}}}
		}
		model, level := a.Model, a.ReasonLevel
		if override := s.KV[StateModel]; override != "" {
			model = override
		}
		if override, err := strconv.Atoi(s.KV[StateReasonLevel]); err == nil && override >= 0 {
			level = override
		}
		return []Effect{{ID: ReplyID(s.Seq), CallModel: &EffectCallModel{
			BaseSeq:     s.Seq,
			Model:       model,
			Instruction: a.Instruction,
			ReasonLevel: level,
			Messages:    BuildMessages(s),
			Tools:       mergeTools(a.Tools, s.Tools),
		}}}
	}
	return nil
}

// mergeTools returns the agent tools followed by the input tools, the agent
// winning on a name clash.
func mergeTools(static, dynamic []ToolSpec) []ToolSpec {
	merged := append([]ToolSpec(nil), static...)
	names := map[string]struct{}{}
	for _, tool := range static {
		names[tool.Name] = struct{}{}
	}
	for _, tool := range dynamic {
		if _, dup := names[tool.Name]; dup {
			continue
		}
		names[tool.Name] = struct{}{}
		merged = append(merged, tool)
	}
	return merged
}

package llmx

import "fmt"

// Agent is the definition of an agent: pure data, shared by every driver of
// every session it serves. What it runs on (models, tools) is given to the
// executors, not to the agent.
type Agent struct {
	Name        string
	Description string

	// Model is the model name a ModelExecutor resolves.
	Model string
	// Instruction is the system instruction sent with every model call.
	Instruction string
	// ReasonLevel is the reasoning effort, 0-10, passed to the model.
	ReasonLevel int
	// Tools declares the tools always visible to the model, whoever runs
	// them. Tools declared by inputs are added to these.
	Tools []ToolSpec
	// MaxTurn bounds the model calls one admitted input may trigger; zero
	// means no bound.
	MaxTurn int
	// Continue decides whether the model is called again once no tool call
	// is pending. Nil means ReactContinue.
	Continue func(*State) bool
}

func (a *Agent) Validate() error {
	if a == nil {
		return fmt.Errorf("agent is nil")
	}
	if a.Model == "" {
		return fmt.Errorf("agent model is empty")
	}
	if a.MaxTurn < 0 {
		return fmt.Errorf("agent max turn must not be negative")
	}
	if a.ReasonLevel < 0 {
		return fmt.Errorf("agent reason level must not be negative")
	}
	names := map[string]struct{}{}
	for index, tool := range a.Tools {
		if tool.Name == "" {
			return fmt.Errorf("agent tool %d name is empty", index)
		}
		if _, dup := names[tool.Name]; dup {
			return fmt.Errorf("duplicate agent tool %q", tool.Name)
		}
		names[tool.Name] = struct{}{}
	}
	return nil
}

func (a *Agent) continues(s *State) bool {
	if a.Continue != nil {
		return a.Continue(s)
	}
	return ReactContinue(s)
}

// ReactContinue is the plain reason-act cycle: the model is called while the
// newest admitted event gives it something to answer -- an input with
// content, or a tool result not marked final.
func ReactContinue(s *State) bool {
	return s.NeedsAnswer()
}

package llmx

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/google/jsonschema-go/jsonschema"
)

// ToolContext is what a tool runs with.
type ToolContext interface {
	context.Context
	// Call is the call being answered.
	Call() ToolCall
	// Answer is set when the tool runs again after it asked; nil otherwise.
	Answer() *ToolAnswer
	// State is the key-value state of the conversation.
	State() map[string]string
	// Events is the conversation so far.
	Events() []*Event
}

// ToolInvoker answers a tool call. It returns a plain value, an *Output, an
// *Ask, or an error.
type ToolInvoker func(ToolContext, string) (any, error)

// Tool is a tool with its implementation. Spec declares it to the model.
type Tool struct {
	Name        string
	Description string
	Parameters  map[string]any
	Invoker     ToolInvoker
}

func (t *Tool) Spec() ToolSpec {
	return ToolSpec{Name: t.Name, Description: t.Description, Parameters: cloneAnyMap(t.Parameters)}
}

// Specs declares tools.
func Specs(tools ...*Tool) []ToolSpec {
	specs := make([]ToolSpec, 0, len(tools))
	for _, tool := range tools {
		specs = append(specs, tool.Spec())
	}
	return specs
}

// Output is what a tool returns when a plain value is not enough.
type Output struct {
	// Value is shown to the model: a string as is, anything else as JSON.
	Value any
	// Instruction adds tagged instructions to the result the model sees.
	Instruction map[string]string
	// Final marks a result the model need not answer.
	Final bool
	// Patch records a key-value state change alongside the result.
	Patch map[string]string
	// Images are pictures returned alongside Value (data URLs or https URLs),
	// shown to the model right after the result.
	Images []string
}

// Ask is what a tool returns to ask a question before answering: the call
// stays pending until a ToolAnswer arrives, then the tool runs again with
// Answer set in its context.
type Ask struct {
	Args map[string]any
}

// NewTool builds a tool whose argument schema is derived from T.
func NewTool[T any](name string, desc string, invoker func(ToolContext, *T) (any, error)) (*Tool, error) {
	if name == "" {
		return nil, fmt.Errorf("tool name is empty")
	}
	if invoker == nil {
		return nil, fmt.Errorf("tool %q invoker is nil", name)
	}
	schema, err := jsonschema.For[T](nil)
	if err != nil {
		return nil, fmt.Errorf("build tool %q parameter schema: %w", name, err)
	}
	resolved, err := schema.Resolve(nil)
	if err != nil {
		return nil, fmt.Errorf("resolve tool %q parameter schema: %w", name, err)
	}
	schemaBytes, err := resolved.Schema().MarshalJSON()
	if err != nil {
		return nil, fmt.Errorf("marshal tool %q parameter schema: %w", name, err)
	}
	parameters := map[string]any{}
	if err := json.Unmarshal(schemaBytes, &parameters); err != nil {
		return nil, fmt.Errorf("decode tool %q parameter schema: %w", name, err)
	}
	return &Tool{
		Name:        name,
		Description: desc,
		Parameters:  parameters,
		Invoker: func(ctx ToolContext, args string) (any, error) {
			param, err := unmarshalToolArgs[T](args)
			if err != nil {
				return nil, fmt.Errorf("invalid tool arguments: %w", err)
			}
			return invoker(ctx, param)
		},
	}, nil
}

func MustNewTool[T any](name string, desc string, invoker func(ToolContext, *T) (any, error)) *Tool {
	tool, err := NewTool(name, desc, invoker)
	if err != nil {
		panic(err)
	}
	return tool
}

// NewRawTool builds a tool from a hand-written JSON schema.
func NewRawTool(name, desc string, parameters map[string]any, invoker ToolInvoker) (*Tool, error) {
	if name == "" {
		return nil, fmt.Errorf("tool name is empty")
	}
	if invoker == nil {
		return nil, fmt.Errorf("tool %q invoker is nil", name)
	}
	return &Tool{Name: name, Description: desc, Parameters: cloneAnyMap(parameters), Invoker: invoker}, nil
}

func unmarshalToolArgs[T any](args string) (*T, error) {
	param := new(T)
	if strings.TrimSpace(args) == "" {
		args = "{}"
	}
	err := json.Unmarshal([]byte(args), param)
	if err == nil {
		return param, nil
	}
	// vllm bug, missing end "}" with nesting json
	var retryArgs strings.Builder
	retryArgs.WriteString(args)
	for i := 0; i < 3; i++ {
		retryArgs.WriteString("}")
		if retryErr := json.Unmarshal([]byte(retryArgs.String()), param); retryErr == nil {
			return param, nil
		}
	}
	return nil, err
}

// ToolExecutor performs the tool calls of the tools it holds. A call that
// asked and has no answer yet is left to a HumanExecutor.
func ToolExecutor(tools ...*Tool) Executor {
	byName := make(map[string]*Tool, len(tools))
	for _, tool := range tools {
		if tool != nil && tool.Name != "" {
			byName[tool.Name] = tool
		}
	}
	return &toolExecutor{tools: byName}
}

type toolExecutor struct {
	tools map[string]*Tool
}

func (x *toolExecutor) Can(e *Effect) bool {
	if e == nil || e.CallTool == nil {
		return false
	}
	if e.CallTool.Ask != nil && e.CallTool.Answer == nil {
		return false
	}
	_, ok := x.tools[e.CallTool.Call.Name]
	return ok
}

func (x *toolExecutor) Exec(ctx context.Context, s *State, e *Effect) (events []*Event, err error) {
	call := e.CallTool.Call
	tool := x.tools[call.Name]
	tctx := &toolContext{Context: ctx, call: call, answer: e.CallTool.Answer, state: s}
	var value any
	func() {
		// A panic anywhere in the tool answers the call instead of taking
		// the process down.
		defer func() {
			if recovered := recover(); recovered != nil {
				err = fmt.Errorf("tool %s panic: %v", call.Name, recovered)
			}
		}()
		value, err = tool.Invoker(tctx, call.Arguments)
	}()
	return ToolOutcome(call, value, err, e.CallTool.Answer != nil)
}

// ToolOutcome turns what a tool returned into the event recording it: a
// ToolAsk when it asked, a ToolResult otherwise. A tool that errs answers
// with the error, whatever it asked. A tool that asks again after it was
// answered is a bug.
func ToolOutcome(call ToolCall, value any, err error, answered bool) ([]*Event, error) {
	if err == nil {
		if ask, ok := value.(*Ask); ok && ask != nil {
			if answered {
				return nil, fmt.Errorf("tool %q asked again after it was answered", call.Name)
			}
			return []*Event{NewToolAskEvent(&ToolAsk{CallID: call.ID, Args: ask.Args})}, nil
		}
	}
	result := &ToolResult{CallID: call.ID, Name: call.Name}
	var events []*Event
	if err != nil {
		result.Error = err.Error()
	} else {
		output, ok := value.(*Output)
		if !ok {
			output = &Output{Value: value}
		}
		if len(output.Patch) > 0 {
			events = append(events, NewPatchEvent(output.Patch, false))
		}
		result.Instruction = output.Instruction
		result.Final = output.Final
		result.Images = output.Images
		if output.Value != nil {
			if text, ok := output.Value.(string); ok {
				result.Value = text
			} else {
				text, err := sonic.MarshalString(output.Value)
				if err != nil {
					return nil, err
				}
				result.Value = text
			}
		}
	}
	return append(events, NewToolResultEvent(result)), nil
}

type toolContext struct {
	context.Context
	call   ToolCall
	answer *ToolAnswer
	state  *State
}

func (c *toolContext) Call() ToolCall           { return c.call }
func (c *toolContext) Answer() *ToolAnswer      { return c.answer }
func (c *toolContext) State() map[string]string { return c.state.KV }
func (c *toolContext) Events() []*Event         { return c.state.Events }

// HumanExecutor answers the questions tools ask. answer is called with the
// call and its question and returns the answer; it may block on a human.
func HumanExecutor(answer func(ctx context.Context, call ToolCall, ask *ToolAsk) (map[string]any, error)) Executor {
	return ExecutorFunc{
		CanFunc: func(e *Effect) bool {
			return e != nil && e.CallTool != nil && e.CallTool.Ask != nil && e.CallTool.Answer == nil
		},
		ExecFunc: func(ctx context.Context, _ *State, e *Effect) ([]*Event, error) {
			result, err := answer(ctx, e.CallTool.Call, e.CallTool.Ask)
			if err != nil {
				return nil, err
			}
			return []*Event{NewToolAnswerEvent(&ToolAnswer{CallID: e.CallTool.Call.ID, Result: result})}, nil
		},
	}
}

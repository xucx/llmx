package llmx

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/google/jsonschema-go/jsonschema"
	"github.com/google/uuid"
	"github.com/xucx/llmx/types"
)

type ToolInvoker func(ToolContext, string) (any, error)

type Tool struct {
	Name        string
	Description string
	Parameters  map[string]any
	Invoker     ToolInvoker
}

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

	invokerWrapp := func(ctx ToolContext, args string) (any, error) {
		param, err := unmarshalToolArgs[T](args)
		if err != nil {
			return nil, fmt.Errorf("invalid tool arguments: %w", err)
		}

		return invoker(ctx, param)
	}

	return &Tool{
		Name:        name,
		Description: desc,
		Parameters:  parameters,
		Invoker:     invokerWrapp,
	}, nil
}

func MustNewTool[T any](name string, desc string, invoker func(ToolContext, *T) (any, error)) *Tool {
	tool, err := NewTool(name, desc, invoker)
	if err != nil {
		panic(err)
	}
	return tool
}

func unmarshalToolArgs[T any](args string) (*T, error) {
	param := new(T)
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

// ToolResult is what a tool returns, by value or by pointer, when a plain
// value is not enough. Exactly one of Result and AskHuman must be set. Any
// other value a tool returns is the Value of its Result. A tool that returns an
// error alongside AskHuman answers with the error: the request is dropped.
type ToolResult struct {
	// Result answers the tool call.
	Result *ToolOutput
	// AskHuman answers nothing yet: a human request is appended instead, and
	// the tool runs again once a human answers it.
	AskHuman *ToolAskHuman
}

// ToolOutput is the answer of a tool call.
type ToolOutput struct {
	// Value is shown to the model: a string as is, anything else as JSON.
	Value any
	// Instruction adds tagged instructions to the result the model sees.
	Instruction map[string]string
	// Final marks a result the model need not answer: alone, it starts no turn.
	Final bool
}

// ToolAskHuman is the human request a tool raises instead of answering.
type ToolAskHuman struct {
	// Arguments tell the human what is asked.
	Arguments map[string]any
}

// asToolResult returns what a tool returned as a ToolResult with exactly one
// of Result and AskHuman set.
func asToolResult(value any) (*ToolResult, error) {
	var result *ToolResult
	switch value := value.(type) {
	case ToolResult:
		result = &value
	case *ToolResult:
		result = value
	default:
		return &ToolResult{Result: &ToolOutput{Value: value}}, nil
	}
	if result == nil {
		return nil, fmt.Errorf("nil *ToolResult")
	}
	if (result.Result == nil) == (result.AskHuman == nil) {
		return nil, fmt.Errorf("ToolResult must set exactly one of Result and AskHuman")
	}
	return result, nil
}

// toolRun is one finished tool call.
type toolRun struct {
	call   *types.MessageToolCall
	result any
	err    error
	// resumed marks a call resuming an answered human request, where asking
	// for a human again is a bug.
	resumed bool
}

func createToolResultMessage(
	call *types.MessageToolCall,
	result any,
	resultErr error,
	instructions map[string]string,
	final bool,
) (*types.MessageToolResult, error) {
	resultString := ""
	if result != nil {
		if value, ok := result.(string); ok {
			resultString = value
		} else {
			value, err := sonic.MarshalString(result)
			if err != nil {
				return nil, err
			}
			resultString = value
		}
	}

	errorString := ""
	if resultErr != nil {
		errorString = resultErr.Error()
	}

	var instruction map[string][]string
	if len(instructions) > 0 {
		instruction = make(map[string][]string, len(instructions))
		for key, value := range instructions {
			instruction[key] = []string{value}
		}
	}

	return &types.MessageToolResult{
		ID:          call.ID,
		Name:        call.Name,
		Result:      resultString,
		Error:       errorString,
		Instruction: instruction,
		Final:       final,
	}, nil
}

// errToolInterrupted answers every tool call an interruption cut short or
// never started.
var errToolInterrupted = errors.New(
	"tool call interrupted: it may not have run, or may have partly taken effect",
)

// runTools runs tool calls concurrently, each with its own context, and
// appends what they produced through ctx: their human requests, then one tool
// message holding every result in call order.
//
// Once ctx is canceled no tool starts, and every call not finished yet gets
// errToolInterrupted as its result, so the stream never keeps a tool call
// without an answer. The cancellation is returned once those results are
// appended.
func (a *Agent) runTools(ctx TurnContext, calls []*toolContext) error {
	type finished struct {
		index int
		run   *toolRun
	}
	runs := make([]*toolRun, len(calls))
	canceled := contextError(ctx)
	if canceled == nil {
		finishedCh := make(chan finished, len(calls))
		for index, call := range calls {
			go func() {
				run := &toolRun{call: call.requested, resumed: call.response != nil}
				// A panic anywhere in the handler chain answers the call
				// instead of taking the process down.
				defer func() {
					if recovered := recover(); recovered != nil {
						run.result = nil
						run.err = fmt.Errorf("tool %s panic: %v", call.requested.Name, recovered)
					}
					finishedCh <- finished{index, run}
				}()
				run.result, run.err = a.toolCallHandler(call)
			}()
		}
		for pending := len(calls); pending > 0 && canceled == nil; pending-- {
			select {
			case <-ctx.Done():
				canceled = contextError(ctx)
			case done := <-finishedCh:
				runs[done.index] = done.run
			}
		}
		// A tool that finished while the cancellation was noticed keeps its
		// result: it ran, so the model must see what it did. One that only
		// reported the cancellation was cut short like the others.
		for drained := false; !drained; {
			select {
			case done := <-finishedCh:
				if errors.Is(done.run.err, context.Canceled) ||
					errors.Is(done.run.err, context.DeadlineExceeded) {
					done.run.result, done.run.err = nil, errToolInterrupted
				}
				runs[done.index] = done.run
			default:
				drained = true
			}
		}
	}
	for index, run := range runs {
		if run == nil {
			runs[index] = &toolRun{call: calls[index].requested, err: errToolInterrupted}
		}
	}

	events, err := toolRunEvents(runs)
	if err != nil {
		return err
	}
	if err := ctx.Append(events...); err != nil {
		return err
	}
	return canceled
}

// callTool is the innermost tool call handler: it invokes the registered tool.
func (a *Agent) callTool(ctx ToolContext) (any, error) {
	call := ctx.ToolCall()
	tool, err := a.getTool(call.Name)
	if err != nil {
		return nil, err
	}
	return tool.Invoker(ctx, call.Arguments)
}

func (a *Agent) getTool(name string) (*Tool, error) {
	for _, tool := range a.options.Tools {
		if tool.Name == name {
			return tool, nil
		}
	}
	return nil, fmt.Errorf("tool %s not found", name)
}

// toolRunEvents turns one batch of tool runs into the events it appends: human
// requests first, then one tool message holding every tool result of the batch
// in call order.
func toolRunEvents(runs []*toolRun) ([]*Event, error) {
	events := make([]*Event, 0, len(runs)+1)
	message := types.NewMessage(types.MessageRoleTool)
	for _, run := range runs {
		if run == nil || run.call == nil {
			return nil, fmt.Errorf("invalid tool run")
		}
		result, err := asToolResult(run.result)
		if err != nil {
			return nil, fmt.Errorf("tool %q returned %w", run.call.Name, err)
		}
		// A failed call answers with its error, whatever it asked alongside.
		if run.err != nil && result.AskHuman != nil {
			result = &ToolResult{Result: &ToolOutput{}}
		}
		if run.resumed && result.AskHuman != nil {
			return nil, fmt.Errorf("resumed human tool %q requested human input again", run.call.Name)
		}
		switch {
		case result.AskHuman != nil:
			events = append(events, NewEventHumanRequest(&EventHumanRequest{
				ID:        uuid.NewString(),
				Arguments: cloneAnyMap(result.AskHuman.Arguments),
				ToolCall:  cloneToolCall(run.call),
			}))
		default:
			toolResult, err := createToolResultMessage(
				run.call,
				result.Result.Value,
				run.err,
				result.Result.Instruction,
				result.Result.Final,
			)
			if err != nil {
				return nil, err
			}
			message.Parts = append(message.Parts, &types.MessagePart{ToolResult: toolResult})
		}
	}

	if len(message.Parts) > 0 {
		events = append(events, NewEventMessage(message))
	}
	return events, nil
}

func cloneToolCall(call *types.MessageToolCall) *types.MessageToolCall {
	if call == nil {
		return nil
	}
	clone := *call
	return &clone
}

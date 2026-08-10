package llmx

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/bytedance/sonic"
	"github.com/google/jsonschema-go/jsonschema"
	"github.com/google/uuid"
	"github.com/xucx/llmx/types"
)

type ToolContext interface {
	context.Context
	ToolCallID() string
	State() map[string]string
	SetState(patch map[string]string) error
	// HumanRequest and HumanResponse return immutable session values. Call Clone
	// before modifying nested fields.
	HumanRequest() *EventHumanRequest
	HumanResponse() *EventHumanResponse
}

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

type ToolResultNotify struct{}
type ToolResultHuman struct {
	Arguments map[string]any
}
type ToolResultWithInstruction struct {
	Result      any
	Instruction map[string]string
}

type ToolRun struct {
	Call         *types.MessageToolCall
	Result       any
	Error        error
	Duration     time.Duration
	StatePatches []map[string]string
}

func createToolResultMessage(
	call *types.MessageToolCall,
	result any,
	resultErr error,
	instructions map[string]string,
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
	}, nil
}

type sessionToolContext struct {
	context.Context

	callID   string
	mu       sync.Mutex
	state    map[string]string
	patches  []map[string]string
	request  *EventHumanRequest
	response *EventHumanResponse
}

func (ctx *sessionToolContext) ToolCallID() string {
	return ctx.callID
}

func (ctx *sessionToolContext) State() map[string]string {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	return cloneStringMap(ctx.state)
}

func (ctx *sessionToolContext) SetState(patch map[string]string) error {
	if err := contextError(ctx.Context); err != nil {
		return err
	}
	if len(patch) == 0 {
		return nil
	}
	if err := validateState(patch); err != nil {
		return err
	}
	copyPatch := cloneStringMap(patch)
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	for key, value := range copyPatch {
		ctx.state[key] = value
	}
	ctx.patches = append(ctx.patches, copyPatch)
	return nil
}

func (ctx *sessionToolContext) HumanRequest() *EventHumanRequest {
	return ctx.request
}

func (ctx *sessionToolContext) HumanResponse() *EventHumanResponse {
	return ctx.response
}

func (ctx *sessionToolContext) statePatches() []map[string]string {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	patches := make([]map[string]string, len(ctx.patches))
	for index, patch := range ctx.patches {
		patches[index] = cloneStringMap(patch)
	}
	return patches
}

type indexedToolRun struct {
	index int
	run   *ToolRun
}

func (s *Session) invokeToolsParallel(
	ctx context.Context,
	calls []*types.MessageToolCall,
	requests map[string]*EventHumanRequest,
	responses map[string]*EventHumanResponse,
) ([]*ToolRun, error) {
	results := make([]*ToolRun, len(calls))
	resultCh := make(chan indexedToolRun, len(calls))
	baseState := s.stateSnapshot()

	for index, call := range calls {
		go func(index int, call *types.MessageToolCall) {
			run := &ToolRun{Call: call}
			if call == nil {
				run.Error = fmt.Errorf("tool call is nil")
				resultCh <- indexedToolRun{index: index, run: run}
				return
			}

			toolCtx := &sessionToolContext{
				Context:  ctx,
				callID:   call.ID,
				state:    cloneStringMap(baseState),
				request:  requestForCall(requests, call.ID),
				response: responseForCall(requests, responses, call.ID),
			}
			handler := func(
				handlerCtx ToolContext,
				call *types.MessageToolCall,
			) (result any, err error) {
				defer func() {
					if recovered := recover(); recovered != nil {
						err = fmt.Errorf("tool %s panic: %v", call.Name, recovered)
					}
				}()
				tool, err := s.getTool(call.Name)
				if err != nil {
					return nil, err
				}
				if tool.Invoker == nil {
					return &ToolResultNotify{}, nil
				}
				return tool.Invoker(handlerCtx, call.Arguments)
			}
			for hookIndex := len(s.options.ToolCallInterceptors) - 1; hookIndex >= 0; hookIndex-- {
				handler = s.options.ToolCallInterceptors[hookIndex](handler)
			}

			start := time.Now()
			run.Result, run.Error = handler(toolCtx, call)
			run.Duration = time.Since(start)
			run.StatePatches = toolCtx.statePatches()
			resultCh <- indexedToolRun{index: index, run: run}
		}(index, call)
	}

	for range calls {
		select {
		case <-ctx.Done():
			return nil, contextError(ctx)
		case result := <-resultCh:
			results[result.index] = result.run
		}
	}
	return results, nil
}

func (s *Session) getTool(name string) (*Tool, error) {
	for _, tool := range s.options.Tools {
		if tool.Name == name {
			return tool, nil
		}
	}
	return nil, fmt.Errorf("tool %s not found", name)
}

// toolRunEvents turns one batch of tool runs into events the session commits.
// It only converts: parallel tools never write the event log themselves, so the
// session goroutine stays the single writer and the event order stays the tool
// call order.
//
// The returned order is: state patches, human requests, and finally one tool
// message holding every tool result of the batch. resumed marks a batch that
// resumes a human request, where asking for human input again is a bug.
func (s *Session) toolRunEvents(
	toolRuns []*ToolRun,
	resumed bool,
) ([]*Event, error) {
	events := make([]*Event, 0, len(toolRuns)+1)
	for _, run := range toolRuns {
		if run == nil || run.Call == nil {
			return nil, fmt.Errorf("invalid tool run")
		}
		for _, patch := range run.StatePatches {
			if len(patch) == 0 {
				continue
			}
			events = append(events, NewEventState(patch, true))
		}
	}

	message := types.NewMessage(types.MessageRoleTool)
	for _, run := range toolRuns {
		switch result := run.Result.(type) {
		case *ToolResultNotify:
			if resumed {
				return nil, fmt.Errorf(
					"resumed human tool %q returned notify without a tool result",
					run.Call.Name,
				)
			}
			continue
		case *ToolResultHuman:
			if resumed {
				return nil, fmt.Errorf(
					"resumed human tool %q requested human input again",
					run.Call.Name,
				)
			}
			events = append(events, NewEventHumanRequest(&EventHumanRequest{
				ID:        uuid.NewString(),
				Arguments: cloneAnyMap(result.Arguments),
				ToolCall:  cloneToolCall(run.Call),
			}))
		case ToolResultWithInstruction:
			toolResult, err := createToolResultMessage(
				run.Call,
				result.Result,
				run.Error,
				result.Instruction,
			)
			if err != nil {
				return nil, err
			}
			message.Parts = append(
				message.Parts,
				&types.MessagePart{ToolResult: toolResult},
			)
		case *ToolResultWithInstruction:
			toolResult, err := createToolResultMessage(
				run.Call,
				result.Result,
				run.Error,
				result.Instruction,
			)
			if err != nil {
				return nil, err
			}
			message.Parts = append(
				message.Parts,
				&types.MessagePart{ToolResult: toolResult},
			)
		default:
			toolResult, err := createToolResultMessage(
				run.Call,
				run.Result,
				run.Error,
				nil,
			)
			if err != nil {
				return nil, err
			}
			message.Parts = append(
				message.Parts,
				&types.MessagePart{ToolResult: toolResult},
			)
		}
	}
	if len(message.Parts) > 0 {
		events = append(events, NewEventMessage(message))
	}
	return events, nil
}

func (s *Session) stateSnapshot() map[string]string {
	return cloneStringMap(s.state)
}

func cloneToolCall(call *types.MessageToolCall) *types.MessageToolCall {
	if call == nil {
		return nil
	}
	clone := *call
	return &clone
}

func requestForCall(
	requests map[string]*EventHumanRequest,
	callID string,
) *EventHumanRequest {
	if requests == nil {
		return nil
	}
	return requests[callID]
}

func responseForCall(
	requests map[string]*EventHumanRequest,
	responses map[string]*EventHumanResponse,
	callID string,
) *EventHumanResponse {
	if requests == nil || responses == nil {
		return nil
	}
	return responses[callID]
}

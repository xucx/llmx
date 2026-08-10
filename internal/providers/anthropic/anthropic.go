package anthropic

import (
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/xucx/llmx/types"
)

const (
	DefaultChatModel = string(anthropic.ModelClaudeSonnet4_0)
)

const (
	RoleSystem    = "system"
	RoleAssistant = "assistant"
	RoleUser      = "user"
)

var (
	_ types.Provider = (*AnthropicProvider)(nil)
)

type AnthropicProvider struct {
	types.ProviderNop
	client *anthropic.Client
}

func NewAnthropicProvider(opts ...types.ProviderOption) (types.Provider, error) {
	options := types.ApplyProviderOptions(nil, opts...)

	config := []option.RequestOption{
		option.WithAPIKey(options.Sk),
	}
	if options.Url != "" {
		config = append(config, option.WithBaseURL(options.Url))
	}
	if options.HTTPClient != nil {
		config = append(config, option.WithHTTPClient(options.HTTPClient))
	}

	client := anthropic.NewClient(config...)
	return &AnthropicProvider{
		client: &client,
	}, nil
}

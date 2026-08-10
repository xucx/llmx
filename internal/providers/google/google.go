package google

import (
	"context"

	"github.com/xucx/llmx/types"
	"google.golang.org/genai"
)

const (
	DefaultChatModel = "gemini-2.5-flash"
)

const (
	RoleSystem = "system"
	RoleModel  = "model"
	RoleUser   = "user"
	RoleTool   = "tool"
)

var (
	_ types.Provider = (*GoogleProvider)(nil)
)

type GoogleProvider struct {
	types.ProviderNop
	client *genai.Client
}

// use gemini
func NewGoogleProvider(opts ...types.ProviderOption) (types.Provider, error) {
	options := types.ApplyProviderOptions(nil, opts...)

	config := &genai.ClientConfig{
		APIKey:     options.Sk,
		HTTPClient: options.HTTPClient,
	}
	if options.Url != "" {
		config.HTTPOptions.BaseURL = options.Url
	}

	client, err := genai.NewClient(context.Background(), config)
	if err != nil {
		return nil, err
	}

	return &GoogleProvider{client: client}, nil
}

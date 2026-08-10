package openai

import (
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/xucx/llmx/internal/httpx"
	"github.com/xucx/llmx/types"
)

const (
	DefaultBaseUrl = "https://api.openai.com/v1"
)

var (
	_ types.Provider = (*OpenaiProvider)(nil)
)

type OpenaiProvider struct {
	types.ProviderNop
	options    *types.ProviderOptions
	httpConfig *httpx.Config
	client     openai.Client
}

func NewOpenaiProvider(opts ...types.ProviderOption) (types.Provider, error) {
	options := types.ApplyProviderOptions(nil, opts...)
	openaiOpts := []option.RequestOption{
		option.WithAPIKey(options.Sk),
	}
	if options.HTTPClient != nil {
		openaiOpts = append(openaiOpts, option.WithHTTPClient(options.HTTPClient))
	}

	if options.Url == "" {
		options.Url = DefaultBaseUrl
	} else {
		if !strings.HasPrefix(options.Url, "http://") && !strings.HasPrefix(options.Url, "https://") {
			options.Url = "https://" + options.Url
		}
	}

	openaiOpts = append(openaiOpts, option.WithBaseURL(options.Url))
	client := openai.NewClient(openaiOpts...)

	return &OpenaiProvider{
		options:    options,
		httpConfig: newHTTPConfig(options),
		client:     client,
	}, nil
}

func newHTTPConfig(options *types.ProviderOptions) *httpx.Config {
	httpOptions := []httpx.Option{
		httpx.WithBaseUrl(options.Url),
		httpx.WithHeader("Authorization", "Bearer "+options.Sk),
	}
	if options.HTTPClient != nil {
		httpOptions = append(httpOptions, httpx.WithClient(options.HTTPClient))
	}
	return httpx.NewConfig(httpOptions...)
}

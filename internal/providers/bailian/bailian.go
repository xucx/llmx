package bailian

import (
	"fmt"

	"github.com/xucx/llmx/internal/httpx"
	"github.com/xucx/llmx/types"
)

var (
	_ types.Provider = (*BailianProvider)(nil)
)

type BailianProvider struct {
	types.ProviderNop
	httpConfig *httpx.Config
}

func NewBailianProvider(opts ...types.ProviderOption) (types.Provider, error) {
	options := types.ApplyProviderOptions(nil, opts...)
	if options.Url == "" {
		return nil, fmt.Errorf("bailian provider URL is empty")
	}

	httpOptions := []httpx.Option{
		httpx.WithBaseUrl(options.Url),
		httpx.WithHeader("Authorization", "Bearer "+options.Sk),
	}
	if options.HTTPClient != nil {
		httpOptions = append(httpOptions, httpx.WithClient(options.HTTPClient))
	}
	return &BailianProvider{
		httpConfig: httpx.NewConfig(httpOptions...),
	}, nil

}

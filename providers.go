package llmx

import (
	"fmt"
	"sync"

	"github.com/xucx/llmx/internal/providers/anthropic"
	"github.com/xucx/llmx/internal/providers/bailian"
	"github.com/xucx/llmx/internal/providers/google"
	"github.com/xucx/llmx/internal/providers/openai"
	"github.com/xucx/llmx/internal/providers/proxy"
	"github.com/xucx/llmx/types"
)

const (
	ProviderProxy     = "proxy"
	ProviderOpenAI    = "openai"
	ProviderAnthropic = "anthropic"
	ProviderGoogle    = "google"
	ProviderBailian   = "bailian"
)

var (
	_allProviders = map[string]ProviderCreator{
		ProviderProxy:     proxy.NewProxyProvider,
		ProviderOpenAI:    openai.NewOpenaiProvider,
		ProviderAnthropic: anthropic.NewAnthropicProvider,
		ProviderGoogle:    google.NewGoogleProvider,
		ProviderBailian:   bailian.NewBailianProvider,
	}
	_allProvidersMu sync.RWMutex
)

type ProviderCreator func(opts ...types.ProviderOption) (types.Provider, error)

func RegisterProviderCreator(name string, creator ProviderCreator) error {
	if name == "" {
		return fmt.Errorf("provider name is empty")
	}
	if creator == nil {
		return fmt.Errorf("provider %q creator is nil", name)
	}
	_allProvidersMu.Lock()
	defer _allProvidersMu.Unlock()
	_allProviders[name] = creator
	return nil
}

func NewProvider(name string, opts ...types.ProviderOption) (types.Provider, error) {
	_allProvidersMu.RLock()
	creator, ok := _allProviders[name]
	_allProvidersMu.RUnlock()
	if ok {
		return creator(opts...)
	}

	return nil, fmt.Errorf("%w: provider %q", types.ErrUnsupported, name)
}

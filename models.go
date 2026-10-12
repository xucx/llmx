package llmx

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/xucx/llmx/types"
	"go.yaml.in/yaml/v3"
)

type ProviderConfig struct {
	Name          string                       `yaml:"name"`
	Provider      string                       `yaml:"provider"`
	Url           string                       `yaml:"url"`
	Insecure      bool                         `yaml:"insecure"`
	Sk            string                       `yaml:"sk"`
	ThinkArgs     *types.ProviderThinkArgs     `yaml:"thinkArgs"`
	EmbeddingArgs *types.ProviderEmbeddingArgs `yaml:"embeddingArgs"`
}

type ModelConfig struct {
	Name     string               `yaml:"name"`
	Model    string               `yaml:"model"`
	Provider string               `yaml:"provider"`
	MaxToken int64                `yaml:"maxToken"`
	Fallback *ModelConfigFallback `yaml:"fallback"`
}

type ModelConfigFallback struct {
	MaxSeconds int                           `yaml:"maxSeconds"`
	Providers  []ModelConfigFallbackProvider `yaml:"providers"`
}

type ModelConfigFallbackProvider struct {
	Provider string `yaml:"provider"`
	Model    string `yaml:"model"`
}

type ModelsConfig struct {
	Providers []ProviderConfig `yaml:"providers"`
	Models    []ModelConfig    `yaml:"models"`
}

type Model struct {
	Name     string
	Model    string
	MaxToken int64
	Provider types.Provider
	Fallback *ModelFallback
}

type ModelFallback struct {
	MaxSeconds int
	Providers  []*ModelFallbackProvider

	mu        sync.Mutex
	lastTime  *time.Time
	lastIndex int
}

type ModelFallbackProvider struct {
	ProviderName string
	Provider     types.Provider
	Model        string
}

type Models struct {
	mu        sync.RWMutex
	providers map[string]types.Provider
	models    map[string]*Model
}

func (m *Model) Embedding(ctx context.Context, input []string, options ...types.EmbeddingOption) (*types.Embedding, error) {
	if ctx == nil {
		return nil, fmt.Errorf("embedding context is nil")
	}
	return callModelWithfallback(ctx, m, func(provider types.Provider, model string) (*types.Embedding, error) {
		return provider.Embedding(ctx, model, input, options...)
	})
}

func (m *Model) Completion(ctx context.Context, messages []*types.Message, options ...types.CompletionOption) (*types.Completion, error) {
	if ctx == nil {
		return nil, fmt.Errorf("completion context is nil")
	}
	return callModelWithfallback(ctx, m, func(provider types.Provider, model string) (*types.Completion, error) {
		return provider.Completion(ctx, model, messages, options...)
	})
}

func (m *Model) Realtime(ctx context.Context, messages []*types.Message, options ...types.RealTimeOption) (types.RealTimeSession, error) {
	if ctx == nil {
		return nil, fmt.Errorf("realtime context is nil")
	}
	return callModelWithfallback(ctx, m, func(provider types.Provider, model string) (types.RealTimeSession, error) {
		return provider.Realtime(ctx, model, messages, options...)
	})
}

func (m *Model) Rerank(ctx context.Context, input *types.RerankInput, options ...types.RerankOption) (*types.RerankOutput, error) {
	if ctx == nil {
		return nil, fmt.Errorf("rerank context is nil")
	}
	return callModelWithfallback(ctx, m, func(provider types.Provider, model string) (*types.RerankOutput, error) {
		return provider.Rerank(ctx, model, input, options...)
	})
}

func callModelWithfallback[T any](ctx context.Context, m *Model, call func(types.Provider, string) (T, error)) (T, error) {
	var zero T
	if m == nil {
		return zero, fmt.Errorf("model is nil")
	}
	if m.Provider == nil {
		return zero, fmt.Errorf("model %q provider is nil", m.Name)
	}
	if m.Fallback == nil {
		return call(m.Provider, m.Model)
	}

	m.Fallback.mu.Lock()
	lastTime := m.Fallback.lastTime
	lastIndex := m.Fallback.lastIndex
	m.Fallback.mu.Unlock()

	fallbackActive := lastTime != nil &&
		m.Fallback.MaxSeconds > 0 &&
		time.Since(*lastTime) <= time.Second*time.Duration(m.Fallback.MaxSeconds)
	if lastTime != nil && !fallbackActive {
		m.Fallback.mu.Lock()
		if m.Fallback.lastTime == lastTime {
			m.Fallback.lastTime = nil
			m.Fallback.lastIndex = 0
		}
		m.Fallback.mu.Unlock()
		lastIndex = 0
	}

	callErrors := make([]error, 0, 1+len(m.Fallback.Providers))
	if !fallbackActive {
		result, err := call(m.Provider, m.Model)
		if err == nil {
			return result, nil
		}
		if ctx.Err() != nil {
			// A canceled call keeps what it received so far.
			return result, context.Cause(ctx)
		}
		if !fallbackEligible(err) {
			return zero, fmt.Errorf("primary model %q: %w", m.Model, err)
		}
		callErrors = append(callErrors, fmt.Errorf("primary model %q: %w", m.Model, err))
		now := time.Now()
		m.Fallback.mu.Lock()
		m.Fallback.lastTime = &now
		m.Fallback.lastIndex = 0
		m.Fallback.mu.Unlock()
		lastIndex = 0
	}

	for i := 0; i < len(m.Fallback.Providers); i++ {
		if ctx.Err() != nil {
			callErrors = append(callErrors, context.Cause(ctx))
			break
		}
		index := (lastIndex + i) % len(m.Fallback.Providers)
		fallback := m.Fallback.Providers[index]
		result, err := call(fallback.Provider, fallback.Model)
		if err == nil {
			m.Fallback.mu.Lock()
			m.Fallback.lastIndex = index
			m.Fallback.mu.Unlock()
			return result, nil
		}
		if ctx.Err() != nil {
			return result, context.Cause(ctx)
		}
		callErrors = append(callErrors, fmt.Errorf(
			"fallback provider %q model %q: %w",
			fallback.ProviderName,
			fallback.Model,
			err,
		))
	}

	if len(callErrors) == 0 {
		return zero, fmt.Errorf("model %q has no fallback providers", m.Model)
	}
	return zero, errors.Join(callErrors...)
}

func fallbackEligible(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, types.ErrInvalidMessage) || errors.Is(err, types.ErrUnsupported) {
		return false
	}
	if statusCode, ok := types.ErrorStatusCode(err); ok {
		switch statusCode {
		case 408, 409, 425, 429:
			return true
		default:
			return statusCode >= 500
		}
	}
	return true
}

func NewModels() *Models {
	return &Models{
		providers: map[string]types.Provider{},
		models:    map[string]*Model{},
	}
}

func NewModelsWithConfigFile(fn string) (*Models, error) {

	fd, err := os.ReadFile(fn)
	if err != nil {
		return nil, err
	}

	conf := struct {
		LLMX   ModelsConfig `yaml:"llmx"`
		LLM    ModelsConfig `yaml:"llm"`
		LLMAPI ModelsConfig `yaml:"llmapi"`
	}{}
	if err := yaml.Unmarshal(fd, &conf); err != nil {
		return nil, err
	}

	config := conf.LLMX
	if len(config.Providers) == 0 && len(config.Models) == 0 {
		config = conf.LLM
	}
	if len(config.Providers) == 0 && len(config.Models) == 0 {
		config = conf.LLMAPI
	}
	return NewModelsWithConfig(config)
}

func NewModelsWithConfig(config ModelsConfig) (*Models, error) {
	models := NewModels()
	for _, p := range config.Providers {
		if _, err := models.AddProvider(p); err != nil {
			_ = models.Close()
			return nil, fmt.Errorf("add provider %q: %w", p.Name, err)
		}
	}

	for _, m := range config.Models {
		if _, err := models.AddModel(m); err != nil {
			_ = models.Close()
			return nil, err
		}
	}

	return models, nil
}

func (m *Models) AddProvider(conf ProviderConfig) (types.Provider, error) {
	if conf.Name == "" {
		return nil, fmt.Errorf("provider name empty")
	}
	if conf.Provider == "" {
		return nil, fmt.Errorf("provider %q type empty", conf.Name)
	}
	m.mu.RLock()
	_, exists := m.providers[conf.Name]
	m.mu.RUnlock()
	if exists {
		return nil, fmt.Errorf("provider %q already exists", conf.Name)
	}

	sk := conf.Sk
	if sk == "" {
		sk = os.Getenv(strings.ToUpper(fmt.Sprintf("%s_SK", conf.Name)))
	}

	providerOpts := []types.ProviderOption{
		types.ProviderWithUrl(conf.Url),
		types.ProviderWithSk(sk),
		types.ProviderWithInsecure(conf.Insecure),
		types.ProviderWithThinkArgs(conf.ThinkArgs),
		types.ProviderWithEmbeddingArgs(conf.EmbeddingArgs),
	}

	provider, err := NewProvider(conf.Provider, providerOpts...)
	if err != nil {
		return nil, err
	}
	if err := m.RegisterProvider(conf.Name, provider); err != nil {
		if closer, ok := provider.(io.Closer); ok {
			_ = closer.Close()
		}
		return nil, err
	}

	return provider, nil
}

func (m *Models) RegisterProvider(name string, provider types.Provider) error {
	if m == nil {
		return fmt.Errorf("models is nil")
	}
	if name == "" {
		return fmt.Errorf("provider name empty")
	}
	if provider == nil {
		return fmt.Errorf("provider %q is nil", name)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.providers[name]; exists {
		return fmt.Errorf("provider %q already exists", name)
	}
	m.providers[name] = provider
	return nil
}

func (m *Models) AddModel(conf ModelConfig) (*Model, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if conf.Name == "" {
		return nil, fmt.Errorf("model name empty")
	}
	if conf.Model == "" {
		return nil, fmt.Errorf("model %q provider model empty", conf.Name)
	}
	if _, exists := m.models[conf.Name]; exists {
		return nil, fmt.Errorf("model %q already exists", conf.Name)
	}

	provider, ok := m.providers[conf.Provider]
	if !ok {
		return nil, fmt.Errorf("provider %s not found", conf.Provider)
	}

	model := &Model{
		Name:     conf.Name,
		Model:    conf.Model,
		MaxToken: conf.MaxToken,
		Provider: provider,
	}

	info := defaultModelInfo(conf.Model)
	if model.MaxToken <= 0 {
		model.MaxToken = info.MaxToken
	}

	if conf.Fallback != nil {
		model.Fallback = &ModelFallback{
			MaxSeconds: conf.Fallback.MaxSeconds,
		}
		for _, p := range conf.Fallback.Providers {
			pp, ok := m.providers[p.Provider]
			if !ok {
				return nil, fmt.Errorf("provider %s not found", p.Provider)
			}
			if p.Model == "" {
				return nil, fmt.Errorf("fallback provider %q model empty", p.Provider)
			}
			model.Fallback.Providers = append(model.Fallback.Providers, &ModelFallbackProvider{
				ProviderName: p.Provider,
				Provider:     pp,
				Model:        p.Model,
			})
		}
	}

	m.models[model.Name] = model

	return model, nil
}

func (m *Models) GetModel(name string) (*Model, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if model, ok := m.models[name]; ok {
		return model, nil
	}

	splitNames := strings.SplitN(name, "/", 2)
	if len(splitNames) == 2 {
		if provider, ok := m.providers[splitNames[0]]; ok {
			info := defaultModelInfo(name)
			model := &Model{
				Name:     name,
				Model:    splitNames[1],
				MaxToken: info.MaxToken,
				Provider: provider,
			}
			return model, nil
		}
	}

	return nil, fmt.Errorf("model %s not found", name)
}

func (m *Models) Close() error {
	if m == nil {
		return nil
	}
	m.mu.RLock()
	providers := make([]types.Provider, 0, len(m.providers))
	for _, provider := range m.providers {
		providers = append(providers, provider)
	}
	m.mu.RUnlock()

	var closeErrors []error
	for _, provider := range providers {
		closer, ok := provider.(io.Closer)
		if !ok {
			continue
		}
		if err := closer.Close(); err != nil {
			closeErrors = append(closeErrors, err)
		}
	}
	return errors.Join(closeErrors...)
}

func (ms *Models) Embedding(ctx context.Context, model string, input []string, options ...types.EmbeddingOption) (*types.Embedding, error) {
	m, err := ms.GetModel(model)
	if err != nil {
		return nil, err
	}
	return m.Embedding(ctx, input, options...)
}

func (ms *Models) Completion(ctx context.Context, model string, messages []*types.Message, options ...types.CompletionOption) (*types.Completion, error) {
	m, err := ms.GetModel(model)
	if err != nil {
		return nil, err
	}
	return m.Completion(ctx, messages, options...)
}

func (ms *Models) Realtime(ctx context.Context, model string, messages []*types.Message, options ...types.RealTimeOption) (types.RealTimeSession, error) {
	m, err := ms.GetModel(model)
	if err != nil {
		return nil, err
	}

	return m.Realtime(ctx, messages, options...)
}

func (ms *Models) Rerank(ctx context.Context, model string, input *types.RerankInput, options ...types.RerankOption) (*types.RerankOutput, error) {
	m, err := ms.GetModel(model)
	if err != nil {
		return nil, err
	}

	return m.Rerank(ctx, input, options...)
}

var DefaultModels *Models = NewModels()
var defaultModelsInitError error

func DefaultModelsInitError() error {
	return defaultModelsInitError
}

func GetModel(name string) (*Model, error) {
	return DefaultModels.GetModel(name)
}

func Embedding(ctx context.Context, model string, input []string, options ...types.EmbeddingOption) (*types.Embedding, error) {
	return DefaultModels.Embedding(ctx, model, input, options...)
}

func Completion(ctx context.Context, model string, messages []*types.Message, options ...types.CompletionOption) (*types.Completion, error) {
	return DefaultModels.Completion(ctx, model, messages, options...)
}

func Realtime(ctx context.Context, model string, messages []*types.Message, options ...types.RealTimeOption) (types.RealTimeSession, error) {
	return DefaultModels.Realtime(ctx, model, messages, options...)
}

func Rerank(ctx context.Context, model string, input *types.RerankInput, options ...types.RerankOption) (*types.RerankOutput, error) {
	return DefaultModels.Rerank(ctx, model, input, options...)
}

type ProviderEnv struct {
	Provider     string
	Url          string
	LegacyURL    string
	ApiKey       string
	LegacyAPIKey string
}

func init() {
	// init default models
	envProviders := map[string]ProviderEnv{
		"proxy": {
			Provider:     ProviderProxy,
			Url:          "LLMX_API_URL",
			LegacyURL:    "LLMAPI_API_URL",
			ApiKey:       "LLMX_API_KEY",
			LegacyAPIKey: "LLMAPI_API_KEY",
		},
		"openai": {
			Provider: ProviderOpenAI,
			ApiKey:   "OPENAI_API_KEY",
		},
		"anthropic": {
			Provider: ProviderAnthropic,
			ApiKey:   "ANTHROPIC_API_KEY",
		},
		"google": {
			Provider: ProviderGoogle,
			ApiKey:   "GOOGLE_API_KEY",
		},
		"gemini": {
			Provider: ProviderGoogle,
			ApiKey:   "GEMINI_API_KEY",
		},
		"deepseek": {
			Provider: ProviderOpenAI,
			Url:      "DEEPSEEK_API_URL",
			ApiKey:   "DEEPSEEK_API_KEY",
		},
	}

	for name, p := range envProviders {
		conf := ProviderConfig{
			Name:     name,
			Provider: p.Provider,
		}
		if p.ApiKey != "" {
			conf.Sk = os.Getenv(p.ApiKey)
			if conf.Sk == "" && p.LegacyAPIKey != "" {
				conf.Sk = os.Getenv(p.LegacyAPIKey)
			}
			if conf.Sk == "" {
				continue
			}
		}
		if p.Url != "" {
			conf.Url = os.Getenv(p.Url)
			if conf.Url == "" && p.LegacyURL != "" {
				conf.Url = os.Getenv(p.LegacyURL)
			}
			if conf.Url == "" {
				continue
			}
		}

		if _, err := DefaultModels.AddProvider(conf); err != nil {
			defaultModelsInitError = errors.Join(
				defaultModelsInitError,
				fmt.Errorf("initialize default provider %q: %w", name, err),
			)
		}
	}

}

package types

import (
	"context"
	"fmt"
	"maps"
	"net/http"
)

type Provider interface {
	Embedding(ctx context.Context, model string, input []string, options ...EmbeddingOption) (*Embedding, error)
	// Completion calls the model. When ctx is canceled during a streaming
	// call, it returns the completion received so far along with the
	// cancellation error; the completion may hold truncated tool calls.
	Completion(ctx context.Context, model string, messages []*Message, options ...CompletionOption) (*Completion, error)
	Realtime(ctx context.Context, model string, messages []*Message, options ...RealTimeOption) (RealTimeSession, error)
	Rerank(ctx context.Context, model string, input *RerankInput, options ...RerankOption) (*RerankOutput, error)
}

type ProviderOptions struct {
	Url           string                 `yaml:"url"`
	Sk            string                 `yaml:"sk"`
	Insecure      bool                   `yaml:"insecure"`
	HTTPClient    *http.Client           `yaml:"-"`
	ThinkArgs     *ProviderThinkArgs     `yaml:"thinkArgs"`
	EmbeddingArgs *ProviderEmbeddingArgs `yaml:"embeddingArgs"`
}

type ProviderThinkArgs struct {
	Enable  map[string]any `yaml:"enable"`
	Disable map[string]any `yaml:"disable"`
}

type ProviderEmbeddingArgs struct {
	DisableDim bool `yaml:"disableDim"`
}

type ProviderOption func(*ProviderOptions) *ProviderOptions

func ProviderWithOptions(options *ProviderOptions) ProviderOption {
	return func(opts *ProviderOptions) *ProviderOptions {
		return options.Clone()
	}
}

func ProviderWithUrl(url string) ProviderOption {
	return ProviderWithURL(url)
}

func ProviderWithURL(url string) ProviderOption {
	return func(opts *ProviderOptions) *ProviderOptions {
		opts.Url = url
		return opts
	}
}

func ProviderWithSk(sk string) ProviderOption {
	return ProviderWithAPIKey(sk)
}

func ProviderWithAPIKey(apiKey string) ProviderOption {
	return func(opts *ProviderOptions) *ProviderOptions {
		opts.Sk = apiKey
		return opts
	}
}

func ProviderWithInsecure(insecure bool) ProviderOption {
	return func(opts *ProviderOptions) *ProviderOptions {
		opts.Insecure = insecure
		return opts
	}
}

func ProviderWithHTTPClient(client *http.Client) ProviderOption {
	return func(opts *ProviderOptions) *ProviderOptions {
		opts.HTTPClient = client
		return opts
	}
}

func ProviderWithThinkArgs(args *ProviderThinkArgs) ProviderOption {
	return func(opts *ProviderOptions) *ProviderOptions {
		opts.ThinkArgs = cloneProviderThinkArgs(args)
		return opts
	}
}

func ProviderWithEmbeddingArgs(args *ProviderEmbeddingArgs) ProviderOption {
	return func(opts *ProviderOptions) *ProviderOptions {
		if args == nil {
			opts.EmbeddingArgs = nil
		} else {
			clone := *args
			opts.EmbeddingArgs = &clone
		}
		return opts
	}
}

func ApplyProviderOptions(opts *ProviderOptions, ops ...ProviderOption) *ProviderOptions {
	if opts == nil {
		opts = &ProviderOptions{}
	} else {
		opts = opts.Clone()
	}
	for _, op := range ops {
		if op == nil {
			continue
		}
		opts = op(opts)
	}
	return opts
}

func (o *ProviderOptions) Clone() *ProviderOptions {
	if o == nil {
		return &ProviderOptions{}
	}
	clone := *o
	clone.ThinkArgs = cloneProviderThinkArgs(o.ThinkArgs)
	return &clone
}

func cloneProviderThinkArgs(args *ProviderThinkArgs) *ProviderThinkArgs {
	if args == nil {
		return nil
	}
	return &ProviderThinkArgs{
		Enable:  maps.Clone(args.Enable),
		Disable: maps.Clone(args.Disable),
	}
}

var _ Provider = (*ProviderNop)(nil)

type ProviderNop struct{}

func (ProviderNop) Embedding(ctx context.Context, model string, input []string, options ...EmbeddingOption) (*Embedding, error) {
	return nil, fmt.Errorf("%w: embedding", ErrUnsupported)
}

func (ProviderNop) Completion(ctx context.Context, model string, messages []*Message, options ...CompletionOption) (*Completion, error) {
	return nil, fmt.Errorf("%w: completion", ErrUnsupported)
}

func (ProviderNop) Realtime(ctx context.Context, model string, messages []*Message, options ...RealTimeOption) (RealTimeSession, error) {
	return nil, fmt.Errorf("%w: realtime", ErrUnsupported)
}

func (ProviderNop) Rerank(ctx context.Context, model string, input *RerankInput, options ...RerankOption) (*RerankOutput, error) {
	return nil, fmt.Errorf("%w: rerank", ErrUnsupported)
}

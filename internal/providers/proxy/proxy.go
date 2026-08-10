package proxy

import (
	"context"
	"crypto/tls"
	"fmt"
	"strings"
	"sync"

	v1 "github.com/xucx/llmx/api/v1"
	"github.com/xucx/llmx/types"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

var (
	_ types.Provider = (*ProxyProvider)(nil)
)

type ProxyProvider struct {
	types.ProviderNop
	mu        sync.Mutex
	options   *types.ProviderOptions
	apiClient v1.ApiServiceClient
	conn      *grpc.ClientConn
}

func NewProxyProvider(opts ...types.ProviderOption) (types.Provider, error) {
	options := types.ApplyProviderOptions(nil, opts...)
	if options.Url == "" {
		return nil, fmt.Errorf("proxy provider URL is empty")
	}
	return &ProxyProvider{options: options}, nil
}

func (p *ProxyProvider) getApiClient() (v1.ApiServiceClient, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.apiClient != nil {
		return p.apiClient, nil
	}

	target := p.options.Url
	secure := !p.options.Insecure
	if strings.HasPrefix(target, "grpcs://") {
		target = strings.TrimPrefix(target, "grpcs://")
		secure = true
	} else if strings.HasPrefix(target, "grpc://") {
		target = strings.TrimPrefix(target, "grpc://")
		secure = false
	}

	dailOptions := []grpc.DialOption{
		grpc.WithPerRPCCredentials(&tokenAuth{
			token:      p.options.Sk,
			requireTLS: secure,
		}),
	}

	if secure {
		dailOptions = append(dailOptions, grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{
			MinVersion: tls.VersionTLS12,
		})))
	} else {
		dailOptions = append(dailOptions, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	conn, err := grpc.NewClient(target, dailOptions...)
	if err != nil {
		return nil, err
	}

	p.conn = conn
	p.apiClient = v1.NewApiServiceClient(conn)

	return p.apiClient, nil
}

type tokenAuth struct {
	token      string
	requireTLS bool
}

func (t *tokenAuth) GetRequestMetadata(ctx context.Context, uri ...string) (map[string]string, error) {
	return map[string]string{
		"Authorization": "Bearer " + t.token,
	}, nil
}

func (t *tokenAuth) RequireTransportSecurity() bool {
	return t.requireTLS
}

func (p *ProxyProvider) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.conn == nil {
		return nil
	}
	err := p.conn.Close()
	p.conn = nil
	p.apiClient = nil
	return err
}

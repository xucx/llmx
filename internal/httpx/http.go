package httpx

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	neturl "net/url"
	"strconv"
	"time"

	"github.com/cenkalti/backoff/v5"
)

var defaultClient = &http.Client{}

type Config struct {
	Client  *http.Client
	BaseUrl string
	Header  http.Header
	Retry   *Retry
}

func NewConfig(opts ...Option) *Config {
	conf := &Config{
		Client: &http.Client{},
		Header: http.Header{},
	}

	for _, opt := range opts {
		opt(conf)
	}

	return conf
}

func (c *Config) Clone() *Config {
	if c == nil {
		return NewConfig()
	}

	clone := &Config{
		Client:  c.Client,
		BaseUrl: c.BaseUrl,
		Header:  c.Header.Clone(),
		Retry:   c.Retry,
	}
	if clone.Header == nil {
		clone.Header = make(http.Header)
	}

	return clone
}

type Retry struct {
	InitInterval time.Duration
	MaxInterval  time.Duration
	Multiplier   float64
	Max          time.Duration
}

type Option func(*Config)

func WithClient(client *http.Client) Option {
	return func(conf *Config) {
		conf.Client = client
	}
}

func WithBaseUrl(url string) Option {
	return func(conf *Config) {
		conf.BaseUrl = url
	}
}

func WithHeader(k, v string) Option {
	return func(conf *Config) {
		if conf.Header == nil {
			conf.Header = make(http.Header)
		}
		conf.Header.Set(k, v)
	}
}

func WithRetry(retryInitInterval, retryMaxInterval, retryMax time.Duration, multiplier float64) Option {
	return func(conf *Config) {
		conf.Retry = &Retry{
			InitInterval: retryInitInterval,
			MaxInterval:  retryMaxInterval,
			Multiplier:   multiplier,
			Max:          retryMax,
		}
	}
}

func (c *Config) Request(ctx context.Context, method, url string, body io.Reader, opts ...Option) (*http.Response, error) {
	if ctx == nil {
		return nil, fmt.Errorf("request context is nil")
	}
	conf := c.Clone()
	for _, opt := range opts {
		opt(conf)
	}

	if conf.Client == nil {
		conf.Client = defaultClient
	}

	if conf.BaseUrl != "" {
		fullUrl, err := neturl.JoinPath(conf.BaseUrl, url)
		if err != nil {
			return nil, err
		}
		url = fullUrl
	}

	var bodyData []byte
	if body != nil {
		var err error
		bodyData, err = io.ReadAll(body)
		if err != nil {
			return nil, err
		}
	}

	makeRequest := func() (*http.Request, error) {
		var requestBody io.Reader
		if bodyData != nil {
			requestBody = bytes.NewReader(bodyData)
		}
		req, err := http.NewRequestWithContext(ctx, method, url, requestBody)
		if err != nil {
			return nil, err
		}
		req.Header = conf.Header.Clone()
		return req, nil
	}

	if conf.Retry == nil {
		req, err := makeRequest()
		if err != nil {
			return nil, err
		}
		return request(conf.Client, req)
	}

	return backoff.Retry(
		ctx,
		func() (*http.Response, error) {
			req, err := makeRequest()
			if err != nil {
				return nil, backoff.Permanent(err)
			}
			rsp, err := request(conf.Client, req)
			if err == nil {
				return rsp, nil
			}

			var statusError *ErrorStatusCode
			if errors.As(err, &statusError) {
				if statusError.Code == http.StatusTooManyRequests {
					if statusError.RetryAfter > 0 {
						return nil, backoff.RetryAfter(statusError.RetryAfter)
					}
					return nil, err
				}
				if statusError.Code >= 500 {
					return nil, err
				}
				return nil, backoff.Permanent(err)
			}
			return nil, err
		},
		backoff.WithBackOff(&backoff.ExponentialBackOff{
			InitialInterval:     conf.Retry.InitInterval,
			RandomizationFactor: 0.5,
			Multiplier:          conf.Retry.Multiplier,
			MaxInterval:         conf.Retry.MaxInterval,
		}),
		backoff.WithMaxElapsedTime(conf.Retry.Max),
	)
}

func request(client *http.Client, r *http.Request) (*http.Response, error) {

	rsp, err := client.Do(r)
	if err != nil {
		return nil, err
	}

	if rsp.StatusCode < 200 || rsp.StatusCode > 299 {
		body, _ := io.ReadAll(rsp.Body)
		rsp.Body.Close()

		if rsp.StatusCode == 429 {
			retryAfter, _ := strconv.ParseInt(rsp.Header.Get("Retry-After"), 10, 64)
			return nil, NewErrorStatusCode429(string(body), int(retryAfter))
		}

		return nil, NewErrorStatusCode(rsp.StatusCode, string(body))
	}

	return rsp, nil
}

func GetWithConfig(ctx context.Context, conf *Config, url string, opts ...Option) (*http.Response, error) {
	return conf.Request(ctx, http.MethodGet, url, nil, opts...)
}

func Get(ctx context.Context, url string, opts ...Option) (*http.Response, error) {
	return GetWithConfig(ctx, NewConfig(), url, opts...)
}

func GetJSONWithConfig[T any](ctx context.Context, conf *Config, url string, opts ...Option) (*T, error) {
	rsp, err := GetWithConfig(ctx, conf, url, opts...)
	if err != nil {
		return nil, err
	}
	defer rsp.Body.Close()

	body, err := io.ReadAll(rsp.Body)
	if err != nil {
		return nil, err
	}

	rspData := new(T)
	if err := json.Unmarshal(body, rspData); err != nil {
		return nil, err
	}

	return rspData, nil
}

func GetJSON[T any](ctx context.Context, url string, opts ...Option) (*T, error) {
	return GetJSONWithConfig[T](ctx, NewConfig(), url, opts...)
}

func PostWithConfig(ctx context.Context, conf *Config, url string, body io.Reader, opts ...Option) (*http.Response, error) {
	return conf.Request(ctx, http.MethodPost, url, body, opts...)
}

func Post(ctx context.Context, url string, body io.Reader, opts ...Option) (*http.Response, error) {
	return PostWithConfig(ctx, NewConfig(), url, body, opts...)
}

func PostJsonWithConfig[T any](ctx context.Context, conf *Config, url string, body any, opts ...Option) (*T, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	opts = append(opts, WithHeader("Content-Type", "application/json"))
	rsp, err := PostWithConfig(ctx, conf, url, bytes.NewReader(data), opts...)
	if err != nil {
		return nil, err
	}
	defer rsp.Body.Close()

	rspBody, err := io.ReadAll(rsp.Body)
	if err != nil {
		return nil, err
	}

	rspData := new(T)
	if err := json.Unmarshal(rspBody, rspData); err != nil {
		return nil, err
	}

	return rspData, nil
}

func PostJson[T any](ctx context.Context, url string, body any, opts ...Option) (*T, error) {
	return PostJsonWithConfig[T](ctx, NewConfig(), url, body, opts...)
}

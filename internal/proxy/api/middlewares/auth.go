package middlewares

import (
	"context"
	"errors"
	"strings"

	"github.com/labstack/echo/v4"
	"github.com/xucx/llmx/internal/proxy/api/middleware"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

const (
	DefaultAuthHeader     = "Authorization"
	DefaultAuthHeaderType = "Bearer"
)

type HttpTokenGetter func(echo.Context) (string, error)

type AuthOpts struct {
	Tokens          []string
	HttpTokenGetter HttpTokenGetter
}

type AuthOpt func(*AuthOpts)

func AuthWithTokens(tokens []string) AuthOpt {
	return func(opts *AuthOpts) {
		opts.Tokens = tokens
	}
}

func AuthWithHttpTokenGetter(getter HttpTokenGetter) AuthOpt {
	return func(opts *AuthOpts) {
		opts.HttpTokenGetter = getter
	}
}

type Auth struct {
	middleware.NopMiddleware
	opts *AuthOpts
}

func NewAuth(opts ...AuthOpt) *Auth {
	option := &AuthOpts{
		HttpTokenGetter: DefaultAuthHttpHeaderGetter,
	}

	for _, opt := range opts {
		opt(option)
	}

	return &Auth{opts: option}
}

func (a *Auth) Unary() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (response interface{}, err error) {
		if err := a.checkGrpcAuth(ctx); err != nil {
			return nil, status.Errorf(codes.Unauthenticated, "Unauthenticated: %v", err)
		}
		return handler(ctx, req)
	}
}

func (a *Auth) Stream() grpc.StreamServerInterceptor {
	return func(srv interface{}, stream grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		if err := a.checkGrpcAuth(stream.Context()); err != nil {
			return status.Errorf(codes.Unauthenticated, "Unauthenticated: %v", err)
		}

		return handler(srv, stream)
	}
}

func (a *Auth) Http() echo.MiddlewareFunc {
	return func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c echo.Context) error {
			if err := a.checkHttpAuth(c); err != nil {
				return echo.ErrUnauthorized
			}
			return next(c)
		}
	}
}

func (a *Auth) checkGrpcAuth(ctx context.Context) error {

	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return errors.New("authorization not found")
	}

	tokens := md.Get(DefaultAuthHeader)
	if len(tokens) == 0 {
		return errors.New("authorization not found")
	}

	ts := strings.Fields(tokens[0])
	if len(ts) != 2 || ts[0] != DefaultAuthHeaderType {
		return errors.New("wrong token")
	}

	return a.checkToken(ts[1])
}

func (a *Auth) checkHttpAuth(ctx echo.Context) error {

	token, err := a.opts.HttpTokenGetter(ctx)
	if err != nil {
		return err
	}

	return a.checkToken(token)
}

func (a *Auth) checkToken(token string) error {
	if len(a.opts.Tokens) == 0 {
		return nil
	}

	for _, t := range a.opts.Tokens {
		if token == t {
			return nil
		}
	}

	return errors.New("wrong token")
}

func DefaultAuthHttpHeaderGetter(ctx echo.Context) (string, error) {
	token := ctx.Request().Header.Get(DefaultAuthHeader)
	ts := strings.Split(token, " ")
	if len(ts) == 2 {
		if ts[0] == DefaultAuthHeaderType {
			return ts[1], nil
		}
	}

	return "", errors.New("authorization not found")
}

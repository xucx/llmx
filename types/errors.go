package types

import (
	"errors"
	"time"
)

var (
	ErrInvalidMessage = errors.New("invalid message")
	ErrUnsupported    = errors.New("unsupported provider capability")
)

type StatusCodeError interface {
	error
	StatusCode() int
}

type RetryAfterError interface {
	error
	RetryAfterDuration() time.Duration
}

func ErrorStatusCode(err error) (int, bool) {
	var statusError StatusCodeError
	if !errors.As(err, &statusError) {
		return 0, false
	}
	return statusError.StatusCode(), true
}

func ErrorRetryAfter(err error) (time.Duration, bool) {
	var retryError RetryAfterError
	if !errors.As(err, &retryError) {
		return 0, false
	}
	retryAfter := retryError.RetryAfterDuration()
	return retryAfter, retryAfter > 0
}

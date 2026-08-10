package httpx

import (
	"fmt"
	"time"
)

type ErrorStatusCode struct {
	Code       int
	Message    string
	RetryAfter int
}

func (e *ErrorStatusCode) Error() string {
	return fmt.Sprintf("http return status code %d: %s", e.Code, e.Message)
}

func (e *ErrorStatusCode) StatusCode() int {
	return e.Code
}

func (e *ErrorStatusCode) RetryAfterDuration() time.Duration {
	return time.Duration(e.RetryAfter) * time.Second
}

func NewErrorStatusCode(code int, message string) error {
	return &ErrorStatusCode{
		Code:    code,
		Message: message,
	}
}

func NewErrorStatusCode429(message string, retryAfter int) error {
	return &ErrorStatusCode{
		Code:       429,
		Message:    message,
		RetryAfter: retryAfter,
	}
}

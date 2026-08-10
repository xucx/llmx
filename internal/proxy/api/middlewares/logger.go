package middlewares

import (
	"github.com/labstack/echo/v4"
	echomiddleware "github.com/labstack/echo/v4/middleware"
	"github.com/xucx/gox/logx"
	"github.com/xucx/llmx/internal/proxy/api/middleware"
)

type Logger struct {
	middleware.NopMiddleware
}

func NewLogger() *Logger {
	return &Logger{}
}

func (*Logger) Http() echo.MiddlewareFunc {
	return echomiddleware.RequestLoggerWithConfig(echomiddleware.RequestLoggerConfig{
		LogLatency:       true,
		LogProtocol:      false,
		LogRemoteIP:      true,
		LogHost:          true,
		LogMethod:        true,
		LogURI:           true,
		LogURIPath:       false,
		LogRoutePath:     false,
		LogRequestID:     true,
		LogReferer:       false,
		LogUserAgent:     true,
		LogStatus:        true,
		LogError:         true,
		LogContentLength: true,
		LogResponseSize:  true,
		LogHeaders:       nil,
		LogQueryParams:   nil,
		LogFormValues:    nil,
		HandleError:      true, // forwards error to the global error handler, so it can decide appropriate status code
		LogValuesFunc: func(c echo.Context, v echomiddleware.RequestLoggerValues) error {
			logx.Infow("http request",

				"method", v.Method,
				"uri", v.URI,
				"status", v.Status,
				"latency", v.Latency,
				"host", v.Host,
				"bytesIn", v.ContentLength,
				"bytesOut", v.ResponseSize,
				"userAgent", v.UserAgent,
				"remoteIP", v.RemoteIP,
				"requestID", v.RequestID,
				"error", v.Error,
			)

			return nil
		},
	})
}

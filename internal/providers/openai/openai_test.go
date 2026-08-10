package openai

import (
	"errors"
	"os"

	"github.com/xucx/llmx/types"
)

var errorNoOpenApiKey = errors.New("No OPENAI_API_KEY")

func InitTestOpenaiProvider() (types.Provider, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, errorNoOpenApiKey
	}

	return NewOpenaiProvider(types.ProviderWithSk(apiKey))
}

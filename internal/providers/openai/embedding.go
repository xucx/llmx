package openai

import (
	"context"
	"fmt"

	"github.com/xucx/llmx/internal/httpx"
	"github.com/xucx/llmx/types"
)

const (
	EmbeddingPath = "/embeddings"
)

type EmbeddingReq struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	Dimensions     int      `json:"dimensions,omitempty"`
	EncodingFormat string   `json:"encoding_format"` //float base64
}

type EmbeddingRsp struct {
	Model   string              `json:"model"`
	Data    []*EmbeddingRspData `json:"data"`
	Usage   CompletionUsage     `json:"usage"`
	Object  string              `json:"object"`
	Message string              `json:"message"`
}

type EmbeddingRspData struct {
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

func (p *OpenaiProvider) Embedding(ctx context.Context, model string, input []string, options ...types.EmbeddingOption) (*types.Embedding, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("no input")
	}

	opts := types.ApplyEmbeddingOptions(nil, options...)
	if opts.Dimensions < 0 {
		return nil, fmt.Errorf("embedding dimensions must not be negative")
	}

	req := EmbeddingReq{
		Model:          model,
		Input:          input,
		Dimensions:     opts.Dimensions,
		EncodingFormat: "float",
	}

	if p.options.EmbeddingArgs != nil {
		if p.options.EmbeddingArgs.DisableDim {
			req.Dimensions = 0
		}
	}

	rsp, err := httpx.PostJsonWithConfig[EmbeddingRsp](ctx, p.httpConfig, EmbeddingPath, &req)
	if err != nil {
		return nil, err
	}

	if rsp.Object == "error" {
		return nil, fmt.Errorf("openai response fail, error is %s", rsp.Message)
	}

	if len(rsp.Data) == 0 {
		return nil, fmt.Errorf("openapi response no data")
	}

	data := []*types.EmbeddingData{}
	for _, d := range rsp.Data {
		data = append(data, &types.EmbeddingData{
			Index:     d.Index,
			Embedding: d.Embedding,
		})
	}
	return &types.Embedding{
		Model:        rsp.Model,
		Data:         data,
		PromptTokens: rsp.Usage.PromptTokens,
		TotalTokens:  rsp.Usage.TotalTokens,
	}, nil

}

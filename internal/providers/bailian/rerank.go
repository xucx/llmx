package bailian

import (
	"context"
	"fmt"

	"github.com/xucx/llmx/internal/httpx"
	"github.com/xucx/llmx/types"
)

const BailianUrlPath = "/reranks"

type BailianRerankOutput struct {
	ReqeustId string `json:"request_id"`
	Code      string `json:"code"`
	Message   string `json:"message"`
	Output    struct {
		Results []*BailianRerankOutputResult `json:"results"`
	} `json:"output"`
	Results []*BailianRerankOutputResult `json:"results"` //兼容接口
}

type BailianRerankOutputResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

func (b *BailianProvider) Rerank(ctx context.Context, model string, input *types.RerankInput, options ...types.RerankOption) (*types.RerankOutput, error) {
	if input == nil {
		return nil, fmt.Errorf("rerank input is nil")
	}
	if len(input.Documents) == 0 {
		return nil, fmt.Errorf("rerank documents are empty")
	}
	if input.TopN < 0 {
		return nil, fmt.Errorf("rerank top_n must not be negative")
	}
	opts := types.ApplyRerankOptions(nil, options...)

	req := map[string]any{
		"model":     model,
		"documents": input.Documents,
		"query":     input.Query,
		"top_n":     input.TopN,
	}
	if opts.Instruction != "" {
		req["instruct"] = opts.Instruction
	}

	rsp, err := httpx.PostJsonWithConfig[BailianRerankOutput](ctx, b.httpConfig, BailianUrlPath, req)
	if err != nil {
		return nil, err
	}

	rspResults := rsp.Results
	if rspResults == nil {
		rspResults = rsp.Output.Results
	}
	if rspResults == nil {
		return nil, fmt.Errorf("ali bailian rerank return no result")
	}

	output := &types.RerankOutput{}
	for _, r := range rspResults {
		item := &types.RerankResultItem{
			Index: r.Index,
			Score: r.RelevanceScore,
		}
		if opts.OutputDoc {
			if r.Index < 0 || r.Index >= len(input.Documents) {
				return nil, fmt.Errorf("bailian rerank returned document index %d out of range", r.Index)
			}
			item.Document = input.Documents[r.Index]
		}

		output.Result = append(output.Result, item)
	}

	return output, nil
}

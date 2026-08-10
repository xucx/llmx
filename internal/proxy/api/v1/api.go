package v1

import (
	context "context"
	"fmt"

	"github.com/xucx/gox/logx"
	"github.com/xucx/llmx"
	"github.com/xucx/llmx/internal/providers/proxy"

	apiv1 "github.com/xucx/llmx/api/v1"
	"github.com/xucx/llmx/types"

	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

var (
	GrpcArgumentError = status.Errorf(codes.InvalidArgument, "Invalid Argument")
	GrpcInternalError = status.Errorf(codes.Internal, "Internal Server Error")
)

type ApiService struct {
	apiv1.UnimplementedApiServiceServer
	models *llmx.Models
}

func NewApiService(models *llmx.Models) *ApiService {
	return &ApiService{
		models: models,
	}
}

func (s *ApiService) Chat(ctx context.Context, req *apiv1.ChatRequest) (*apiv1.ChatResponse, error) {
	if req.ChatParams == nil {
		return nil, GrpcArgumentError
	}

	messages := []*types.Message{}
	for _, m := range req.ChatParams.Messages {
		msg, err := proxy.ToMessage(m)
		if err != nil {
			return nil, err
		}
		messages = append(messages, msg)
	}

	options, err := proxy.ChatParamsToOptions(req.ChatParams)
	if err != nil {
		return nil, GrpcArgumentError
	}

	completion, err := s.models.Completion(ctx, req.ChatParams.Model, messages, types.CompletionWithOptions(options))
	if err != nil {
		logx.Errorw("llm chat fail", "model", req.ChatParams.Model, "error", err)
		return nil, GrpcInternalError
	}

	retCompletion, err := proxy.FromChatCompletion(completion)
	if err != nil {
		return nil, GrpcInternalError
	}

	return &apiv1.ChatResponse{
		ChatCompletion: retCompletion,
	}, nil
}

func (s *ApiService) ChatStream(req *apiv1.ChatStreamRequest, stream apiv1.ApiService_ChatStreamServer) error {

	if req.ChatParams == nil {
		logx.Warn("llm stream chat fail with wrong argument, ChatParams is nil ")
		return GrpcArgumentError
	}

	messages := []*types.Message{}
	for _, m := range req.ChatParams.Messages {
		msg, err := proxy.ToMessage(m)
		if err != nil {
			logx.Warnf("llm stream chat fail with wrong argument, convert message fail, error is %v", err)
			return err
		}
		messages = append(messages, msg)
	}

	options, err := proxy.ChatParamsToOptions(req.ChatParams)
	if err != nil {
		logx.Warnf("llm stream chat fail with wrong argument, parse options fail, error is %v", err)
		return GrpcArgumentError
	}

	completion, err := s.models.Completion(stream.Context(), req.ChatParams.Model, messages, types.CompletionWithOptions(options),
		types.CompletionWithStreamingFunc(func(ctx context.Context, c *types.Completion) error {
			deltaCompletion, err := proxy.FromChatCompletion(c)
			if err != nil {
				logx.Warnw("llm stream chat fail, convert stream completion fail", "error", err)
				return err
			}

			return stream.Send(&apiv1.ChatStreamResponse{
				ChatCompletion: deltaCompletion,
			})
		}),
	)

	if err != nil {
		logx.Warnw("llm stream chat fail", "model", req.ChatParams.Model, "error", err)
		return GrpcInternalError
	}

	fullCompletion, err := proxy.FromChatCompletion(completion)
	if err != nil {
		logx.Warnw("llm stream chat fail, convert final completion fail", "error", err)
		return GrpcInternalError
	}

	return stream.Send(&apiv1.ChatStreamResponse{
		ChatCompletion: fullCompletion,
	})
}

func (s *ApiService) ChatRealtime(stream apiv1.ApiService_ChatRealtimeServer) error {
	ctx, cancel := context.WithCancel(stream.Context())
	defer cancel()

	session, err := s.initRealtime(stream)
	if err != nil {
		return GrpcInternalError
	}

	go func() {
		defer cancel()

		for {
			req, err := stream.Recv()
			if err != nil {
				return
			}

			message, err := proxy.ToMessage(req.Message)
			if err != nil {
				return
			}

			if err := session.Send(ctx, message); err != nil {
				return
			}
		}
	}()

	for {
		rsp, err := session.Recv(ctx)
		if err != nil {
			break
		}

		completion, err := proxy.FromChatCompletion(rsp)
		if err != nil {
			return err
		}

		if err := stream.Send(&apiv1.ChatRealtimeResponse{
			ChatCompletion: completion,
		}); err != nil {
			return err
		}
	}

	return nil
}

func (s *ApiService) initRealtime(stream apiv1.ApiService_ChatRealtimeServer) (types.RealTimeSession, error) {
	ctx := stream.Context()

	req, err := stream.Recv()
	if err != nil {
		return nil, err
	}

	if req.Init == nil {
		return nil, fmt.Errorf("no init message")
	}

	messages := []*types.Message{}
	for _, m := range req.Init.ChatParams.Messages {
		msg, err := proxy.ToMessage(m)
		if err != nil {
			return nil, err
		}
		messages = append(messages, msg)
	}

	options := []types.RealTimeOption{}
	if req.Init.ChatParams.Instructions != "" {
		options = append(options, types.RealTimeWithInstructions(req.Init.ChatParams.Instructions))
	}

	if req.Init.ChatParams.Tools != nil {
		tools, err := proxy.ToChatTools(req.Init.ChatParams.Tools)
		if err != nil {
			return nil, err
		}
		options = append(options, types.RealTimeWithTools(tools))
	}
	if req.Init.ChatParams.Voice != "" {
		voice, err := proxy.ToChatVoice(req.Init.ChatParams.Voice)
		if err != nil {
			return nil, err
		}
		options = append(options, types.RealTimeWithAudioVoice(voice))
	}

	session, err := s.models.Realtime(ctx, req.Init.ChatParams.Model, messages, options...)
	if err != nil {
		return nil, err
	}

	return session, nil
}

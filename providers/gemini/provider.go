package gemini

import (
	"context"
	"net/http"
	"os"
	"time"

	llmrouter "github.com/bluefunda/llmrouter"
	"google.golang.org/genai"
)

// Provider handles the Google Gemini API through the official google.golang.org/genai SDK.
//
// It replaced github.com/google/generative-ai-go, which Google deprecated and which can no
// longer parse Gemini 3.x streams: every stream ended in "invalid character ']'", and
// function-calling turns came back as an unknown finish reason with no content.
type Provider struct {
	client *genai.Client
	model  string
	models []string
}

// DefaultModels is the list of available Gemini models
var DefaultModels = []string{
	"gemini-3.5-flash",
	"gemini-3.5-flash-lite",
}

// New creates a new Gemini provider.
// The SDK takes a context for client construction; context.Background() is used so the
// provider lifetime is not tied to a caller's context. cfg.BaseURL overrides the API
// endpoint (used by tests); cfg.CustomHeaders are sent on every request.
func New(cfg llmrouter.ProviderConfig) (*Provider, error) {
	model := cfg.Model
	if model == "" {
		model = "gemini-3.5-flash-lite"
	}

	models := cfg.Models
	if len(models) == 0 {
		models = DefaultModels
	}

	cc := &genai.ClientConfig{
		APIKey:  cfg.APIKey,
		Backend: genai.BackendGeminiAPI,
	}
	if cfg.BaseURL != "" {
		cc.HTTPOptions.BaseURL = cfg.BaseURL
	}
	if len(cfg.CustomHeaders) > 0 {
		cc.HTTPOptions.Headers = http.Header{}
		for key, value := range cfg.CustomHeaders {
			cc.HTTPOptions.Headers.Set(key, value)
		}
	}

	client, err := genai.NewClient(context.Background(), cc)
	if err != nil {
		return nil, err
	}

	return &Provider{
		client: client,
		model:  model,
		models: models,
	}, nil
}

// NewFromEnv creates a provider using the GEMINI_API_KEY environment variable
func NewFromEnv() (*Provider, error) {
	return New(llmrouter.ProviderConfig{
		APIKey: os.Getenv("GEMINI_API_KEY"),
	})
}

// Close releases the provider. The genai client holds no resources that need closing.
func (p *Provider) Close() error {
	return nil
}

func (p *Provider) Name() string {
	return "gemini"
}

func (p *Provider) Models() []string {
	out := make([]string, len(p.models))
	copy(out, p.models)
	return out
}

func (p *Provider) modelName(req *llmrouter.Request) string {
	if req.Model != "" {
		return req.Model
	}
	return p.model
}

func (p *Provider) Complete(ctx context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	modelName := p.modelName(req)
	contents, config := buildRequest(req)

	resp, err := p.client.Models.GenerateContent(ctx, modelName, contents, config)
	if err != nil {
		return nil, wrapError(err)
	}

	acc := newAccumulator()
	if err := acc.add(resp, func(llmrouter.Event) {}); err != nil {
		return nil, err
	}
	return acc.response(modelName, p.Name()), nil
}

func (p *Provider) Stream(ctx context.Context, req *llmrouter.Request) (*llmrouter.StreamResult, error) {
	modelName := p.modelName(req)
	contents, config := buildRequest(req)

	ctx, cancel := context.WithCancel(ctx)
	ch := make(chan llmrouter.Event)
	res := llmrouter.NewStreamResult(ch)
	res.OnClose(func() error { cancel(); return nil })

	go func() {
		defer close(ch)
		defer cancel()

		emit := func(ev llmrouter.Event) {
			select {
			case ch <- ev:
			case <-ctx.Done():
			}
		}

		acc := newAccumulator()
		for resp, err := range p.client.Models.GenerateContentStream(ctx, modelName, contents, config) {
			if err != nil {
				emit(llmrouter.Event{Type: llmrouter.EventError, Error: wrapError(err)})
				return
			}
			if err := acc.add(resp, emit); err != nil {
				emit(llmrouter.Event{Type: llmrouter.EventError, Error: err})
				return
			}
		}
		emit(llmrouter.Event{
			Type:     llmrouter.EventDone,
			Response: acc.response(modelName, p.Name()),
		})
	}()

	return res, nil
}

// buildRequest converts an llmrouter request into Gemini contents and generation config.
func buildRequest(req *llmrouter.Request) ([]*genai.Content, *genai.GenerateContentConfig) {
	toolsDeclared := len(req.Tools) > 0
	contents, system := convertMessages(req.Messages, toolsDeclared)

	config := &genai.GenerateContentConfig{
		SystemInstruction: system,
		MaxOutputTokens:   16384,
		StopSequences:     req.Stop,
	}
	if req.MaxTokens != nil {
		config.MaxOutputTokens = int32(*req.MaxTokens)
	}
	if req.Temperature != nil {
		temp := float32(*req.Temperature)
		config.Temperature = &temp
	}
	if req.TopP != nil {
		topP := float32(*req.TopP)
		config.TopP = &topP
	}
	if toolsDeclared {
		config.Tools = convertTools(req.Tools)
	}
	return contents, config
}

// response builds the final OpenAI-compatible response once the stream ends.
func (a *accumulator) response(model, provider string) *llmrouter.Response {
	return &llmrouter.Response{
		Model:    model,
		Provider: provider,
		Object:   "chat.completion",
		Created:  time.Now().Unix(),
		Choices: []llmrouter.Choice{
			{
				Index: 0,
				Message: &llmrouter.Message{
					Role:      llmrouter.RoleAssistant,
					Content:   a.content.String(),
					ToolCalls: a.toolCalls,
				},
				FinishReason: a.finishReason(),
			},
		},
		Usage: a.usage(),
	}
}

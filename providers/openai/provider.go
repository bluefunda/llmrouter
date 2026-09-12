package openai

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"time"

	llmrouter "github.com/bluefunda/llmrouter"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/respjson"
)

// Presets contains default configurations for OpenAI-compatible providers
var Presets = map[string]struct {
	BaseURL      string
	DefaultModel string
	Models       []string
}{
	"openai": {
		BaseURL:      "https://api.openai.com/v1/",
		DefaultModel: "gpt-4.1-mini",
		Models:       []string{"gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini", "o4-mini"},
	},
	"deepseek": {
		BaseURL:      "https://api.deepseek.com/",
		DefaultModel: "deepseek-chat",
		Models:       []string{"deepseek-chat", "deepseek-coder"},
	},
	"groq": {
		BaseURL:      "https://api.groq.com/openai/v1/",
		DefaultModel: "llama-3.3-70b-versatile",
		Models:       []string{"llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"},
	},
	"together": {
		BaseURL:      "https://api.together.xyz/v1/",
		DefaultModel: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
		Models:       []string{"meta-llama/Llama-3.3-70B-Instruct-Turbo", "mistralai/Mixtral-8x7B-Instruct-v0.1"},
	},
	"ollama": {
		BaseURL:      "http://localhost:11434/v1/",
		DefaultModel: "llama3.2",
		Models:       []string{}, // Dynamic based on what's installed
	},
	"sarvam": {
		BaseURL:      "https://api.sarvam.ai/v1/",
		DefaultModel: "sarvam-30b",
		Models:       []string{"sarvam-m", "sarvam-30b", "sarvam-105b"},
	},
}

// Provider handles OpenAI and OpenAI-compatible APIs
type Provider struct {
	client            openai.Client
	name              string
	model             string
	models            []string
	stringContentOnly bool
}

// New creates a new OpenAI-compatible provider
func New(cfg llmrouter.ProviderConfig) *Provider {
	preset, hasPreset := Presets[cfg.Name]

	baseURL := cfg.BaseURL
	if baseURL == "" && hasPreset {
		baseURL = preset.BaseURL
	}
	// Ensure trailing slash so url.Parse resolves paths correctly
	if baseURL != "" && !strings.HasSuffix(baseURL, "/") {
		baseURL += "/"
	}

	model := cfg.Model
	if model == "" && hasPreset {
		model = preset.DefaultModel
	}

	opts := []option.RequestOption{}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.Timeout > 0 {
		opts = append(opts, option.WithRequestTimeout(cfg.Timeout))
	}
	for key, value := range cfg.CustomHeaders {
		opts = append(opts, option.WithHeader(key, value))
	}

	models := cfg.Models
	if len(models) == 0 && hasPreset {
		models = preset.Models
	}

	return &Provider{
		client:            openai.NewClient(opts...),
		name:              cfg.Name,
		model:             model,
		models:            models,
		stringContentOnly: cfg.StringContentOnly,
	}
}

// NewFromEnv creates a provider using environment variable for API key
func NewFromEnv(name string, envKey string) *Provider {
	return New(llmrouter.ProviderConfig{
		Name:   name,
		APIKey: os.Getenv(envKey),
	})
}

// NewOpenAI creates a standard OpenAI provider
func NewOpenAI(apiKey string) *Provider {
	return New(llmrouter.ProviderConfig{
		Name:   "openai",
		APIKey: apiKey,
	})
}

// NewDeepSeek creates a DeepSeek provider
func NewDeepSeek(apiKey string) *Provider {
	return New(llmrouter.ProviderConfig{
		Name:   "deepseek",
		APIKey: apiKey,
	})
}

// NewGroq creates a Groq provider
func NewGroq(apiKey string) *Provider {
	return New(llmrouter.ProviderConfig{
		Name:   "groq",
		APIKey: apiKey,
	})
}

// NewTogether creates a Together AI provider
func NewTogether(apiKey string) *Provider {
	return New(llmrouter.ProviderConfig{
		Name:   "together",
		APIKey: apiKey,
	})
}

// NewOllama creates an Ollama provider
func NewOllama(baseURL string) *Provider {
	if baseURL == "" {
		baseURL = "http://localhost:11434/v1"
	}
	return New(llmrouter.ProviderConfig{
		Name:    "ollama",
		BaseURL: baseURL,
		APIKey:  "ollama", // Ollama doesn't require a real key but needs something
	})
}

func (p *Provider) Name() string {
	return p.name
}

func (p *Provider) Models() []string {
	out := make([]string, len(p.models))
	copy(out, p.models)
	return out
}

// resolveModel returns the model name to use for a request.
func (p *Provider) resolveModel(req *llmrouter.Request) string {
	if req.Model == "" || req.Model == p.name {
		return p.model
	}
	return req.Model
}

// buildParams constructs the API params shared by Complete and Stream.
func (p *Provider) buildParams(req *llmrouter.Request) openai.ChatCompletionNewParams {
	params := openai.ChatCompletionNewParams{
		Model:    p.resolveModel(req),
		Messages: convertMessages(req.Messages, p.stringContentOnly),
	}

	if req.Temperature != nil {
		params.Temperature = openai.Float(*req.Temperature)
	}
	if req.MaxTokens != nil {
		params.MaxCompletionTokens = openai.Int(int64(*req.MaxTokens))
	}
	if req.TopP != nil {
		params.TopP = openai.Float(*req.TopP)
	}
	if len(req.Stop) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: req.Stop}
	}
	if len(req.Tools) > 0 {
		params.Tools = convertTools(req.Tools)
	}
	if req.ToolChoice != nil {
		params.ToolChoice = convertToolChoice(req.ToolChoice)
	}

	return params
}

// extraStringField reads a string-valued field out of ChoiceDelta's untyped
// ExtraFields — used for provider-specific delta fields that aren't part of
// OpenAI's own typed schema (e.g. Groq's "reasoning" field).
//
// Deliberately does NOT gate on field.Valid(): the apijson decoder only
// marks an extra field "valid" when the struct declares a typed `,extras`
// destination map to decode into. ChatCompletionChunkChoiceDelta has no such
// field (only the metadata-only JSON.ExtraFields), so every untyped extra
// field it captures is unconditionally stamped status=invalid regardless of
// its actual content — Valid() would always be false here even for a
// perfectly well-formed string value. Raw() is unaffected by that status
// (it only special-cases the omitted case), so it's the correct signal to
// use instead: empty means the key was never present, "null" means an
// explicit JSON null, and everything else is real content.
func extraStringField(extra map[string]respjson.Field, key string) (string, bool) {
	field, ok := extra[key]
	if !ok {
		return "", false
	}
	raw := field.Raw()
	if raw == "" || raw == "null" {
		return "", false
	}
	var s string
	if err := json.Unmarshal([]byte(raw), &s); err != nil {
		return "", false
	}
	return s, true
}

func (p *Provider) Complete(ctx context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	params := p.buildParams(req)

	resp, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, wrapError(p.name, err)
	}

	return convertResponse(resp, p.name), nil
}

func (p *Provider) Stream(ctx context.Context, req *llmrouter.Request) (*llmrouter.StreamResult, error) {
	model := p.resolveModel(req)
	params := p.buildParams(req)
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: openai.Bool(true)}

	ctx, cancel := context.WithCancel(ctx)
	ch := make(chan llmrouter.Event)
	res := llmrouter.NewStreamResult(ch)
	res.OnClose(func() error { cancel(); return nil })

	go func() {
		defer close(ch)
		defer cancel()

		stream := p.client.Chat.Completions.NewStreaming(ctx, params)

		// reasoningActive tracks whether we're currently inside a reasoning
		// span for providers that expose one (bluefunda/cai-llm-router#325 —
		// Groq's openai/gpt-oss-120b sends a non-standard "reasoning" delta
		// field, undeclared in the OpenAI SDK's typed schema, so it only
		// surfaces via ChoiceDelta.JSON.ExtraFields). There's no explicit
		// start/stop marker in the OpenAI-compatible streaming protocol the
		// way Anthropic has content_block_start/stop, so boundaries are
		// inferred: reasoning begins on the first chunk that carries it, and
		// ends the moment real content starts arriving (the model has
		// finished reasoning and moved on to its answer).
		reasoningActive := false

		var lastChunk *openai.ChatCompletionChunk
		for stream.Next() {
			chunk := stream.Current()
			lastChunk = &chunk

			if len(chunk.Choices) > 0 {
				delta := chunk.Choices[0].Delta

				if reasoningText, ok := extraStringField(delta.JSON.ExtraFields, "reasoning"); ok && reasoningText != "" {
					if !reasoningActive {
						reasoningActive = true
						ch <- llmrouter.Event{Type: llmrouter.EventThinkingStart}
					}
					ch <- llmrouter.Event{Type: llmrouter.EventThinkingDelta, Content: reasoningText}
				}

				if delta.Content != "" {
					if reasoningActive {
						reasoningActive = false
						ch <- llmrouter.Event{Type: llmrouter.EventThinkingStop}
					}
					ch <- llmrouter.Event{
						Type:    llmrouter.EventContentDelta,
						Content: delta.Content,
					}
				}

				if len(delta.ToolCalls) > 0 {
					ch <- llmrouter.Event{
						Type: llmrouter.EventToolCallDelta,
						Delta: &llmrouter.Delta{
							ToolCalls: convertStreamToolCalls(delta.ToolCalls),
						},
					}
				}
			}
		}

		if reasoningActive {
			ch <- llmrouter.Event{Type: llmrouter.EventThinkingStop}
		}

		if err := stream.Err(); err != nil {
			ch <- llmrouter.Event{
				Type:  llmrouter.EventError,
				Error: wrapError(p.name, err),
			}
			return
		}

		if lastChunk != nil {
			ch <- llmrouter.Event{
				Type:     llmrouter.EventDone,
				Response: convertChunkResponse(lastChunk, p.name),
			}
		} else {
			ch <- llmrouter.Event{
				Type: llmrouter.EventDone,
				Response: &llmrouter.Response{
					Provider: p.name,
					Model:    model,
					Object:   "chat.completion",
					Created:  time.Now().Unix(),
				},
			}
		}
	}()

	return res, nil
}

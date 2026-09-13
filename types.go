package llmrouter

import (
	"encoding/json"
	"time"
)

// StreamResult is the iterator type for streaming LLM responses.
// Usage:
//
//	stream, err := router.Stream(ctx, req)
//	if err != nil { return err }
//	defer stream.Close()
//	for stream.Next() {
//	    event := stream.Event()
//	    // handle event
//	}
//	if err := stream.Err(); err != nil { ... }
type StreamResult struct {
	ch      <-chan Event
	closers []func() error
	current Event
	err     error
	done    bool
}

// NewStreamResult creates a StreamResult from an event channel.
// Providers and middleware use this to construct a StreamResult.
func NewStreamResult(ch <-chan Event) *StreamResult {
	return &StreamResult{ch: ch}
}

// OnClose registers a function called when Close is invoked (e.g. context cancel).
func (s *StreamResult) OnClose(fn func() error) {
	s.closers = append(s.closers, fn)
}

// Next advances to the next event. Returns false when the stream ends or an error occurs.
func (s *StreamResult) Next() bool {
	if s.done {
		return false
	}
	event, ok := <-s.ch
	if !ok {
		s.done = true
		return false
	}
	if event.Type == EventError {
		s.err = event.Error
		s.done = true
		return false
	}
	s.current = event
	if event.Type == EventDone {
		s.done = true
		return true
	}
	return true
}

// Event returns the current event (valid after Next returns true).
func (s *StreamResult) Event() Event { return s.current }

// Err returns the streaming error, if any. Check after Next returns false.
func (s *StreamResult) Err() error { return s.err }

// Close stops the stream and releases resources. Safe to call multiple times.
func (s *StreamResult) Close() error {
	s.done = true
	var firstErr error
	for _, fn := range s.closers {
		if err := fn(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	s.closers = nil
	// Drain remaining events to allow producer goroutines to exit.
	for range s.ch {
	}
	return firstErr
}

// Request represents a unified LLM request
type Request struct {
	Messages    []Message      `json:"messages"`
	Model       string         `json:"model,omitempty"`
	Tools       []Tool         `json:"tools,omitempty"`
	ToolChoice  *ToolChoice    `json:"tool_choice,omitempty"`
	Temperature *float64       `json:"temperature,omitempty"`
	MaxTokens   *int           `json:"max_tokens,omitempty"`
	TopP        *float64       `json:"top_p,omitempty"`
	Stop        []string       `json:"stop,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

// Message represents a chat message
type Message struct {
	Role         Role          `json:"role"`
	Content      string        `json:"content"`
	ContentParts []ContentPart `json:"content_parts,omitempty"`
	Name         string        `json:"name,omitempty"`
	ToolCalls    []ToolCall    `json:"tool_calls,omitempty"`
	ToolCallID   string        `json:"tool_call_id,omitempty"`
	// CacheControl marks this message's content for prompt caching (Anthropic only).
	// For user messages with ContentParts, set CacheControl on individual parts instead.
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// ContentPart represents a part of a multimodal message
type ContentPart struct {
	Type         string        `json:"type"`                   // "text", "image_url", or "document"
	Text         string        `json:"text,omitempty"`
	ImageURL     *ImageURL     `json:"image_url,omitempty"`
	Document     *Document     `json:"document,omitempty"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// CacheControl marks a content block for provider-level prompt caching.
// Only "ephemeral" is currently supported. OpenAI and Gemini cache automatically
// and ignore this field; set it only when targeting Anthropic.
type CacheControl struct {
	Type string `json:"type"` // "ephemeral"
}

// ImageURL represents an image reference with both URL and base64 forms
type ImageURL struct {
	URL       string `json:"url"`
	Detail    string `json:"detail,omitempty"`
	Base64    string `json:"base64,omitempty"`
	MediaType string `json:"media_type,omitempty"`
}

// Document represents a document (PDF, etc.) for providers that support it natively
type Document struct {
	Base64    string `json:"base64"`
	MediaType string `json:"media_type"` // e.g. "application/pdf"
}

// Role represents the message role
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Response represents a unified LLM response (OpenAI-compatible)
type Response struct {
	ID       string   `json:"id"`
	Object   string   `json:"object"`
	Created  int64    `json:"created"`
	Model    string   `json:"model"`
	Choices  []Choice `json:"choices"`
	Usage    *Usage   `json:"usage,omitempty"`
	Provider string   `json:"provider"`
}

// Choice represents a completion choice
type Choice struct {
	Index        int      `json:"index"`
	Message      *Message `json:"message,omitempty"`
	Delta        *Delta   `json:"delta,omitempty"`
	FinishReason string   `json:"finish_reason,omitempty"`
}

// Delta represents streaming content delta
type Delta struct {
	Role      Role       `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// Usage represents token usage
type Usage struct {
	PromptTokens        int     `json:"prompt_tokens"`
	CompletionTokens    int     `json:"completion_tokens"`
	TotalTokens         int     `json:"total_tokens"`
	CachedPromptTokens  int     `json:"cached_prompt_tokens,omitempty"`  // tokens served from cache (all providers)
	CacheCreationTokens int     `json:"cache_creation_tokens,omitempty"` // tokens written to cache (Anthropic only)
	Cost                float64 `json:"cost_usd,omitempty"`              // estimated USD cost; 0 if model not in price table
}

// CacheHitRate returns the fraction of prompt tokens served from cache (0–1).
// Returns 0 if no prompt tokens were recorded.
func (u *Usage) CacheHitRate() float64 {
	if u == nil || u.PromptTokens == 0 {
		return 0
	}
	return float64(u.CachedPromptTokens) / float64(u.PromptTokens)
}

// Event represents a streaming event
type Event struct {
	Type     EventType
	Content  string
	Delta    *Delta
	Response *Response
	Error    error
	// Index is the provider's content-block index, set only on the Thinking*
	// event types below — a stable id for one reasoning segment within the
	// current turn (see bluefunda/cai-llm-router#325).
	Index int64
}

// EventType represents the type of streaming event
type EventType int

const (
	EventContentDelta  EventType = iota // Text content chunk
	EventToolCallDelta                  // Tool call chunk
	EventThinkingStart                  // A new reasoning segment began (Index set)
	EventThinkingDelta                  // Incremental reasoning text (Index + Content set)
	EventThinkingStop                   // A reasoning segment finished (Index set)
	EventDone                           // Stream completed
	EventError                          // Error occurred
)

// Tool represents a function/tool definition
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function represents a function definition
type Function struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ToolCall represents a tool invocation
type ToolCall struct {
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Function FuncCall `json:"function"`
	Index    *int     `json:"index,omitempty"`
}

// FuncCall represents a function call
type FuncCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolChoice controls tool selection
type ToolChoice struct {
	Type     string   `json:"type,omitempty"`
	Function *FuncRef `json:"function,omitempty"`
}

// FuncRef references a specific function
type FuncRef struct {
	Name string `json:"name"`
}

// ProviderConfig holds common configuration for providers
type ProviderConfig struct {
	Name          string
	APIKey        string
	BaseURL       string
	Model         string
	Models        []string
	Timeout       time.Duration
	CustomHeaders map[string]string // custom HTTP headers (e.g. api-subscription-key)
	// StringContentOnly forces message content to be sent as plain strings
	// instead of structured arrays. Required for some OpenAI-compatible APIs
	// (e.g. Sarvam) that don't support the array content format.
	StringContentOnly bool
}

package anthropic

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"

	llmrouter "github.com/bluefunda/llmrouter"
	ant "github.com/anthropics/anthropic-sdk-go"
)

// makeAntError builds an ant.Error with populated Request/Response so
// Error() doesn't nil-deref inside the SDK.
func makeAntError(statusCode int) *ant.Error {
	req, _ := http.NewRequest(http.MethodPost, "https://api.anthropic.com/v1/messages", nil)
	resp := &http.Response{StatusCode: statusCode, Request: req}
	return &ant.Error{StatusCode: statusCode, Request: req, Response: resp}
}

// unmarshalMessage unmarshals raw JSON into an ant.Message so AsAny() works
// (it relies on the internal JSON.raw field populated during unmarshal).
func unmarshalMessage(t *testing.T, raw string) *ant.Message {
	t.Helper()
	var msg ant.Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}
	return &msg
}

// ---- convertMessages ----

func TestConvertMessages_System(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleSystem, Content: "you are helpful"},
	}
	_, sys := convertMessages(msgs)
	if len(sys) != 1 {
		t.Fatalf("expected 1 system block, got %d", len(sys))
	}
	if sys[0].Text != "you are helpful" {
		t.Errorf("unexpected text: %s", sys[0].Text)
	}
}

func TestConvertMessages_SystemWithCache(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleSystem, Content: "system prompt", CacheControl: &llmrouter.CacheControl{Type: "ephemeral"}},
	}
	_, sys := convertMessages(msgs)
	if len(sys) != 1 {
		t.Fatalf("expected 1 system block, got %d", len(sys))
	}
	expected := ant.NewCacheControlEphemeralParam()
	got := sys[0].CacheControl
	// both should be the same struct value
	if got != expected {
		t.Errorf("cache control mismatch: got %+v, want %+v", got, expected)
	}
}

func TestConvertMessages_UserPlainText(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleUser, Content: "hello"},
	}
	result, _ := convertMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].Role != ant.MessageParamRoleUser {
		t.Errorf("expected user role, got %s", result[0].Role)
	}
}

func TestConvertMessages_UserPlainTextWithCache(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleUser, Content: "hello", CacheControl: &llmrouter.CacheControl{Type: "ephemeral"}},
	}
	result, _ := convertMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	blocks := result[0].Content
	if len(blocks) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(blocks))
	}
	if blocks[0].OfText == nil {
		t.Fatal("expected text block")
	}
	if blocks[0].OfText.CacheControl != ant.NewCacheControlEphemeralParam() {
		t.Error("expected cache control to be set on text block")
	}
}

func TestConvertMessages_UserTextPart(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleUser, ContentParts: []llmrouter.ContentPart{
			{Type: "text", Text: "hello"},
		}},
	}
	result, _ := convertMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	blocks := result[0].Content
	if len(blocks) != 1 {
		t.Fatalf("expected 1 block, got %d", len(blocks))
	}
	if blocks[0].OfText == nil {
		t.Error("expected text block")
	}
}

func TestConvertMessages_UserTextPartWithCache(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleUser, ContentParts: []llmrouter.ContentPart{
			{Type: "text", Text: "cached text", CacheControl: &llmrouter.CacheControl{Type: "ephemeral"}},
		}},
	}
	result, _ := convertMessages(msgs)
	blocks := result[0].Content
	if blocks[0].OfText == nil {
		t.Fatal("expected text block")
	}
	if blocks[0].OfText.CacheControl != ant.NewCacheControlEphemeralParam() {
		t.Error("expected cache control")
	}
}

func TestConvertMessages_UserImagePart(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleUser, ContentParts: []llmrouter.ContentPart{
			{Type: "image_url", ImageURL: &llmrouter.ImageURL{Base64: "aGVsbG8=", MediaType: "image/png"}},
		}},
	}
	result, _ := convertMessages(msgs)
	blocks := result[0].Content
	if len(blocks) != 1 {
		t.Fatalf("expected 1 block, got %d", len(blocks))
	}
	if blocks[0].OfImage == nil {
		t.Error("expected image block")
	}
}

func TestConvertMessages_UserImagePartEmptyBase64Skipped(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleUser, ContentParts: []llmrouter.ContentPart{
			{Type: "text", Text: "hi"},
			{Type: "image_url", ImageURL: &llmrouter.ImageURL{Base64: ""}},
		}},
	}
	result, _ := convertMessages(msgs)
	// image with empty base64 is skipped; only text block remains
	blocks := result[0].Content
	if len(blocks) != 1 {
		t.Fatalf("expected 1 block (skipped empty image), got %d", len(blocks))
	}
}

func TestConvertMessages_UserDocumentPart(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleUser, ContentParts: []llmrouter.ContentPart{
			{Type: "document", Document: &llmrouter.Document{Base64: "cGRm", MediaType: "application/pdf"}},
		}},
	}
	result, _ := convertMessages(msgs)
	blocks := result[0].Content
	if len(blocks) != 1 {
		t.Fatalf("expected 1 block, got %d", len(blocks))
	}
	if blocks[0].OfDocument == nil {
		t.Error("expected document block")
	}
}

func TestConvertMessages_AssistantNoTools(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleAssistant, Content: "sure, I can help"},
	}
	result, _ := convertMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("expected 1, got %d", len(result))
	}
	if result[0].Role != ant.MessageParamRoleAssistant {
		t.Errorf("expected assistant, got %s", result[0].Role)
	}
	blocks := result[0].Content
	if blocks[0].OfText == nil {
		t.Error("expected text block")
	}
}

func TestConvertMessages_AssistantWithTools(t *testing.T) {
	msgs := []llmrouter.Message{
		{
			Role:    llmrouter.RoleAssistant,
			Content: "using tool",
			ToolCalls: []llmrouter.ToolCall{
				{
					ID:   "tu_abc",
					Type: "function",
					Function: llmrouter.FuncCall{
						Name:      "get_weather",
						Arguments: `{"location":"NYC"}`,
					},
				},
			},
		},
	}
	result, _ := convertMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("expected 1, got %d", len(result))
	}
	blocks := result[0].Content
	// content: text block + tool use block
	if len(blocks) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(blocks))
	}
	if blocks[1].OfToolUse == nil {
		t.Error("expected tool use block")
	}
	if blocks[1].OfToolUse.ID != "tu_abc" {
		t.Errorf("unexpected tool use ID: %s", blocks[1].OfToolUse.ID)
	}
}

func TestConvertMessages_AssistantToolsNoContent(t *testing.T) {
	msgs := []llmrouter.Message{
		{
			Role: llmrouter.RoleAssistant,
			ToolCalls: []llmrouter.ToolCall{
				{ID: "tu_1", Function: llmrouter.FuncCall{Name: "f", Arguments: "{}"}},
			},
		},
	}
	result, _ := convertMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("expected 1, got %d", len(result))
	}
	blocks := result[0].Content
	// no text content, only tool use
	if len(blocks) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(blocks))
	}
	if blocks[0].OfToolUse == nil {
		t.Error("expected tool use block")
	}
}

func TestConvertMessages_ToolResult(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleTool, Content: `{"temp":72}`, ToolCallID: "tu_abc"},
	}
	result, _ := convertMessages(msgs)
	if len(result) != 1 {
		t.Fatalf("expected 1, got %d", len(result))
	}
	if result[0].Role != ant.MessageParamRoleUser {
		t.Errorf("expected user (tool result), got %s", result[0].Role)
	}
	blocks := result[0].Content
	if blocks[0].OfToolResult == nil {
		t.Error("expected tool result block")
	}
	if blocks[0].OfToolResult.ToolUseID != "tu_abc" {
		t.Errorf("unexpected tool use ID: %s", blocks[0].OfToolResult.ToolUseID)
	}
}

// ---- convertTools ----

func TestConvertTools_Basic(t *testing.T) {
	raw := json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`)
	tools := []llmrouter.Tool{
		{
			Function: llmrouter.Function{
				Name:        "search",
				Description: "search the web",
				Parameters:  raw,
			},
		},
	}
	result := convertTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].OfTool == nil {
		t.Fatal("expected OfTool to be set")
	}
	if result[0].OfTool.Name != "search" {
		t.Errorf("expected name search, got %s", result[0].OfTool.Name)
	}
	if !result[0].OfTool.Description.Valid() || result[0].OfTool.Description.Value != "search the web" {
		t.Errorf("unexpected description: %v", result[0].OfTool.Description)
	}
}

func TestConvertTools_NoDescription(t *testing.T) {
	tools := []llmrouter.Tool{
		{Function: llmrouter.Function{Name: "noop"}},
	}
	result := convertTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].OfTool.Description.Valid() {
		t.Error("expected unset description for empty string")
	}
}

func TestConvertTools_NilParameters(t *testing.T) {
	tools := []llmrouter.Tool{
		{Function: llmrouter.Function{Name: "noop", Description: "does nothing"}},
	}
	result := convertTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1, got %d", len(result))
	}
}

// ---- convertToolChoice ----

func TestConvertToolChoice_Auto(t *testing.T) {
	tc := &llmrouter.ToolChoice{Type: "auto"}
	result := convertToolChoice(tc)
	if result.OfAuto == nil {
		t.Error("expected auto")
	}
}

func TestConvertToolChoice_None(t *testing.T) {
	tc := &llmrouter.ToolChoice{Type: "none"}
	result := convertToolChoice(tc)
	if result.OfAuto == nil {
		t.Error("expected auto (none maps to auto)")
	}
}

func TestConvertToolChoice_Required(t *testing.T) {
	tc := &llmrouter.ToolChoice{Type: "required"}
	result := convertToolChoice(tc)
	if result.OfAny == nil {
		t.Error("expected any for required")
	}
}

func TestConvertToolChoice_Any(t *testing.T) {
	tc := &llmrouter.ToolChoice{Type: "any"}
	result := convertToolChoice(tc)
	if result.OfAny == nil {
		t.Error("expected any")
	}
}

func TestConvertToolChoice_Function(t *testing.T) {
	tc := &llmrouter.ToolChoice{Type: "function", Function: &llmrouter.FuncRef{Name: "get_weather"}}
	result := convertToolChoice(tc)
	if result.OfTool == nil {
		t.Fatal("expected tool choice")
	}
	if result.OfTool.Name != "get_weather" {
		t.Errorf("expected get_weather, got %s", result.OfTool.Name)
	}
}

func TestConvertToolChoice_Nil(t *testing.T) {
	result := convertToolChoice(nil)
	_ = result // should not panic; returns zero value
}

// ---- convertToOpenAIResponse ----

func TestConvertToOpenAIResponse_TextOnly(t *testing.T) {
	msg := unmarshalMessage(t, `{
		"id": "msg_123",
		"type": "message",
		"model": "claude-sonnet-4-20250514",
		"role": "assistant",
		"content": [{"type": "text", "text": "hello"}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`)
	result := convertToOpenAIResponse(msg, "anthropic")
	if result.ID != "msg_123" {
		t.Errorf("unexpected ID: %s", result.ID)
	}
	if result.Provider != "anthropic" {
		t.Errorf("unexpected provider: %s", result.Provider)
	}
	if len(result.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(result.Choices))
	}
	if result.Choices[0].Message.Content != "hello" {
		t.Errorf("unexpected content: %s", result.Choices[0].Message.Content)
	}
	if result.Choices[0].FinishReason != "stop" {
		t.Errorf("unexpected finish reason: %s", result.Choices[0].FinishReason)
	}
	if result.Usage == nil {
		t.Fatal("expected usage")
	}
	if result.Usage.PromptTokens != 10 {
		t.Errorf("expected 10 prompt tokens, got %d", result.Usage.PromptTokens)
	}
}

func TestConvertToOpenAIResponse_ToolUse(t *testing.T) {
	msg := unmarshalMessage(t, `{
		"id": "msg_456",
		"type": "message",
		"model": "claude-sonnet-4-20250514",
		"role": "assistant",
		"content": [{"type": "tool_use", "id": "tu_xyz", "name": "get_weather", "input": {"location": "NYC"}}],
		"stop_reason": "tool_use",
		"usage": {"input_tokens": 20, "output_tokens": 10}
	}`)
	result := convertToOpenAIResponse(msg, "anthropic")
	if result.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("expected tool_calls, got %s", result.Choices[0].FinishReason)
	}
	if len(result.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result.Choices[0].Message.ToolCalls))
	}
	tc := result.Choices[0].Message.ToolCalls[0]
	if tc.ID != "tu_xyz" {
		t.Errorf("unexpected ID: %s", tc.ID)
	}
	if tc.Function.Name != "get_weather" {
		t.Errorf("unexpected name: %s", tc.Function.Name)
	}
}

func TestConvertToOpenAIResponse_MaxTokens(t *testing.T) {
	msg := unmarshalMessage(t, `{
		"id": "msg_789",
		"type": "message",
		"model": "claude-3-5-haiku-20241022",
		"role": "assistant",
		"content": [{"type": "text", "text": "truncated..."}],
		"stop_reason": "max_tokens",
		"usage": {"input_tokens": 5, "output_tokens": 100}
	}`)
	result := convertToOpenAIResponse(msg, "anthropic")
	if result.Choices[0].FinishReason != "length" {
		t.Errorf("expected length, got %s", result.Choices[0].FinishReason)
	}
}

// ---- wrapError ----

func TestWrapError_Nil(t *testing.T) {
	if wrapError(nil) != nil {
		t.Error("expected nil")
	}
}

func TestWrapError_Generic(t *testing.T) {
	err := errors.New("something went wrong")
	wrapped := wrapError(err)
	var apiErr *llmrouter.APIError
	if !errors.As(wrapped, &apiErr) {
		t.Fatal("expected APIError")
	}
	if apiErr.Provider != "anthropic" {
		t.Errorf("expected provider anthropic, got %s", apiErr.Provider)
	}
}

func TestWrapError_RateLimited(t *testing.T) {
	wrapped := wrapError(makeAntError(http.StatusTooManyRequests))
	if !errors.Is(wrapped, llmrouter.ErrRateLimited) {
		t.Errorf("expected ErrRateLimited, got %v", wrapped)
	}
}

func TestWrapError_AuthFailed_401(t *testing.T) {
	wrapped := wrapError(makeAntError(http.StatusUnauthorized))
	if !errors.Is(wrapped, llmrouter.ErrAuthFailed) {
		t.Errorf("expected ErrAuthFailed, got %v", wrapped)
	}
}

func TestWrapError_AuthFailed_403(t *testing.T) {
	wrapped := wrapError(makeAntError(http.StatusForbidden))
	if !errors.Is(wrapped, llmrouter.ErrAuthFailed) {
		t.Errorf("expected ErrAuthFailed, got %v", wrapped)
	}
}

func TestWrapError_InvalidRequest(t *testing.T) {
	wrapped := wrapError(makeAntError(http.StatusBadRequest))
	if !errors.Is(wrapped, llmrouter.ErrInvalidRequest) {
		t.Errorf("expected ErrInvalidRequest, got %v", wrapped)
	}
}

// ---- cache control on assistant / tool messages and tools ----

// cacheBreakpoints marshals v and returns the number of "cache_control" keys
// found in the wire JSON.
func cacheBreakpoints(t *testing.T, v any) int {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return strings.Count(string(b), `"cache_control":{"type":"ephemeral"}`)
}

func TestConvertMessages_AssistantTextWithCache(t *testing.T) {
	eph := &llmrouter.CacheControl{Type: "ephemeral"}
	msgs, _ := convertMessages([]llmrouter.Message{
		{Role: llmrouter.RoleUser, Content: "hi"},
		{Role: llmrouter.RoleAssistant, Content: "hello", CacheControl: eph},
	})
	if got := cacheBreakpoints(t, msgs[0]); got != 0 {
		t.Errorf("user message: expected 0 breakpoints, got %d", got)
	}
	if got := cacheBreakpoints(t, msgs[1]); got != 1 {
		t.Errorf("assistant message: expected 1 breakpoint, got %d", got)
	}
}

func TestConvertMessages_AssistantWithoutCache(t *testing.T) {
	msgs, _ := convertMessages([]llmrouter.Message{
		{Role: llmrouter.RoleAssistant, Content: "hello"},
	})
	if got := cacheBreakpoints(t, msgs); got != 0 {
		t.Errorf("expected 0 breakpoints, got %d", got)
	}
}

func TestConvertMessages_AssistantToolCallsCacheOnLastBlock(t *testing.T) {
	eph := &llmrouter.CacheControl{Type: "ephemeral"}
	msgs, _ := convertMessages([]llmrouter.Message{{
		Role:    llmrouter.RoleAssistant,
		Content: "checking",
		ToolCalls: []llmrouter.ToolCall{
			{ID: "t1", Function: llmrouter.FuncCall{Name: "a", Arguments: `{}`}},
			{ID: "t2", Function: llmrouter.FuncCall{Name: "b", Arguments: `{}`}},
		},
		CacheControl: eph,
	}})
	blocks := msgs[0].Content
	if len(blocks) != 3 {
		t.Fatalf("expected 3 blocks, got %d", len(blocks))
	}
	for i, want := range []int{0, 0, 1} {
		if got := cacheBreakpoints(t, blocks[i]); got != want {
			t.Errorf("block %d: expected %d breakpoints, got %d", i, want, got)
		}
	}
}

func TestConvertMessages_ToolResultWithCache(t *testing.T) {
	eph := &llmrouter.CacheControl{Type: "ephemeral"}
	msgs, _ := convertMessages([]llmrouter.Message{
		{Role: llmrouter.RoleTool, ToolCallID: "t1", Content: "r1"},
		{Role: llmrouter.RoleTool, ToolCallID: "t2", Content: "r2", CacheControl: eph},
	})
	if got := cacheBreakpoints(t, msgs[0]); got != 0 {
		t.Errorf("first tool result: expected 0 breakpoints, got %d", got)
	}
	if got := cacheBreakpoints(t, msgs[1]); got != 1 {
		t.Errorf("second tool result: expected 1 breakpoint, got %d", got)
	}
}

func TestConvertTools_CacheControl(t *testing.T) {
	tools := convertTools([]llmrouter.Tool{
		{Type: "function", Function: llmrouter.Function{Name: "a"}},
		{Type: "function", Function: llmrouter.Function{Name: "b"}, CacheControl: &llmrouter.CacheControl{Type: "ephemeral"}},
	})
	if got := cacheBreakpoints(t, tools[0]); got != 0 {
		t.Errorf("first tool: expected 0 breakpoints, got %d", got)
	}
	if got := cacheBreakpoints(t, tools[1]); got != 1 {
		t.Errorf("last tool: expected 1 breakpoint, got %d", got)
	}
}

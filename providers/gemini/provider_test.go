package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llmrouter "github.com/bluefunda/llmrouter"
	"google.golang.org/genai"
)

// fakeGemini serves canned Gemini API responses and records each request body.
type fakeGemini struct {
	server *httptest.Server
	bodies []map[string]any
}

// newFakeGemini streams chunks as server-sent events for streamGenerateContent, and
// answers generateContent with the last chunk as one JSON body. status != 200 returns errBody.
func newFakeGemini(t *testing.T, status int, errBody string, chunks ...string) *fakeGemini {
	t.Helper()
	f := &fakeGemini{}
	f.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		f.bodies = append(f.bodies, body)

		if status != http.StatusOK {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			_, _ = io.WriteString(w, errBody)
			return
		}
		if strings.Contains(r.URL.Path, ":streamGenerateContent") {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, c := range chunks {
				_, _ = fmt.Fprintf(w, "data: %s\r\n\r\n", c)
			}
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, chunks[len(chunks)-1])
	}))
	t.Cleanup(f.server.Close)
	return f
}

func (f *fakeGemini) provider(t *testing.T) *Provider {
	t.Helper()
	p, err := New(llmrouter.ProviderConfig{APIKey: "test-key", BaseURL: f.server.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func collect(t *testing.T, p *Provider, req *llmrouter.Request) []llmrouter.Event {
	t.Helper()
	stream, err := p.Stream(context.Background(), req)
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer func() { _ = stream.Close() }()
	var events []llmrouter.Event
	for stream.Next() {
		events = append(events, stream.Event())
	}
	return events
}

func userRequest(text string) *llmrouter.Request {
	return &llmrouter.Request{
		Model:    "gemini-3.5-flash-lite",
		Messages: []llmrouter.Message{{Role: llmrouter.RoleUser, Content: text}},
	}
}

// The deprecated SDK failed at the end of every Gemini 3 stream ("invalid character ']'"),
// which turned complete answers into errors. A normal stream must end in EventDone only.
func TestStream_TextAnswerEndsCleanly(t *testing.T) {
	f := newFakeGemini(t, http.StatusOK, "",
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello, "}]}}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"world."}]},"finishReason":"STOP"}],`+
			`"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4,"thoughtsTokenCount":6,"totalTokenCount":20,"cachedContentTokenCount":3}}`,
	)

	events := collect(t, f.provider(t), userRequest("Say hello"))

	var text strings.Builder
	var done *llmrouter.Response
	for _, ev := range events {
		switch ev.Type {
		case llmrouter.EventError:
			t.Fatalf("unexpected error event: %v", ev.Error)
		case llmrouter.EventContentDelta:
			text.WriteString(ev.Content)
		case llmrouter.EventDone:
			done = ev.Response
		}
	}
	if text.String() != "Hello, world." {
		t.Fatalf("streamed text = %q", text.String())
	}
	if done == nil {
		t.Fatal("no EventDone")
	}
	if got := done.Choices[0].FinishReason; got != "stop" {
		t.Errorf("finish reason = %q, want stop", got)
	}
	if got := done.Choices[0].Message.Content; got != "Hello, world." {
		t.Errorf("final content = %q", got)
	}
	u := done.Usage
	if u == nil || u.PromptTokens != 10 || u.CompletionTokens != 10 || u.TotalTokens != 20 || u.CachedPromptTokens != 3 {
		t.Errorf("usage = %+v, want prompt 10, completion 10 (4 answer + 6 thinking), total 20, cached 3", u)
	}
}

// Gemini 3 issues a thought signature with a function call and rejects the next request if it is
// not sent back. Two parallel calls to the same tool must also stay two distinct calls: the old
// provider used the function name as the ID, so they collided (#294 in cai-llm-router).
func TestStream_FunctionCallsKeepSignatureAndUniqueIDs(t *testing.T) {
	f := newFakeGemini(t, http.StatusOK, "",
		`{"candidates":[{"content":{"role":"model","parts":[`+
			`{"functionCall":{"name":"fetch","args":{"url":"https://a.example"}},"thoughtSignature":"c2lnLWE="},`+
			`{"functionCall":{"name":"fetch","args":{"url":"https://b.example"}}}`+
			`]},"finishReason":"STOP"}]}`,
	)

	events := collect(t, f.provider(t), userRequest("Fetch both"))

	var deltas []llmrouter.ToolCall
	var done *llmrouter.Response
	for _, ev := range events {
		switch ev.Type {
		case llmrouter.EventError:
			t.Fatalf("unexpected error event: %v", ev.Error)
		case llmrouter.EventToolCallDelta:
			deltas = append(deltas, ev.Delta.ToolCalls...)
		case llmrouter.EventDone:
			done = ev.Response
		}
	}
	if len(deltas) != 2 {
		t.Fatalf("got %d tool-call deltas, want 2", len(deltas))
	}
	if deltas[0].ID == deltas[1].ID {
		t.Errorf("parallel calls share ID %q", deltas[0].ID)
	}
	if *deltas[0].Index != 0 || *deltas[1].Index != 1 {
		t.Errorf("indexes = %d, %d; want 0, 1", *deltas[0].Index, *deltas[1].Index)
	}
	if string(deltas[0].ThoughtSignature) != "sig-a" {
		t.Errorf("first call signature = %q, want sig-a", deltas[0].ThoughtSignature)
	}
	if deltas[1].Function.Arguments != `{"url":"https://b.example"}` {
		t.Errorf("second call args = %s", deltas[1].Function.Arguments)
	}
	if done == nil || done.Choices[0].FinishReason != "tool_calls" || len(done.Choices[0].Message.ToolCalls) != 2 {
		t.Fatalf("EventDone = %+v, want finish tool_calls with 2 calls", done)
	}
}

// Thought parts must never be mixed into the answer text.
func TestStream_ThoughtPartsAreNotAnswerText(t *testing.T) {
	f := newFakeGemini(t, http.StatusOK, "",
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Let me think about this.","thought":true},{"text":"The answer is 42."}]},"finishReason":"STOP"}]}`,
	)

	events := collect(t, f.provider(t), userRequest("What is the answer?"))

	for _, ev := range events {
		if ev.Type == llmrouter.EventContentDelta && strings.Contains(ev.Content, "think") {
			t.Fatalf("thought leaked into answer: %q", ev.Content)
		}
		if ev.Type == llmrouter.EventDone && ev.Response.Choices[0].Message.Content != "The answer is 42." {
			t.Fatalf("final content = %q", ev.Response.Choices[0].Message.Content)
		}
	}
}

// A tool-loop follow-up: the request on the wire must replay the signature, group parallel
// results in one user turn, name a result whose message lacks Name, and pass the tool's JSON
// schema through untouched (the old converter dropped keywords such as "format").
func TestStream_FollowUpRequestOnTheWire(t *testing.T) {
	f := newFakeGemini(t, http.StatusOK, "",
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Done."}]},"finishReason":"STOP"}]}`,
	)
	idx0, idx1 := 0, 1
	req := &llmrouter.Request{
		Model: "gemini-3.5-flash-lite",
		Messages: []llmrouter.Message{
			{Role: llmrouter.RoleSystem, Content: "Be brief."},
			{Role: llmrouter.RoleUser, Content: "Fetch both"},
			{Role: llmrouter.RoleAssistant, ToolCalls: []llmrouter.ToolCall{
				{ID: "call_0_fetch", Type: "function", Index: &idx0, ThoughtSignature: []byte("sig-a"),
					Function: llmrouter.FuncCall{Name: "fetch", Arguments: `{"url":"https://a.example"}`}},
				{ID: "call_1_fetch", Type: "function", Index: &idx1,
					Function: llmrouter.FuncCall{Name: "fetch", Arguments: `{"url":"https://b.example"}`}},
			}},
			{Role: llmrouter.RoleTool, ToolCallID: "call_0_fetch", Name: "fetch", Content: "page A"},
			{Role: llmrouter.RoleTool, ToolCallID: "call_1_fetch", Content: `{"status":"ok"}`},
		},
		Tools: []llmrouter.Tool{{Type: "function", Function: llmrouter.Function{
			Name:        "fetch",
			Description: "Fetch a URL",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"url":{"type":"string","format":"uri"}},"required":["url"]}`),
		}}},
	}

	for _, ev := range collect(t, f.provider(t), req) {
		if ev.Type == llmrouter.EventError {
			t.Fatalf("unexpected error event: %v", ev.Error)
		}
	}
	if len(f.bodies) != 1 {
		t.Fatalf("got %d requests, want 1", len(f.bodies))
	}
	body := f.bodies[0]

	if got := jsonPath(body, "systemInstruction", "parts", 0, "text"); got != "Be brief." {
		t.Errorf("system instruction = %v", got)
	}
	contents, _ := body["contents"].([]any)
	if len(contents) != 3 {
		t.Fatalf("got %d contents, want 3 (user, model calls, grouped results): %v", len(contents), contents)
	}
	if got := jsonPath(body, "contents", 1, "parts", 0, "thoughtSignature"); got != "c2lnLWE=" {
		t.Errorf("replayed signature = %v, want base64 of sig-a", got)
	}
	if got := jsonPath(body, "contents", 1, "parts", 1, "functionCall", "args", "url"); got != "https://b.example" {
		t.Errorf("second call url = %v", got)
	}
	results, _ := jsonPath(body, "contents", 2, "parts").([]any)
	if len(results) != 2 {
		t.Fatalf("parallel results not grouped in one turn: %v", contents[2])
	}
	if got := jsonPath(body, "contents", 2, "parts", 0, "functionResponse", "response", "output"); got != "page A" {
		t.Errorf("text result = %v, want under \"output\"", got)
	}
	if got := jsonPath(body, "contents", 2, "parts", 1, "functionResponse", "name"); got != "fetch" {
		t.Errorf("unnamed result resolved to %v, want fetch", got)
	}
	if got := jsonPath(body, "contents", 2, "parts", 1, "functionResponse", "response", "status"); got != "ok" {
		t.Errorf("JSON result = %v, want object passed through", got)
	}
	schema := jsonPath(body, "tools", 0, "functionDeclarations", 0, "parametersJsonSchema", "properties", "url", "format")
	if schema != "uri" {
		t.Errorf("tool schema keyword lost: format = %v", schema)
	}
}

// Without declared tools (a tool-free retry after a rejected tool call), Gemini rejects
// function-call history, so earlier tool exchanges are replayed as text.
func TestConvertMessages_WithoutToolsReplaysToolExchangesAsText(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleUser, Content: "Fetch it"},
		{Role: llmrouter.RoleAssistant, ToolCalls: []llmrouter.ToolCall{
			{ID: "call_0_fetch", Function: llmrouter.FuncCall{Name: "fetch", Arguments: `{"url":"https://a.example"}`}},
		}},
		{Role: llmrouter.RoleTool, ToolCallID: "call_0_fetch", Content: "page A"},
	}

	contents, _ := convertMessages(msgs, false)

	for _, c := range contents {
		for _, p := range c.Parts {
			if p.FunctionCall != nil || p.FunctionResponse != nil {
				t.Fatalf("function part sent without declared tools: %+v", p)
			}
		}
	}
	if len(contents) != 3 {
		t.Fatalf("got %d contents, want 3", len(contents))
	}
	if !strings.Contains(contents[1].Parts[0].Text, "fetch") || !strings.Contains(contents[2].Parts[0].Text, "page A") {
		t.Errorf("tool exchange not replayed as text: %q / %q", contents[1].Parts[0].Text, contents[2].Parts[0].Text)
	}
}

// Gemini's real error must reach the caller (the old SDK hid it behind a JSON parse error).
func TestStream_APIErrorIsReported(t *testing.T) {
	f := newFakeGemini(t, http.StatusNotFound,
		`{"error":{"code":404,"message":"models/gemini-x is not found","status":"NOT_FOUND"}}`)

	stream, err := f.provider(t).Stream(context.Background(), userRequest("hi"))
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer func() { _ = stream.Close() }()
	for stream.Next() {
		t.Fatalf("unexpected event before the error: %+v", stream.Event())
	}

	// StreamResult surfaces an EventError through Err(), ending the stream.
	var apiErr *llmrouter.APIError
	if !errors.As(stream.Err(), &apiErr) || apiErr.StatusCode != http.StatusNotFound ||
		!strings.Contains(apiErr.Message, "not found") {
		t.Fatalf("error = %#v, want gemini APIError 404 with the server message", stream.Err())
	}
}

func TestComplete_ReturnsAnswer(t *testing.T) {
	f := newFakeGemini(t, http.StatusOK, "",
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Hi there."}]},"finishReason":"MAX_TOKENS"}]}`,
	)

	resp, err := f.provider(t).Complete(context.Background(), userRequest("hi"))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Choices[0].Message.Content != "Hi there." || resp.Choices[0].FinishReason != "length" {
		t.Fatalf("response = %+v", resp.Choices[0])
	}
}

func TestFinishReason_Mapping(t *testing.T) {
	cases := map[string]string{
		"STOP":                    "stop",
		"":                        "stop",
		"MAX_TOKENS":              "length",
		"SAFETY":                  "content_filter",
		"MALFORMED_FUNCTION_CALL": "malformed_function_call",
	}
	for reason, want := range cases {
		a := newAccumulator()
		a.finish = genaiFinish(reason)
		if got := a.finishReason(); got != want {
			t.Errorf("%q → %q, want %q", reason, got, want)
		}
	}
}

// jsonPath walks a decoded JSON value by object keys (string) and array indexes (int).
func jsonPath(v any, path ...any) any {
	for _, step := range path {
		switch key := step.(type) {
		case string:
			m, ok := v.(map[string]any)
			if !ok {
				return nil
			}
			v = m[key]
		case int:
			a, ok := v.([]any)
			if !ok || key >= len(a) {
				return nil
			}
			v = a[key]
		}
	}
	return v
}

func genaiFinish(s string) genai.FinishReason { return genai.FinishReason(s) }

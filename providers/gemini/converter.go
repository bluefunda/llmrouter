package gemini

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	llmrouter "github.com/bluefunda/llmrouter"
	"google.golang.org/genai"
)

// convertMessages converts llmrouter messages to Gemini contents plus the system instruction.
//
// toolsDeclared reports whether the request declares tools. Gemini rejects function-call and
// function-response history when no tools are declared, so without tools (e.g. a tool-free
// retry after a rejected tool call) earlier tool exchanges are replayed as plain text instead.
func convertMessages(msgs []llmrouter.Message, toolsDeclared bool) ([]*genai.Content, *genai.Content) {
	var contents []*genai.Content
	var system []string
	toolNames := map[string]string{} // tool-call ID → function name, for results missing Name

	for _, msg := range msgs {
		switch msg.Role {
		case llmrouter.RoleSystem:
			if strings.TrimSpace(msg.Content) != "" {
				system = append(system, msg.Content)
			}

		case llmrouter.RoleUser:
			if parts := buildUserParts(msg); len(parts) > 0 {
				contents = append(contents, &genai.Content{Role: genai.RoleUser, Parts: parts})
			}

		case llmrouter.RoleAssistant:
			var parts []*genai.Part
			if msg.Content != "" {
				parts = append(parts, &genai.Part{Text: msg.Content})
			}
			for _, tc := range msg.ToolCalls {
				toolNames[tc.ID] = tc.Function.Name
				if toolsDeclared {
					parts = append(parts, &genai.Part{
						FunctionCall: &genai.FunctionCall{
							Name: tc.Function.Name,
							Args: parseArgs(tc.Function.Arguments),
						},
						// Gemini 3 rejects function-call history without the signature it issued.
						ThoughtSignature: tc.ThoughtSignature,
					})
				} else {
					parts = append(parts, &genai.Part{
						Text: fmt.Sprintf("[Called tool %s with arguments %s]", tc.Function.Name, tc.Function.Arguments),
					})
				}
			}
			if len(parts) > 0 {
				contents = append(contents, &genai.Content{Role: genai.RoleModel, Parts: parts})
			}

		case llmrouter.RoleTool:
			name := msg.Name
			if name == "" {
				name = toolNames[msg.ToolCallID]
			}
			if !toolsDeclared {
				contents = append(contents, &genai.Content{
					Role:  genai.RoleUser,
					Parts: []*genai.Part{{Text: fmt.Sprintf("[Result of tool %s]\n%s", name, msg.Content)}},
				})
				continue
			}
			part := &genai.Part{FunctionResponse: &genai.FunctionResponse{
				Name:     name,
				Response: toolResult(msg.Content),
			}}
			// Results of parallel calls go back together in one user turn.
			if last := lastContent(contents); last != nil && isFunctionResponseTurn(last) {
				last.Parts = append(last.Parts, part)
				continue
			}
			contents = append(contents, &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{part}})
		}
	}

	var instruction *genai.Content
	if len(system) > 0 {
		instruction = &genai.Content{Parts: []*genai.Part{{Text: strings.Join(system, "\n\n")}}}
	}
	return contents, instruction
}

func lastContent(contents []*genai.Content) *genai.Content {
	if len(contents) == 0 {
		return nil
	}
	return contents[len(contents)-1]
}

func isFunctionResponseTurn(c *genai.Content) bool {
	if c.Role != genai.RoleUser || len(c.Parts) == 0 {
		return false
	}
	for _, p := range c.Parts {
		if p.FunctionResponse == nil {
			return false
		}
	}
	return true
}

// parseArgs decodes a tool call's JSON arguments; malformed or empty arguments become {}.
func parseArgs(arguments string) map[string]any {
	args := map[string]any{}
	if strings.TrimSpace(arguments) != "" {
		_ = json.Unmarshal([]byte(arguments), &args)
	}
	return args
}

// toolResult wraps a tool's output as a Gemini function response: a JSON object is passed
// through, anything else goes under "output".
func toolResult(content string) map[string]any {
	var obj map[string]any
	if err := json.Unmarshal([]byte(content), &obj); err == nil && obj != nil {
		return obj
	}
	return map[string]any{"output": content}
}

// buildUserParts converts a user message (text-only or multimodal) to Gemini parts.
func buildUserParts(msg llmrouter.Message) []*genai.Part {
	if len(msg.ContentParts) == 0 {
		if msg.Content == "" {
			return nil
		}
		return []*genai.Part{{Text: msg.Content}}
	}

	parts := make([]*genai.Part, 0, len(msg.ContentParts))
	for _, p := range msg.ContentParts {
		switch p.Type {
		case "text":
			if p.Text != "" {
				parts = append(parts, &genai.Part{Text: p.Text})
			}
		case "image_url":
			if p.ImageURL != nil && p.ImageURL.Base64 != "" {
				if data, err := base64.StdEncoding.DecodeString(p.ImageURL.Base64); err == nil {
					parts = append(parts, &genai.Part{InlineData: &genai.Blob{MIMEType: p.ImageURL.MediaType, Data: data}})
				}
			}
		}
	}
	return parts
}

// convertTools converts llmrouter tools to Gemini function declarations. Parameters are passed
// as raw JSON Schema (ParametersJsonSchema), so nothing in the schema is lost in conversion.
func convertTools(tools []llmrouter.Tool) []*genai.Tool {
	decls := make([]*genai.FunctionDeclaration, 0, len(tools))
	for _, tool := range tools {
		decl := &genai.FunctionDeclaration{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
		}
		if len(tool.Function.Parameters) > 0 {
			var schema any
			if err := json.Unmarshal(tool.Function.Parameters, &schema); err == nil && schema != nil {
				decl.ParametersJsonSchema = schema
			}
		}
		decls = append(decls, decl)
	}
	return []*genai.Tool{{FunctionDeclarations: decls}}
}

// accumulator turns Gemini response chunks into llmrouter events and the final response.
type accumulator struct {
	content   strings.Builder
	toolCalls []llmrouter.ToolCall
	finish    genai.FinishReason
	meta      *genai.GenerateContentResponseUsageMetadata
}

func newAccumulator() *accumulator { return &accumulator{} }

// add processes one response chunk, emitting events for text and tool calls.
func (a *accumulator) add(resp *genai.GenerateContentResponse, emit func(llmrouter.Event)) error {
	if resp == nil {
		return nil
	}
	if resp.UsageMetadata != nil {
		a.meta = resp.UsageMetadata
	}
	if len(resp.Candidates) == 0 {
		if fb := resp.PromptFeedback; fb != nil && fb.BlockReason != "" {
			return &llmrouter.APIError{
				Provider: "gemini",
				Type:     "prompt_blocked",
				Message:  fmt.Sprintf("prompt blocked: %s %s", fb.BlockReason, fb.BlockReasonMessage),
			}
		}
		return nil
	}

	cand := resp.Candidates[0]
	if cand.FinishReason != "" {
		a.finish = cand.FinishReason
	}
	if cand.Content == nil {
		return nil
	}
	for _, part := range cand.Content.Parts {
		switch {
		case part == nil:
		case part.FunctionCall != nil:
			tc := a.addToolCall(part)
			emit(llmrouter.Event{
				Type:  llmrouter.EventToolCallDelta,
				Delta: &llmrouter.Delta{ToolCalls: []llmrouter.ToolCall{tc}},
			})
		case part.Thought:
			// Thought summaries are not requested; skip any so they never leak into the answer.
		case part.Text != "":
			a.content.WriteString(part.Text)
			emit(llmrouter.Event{Type: llmrouter.EventContentDelta, Content: part.Text})
		}
	}
	return nil
}

// addToolCall records a function call. Each call gets an ID that is unique across the whole
// conversation even when Gemini omits one: callers key per-call state on it (cai-llm-router's
// thinking-card steps), and a per-response counter like "call_0_fetch" repeats on every tool
// round, so a later round's call overwrote the earlier one.
func (a *accumulator) addToolCall(part *genai.Part) llmrouter.ToolCall {
	fc := part.FunctionCall
	index := len(a.toolCalls)
	id := fc.ID
	if id == "" {
		id = newCallID()
	}
	args := "{}"
	if len(fc.Args) > 0 {
		if b, err := json.Marshal(fc.Args); err == nil {
			args = string(b)
		}
	}
	tc := llmrouter.ToolCall{
		ID:               id,
		Type:             "function",
		Function:         llmrouter.FuncCall{Name: fc.Name, Arguments: args},
		Index:            &index,
		ThoughtSignature: part.ThoughtSignature,
	}
	a.toolCalls = append(a.toolCalls, tc)
	return tc
}

// finishReason maps Gemini's finish reason to the OpenAI-compatible value.
func (a *accumulator) finishReason() string {
	if len(a.toolCalls) > 0 {
		return "tool_calls"
	}
	switch a.finish {
	case "", genai.FinishReasonStop:
		return "stop"
	case genai.FinishReasonMaxTokens:
		return "length"
	case genai.FinishReasonSafety, genai.FinishReasonRecitation, genai.FinishReasonBlocklist,
		genai.FinishReasonProhibitedContent, genai.FinishReasonSPII, genai.FinishReasonImageSafety:
		return "content_filter"
	default:
		// e.g. MALFORMED_FUNCTION_CALL — surfaced as-is so callers can see what happened.
		return strings.ToLower(string(a.finish))
	}
}

// usage reports token usage; thinking tokens count as completion tokens (they are billed so).
func (a *accumulator) usage() *llmrouter.Usage {
	if a.meta == nil {
		return nil
	}
	return &llmrouter.Usage{
		PromptTokens:       int(a.meta.PromptTokenCount),
		CompletionTokens:   int(a.meta.CandidatesTokenCount + a.meta.ThoughtsTokenCount),
		TotalTokens:        int(a.meta.TotalTokenCount),
		CachedPromptTokens: int(a.meta.CachedContentTokenCount),
	}
}

// wrapError wraps Gemini errors, keeping the HTTP status when the SDK reports one.
func wrapError(err error) error {
	if err == nil {
		return nil
	}
	apiErr := &llmrouter.APIError{Provider: "gemini", Message: err.Error(), Err: err}
	var gErr genai.APIError
	if errors.As(err, &gErr) {
		apiErr.StatusCode = gErr.Code
		apiErr.Type = gErr.Status
		apiErr.Message = gErr.Message
	}
	return apiErr
}

// newCallID returns a random tool-call ID ("call_" + 24 hex chars), the same shape as OpenAI's.
func newCallID() string {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("call_%x", time.Now().UnixNano())
	}
	return "call_" + hex.EncodeToString(b[:])
}

package anthropic

import (
	"encoding/json"
	"net/http"

	llmrouter "github.com/bluefunda/llmrouter"
	"github.com/anthropics/anthropic-sdk-go"
)

// convertMessages converts llmrouter messages to Anthropic format.
// Returns the non-system messages and the system blocks (with optional cache control).
func convertMessages(msgs []llmrouter.Message) ([]anthropic.MessageParam, []anthropic.TextBlockParam) {
	var systemBlocks []anthropic.TextBlockParam
	var messages []anthropic.MessageParam

	for _, msg := range msgs {
		switch msg.Role {
		case llmrouter.RoleSystem:
			block := anthropic.TextBlockParam{Text: msg.Content}
			if msg.CacheControl != nil {
				block.CacheControl = anthropic.NewCacheControlEphemeralParam()
			}
			systemBlocks = append(systemBlocks, block)

		case llmrouter.RoleUser:
			if len(msg.ContentParts) > 0 {
				blocks := make([]anthropic.ContentBlockParamUnion, 0, len(msg.ContentParts))
				for _, p := range msg.ContentParts {
					switch p.Type {
					case "text":
						tb := &anthropic.TextBlockParam{Text: p.Text}
						if p.CacheControl != nil {
							tb.CacheControl = anthropic.NewCacheControlEphemeralParam()
						}
						blocks = append(blocks, anthropic.ContentBlockParamUnion{OfText: tb})
					case "image_url":
						if p.ImageURL != nil && p.ImageURL.Base64 != "" {
							block := anthropic.NewImageBlockBase64(p.ImageURL.MediaType, p.ImageURL.Base64)
							if p.CacheControl != nil && block.OfImage != nil {
								block.OfImage.CacheControl = anthropic.NewCacheControlEphemeralParam()
							}
							blocks = append(blocks, block)
						}
					case "document":
						if p.Document != nil && p.Document.Base64 != "" {
							doc := &anthropic.DocumentBlockParam{
								Source: anthropic.DocumentBlockParamSourceUnion{
									OfBase64: &anthropic.Base64PDFSourceParam{Data: p.Document.Base64},
								},
							}
							if p.CacheControl != nil {
								doc.CacheControl = anthropic.NewCacheControlEphemeralParam()
							}
							blocks = append(blocks, anthropic.ContentBlockParamUnion{OfDocument: doc})
						}
					}
				}
				messages = append(messages, anthropic.NewUserMessage(blocks...))
			} else {
				var block anthropic.ContentBlockParamUnion
				if msg.CacheControl != nil {
					block = anthropic.ContentBlockParamUnion{OfText: &anthropic.TextBlockParam{
						Text:         msg.Content,
						CacheControl: anthropic.NewCacheControlEphemeralParam(),
					}}
				} else {
					block = anthropic.NewTextBlock(msg.Content)
				}
				messages = append(messages, anthropic.NewUserMessage(block))
			}

		case llmrouter.RoleAssistant:
			if len(msg.ToolCalls) > 0 {
				blocks := []anthropic.ContentBlockParamUnion{}
				if msg.Content != "" {
					blocks = append(blocks, anthropic.NewTextBlock(msg.Content))
				}
				for _, tc := range msg.ToolCalls {
					// Anthropic rejects a null `input` ("Input should be an
					// object") — arguments come back empty for no-argument
					// tool calls (e.g. github-mcp's get_me), which leaves the
					// unmarshal target nil unless it starts as {}.
					input := map[string]interface{}{}
					if tc.Function.Arguments != "" {
						_ = json.Unmarshal([]byte(tc.Function.Arguments), &input)
					}
					blocks = append(blocks, anthropic.NewToolUseBlock(tc.ID, input, tc.Function.Name))
				}
				messages = append(messages, anthropic.NewAssistantMessage(blocks...))
			} else {
				messages = append(messages, anthropic.NewAssistantMessage(
					anthropic.NewTextBlock(msg.Content),
				))
			}

		case llmrouter.RoleTool:
			messages = append(messages, anthropic.NewUserMessage(
				anthropic.NewToolResultBlock(msg.ToolCallID, msg.Content, false),
			))
		}
	}

	return messages, systemBlocks
}

// convertTools converts llmrouter tools to Anthropic format
func convertTools(tools []llmrouter.Tool) []anthropic.ToolUnionParam {
	result := make([]anthropic.ToolUnionParam, len(tools))

	for i, tool := range tools {
		schema := anthropic.ToolInputSchemaParam{}
		if tool.Function.Parameters != nil {
			var params map[string]interface{}
			if err := json.Unmarshal(tool.Function.Parameters, &params); err == nil && params != nil {
				schema.Properties = params["properties"]
				if required, ok := params["required"].([]interface{}); ok {
					for _, r := range required {
						if s, ok := r.(string); ok {
							schema.Required = append(schema.Required, s)
						}
					}
				}
			}
		}

		t := anthropic.ToolUnionParamOfTool(schema, tool.Function.Name)
		if tool.Function.Description != "" {
			t.OfTool.Description = anthropic.String(tool.Function.Description)
		}
		result[i] = t
	}

	return result
}

// convertToolChoice converts llmrouter tool choice to Anthropic format
func convertToolChoice(tc *llmrouter.ToolChoice) anthropic.ToolChoiceUnionParam {
	if tc == nil {
		return anthropic.ToolChoiceUnionParam{}
	}

	switch tc.Type {
	case "auto", "none":
		return anthropic.ToolChoiceUnionParam{OfAuto: &anthropic.ToolChoiceAutoParam{}}
	case "required", "any":
		return anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{}}
	case "function":
		if tc.Function != nil {
			return anthropic.ToolChoiceParamOfTool(tc.Function.Name)
		}
	}

	return anthropic.ToolChoiceUnionParam{}
}

// convertToOpenAIResponse converts Anthropic response to OpenAI-compatible format
func convertToOpenAIResponse(msg *anthropic.Message, provider string) *llmrouter.Response {
	var content string
	var toolCalls []llmrouter.ToolCall

	for _, block := range msg.Content {
		switch b := block.AsAny().(type) {
		case anthropic.TextBlock:
			content += b.Text
		case anthropic.ToolUseBlock:
			args, _ := json.Marshal(b.Input)
			toolCalls = append(toolCalls, llmrouter.ToolCall{
				ID:   b.ID,
				Type: "function",
				Function: llmrouter.FuncCall{
					Name:      b.Name,
					Arguments: string(args),
				},
			})
		}
	}

	finishReason := "stop"
	switch msg.StopReason {
	case anthropic.StopReasonToolUse:
		finishReason = "tool_calls"
	case anthropic.StopReasonMaxTokens:
		finishReason = "length"
	case anthropic.StopReasonStopSequence:
		finishReason = "stop"
	}

	return &llmrouter.Response{
		ID:       msg.ID,
		Object:   "chat.completion",
		Model:    string(msg.Model),
		Provider: provider,
		Choices: []llmrouter.Choice{
			{
				Index: 0,
				Message: &llmrouter.Message{
					Role:      llmrouter.RoleAssistant,
					Content:   content,
					ToolCalls: toolCalls,
				},
				FinishReason: finishReason,
			},
		},
		Usage: &llmrouter.Usage{
			PromptTokens:        int(msg.Usage.InputTokens),
			CompletionTokens:    int(msg.Usage.OutputTokens),
			TotalTokens:         int(msg.Usage.InputTokens + msg.Usage.OutputTokens),
			CachedPromptTokens:  int(msg.Usage.CacheReadInputTokens),
			CacheCreationTokens: int(msg.Usage.CacheCreationInputTokens),
		},
	}
}

// wrapError wraps Anthropic errors
func wrapError(err error) error {
	if err == nil {
		return nil
	}

	apiErr := &llmrouter.APIError{
		Provider: "anthropic",
		Message:  err.Error(),
		Err:      err,
	}

	// Check for Anthropic-specific error types
	if antErr, ok := err.(*anthropic.Error); ok {
		apiErr.StatusCode = antErr.StatusCode

		switch antErr.StatusCode {
		case http.StatusUnauthorized, http.StatusForbidden:
			apiErr.Err = llmrouter.ErrAuthFailed
		case http.StatusTooManyRequests:
			apiErr.Err = llmrouter.ErrRateLimited
		case http.StatusBadRequest:
			apiErr.Err = llmrouter.ErrInvalidRequest
		}
	}

	return apiErr
}

// Package gemini implements the llmrouter.Provider interface for Google
// Gemini models using the official Google Gen AI Go SDK (google.golang.org/genai).
//
// Create a provider from the GEMINI_API_KEY environment variable:
//
//	p, err := gemini.NewFromEnv()
//
// Or with an explicit key:
//
//	p, err := gemini.New(llmrouter.ProviderConfig{APIKey: "..."})
//
// The provider supports streaming, tool calling (including Gemini 3 thought
// signatures, carried on llmrouter.ToolCall.ThoughtSignature), and text and
// image inputs.
package gemini

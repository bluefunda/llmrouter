# Changelog

## [0.6.0](https://github.com/bluefunda/llmrouter/compare/v0.5.1...v0.6.0) (2026-09-13)


### Features

* surface real provider reasoning as Thinking* stream events ([#113](https://github.com/bluefunda/llmrouter/issues/113)) ([3652d24](https://github.com/bluefunda/llmrouter/commit/3652d24bff088158d5fb83553fe6c7d497547165))

## [0.5.1](https://github.com/bluefunda/llmrouter/compare/v0.5.0...v0.5.1) (2026-08-05)


### Bug Fixes

* **gemini:** use user role for function response history entries ([#101](https://github.com/bluefunda/llmrouter/issues/101)) ([ee251cb](https://github.com/bluefunda/llmrouter/commit/ee251cbe57ef7053954ee20532374d3c9236cc45))

## [0.5.0](https://github.com/bluefunda/llmrouter/compare/v0.4.4...v0.5.0) (2026-07-26)


### Features

* add optional RoutingPolicy layer with heuristic policies ([#96](https://github.com/bluefunda/llmrouter/issues/96)) ([fdf433b](https://github.com/bluefunda/llmrouter/commit/fdf433bf3eb242bd4e480b79985d18a8fcc1c078))

## [0.4.4](https://github.com/bluefunda/llmrouter/compare/v0.4.3...v0.4.4) (2026-07-12)


### Bug Fixes

* remove docker deployment from release workflow ([#88](https://github.com/bluefunda/llmrouter/issues/88)) ([6ac5264](https://github.com/bluefunda/llmrouter/commit/6ac52649c53f387772fca4267a7dea4c2d9d0234))

## [0.4.3](https://github.com/bluefunda/llmrouter/compare/v0.4.2...v0.4.3) (2026-07-12)


### Bug Fixes

* surface DeepSeek context-cache hit tokens in Usage.CachedPromptTokens ([#85](https://github.com/bluefunda/llmrouter/issues/85)) ([8b9614a](https://github.com/bluefunda/llmrouter/commit/8b9614a3df3c6c1e946494ccd1c5546e22d9a8dc))

## [0.4.2](https://github.com/bluefunda/llmrouter/compare/v0.4.1...v0.4.2) (2026-05-31)


### Bug Fixes

* enable stream_options.include_usage for OpenAI streaming ([#68](https://github.com/bluefunda/llmrouter/issues/68)) ([390e3f0](https://github.com/bluefunda/llmrouter/commit/390e3f099d85a37b7bfe39cf6f0fd946966360f9))

## [0.4.1](https://github.com/bluefunda/llmrouter/compare/v0.4.0...v0.4.1) (2026-05-31)


### Features

* USD cost calculation and observability hooks middleware ([#64](https://github.com/bluefunda/llmrouter/issues/64)) ([e67c097](https://github.com/bluefunda/llmrouter/commit/e67c097099d18c10fd851a55e561db4b461ded7d))

## [0.4.0](https://github.com/bluefunda/llmrouter/compare/v0.3.1...v0.4.0) (2026-05-22)


### ⚠ BREAKING CHANGES

* replace Middleware interface with MiddlewareFunc function type ([#51](https://github.com/bluefunda/llmrouter/issues/51))
* remove dead API surface (ToolsProvider, MaxRetries, CB name param) ([#45](https://github.com/bluefunda/llmrouter/issues/45))

### Features

* replace Middleware interface with MiddlewareFunc function type ([#51](https://github.com/bluefunda/llmrouter/issues/51)) ([0efb1da](https://github.com/bluefunda/llmrouter/commit/0efb1da4e1d941748b86adde82c5c629bb230359))


### Bug Fixes

* buildParams single return value; comprehensive doc sweep ([#53](https://github.com/bluefunda/llmrouter/issues/53)) ([e0693b8](https://github.com/bluefunda/llmrouter/commit/e0693b8beaf3822ab736ce5f2b806c5bd813d81d))
* circuit breaker Allow/Record, Router.Close, defensive Models copy ([#49](https://github.com/bluefunda/llmrouter/issues/49)) ([d1d133a](https://github.com/bluefunda/llmrouter/commit/d1d133afa54211bd301509535627e5b880713d19))
* **release:** prevent v1.0.0 bump, honour config file bump rules ([#54](https://github.com/bluefunda/llmrouter/issues/54)) ([bc47533](https://github.com/bluefunda/llmrouter/commit/bc475331385e5330658249c82f69888274e6c146))


### Miscellaneous Chores

* remove dead API surface (ToolsProvider, MaxRetries, CB name param) ([#45](https://github.com/bluefunda/llmrouter/issues/45)) ([f3fb501](https://github.com/bluefunda/llmrouter/commit/f3fb5013138e4308ede63f917c13dbae9f22ce0b))

## [0.3.0](https://github.com/bluefunda/llmrouter/compare/v0.2.1...v0.3.0) (2026-05-21)


### Features

* v0.3.1 — StreamResult API, stdlib circuit breaker, lean core ([#31](https://github.com/bluefunda/llmrouter/issues/31)) ([d518314](https://github.com/bluefunda/llmrouter/commit/d51831447f5837c261e71ac0c1b99ff73ac8cc85))


### Bug Fixes

* **lint:** suppress errcheck on defer stream.Close() ([#32](https://github.com/bluefunda/llmrouter/issues/32)) ([fee4192](https://github.com/bluefunda/llmrouter/commit/fee419297360442399405291ca51aacfdfadc0a7))
* pin golangci-lint to v2.12.2 and add toolchain directive ([#29](https://github.com/bluefunda/llmrouter/issues/29)) ([12af2cb](https://github.com/bluefunda/llmrouter/commit/12af2cb0e301a1983bb0bf05478fe4ef952dc23c))

## [0.2.1](https://github.com/bluefunda/llmrouter/compare/v0.2.0...v0.2.1) (2026-05-21)


### Bug Fixes

* wire fallback routing in Complete and Route ([73d3b98](https://github.com/bluefunda/llmrouter/commit/73d3b98e28b00f36b5a62fbeb80e0b61b3ac034d))

## [0.2.0](https://github.com/bluefunda/llm-router/compare/v0.1.3...v0.2.0) (2026-05-12)


### Features

* add prompt caching support across all providers ([#18](https://github.com/bluefunda/llm-router/issues/18)) ([5064425](https://github.com/bluefunda/llm-router/commit/50644251c4649f47878db8003f616fc0448d3a63))

## [0.1.3](https://github.com/bluefunda/llm-router/compare/v0.1.2...v0.1.3) (2026-05-12)


### Bug Fixes

* bump vulnerable indirect deps to resolve 8 Dependabot alerts ([#15](https://github.com/bluefunda/llm-router/issues/15)) ([1d348a3](https://github.com/bluefunda/llm-router/commit/1d348a30058f07d2f30d551c76d7dc26d592900d))

## [0.1.2](https://github.com/bluefunda/llm-router/compare/v0.1.1...v0.1.2) (2026-03-03)


### Bug Fixes

* add StringContentOnly option for OpenAI-compatible APIs ([#8](https://github.com/bluefunda/llm-router/issues/8)) ([f23e435](https://github.com/bluefunda/llm-router/commit/f23e4359e894b0d22bcd75d8dd3df9a42ed562bc))

## [0.1.1](https://github.com/bluefunda/llm-router/compare/v0.1.0...v0.1.1) (2026-02-25)


### Bug Fixes

* check json.Encode error return in test (errcheck lint) ([2b822b2](https://github.com/bluefunda/llm-router/commit/2b822b21bcab138db0a9f8159b1008e3594be566))
* default sarvam preset to sarvam-30b, add all model tiers ([10897a4](https://github.com/bluefunda/llm-router/commit/10897a4986b6d843f97e4cf4a17594b7f492ddc8))

## [0.1.0](https://github.com/bluefunda/llm-router/compare/v0.0.0...v0.1.0) (2026-02-25)


### Features

* add CustomHeaders support and Sarvam AI preset ([373cdde](https://github.com/bluefunda/llm-router/commit/373cddef2f6a6d518ddfc1efefdde92c60c37ed6))

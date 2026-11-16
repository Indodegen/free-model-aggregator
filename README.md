# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Indodegen/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Indodegen/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

A live free-model aggregator: one server, two API wires, real free models behind every answer.

## Run it

```sh
python3 main.py
```

Point a client at it (OpenAI Chat Completions at `POST /v1/chat/completions`,
Anthropic Messages at `POST /v1/messages`, model lists at `GET /v1/models`),
and every completion it answers comes from a real free OpenRouter model
dialled through the gateway on `OPENROUTER_API_KEY`. `python3 main.py --list`
prints the catalog of the fourteen free models it can serve.

## How it picks a model

For each prompt it ranks the free models that meet the prompt's requirements
(tools, vision, JSON mode, context) by the house's published quality rating
and dials the best one. A gateway 429 or 5xx cools that model down for the
rest of the trial and the prompt fails over to the next best model, at most
four models per prompt; a 401 or 400 stops the prompt at once. Refusals it
recovered from are reported in the `x-makefun-upstream-failures` header.

---

Entered on [make.fun](https://make.fun/?ref=&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards

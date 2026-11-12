# ADR-002: Rules-first parsing, LLM as fallback

- **Status:** accepted
- **Context:** Atlas AI can run as a pure tool-calling LLM, but small local models are slow (seconds per tool round)
  and inexact on free-form requests. We fine-tuned Ministral 3 8B (LoRA) on 484 validated tool-use examples and evaled
  on a sealed 150-task set: the fine-tune completed 57% of tasks, few-shot prompting 46–48%, untuned 15%. Most remaining
  failures were rules the model could not see.
- **Decision:** `librarian.py` is a deterministic parser for the common requests (filters, zoom, "books like X",
  authors, topics, top rated, recommendation wizard). Only unparsed requests go to the LLM loop in `agent.py`. The
  agent guide (`docs/AGENT_GUIDE.md`) is the prompt-side half of the same investment: clear instructions and tools
  that apply the rules themselves.
- **Consequences:** common commands are instant, exact and work with no LLM installed (rules mode); the LLM only
  handles genuinely free-form requests. Trade-off: new phrasings must be added to the parser or the guide to be
  reliable — the eval loop in CI (roadmap item 5) tracks this.

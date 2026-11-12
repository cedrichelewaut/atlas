# ADR-001: One shared FlexViz state for the human and the AI

- **Status:** accepted
- **Context:** chat assistants usually answer *beside* the UI. The user asks "books like Dracula" and cannot see what
  the assistant picked unless it describes it in words. Separate agent state also means the agent cannot see what the
  user is looking at.
- **Decision:** the agent mutates the same FlexViz state the browser renders, through tools that are also exposed over
  HTTP (`GET /api/tools`, `POST /api/tools/<name>`) for external agents. The server pushes each change to all browsers
  as an SSE "apply" patch; the browser applies it and POSTs an acknowledgement, so the agent's answer refers to what is
  actually on screen. Highlights (orange = looked up, yellow = recommended) are part of that shared state.
- **Consequences:** answers are grounded in the visible view; external agents can drive the map without new code; the
  ack protocol must be kept in sync (tests in `test_routes.py` cover it). Trade-off: the browser and server must agree
  on the patch format — acceptable, since both live in this repo.

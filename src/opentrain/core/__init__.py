"""The OpenTrain harness core: event-sourced state, the agent engine, tools, permissions, agents.

Shape follows opencode: one server owns everything; the TUI, the web UI, ``opentrain run`` and the
SDK are clients. What the agent works on is a model, not a codebase.
"""

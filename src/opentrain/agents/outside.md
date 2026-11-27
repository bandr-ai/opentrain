---
description: A coding agent outside OpenTrain (Claude Code, Codex, any MCP client) using OpenTrain's tools through `opentrain mcp`. Its own permission prompts are the approval; the Trainfile still binds every call.
mode: primary
hidden: true
permissions:
  "*": allow
  "read": deny
  "glob": deny
  "grep": deny
  "shell": deny
  "edit": deny
  "write": deny
  "webfetch": deny
  "propose": deny
  "question": deny
  "subagent": deny
  "todo": deny
  "heldout.read": deny
---
Calls from an outside agent. This prompt is not sent to any model; the outside agent reads the
tool descriptions and the `opentrain` MCP server's instructions.

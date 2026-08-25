# Codex skills/plugins/MCP guide for this report workflow

## What official Codex supports
- Skills are reusable folders containing a required `SKILL.md` plus optional scripts/references/assets.
- Codex CLI/IDE can invoke skills explicitly with `$skill-name` or `/skills`; Codex may also select them from the skill description.
- The built-in `$skill-creator` can scaffold and validate a reusable skill.
- Codex CLI has `/plugins` for a plugin browser. Plugins can contain skills, connectors, and MCP tools. Start a new session after installing a plugin.
- `AGENTS.md` is automatically read by Codex from the project hierarchy and is the right place for durable repository rules.
- MCP is useful for third-party tools/context; local Codex also already has shell/file access, so ordinary PDF validation is better implemented as repo scripts rather than depending on a remote MCP service.

## Paper-writing finding
The official materials reviewed do not expose a dedicated general-purpose “academic paper / IEEE LaTeX writer” skill that should be trusted as a drop-in solution. The recommended route is:
1. use `$skill-creator` / local skills;
2. install/use a Documents plugin if available for document workflows;
3. use the GitHub plugin or `gh` CLI for repo operations;
4. encode CVio-specific academic and validation rules in the two local skills included in this bundle.

## Recommended explicit skill calls
- `$cvio-academic-report` - rebuild/write the thesis from evidence using native XeLaTeX.
- `$cvio-pdf-preflight` - enforce render/preflight/layout/font/page-number gates.

## MCP recommendation
MCP is optional for this task. Do not add an MCP server merely to run local commands. If the agent needs a connected external source, configure a trusted MCP server in `~/.codex/config.toml` or project `.codex/config.toml`. For GitHub delivery, `gh` CLI is simpler and fully auditable.

# AGENTS Policy

## Skill Compliance (Mandatory)
- If the user names a skill (for example `$skill-name` or plain text), the agent MUST use that skill in the same turn.
- If the task clearly matches an available skill description, the agent MUST use that skill even if not explicitly named.
- If multiple skills apply, the agent MUST use the minimal set that fully covers the task.
- The agent MUST open each selected skill `SKILL.md` and follow its workflow before taking substantial actions.
- The agent MUST NOT claim a skill is active unless it has actually loaded the relevant `SKILL.md`.

## Turn-Level Reporting (Mandatory)
- In every substantive turn, the agent MUST state one of:
- `Skills used: <comma-separated skill names>`
- `Skills used: none (reason: <short reason>)`
- If a named skill cannot be used (missing file/path/tooling), the agent MUST say so briefly and continue with the best fallback.

## Conflict Resolution
- If this file conflicts with weaker style preferences, this file takes precedence for skill usage and reporting.
- If another instruction would prevent using a required skill, the agent MUST report the conflict explicitly and ask for direction.

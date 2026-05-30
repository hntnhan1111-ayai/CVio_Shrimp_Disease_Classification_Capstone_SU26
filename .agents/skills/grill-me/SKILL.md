---
name: grill-me
description: Interview the user relentlessly about a plan, design, refactor, experiment, or research-code implementation until goals, constraints, risks, dependencies, and success criteria are clear. Use when the user says "grill me", "interview me", "stress-test this plan", "ask questions before coding", or when requirements are ambiguous.
---

Interview the user relentlessly about the proposed plan or design until reaching shared understanding.

Rules:
- Ask one question at a time.
- For each question, provide your recommended answer.
- Walk down each branch of the design tree.
- Resolve dependencies between decisions before moving forward.
- Challenge weak assumptions, unsupported claims, vague success criteria, risky implementation choices, and hidden constraints.
- If a question can be answered by exploring the codebase, inspect the codebase instead of asking the user.
- Do not implement code during this skill unless the user explicitly approves the plan.
- When enough information is collected, produce a PLAN.md-style summary with:
  - goal
  - context inspected
  - decisions made
  - unresolved questions
  - implementation phases
  - validation commands
  - stop conditions

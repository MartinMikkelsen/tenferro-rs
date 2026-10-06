# Review decision records

tenferro keeps three different kinds of long-lived engineering notes:

- `docs/plans/`: historical implementation plans and prior discussions. These
  files can become stale and should not be updated to match current code.
- `docs/worklogs/`: curated records of completed nontrivial work. These explain
  what the implementer read, what reference code informed the change, which
  design was chosen, which alternatives were rejected or deferred, and what
  risks remain.
- `docs/design/`: durable design intent that future implementation and review
  should continue to follow.

## PR workflow

Nontrivial refactors, cleanup streams, AI-assisted implementation, and PRs that
make explicit tradeoffs should add or update a work log. The PR body should
link that work log so reviewers can evaluate the diff against the actual design
context instead of inferring intent from the changed files alone.

If the PR establishes a design rule that should outlive the session, the same
PR should also update `docs/design/`. A work log may explain why a decision was
made in one session, but design docs are the source reviewers should use when
the decision applies to future code.

## Review workflow

Before challenging a nontrivial abstraction, module split, macro/codegen choice,
public API boundary, or deferral, reviewers should read the linked work log and
any linked design docs. A review can still disagree with the decision, but it
should address the recorded rationale directly.

This keeps review feedback aligned with repository intent and avoids repeatedly
re-litigating decisions that were already made explicit.

## Recorded decisions

Short API-contract decisions that have no larger design document of their own.

### Unused placeholder in `compile_with_input_specs` (#1968)

`GraphCompiler::compile_with_input_specs` rejects a declared placeholder that
the output does not depend on (`Validation` / `InvalidArgument`, phase
`Compile`). Before, such a placeholder was silently dropped from the program
inputs, so a tensor passed for it at run time was either rejected as an extra
input or, when the input counts happened to match, bound to a retained constant
input in its place. Keeping unused placeholders as ignored program inputs was
rejected because it needs a semantic input that no operation reads; rejecting
the declaration is the fail-fast contract. Callers drop the binding, as the
kdv-pinn `pde::tests` do for the constant third derivative of `x^3`.

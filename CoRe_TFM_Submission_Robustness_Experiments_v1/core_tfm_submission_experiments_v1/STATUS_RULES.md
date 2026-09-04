# Evidence Status Rules

Use these labels exactly in your notes.

## FROZEN_COMPLETED
Historical evidence already completed and audited.

## NEW_COMPLETE
A new experiment for which:
- the requested execution finished,
- expected artifacts exist,
- the completion gate passed,
- checksums were generated.

## OPERATIONALLY_UNSUPPORTED
A pre-specified run that could not execute because of a documented implementation,
license, memory, model, or runtime constraint. Keep the failure log.

## PARTIAL_DO_NOT_CLAIM
A run directory exists, but the completion gate has not passed.

## PLANNED_NOT_RUN
Configuration exists but no successful evidence package exists.

Never convert `PARTIAL_DO_NOT_CLAIM` or `PLANNED_NOT_RUN` into a numerical paper
claim.

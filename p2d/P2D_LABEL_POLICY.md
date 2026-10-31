# Cadence Flash p2d label policy

## Precedence: stated precedence vs content
Each example has a **content level** (from the message content) and an optional **stated precedence** (FLASH / IMMEDIATE / PRIORITY / ROUTINE or an abbreviation of one).

1. No stated precedence: the label is the content level.
2. Stated and content are within **one level** of each other: the **stated** precedence wins. This is the originator's call.
   Example: `PRIORITY SITREP: all quiet, PACE green` -> SITREP / **Priority** (content Routine, stated Priority).
3. Stated and content are **two or more levels** apart: the **content** wins. This is the p2c decoy rule; every p2c decoy pair
   (Flash+ROUTINE/PRIORITY, Immediate+ROUTINE, Priority+FLASH, Routine+FLASH/IMMEDIATE) is 2+ levels apart, so p2c labels are unchanged.
   Example: `ROUTINE TIC grid NV 4512 6789` -> SPOTREP / **Flash**.
4. Flash content is never lowered. `IMMEDIATE` stated on Flash content stays Flash (life and safety floor).
5. Routine admin is never labeled above Priority. Type calls that would produce admin/Immediate or admin/Flash are not generated, so there are no new type x precedence cells.

Consistency check against p2c: p2c had no one-level conflicts, so rule 2 is new but contradicts no existing label.

## Content level for orders and planning messages: the time-to-effect rule
This matches the p2 FRAGO labels: "effective immediately / attack imminent" = Flash, "NLT 1800L" (today) = Immediate, "tomorrow" = Priority, "next week" = Routine.
- Flash: effective now, or an enemy attack is imminent or underway.
- Immediate: effect the same day, within about 12 hours.
- Priority: effect in about 12 to 96 hours (tomorrow, 48 hours, 72 hours).
- Routine: effect in more than 96 hours, or no suspense.
Running estimates use severity: a shortfall that blocks an operation within hours = Immediate, a projected shortfall in 48 to 72 hours = Priority,
within tolerance = Routine. Only the S2 imminent-attack estimate is Flash.

## Type mapping for FM 5-0 products
- WARNORD, OPORD paragraph excerpts, FRAGO -> `FRAGO/order`
- Running estimate updates (S1, S2, S4, S6, FSO) -> `SITREP` (a staff status and assessment report)
- Mission analysis, COA, decision, and wargame briefs -> `other`
- Short type calls -> the stated type keyword (SITREP, SPOTREP, ...), even when the content mentions another type
  (for example, `SITREP: MEDEVAC complete ...` -> SITREP).

## Known label noise in the existing p2 FM 5-0 test set (113 WARNORDs)
Precedence there cannot be recovered from the content. `NLT 0600 tomorrow` is labeled Immediate 15 times and Priority 14 times,
and `in 72 hours` is labeled Immediate 13, Priority 10, and Routine 6 times. Under this policy, both are Priority.
The frozen eval keeps those rows unchanged as `fm50_p2orig`, so the history stays comparable. Use `fm50_p2d` (policy-consistent, held-out templates) for FM 5-0 progress.
Relabeling the original 113 rows is a maintainer decision.

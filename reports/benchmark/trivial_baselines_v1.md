# Trivial-policy baselines for A, B (D's is already reported in the paper)

## A (evidence grounding), N=48, gold distribution: {'ESCALATE': 42, 'COMMIT': 6}

- always-COMMIT TDA: 0.125
- always-ESCALATE TDA: 0.875
- always-ABORT TDA: 0.0

GSR baseline is trivially 0 for every fixed policy: none of these deliver or use
evidence, so the grounding contract is never satisfied regardless of the terminal
action -- a fixed policy cannot achieve nonzero GSR.

## B (evidence acquisition), N=66 (STOP=21, ACQUIRE+available=33, ACQUIRE+unavailable=12)

- never-acquire + always-COMMIT, B_AGS: 0.3182
- always-COMMIT (regardless of acquisition), TDA_post: 0.8182

## D (relational physical grounding) -- already in the paper, Section 4.2

- always-ESCALATE terminal-decision accuracy: 0.8

## For the main table / results section

**Every evaluated model's TDA (62.8-68.8%) is BELOW A's always-ESCALATE baseline (87.5%).** The 42/48 ESCALATE gold imbalance means a label-agnostic constant policy beats every model on TDA alone -- this is exactly why TDA is not the headline metric and why GSR (which no fixed policy can score above 0 on, since none of them deliver or use evidence) is reported as primary in the main table. Cite this baseline explicitly wherever TDA is shown, so a raw percentage is never read as 'good' without this anchor.

B_AGS should be read against 31.8% (never-acquire + always-COMMIT) as its floor, not 0 -- every model in the main table exceeds this floor.

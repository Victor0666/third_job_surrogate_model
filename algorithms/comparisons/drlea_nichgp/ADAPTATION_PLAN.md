# Adaptation plan and implemented stages

## Scope

The current project is the only source for workflows, Poisson arrivals,
deadline cache, cloud/edge topology, triangular fuzzy VM capability/bandwidth,
three timelines, SPECpower integration, discrete events and final metrics.
The upstream dataset generator, XML reader, server/device classes, mobile
devices, local-execution action and deadline estimator are excluded.

## Implemented stages

1. **RA:** stable FCFS selects one global-ready task. A flat masked Dueling
   Double DQN selects a global VM. Busy VM queueing is disabled by default.
2. **Niching GP:** the validation-selected RA is frozen. Training seeds provide
   global-ready situations, with configurable threshold (formal default 6).
   Clearing, tournament, elitism and a four-rule archive use the common tuple.
3. **SA:** frozen RA plus four frozen rules. A second Dueling Double DQN has
   exactly four actions; the selected rule ranks all global-ready tasks.
4. **Evaluation:** frozen RA/rules/SA run on disjoint test seeds. Per-workflow
   and per-seed JSON/CSV retain worst-seed fields.

The comparison never creates a Host Agent and never adopts the primary
Manager–Host–VM hierarchy. There is no LLM or safety-RL value/controller.

## Validation and budgets

Train seeds generate replay and GP situations. Validation seeds alone select
RA checkpoints, GP individuals/archive, SA checkpoints and, when reward
variants are compared, the formal reward mode. Test seeds are read only by
final evaluation. Manifests record episodes, GP population/generations,
seeds, source hash, invalid/failure counts and wall time. Checkpoints record
the PyTorch device-independent state; equal-budget experiments should also
match validation calls, search candidates, workflow counts and CPU/GPU wall
time against the primary method's manifest.

The smoke profile is not a scientific experiment: RA episodes 2, GP population
8, GP generations 2, SA episodes 2, workflows 3.

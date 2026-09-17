# PD3QN Upstream Mapping

- Original state: global VM observation with an action-legality mask.
- Action: FCFS chooses the task and Double-Dueling DQN selects one legal global VM.
- Network: prioritized-replay Double-Dueling DQN using the existing project agent.
- Training flow: replay-based temporal-difference updates; validation every 25 episodes selects the best checkpoint only.
- Current environment adaptation: the shared fuzzy environment supplies deterministic workloads, DDL data, global VM observations, masks, and final fuzzy metrics. The FCFS task order and PD3QN action structure are unchanged.

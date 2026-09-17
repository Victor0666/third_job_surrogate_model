# IRWS Upstream Mapping

- Original state: ready-task descriptors plus a global resource state used by the assignment policy.
- Action: learned ready-task ordering followed by one legal global VM action.
- Network: an eight-feature task ranker and masked PPO VM actor-critic.
- Training flow: on-policy episode collection; validation every 25 episodes selects the best checkpoint only.
- Current environment adaptation: the shared fuzzy environment supplies ready-task features, legal VM masks, deterministic scenario instances, DDL outcomes, and final fuzzy metrics. The IRWS task/routing structure is unchanged.

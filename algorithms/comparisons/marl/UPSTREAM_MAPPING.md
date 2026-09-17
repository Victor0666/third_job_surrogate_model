# MARL Upstream Mapping

- Original state: global Host state for the Host actor-critic and Host-local VM state for the shared VM actor with Host-specific critics.
- Action: FCFS chooses the task, then the Host policy chooses a legal Host and the VM policy chooses a legal local VM.
- Network: masked Host A2C plus a shared VM actor and one critic per Host.
- Training flow: assignment transitions update the Host policy and the selected Host critic; validation every 25 episodes selects the best checkpoint only.
- Current environment adaptation: fuzzy DDL/resource data and legal masks come from the shared environment. Pending VM transitions preserve the same Host-local state domain for state and next_state. Task order and Host/VM action hierarchy are unchanged.

# Upstream DRL-EA/Niching-GP mapping

Source archive: `NichingGP_for_fog_computing(1.3）.zip`. The original source is
kept outside this project; this file is the traceable mapping record.

| Upstream file | New module | Treatment |
|---|---|---|
| `D3QN.py` | `d3qn.py` | Retained dueling streams, Double DQN, online/target networks, soft update and epsilon-greedy; added legal masks. |
| `buffer.py` | `replay_buffer.py` | Retained replay sampling; added current/next masks and schema checks. |
| `Training_R.py` | `train_ra.py`, `routing_agent.py` | Replaced upstream simulator calls with global FCFS plus current CEWS adapter. |
| `NichingGP.py` | `gp_primitives.py`, `niching_gp.py` | Retained arithmetic/protected division/min/max, tournament, elitism, clearing and four-rule archive; replaced weighted tardiness with the common tuple. |
| `getDecisionSituations.py` | `decision_situations.py` | Replaced server-local queues with global-ready sets and frozen RA. |
| `getPriority.py` | `gp_features.py` | Replaced upstream terminals with the specified versioned 14 terminals. |
| `Training_S.py` | `train_sa.py`, `sequencing_agent.py` | Retained four-action D3QN; actions select global-ready rules. |
| `obs.py` | `features.py` | Replaced mobile/server observations with normalized current-project schemas. |
| `reward.py` | `rewards.py` | Replaced modal/local estimates with fuzzy risk finish and incremental fuzzy energy. |
| `CreateDataset.py`, `ReadXML.py`, `Server.py`, `Workflow.py`, `Event.py`, `SimulationEnvironment.py`, `Simulator.py` | `env_adapter.py` | Not ported; current CEWS simulator is reused. |
| `RoutingAction_with_deadline.py`, `SequencingAction_with_deadline.py` | none | Upstream handcrafted local/mobile actions are excluded. |

No upstream data file is copied or read at runtime.

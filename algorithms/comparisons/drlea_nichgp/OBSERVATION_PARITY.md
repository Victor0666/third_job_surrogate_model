# Observation and information parity

The adapter exposes the scheduler's raw problem state: arrived DAGs, ready
tasks, clock, deterministic workflow deadlines, VM availability, Host/VM
topology, triangular fuzzy capability/bandwidth, read-only
finish/communication/energy predictions, upward rank and remaining work. It
does not expose future arrivals or outcomes, test aggregates, primary-agent Q
values, actions or checkpoints.

| Feature group | Current environment source | Primary-agent raw access | Future/test information |
|---|---|---|---|
| task MI/input/output | current arrived task arrays | yes | no |
| ready wait/current time | ready timestamp/current clock | yes | no |
| fuzzy slack/risk finish | public three-timeline predictors | yes | prediction only |
| upward rank/remaining work | public DAG helpers | yes | DAG structure only |
| VM legality/availability | feasible VM + current availability | yes | no |
| execution/communication | public modal duration components | yes | no |
| uncertainty | triangular duration standard deviation | yes | no |
| incremental fuzzy energy | public SPECpower energy predictor | yes | prediction only |
| Host utilization/type | current busy ratio/cloud-edge category | yes | no |
| workflow completion | finished/total tasks of arrived DAG | yes | no |
| GP aggregates | same fields over the current ready set | yes | no |

IDs are used only for stable ordering/tie-breaking and are not continuous
features. VM-count scenarios train separate RA networks. A final instance
fingerprint covers config, seed, arrival sequence, fuzzy resources, selected
DAX sequence and deadlines.

Thus both methods share instances, dynamics, action feasibility, public
information, target direction and metrics. State encoding and shaping may
differ without granting extra raw information.

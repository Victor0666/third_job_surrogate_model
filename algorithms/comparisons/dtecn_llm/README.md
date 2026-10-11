# DTECN-LLM：MADDPG 引导的在线上下文资源决策

对应 Wu et al., *Large Language Model-Based Task Offloading and Resource
Allocation for Digital Twin Edge Computing Networks*, TMC 2026,
DOI `10.1109/TMC.2026.3664866`。上游：
https://github.com/qiongwu86/-LLM-Based-Task-Offloading-and-Resource-Allocation-for-DTECN

本方法只与 LLM-Safe-HRL 比较。它独立实现 MADDPG，不调用项目其他对比策略，
不使用 SeEvo、CMA-ES、surrogate、Q_c、拉格朗日更新或安全 shield。
共享 fuzzy_common 仅作为模拟器/协议/指标基础设施，不复用其他算法策略。

## 原文保留与必要适配

原文既没有 Host 选择也没有 VM 选择；它输出连续卸载比例和 CPU 份额。
本项目的任务不可拆分、VM 能力固定，因此固定完整卸载，动作改为长度等于 VM
数的 `[0,1]` 连续偏好向量。在空闲且资源适配的 VM 中取最大偏好，平局取稳定
索引最小者。Host 从 VM 归属推导。每次只有一次联合资源决策。
这个投影不是原文连续控制问题的等价变换，也不是连续比例的严格复现。

FCFS 按当前 ready time、task ID 排序，完全沿用环境的默认阶段顺序。
策略不得改变任务顺序、选择忙 VM 排队或获得未来工作流。
通信为 `combined_input_compute_output_v1`，所有时间、模糊参数、DAX、到达过程、
Host SPECpower 能耗、DDL 缓存与 LLM-Safe-HRL 共用。

原文车辆智能体映射为工作流智能体，固定 50 个槽位。只有当前 FCFS 任务所属
工作流执行动作。未到达槽位观察和动作均为零，完成槽位终止 bootstrap。
Actor 的连续偏好及执行索引同时记录。Critic 输入联合状态/连续动作，其他 Actor
更新时不参与梯度。保留目标 Actor/Critic、联合 replay、确定性策略梯度和软更新。
本地观察含到达工作流的活动标识、当前 owner、剩余工作量、已过去时延、家族、
当前任务计算量/输入输出/子截止期、五类队列与 VM pc/bw/busy 状态。
LLM 接收当前 owner 的资源观察、队列/合法候选，以及所有已到达工作流的本地状态，
保留原文LLM使用全局状态、Actor使用本地状态的区别；不接收未来信息或最优动作。

网络遵循作者发布代码中的 ReLU 隐藏层、tanh Actor 输出，未采用正文描述的
softmax。经用户确认，Actor 与 Critic 均采用两层隐藏层512→256；Actor接VM偏好输出并映射到 `[0,1]`，Critic接标量Q输出。这是项目网络容量适配，不是作者原始64/64/6与64/64/1结构。
使用 float32，修复上游原地修改 Critic 动作输入，并增加终止/活动槽位校验。
配置保留 gamma=0.95、tau=0.01、Actor lr=5e-4、Critic lr=1e-3、batch=64、
replay=200000、epsilon/noise 初值0.1及原衰减规则。训练至少 50000 次分配，
跨过预算后完成当前 episode；验证每跨过100分配在完整 episode 边界执行。
不按原车辆实验的100时间槽强制截断 DAG。训练预算不是600 episodes。

## Lyapunov 与反馈

队列按 CyberShake/Epigenomics/Ligo/Montage/Sipht 家族归类。
Q 是所有已到达任务的未完成 CPU 工作量（包含未满足依赖和正在计算的剩余部分）。
从共享 assignment history 的 input-arrival 开始计算 CPU 服务，至 compute-end
停止；输入和输出通信均不扣减 CPU 工作量。所有 CPU 服务截断在已观察时钟。
每次动作至下次决策（最后一次至 episode 完成）检查 `Q_next = Q + A - S`。
同一时刻的多个分配有零长度反馈区间，不虚构未来 CPU 服务。

Eq.24 适配目标：

```
J = beta * dot(Q, A - S) + observed_modal_energy_J - system_QoS
```

Q/A/S 单位 MI，能耗单位 J，beta=1。不再分别缩放三项；原来的缩放改变了权衡，不能称作等价单位换算。这里保留目标公式结构，不声称与车辆连续卸载问题等价。
能耗由公共功率积分函数计算，仅裁剪已观察 modal 记录，不新增能耗系数。
QoS=1-(完成时刻或当前已过去时刻-arrival)/(workflow_DDL-arrival)，
系统 QoS 对已到达工作流取均值。在线反馈为 modal 已发生值，最终成绩仍是项目
共用三时间线模糊指标，不把 modal 目标伪装成最终 fuzzy 目标。

MADDPG 奖励保留 Eq.27 的 QoS 减未使用资源结构：
`r_i = QoS_i - eta * unused_CPU_capacity_fraction`，eta=1。
原频率剩余量转换成容量比例以统一单位；只计算实际 CPU 区间占用。
J 与 r 分别记录，绝不将奖励偷换为 -J。LLM 提示词包含 J、队列与示例反馈；
案例不再含improved_objective标记：不同状态/区间的J降低不能证明动作改善。保留原始反馈和在线追加，不新增搜索。
重新校验队列守恒和实际二次漂移，不沿用论文 Eq.20 的错误不等式，
也不声称有限 episode 或近似 LLM 决策具有长期稳定性证明。

## 案例与在线调用

初始案例只能由训练种子1–5、协议允许的训练场景回放已选 MADDPG 得到。
包含状态、连续动作、执行 VM、Eq.27 奖励、已发生队列反馈、J、来源和种子。
在全部训练案例中固定等距选取最多8条，不按奖励挑最好、不做最近邻检索。
提示词保留初始示例和最近8条本 episode 的在线反馈。
每个验证/测试 seed 重置在线库，禁止跨 seed 更新和测试后调参；不更新神经网络。
上下文使用固定48000字符预算，先丢最早在线例，再丢后部初始例，至少留一条；
字符预算只是保守上限，服务返回超上下文错误会明确停止而非静默截断状态。

默认使用原文 `meta/llama-3.1-8b-instruct`，temperature=0.2、top_p=0.7。
已添加 GPT6.1（`gpt-6.1-sol`）与当前千问 Max（`qwen3.8-max`）调用适配。
GPT6.1 使用 `reasoning_effort` 与 `max_completion_tokens`，不发送temperature/top_p。
Qwen3.8-Max 默认 medium reasoning、enable_thinking=true、preserve_thinking=false，
与主方法现行请求设置一致。两者使用流式响应，只取最终content，忽略reasoning_content，
采集末尾usage并校验finish_reason；输出截断不会被误判为合法完成。
普通Llama仍使用原采样设置。`max_tokens` 对推理模型是包含思考的总输出预算，
不要当作仅JSON长度。默认4096保留原配置；同模型正式比较应显式对齐预算。
此次接口测试使用模拟SDK流，不代表在线账号已授权或服务已验证。

同模型实验指定精确ID，所有stage使用同一组选项，结果自动隔离：

```powershell
# 先设置OPENAI_API_KEY；API凭据与Codex登录分开。
python -m algorithms.comparisons.dtecn_llm.run train --scenario SS --ddl T --model gpt-6.1-sol --reasoning-effort medium --max-tokens 32768 --device cuda
python -m algorithms.comparisons.dtecn_llm.run collect --scenario SS --ddl T --model gpt-6.1-sol --reasoning-effort medium --max-tokens 32768 --device cuda
python -m algorithms.comparisons.dtecn_llm.run validate --scenario SS --ddl T --model gpt-6.1-sol --reasoning-effort medium --max-tokens 32768
python -m algorithms.comparisons.dtecn_llm.run evaluate --scenario SS --ddl T --model gpt-6.1-sol --reasoning-effort medium --max-tokens 32768
# 千问设置QWEN_API_KEY，并在上述全部命令中把模型改为qwen3.8-max。
```

32768只是示例，不声称主方法已使用此上限。主方法与本方法须记录实际相同模型、
推理设置、服务endpoint及预算；不得将两种模型的结果混入一组比较。
新调用代码/config已改变工件身份，旧checkpoint/案例不能静默复用。
使用独立客户端以避免修改主算法全局客户端。GPT读取OPENAI_API_KEY/OPENAI_BASE_URL；千问读取QWEN_API_KEY/QWEN_API_BASE_URL（默认北京DashScope兼容端点）；Llama读取CUSTOM_API_KEY/CUSTOM_API_BASE_URL（默认localhost:8000/v1，未设置custom key时为EMPTY）。密钥不写入工件。记录去除凭据和query的endpoint、SDK版本、服务返回model及system_fingerprint（若提供）；模型别名和权重版本的对应仍须部署者确认。
模型/endpoint的实际部署应由实验者保证；manifest记录模型与参数，需另外披露服务
版本与硬件。不会自动下载/安装模型，也不会自动启动付费调用。
每次最多一请求，输出JSON `{"resource_preferences":[...]}`，超时60秒。
仅一个合法 VM 时直接执行。无多轮优化、修复对话或未来 rollout。
临时服务失败/非法输出使用最小合法索引，并明确记录回退次数和原因；
认证、模型/接口配置错误直接停止。SDK自动重试关闭。

## 使用

从仓库根目录执行；先完成训练与案例冻结，再显式执行在线验证/最终测试：

```powershell
python -m algorithms.comparisons.dtecn_llm.run train --scenario SS --ddl T --device cuda
python -m algorithms.comparisons.dtecn_llm.run collect --scenario SS --ddl T --device cuda
$env:CUSTOM_API_BASE_URL = 'http://localhost:8000/v1/'
python -m algorithms.comparisons.dtecn_llm.run validate --scenario SS --ddl T
python -m algorithms.comparisons.dtecn_llm.run evaluate --scenario SS --ddl T
```

Single源场景为SS/SM/SL，对同资源S/M/L任务尺度评估。Multi使用
`--protocol multi --resource-scale S`（或M/L），训练和测试均限定对应资源的三个场景。
所有stage必须使用同一config、optimizer seed、DDL缓存覆盖。
支持`--deadline-cache SCENARIO=PATH`复用项目缓存校验规则。
验证种子101–103只用于选择MADDPG checkpoint；最终种子201–230从未参与训练
或初始案例收集。ICL参数冻结，在线验证不搜索提示词或重新挑选神经网络。

输出：`out/comparisons/dtecn_llm/combined_v1/artifacts_v3/<single|multi>/<source|scale>/<ddl>/s<seed>/`。
包含训练/验证日志、MADDPG checkpoint、案例JSONL、manifest、ICL逐seed指标、
汇总CSV和每次决策反馈JSONL。绑定源文件、共享输入、配置、提示词和工件hash；
不接受旧/混合物理模型。已有正式结果不覆盖；失败评估的部分trace保留供诊断。
神经checkpoint用于案例生成而不是正式测试动作。训练与采集耗时存manifest，
在线调用次数/耗时/token缺失/回退率进入ICL结果，成本必须完整报告。

离线全流程检查（模拟LLM，输出隔离在smoke，不作为论文成绩）：

```powershell
python -m algorithms.comparisons.dtecn_llm.run smoke --optimizer-seed 10
python -m unittest tests.test_dtecn_llm -v
```

smoke仅1个SS工作流，不用缩小的混合M/L工作负载冒充正式50工作流DDL缓存。
正式训练可占用较多内存（联合状态replay及每工作流Critic）；训练收敛、远程服务
重复性和50工作流完整在线成本需真实实验验证。

## 实现验收（2026-10-11）

新增测试覆盖CPU服务区间、队列守恒、独立奖励、MADDPG联合更新及恢复、
未来槽位隔离、在线库重置、合法投影/JSON、回退及token统计、全局上下文预算，
以及相同动作下共享模拟器时间线和能耗逐位一致。
离线smoke完整运行训练、验证选择、案例采集、模拟LLM测试和JSON/CSV导出。
50工作流的SS/MS/LS模拟LLM联调分别完成1410/2195/3504次分配，队列终值均0。
这些是实现检查，不是模型性能结果；没有进行50000步正式训练、真实在线调用，
也没有给出长期队列稳定性或优于LLM-Safe-HRL的结论。

## 修正验收与工件边界（2026-10-11）

config、checkpoint和manifest身份使用schema 3。历史schema 1/2结果保留，但不能恢复/生成新案例；需重训。新结果隔离到artifacts_v3。训练报告新增Actor/Critic实际更新次数和可训练参数总量、验证次数及验证交互量；案例报告新增采集交互量/episode数。在线报告关联全部准备成本。token不完整时total_tokens=null，保留已报告计数和usage_missing，不能当成完整成本。

此次修正只影响本方法网络、目标反馈及审计信息，未改其他方法、共享环境、合法动作、FCFS、seed和最终指标。接口模拟验证不代表真实GPT/千问服务已验收。连续偏好离散投影和按任务决策的学习效果仍需正式训练验证，未改成其他算法，也未擅自改变折扣或奖励。

网络调整：Actor为输入→512→256→VM数，Critic为联合状态/动作→512→256→1。MADDPG更新、奖励、目标、协议和LLM接口不变；目标网络通过深拷贝自动采用新结构。参数量继续记录到manifest。正式50工作流的集中式Critic显著增大，需监控训练内存及运行成本，不能将规模调整视为性能保证。

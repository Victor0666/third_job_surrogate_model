# CEWS 安全启发式准入报告

本目录只保存由
`algorithms/llm_safe_hrl/LLM/problems/cews_task_constructive/eval.py`
协议 v2 生成的不可执行
`RESULT_JSON` 报告或 stdout 文本。Manager 规则库会同时校验报告文件
SHA-256、报告内候选源码 SHA-256、配置指纹和准入记录指纹。

推荐流程：

```powershell
python .\algorithms\llm_safe_hrl\LLM\problems\cews_task_constructive\eval.py `
  --candidate .\algorithms\llm_safe_hrl\LLM\problems\cews_task_constructive\generated\candidate_iterN_indM.py `
  --config .\algorithms\llm_safe_hrl\LLM\cfg\problem\cews_task_constructive_hrl_ss_admission.yaml `
  > .\algorithms\llm_safe_hrl\LLM\problems\cews_task_constructive\admission_reports\iterN_indM.txt

python .\tools\manage_safe_heuristic_manifest.py `
  --manifest .\algorithms\llm_safe_hrl\LLM\problems\cews_task_constructive\safe_heuristic_library_resS.json `
  --candidate .\algorithms\llm_safe_hrl\LLM\problems\cews_task_constructive\generated\candidate_iterN_indM.py `
  --evaluation-report .\algorithms\llm_safe_hrl\LLM\problems\cews_task_constructive\admission_reports\iterN_indM.txt `
  --config .\algorithms\llm_safe_hrl\LLM\cfg\problem\cews_task_constructive_hrl_ss_admission.yaml `
  --heuristic-id seevo_iterN_indM_hrl_ss_v1 `
  --seevo-iteration N `
  --seevo-individual M
```

注册器不会导入或执行候选。未准入记录也会写入 manifest 并保留拒绝原因；
重复 `heuristic_id` 或版本会被明确拒绝，避免覆盖历史证据。

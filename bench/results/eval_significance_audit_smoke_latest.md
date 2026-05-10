# Eval Significance Audit

- Status: pass
- Reports: 1
- Scopes: 4
- Significant HolyC losses: 0

## Scopes

| Scope | Dataset | Split | Records | Delta | Discordant | p-value | Significant HolyC loss |
|---|---|---|---:|---:|---:|---:|---|
| overall | smoke-eval | validation | 3 | 0 | 0 | 1 | False |
| dataset_split | arc-smoke | validation | 1 | 0 | 0 | 1 | False |
| dataset_split | hellaswag-smoke | validation | 1 | 0 | 0 | 1 | False |
| dataset_split | truthfulqa-smoke | validation | 1 | 0 | 0 | 1 | False |

## Findings

- warning no_discordant_pairs overall smoke-eval/validation: McNemar evidence is uninformative because both engines have identical correctness on this scope
- warning no_discordant_pairs dataset_split arc-smoke/validation: McNemar evidence is uninformative because both engines have identical correctness on this scope
- warning no_discordant_pairs dataset_split hellaswag-smoke/validation: McNemar evidence is uninformative because both engines have identical correctness on this scope
- warning no_discordant_pairs dataset_split truthfulqa-smoke/validation: McNemar evidence is uninformative because both engines have identical correctness on this scope

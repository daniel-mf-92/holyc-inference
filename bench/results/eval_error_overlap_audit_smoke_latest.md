# Eval Error Overlap Audit

Status: **pass**

| Metric | Value |
| --- | ---: |
| Gold records | 3 |
| Paired records | 3 |
| HolyC errors | 2 |
| llama.cpp errors | 1 |
| Shared errors | 1 |
| HolyC unique errors | 1 |
| llama.cpp unique errors | 0 |
| Error Jaccard | 0.500000 |
| Findings | 0 |

## Error Rows

| Record ID | Dataset | Answer | HolyC | llama.cpp | Class |
| --- | --- | ---: | ---: | ---: | --- |
| smoke-arc-1 | arc-smoke/validation | 0 | 1 | 2 | shared_error |
| smoke-truthfulqa-1 | truthfulqa-smoke/validation | 0 | 1 | 0 | holyc_unique_error |

## Findings

No eval error-overlap findings.

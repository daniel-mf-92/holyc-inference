# Eval Artifact Identity Audit

- status: pass
- holyc_records: 3
- llama_records: 3
- paired_records: 3
- required_fields: model_sha256, quantization, tokenizer_sha256
- findings: 0

## Identity Coverage

| Source | Field | Present | Missing | Distinct |
| --- | --- | ---: | ---: | ---: |
| holyc | model | 3 | 0 | 1 |
| holyc | model_sha256 | 3 | 0 | 1 |
| holyc | tokenizer_sha256 | 3 | 0 | 1 |
| holyc | quantization | 3 | 0 | 1 |
| holyc | quantization_sha256 | 0 | 3 | 0 |
| holyc | prompt_template_sha256 | 3 | 0 | 1 |
| llama.cpp | model | 3 | 0 | 1 |
| llama.cpp | model_sha256 | 3 | 0 | 1 |
| llama.cpp | tokenizer_sha256 | 3 | 0 | 1 |
| llama.cpp | quantization | 3 | 0 | 1 |
| llama.cpp | quantization_sha256 | 0 | 3 | 0 |
| llama.cpp | prompt_template_sha256 | 3 | 0 | 1 |

No eval artifact identity findings.

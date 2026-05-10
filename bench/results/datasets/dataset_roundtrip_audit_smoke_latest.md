# Dataset Roundtrip Audit

- Status: pass
- Records: 3
- Dataset metadata: smoke-eval
- Split metadata: validation
- Errors: 0
- Warnings: 0

## Binary Layout

| field | expected | actual |
| --- | ---: | ---: |
| binary_bytes | 925 | 925 |
| body_bytes | 778 | 778 |
| choice_length_prefix_bytes | 48 | 48 |
| fixed_header_bytes | 52 | 52 |
| metadata_bytes | 95 | 95 |
| record_count | 3 | 3 |
| record_header_bytes | 72 | 72 |
| record_payload_bytes | 706 | 706 |

## Findings

No roundtrip findings.

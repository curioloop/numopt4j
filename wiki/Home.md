# numopt4j Wiki

Error code reference for `OptimizationException`. Each page explains the cause, shows a minimal failing example, and provides a fix.

## Error Codes

| Code | Trigger |
|------|---------|
| [MISSING_PARAM](MISSING_PARAM) | Required parameter not set before `.solve()` |
| [INVALID_INPUT](INVALID_INPUT) | Initial point contains `NaN` or `Infinity` |
| [DIMENSION_MISMATCH](DIMENSION_MISMATCH) | Array length does not match problem dimension |
| [INVALID_VALUE](INVALID_VALUE) | Parameter value violates documented constraint |

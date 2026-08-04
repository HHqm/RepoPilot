# Buggy Calculator

This deliberately small repository is the deterministic RepoPilot demo target.
`safe_divide` promises a domain-level `ValueError`, but currently leaks Python's
`ZeroDivisionError`.


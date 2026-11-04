"""The ruleset as records: every rule says where it is enforced, and proves it can fail.

A rule that exists only as prose cannot say whether anything enforces it, and a
check nobody has seen go red cannot say whether it would.  ``registry/*.toml``
holds the records, TOML because the hook and push paths must not fail on a
missing package.  ``python -m tools.rules audit`` turns each claim into a checked
statement: exit 1 is a finding, exit 2 means something could not be measured.
"""

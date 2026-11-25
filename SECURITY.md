# Security

Report suspected vulnerabilities privately to the maintainer by email rather than
in a public issue. Include the affected version or commit, the command or input
that triggers the problem, and the observed effect.

The project reads and writes data files and runs third-party simulators. Inputs
from untrusted sources should be checked with the schema validators before use,
and simulators should run in an isolated environment.

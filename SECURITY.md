# Security

Blue Compressor parses untrusted, complex binary files, so memory-safety and
denial-of-service issues in the parsers matter.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's **"Report a vulnerability"**
button on the repository's Security tab, not in a public issue. Include a description, the
affected version and, if possible, a minimal file that triggers the problem.

You can expect an acknowledgement within a week. Fixes are released as soon as
practical, and reporters are credited unless they prefer otherwise.

## Scope

In scope: crashes, hangs, excessive memory use or out-of-bounds access caused by a
crafted `.NEV` or `.R3D` file, and anything that could make the tool modify or delete
files other than its own outputs.

Out of scope: issues in third-party software (such as WebView engines), which should
be reported to their vendors.

**Issue number:**

Partially addresses jepiyush/bottlerocket#12.

**Description of changes:**

Update ten packages: aws-signing-helper 1.8.5, aws-otel-collector 0.50.0, Bash 5.3.20, iproute 7.2.0, strace 7.2, and the five Kubernetes releases below. Include the required Go 1.26 selection for Kubernetes 1.34/1.35, remove the upstreamed 1.34 go-proxyproto patch, and correct the signing helper's version linker symbol.

Kubernetes 1.36, Kubernetes 1.37, and SELinux changes are excluded. Those directories match baseline `209347ba3`. The history contains ten signed package commits named `{package-name}: update to {version}`, plus one validation-evidence commit. The canceled updates and defer commit are removed. No per-file source-fetch overrides or alternate GNU mirror remain.

**Testing done:**

**Fresh validation, 2026-10-06 UTC**

All ten conformance runs were launched independently in parallel, with utility checks alongside them. Each conformance run verified 15 Ready nodes against its AMI, architecture, and kubelet version, then used 15 test processes and at most three attempts per test.

| Kubernetes | EKS-D release | x86_64 passed | aarch64 passed | Tests needing retries |
| --- | ---: | ---: | ---: | ---: |
| 1.31.14 | 50 | 362 | 362 | 0 |
| 1.32.13 | 43 | 372 | 372 | 0 |
| 1.33.13 | 33 | 380 | 380 | 0 |
| 1.34.11 | 24 | 385 | 385 | 0 |
| 1.35.8 | 15 | 402 | 402 | 0 |

**3,802 selected conformance tests passed; zero failures and zero selected skips.** No selected test required a retry. The unchanged filter selects `[Conformance]` and excludes `[Serial]`, `[Disruptive]`, `[Slow]`, and `Garbage.collector`. This is filtered conformance, not full certification.

| Package | Fresh live checks | x86_64 | aarch64 |
| --- | --- | --- | --- |
| aws-signing-helper | Version/aliases, RSA and ECDSA signatures, independent verification and tamper rejection, credential-process and brush policy | Passed | Passed |
| Bash | Version, builtins, and command behavior | Passed | Passed |
| iproute | Version and address inspection | Passed | Passed |
| strace | Version and tracing a command | Passed | Passed |
| aws-otel-collector | Trace/metric forwarding, configuration header, restart behavior, and restoration | Passed | Passed |

Live binary hashes matched the verified kit payloads. All twelve accepted image builds passed artifact, inventory, signing-input, dependency, and source checks. Symphony `make integ`, including the updated skills, passed.

**Inputs and validation notes**

- Kit: `17.0.1-symphony.12.build-6b1b4690`, index `sha256:f683502260a1a3862671b94a9655fabcc10cef5437a9ddfc1b51d8cb7fcbc9fd`. The two verified kit archives were reused; the package tree is identical to this rewritten PR. The accepted images and live test runs are fresh.
- Kubernetes image consumer: `feeffd6b45fc60362938a1f2acff20791fe1e0ad` plus the recorded refreshed `Twoliter.lock` SHA-256 `a1bb83b0a1e180a767274ad837c65e1098d473c661d01ebe232c3d754887ef9d`. Its bytes and every other tracked source file were verified against the recorded inputs.
- Utility image consumer: `a77a7b3d7c83b9624c11b71d256d99ed1a48a40b`, clean. The initial two aws-dev images lacked the collector; the temporary test variant was corrected and only those two images were rebuilt.
- The reused kit builds used SHA-512-verified cached Bash patches. Fresh retrieval from GNU was not established.
- Collector tests used loopback OTLP with a temporary plaintext exporter; external TLS, live Roles Anywhere, and hardware signers were not tested.
- The single final review already ran. Combined kit builds plus RPM inspection remain the disclosed substitute for the individual package-job matrix.

<details>
<summary>Fresh AMI identities and cleanup</summary>

| Variant | x86_64 AMI | aarch64 AMI |
| --- | --- | --- |
| aws-k8s-1.31 | `ami-035807f0cc56ee04a` | `ami-0096662fab20969dd` |
| aws-k8s-1.32 | `ami-0bbe65ed7a437a642` | `ami-0f86e9f23d1c62c94` |
| aws-k8s-1.33 | `ami-03dc400b61af37df2` | `ami-0b5591f38aaeec3f6` |
| aws-k8s-1.34 | `ami-0a9b8086dffcdd169` | `ami-030a9dea9f48a120e` |
| aws-k8s-1.35 | `ami-0991367f5e08caa15` | `ami-028318613bb4966d9` |
| aws-dev | `ami-0c02c2542fbebcb46` | `ami-0656f3efe14fca276` |

All test clusters, nodes, volumes, networking and IAM resources were independently checked after cleanup. Both standalone instances and their network were removed. All twelve private AMIs, twenty-four snapshots, and the issue-owned local registry were deleted after validation.

</details>

[Download the test logs, JUnit reports, build/source receipts, and cleanup evidence](http://127.0.0.1:3100/attachments/27368a13-912f-4ecd-a0f0-7a3fc88b82ac) (SHA-256 `b6459175b2b17062490860b396c539e0929a03ae1a5ed5b883f062543a74852a`). The committed summary is `docs/evidence/2026-10-06-package-updates/fresh-rerun.json`.

**Terms of contribution:**

By submitting this pull request, I agree that this contribution is dual-licensed under the terms of both the Apache License, version 2.0, and the MIT license.

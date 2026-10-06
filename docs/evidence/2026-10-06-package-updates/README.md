# Issue 12 package update validation

## Fresh ten-package PR validation

The requested rerun passed on both architectures. The ten independent filtered
conformance runs passed 3,802 selected tests, with zero failures or selected
skips. All five package runtime targets passed on each architecture. Test
resources, private AMIs and snapshots, and the issue registry were cleaned up.

Kubernetes 1.36, Kubernetes 1.37, and SELinux remain excluded. The rewritten
package tree matches the verified kit input. See [fresh-rerun.md](fresh-rerun.md)
and [fresh-rerun.json](fresh-rerun.json) for results, retries, input identities,
validation limitations, and the downloadable evidence archive.

The sections below preserve the broader campaign at tested revision `ac8104fa`
and the review response at `6c01657e8`. The current ten-package handoff above
supersedes their pending scope/build decisions; it does not relabel their runtime
results as final-tree runs. See [review-response.json](review-response.json).

Validation and cleanup of the previously tested revision completed on
2026-10-06. The corrected
Kubernetes matrix passed 5,462 selected tests across 14 runs. Five additional
package behavior targets passed on each architecture. Two tests in the 1.35 ARM
run passed on their second attempt; no selected test remained failed or skipped.

The tested core-kit revision is
`ac8104fadeba8e18ecd4d266314a2e66e3876368`, based on
`209347ba321f023c242f48cb1afc590e5f9793bb`. The later source-fetch cleanup
preserves every external-file checksum. [source.json](source.json) records
every package version, changed file hash, source revision, and implementation
adjustment.

## Updates and conformance results

The changes update aws-signing-helper to 1.8.5, aws-otel-collector to 0.50.0,
Bash to 5.3.20, iproute to 7.2.0, and strace to 7.2. Bash's actual repository
baseline was 5.3.15; patches 016 through 020 complete the requested update.

| Package | EKS-D release | Kubernetes version | x86_64 passes | aarch64 passes |
| --- | --- | --- | ---: | ---: |
| kubernetes-1.31 | 50 | 1.31.14 | 362 | 362 |
| kubernetes-1.32 | 43 | 1.32.13 | 372 | 372 |
| kubernetes-1.33 | 33 | 1.33.13 | 380 | 380 |
| kubernetes-1.34 | 24 | 1.34.11 | 385 | 385 |
| kubernetes-1.35 | 15 | 1.35.8 | 402 | 402 |
| kubernetes-1.36 | 9 | 1.36.4 | 407 | 407 |
| kubernetes-1.37 | 3 | 1.37.0 | 423 | 423 |

The `conformance-runner` skill's filtered suite ran on EKS in `us-east-2`.
Each run verified 15 Ready nodes against its intended AMI, architecture, and
kubelet version before testing, with 15 parallel test processes and at most
three attempts per test. The focus was `\[Conformance\]`; the skip expression
was `\[Serial\]|\[Disruptive\]|\[Slow\]|Garbage.collector`. This is filtered
conformance, not full certification.

[conformance.json](conformance.json) contains the exact run IDs, test releases,
build IDs, AMIs, all 210 tested instance IDs, cleanup instance and volume IDs,
retry details, and hashes of acceptance receipts. Those receipts bind the
JUnit reports, command settings, node identity observations, logs, and cleanup
checks. All 14 runs passed 23 independent cleanup checks each.

Only the 1.35 ARM run reported retries: the monotonically increasing probe
restart-count test and the node-to-pod UDP communication test each passed on
attempt two within the configured limit.

## Build and runtime identity

Both corrected core-kit archives and all 16 corrected images passed their
builds and artifact verification. The image set covers seven Kubernetes
variants and `aws-dev`, each on x86_64 and aarch64. The verified local kit is
`17.0.1-symphony.12.build-ac8104fa`, with index digest
`sha256:a415a97c0e383096efd63d43b7831d3be3d61fd8127c304f99e4e4fc8ef129b1`.

Images used Bottlerocket revision
`4a3a643456536696b5adcdf2c663f07ac4f5a774` and private release
`1.66.0-symphony.12`. [artifacts.json](artifacts.json) records archive hashes,
image input hashes, compressed and raw disk hashes, UEFI hashes, package
inventories, private AMIs, and snapshots. Each kit archive had 134 verified OCI
blobs; each image result had 36 verified returned files.

The design's 24 individual package build cells were not completed as an
accepted matrix. Its retained record has six compilation passes awaiting RPM
inspection and 18 cells still marked requested. The exporter searched
`build/packages`, while Twoliter 0.25.0 wrote `build/rpms`, so those package
results contained no RPMs. Two combined kit builds and offline inspection of
all 12 packages on both architectures replaced that planned matrix. This is
a validation-plan substitution, not 24 individually accepted package jobs.

[runtime.json](runtime.json) records the SSM command IDs and live binary hashes,
which matched the checked kit payloads. Checks covered:

- Bash version and command execution, iproute version and address inspection,
  and strace version plus tracing a real command.
- Collector traces and metrics forwarded before and after a service restart,
  with the expected environment header and restored packaged configuration.
- Signing-helper version aliases and links, brush restrictions, RSA-2048 and
  ECDSA P-256 signatures checked independently, tamper rejection, and rejection
  of invalid credential-process inputs.

Standalone runtime instances were `i-0ac185f4516422aad` (x86_64) and
`i-0a871d499b7a05b88` (aarch64). Signing checks reused conformance nodes
`i-001c8338c23e34039` and `i-01b338113a1ffe964`, respectively.
Collector forwarding used loopback OTLP with a temporary plaintext exporter
override; external TLS was not tested. Signing checks used disposable local
keys, without a live Roles Anywhere session or hardware signer.

The standalone aws-dev launches overlapped conformance rather than running
serially afterward as designed. The recorded parallel authorization covered
the conformance matrix, not this scheduling deviation. Both standalone
instances and their resources were verified cleaned up; this disclosure does
not retroactively expand that authorization.

## Compatibility fixes and validation adjustments

At core-kit `ac8104fa`, `packages/aws-signing-helper/aws-signing-helper.spec`
targets the upstream `cmd.Version` linker symbol, so installed aliases report
the requested version. Kubernetes Go selections were aligned with upstream
requirements; `packages/kubernetes-1.34/kubernetes-1.34.spec` selects Go 1.26
before code generation and drops the obsolete go-proxyproto patch. The
previously tested source contained 18 `force-upstream` entries and used the
GNU kernel.org mirror for Bash patches 016-020. Those were environment
workarounds: commit `6c01657e8` removes the entries and restores ftp.gnu.org.
All source pins are unchanged, and the five retained patch files were
rechecked against their SHA-512 pins. Core-kit's `make build` supports
`UPSTREAM_SOURCE_FALLBACK=true`, but defaults that option to false. The earlier
review incorrectly described fallback as the default. Symphony's direct-kit
wrapper omits the flag and its request schema exposes no fallback switch. This harness mismatch is reported
in the review response; Symphony itself was not edited.

The original image campaign had ten passing suites and four failing suites:
both architectures of 1.36 and 1.37 failed an ImageVolume sandbox test that
requests SELinux type `svirt_lxc_net_t`. At the tested revision,
`packages/selinux-policy/subject.cil` aliases that legacy name to the existing
confined `container_t`. [policy.json](policy.json) records compiled-policy
comparisons on both architectures: the alias is the sole semantic difference,
with no added allow rules, transitions, or permissive domains. All corrected
images and runtime tests were repeated after this fix. Original results remain
historical evidence and are identified separately in [cleanup.json](cleanup.json).

The baseline and first updated `packages/selinux-policy` directories are
identical and lack this alias. The retained `image_volume.go:48` hardcodes
`svirt_lxc_net_t`, and line 91 declares `Release: v1.36`. This supports the
inference that the policy/test incompatibility predates the package updates
and explains why the earlier-minor suites did not exercise it. No live run of
an unchanged baseline with identical test inputs was performed. The alias
changes shipped policy beyond the 12 named packages; it remains pending the
explicit human scope decision requested by the final review.

[validation-adjustments.json](validation-adjustments.json) records authorized
parallel execution, bounded node stabilization, native AWS capacity/deletion
retries, and issue-local evidence parser corrections. The latter handle the
collector's known gRPC preamble, recycled EIP addresses by allocation identity,
startup replacement nodes in cleanup, and Ginkgo's optional `Flaked` summary.
The collector and Ginkgo summary parsers were changed after inspecting the
outputs they initially rejected, then used to accept those saved outputs.
They were therefore fitted to observed output; independent checks and negative
cases support the results, rather than an unchanged pre-run acceptance parser.
Saved results were reconciled without replaying successful tests or SSM
commands, changing filters, or modifying the Symphony harness.

[prerequisites.json](prerequisites.json) preserves the 40 verified source
inputs, eight signature checks, and prior successful `make integ` receipt.
Those historical gate receipts checked the existing harness working tree;
they were not pristine harness-commit qualifications. The requested fresh gate
receipt is recorded separately in the review response.

## Cleanup and evidence retention

All 217 recorded corrected conformance instances, including seven startup
replacements, and 434 volumes were cleaned up. The two standalone runtime
instances, their four volumes, and their network/IAM stack were also removed.
The 32 private AMIs and 64 snapshots from both image campaigns were removed
after validation. Registry cleanup job
`31090cee-09ef-826c-aa9e-d80c8bf8af26` confirmed removal of `localhost:6431`.

Bottlerocket commit `2c7614b2fedb7dcb3126c2718e08d1bc4f647292` restores
`Release.toml`, `Twoliter.toml`, `Twoliter.lock`, and
`variants/aws-dev/Cargo.toml` byte-for-byte to original baseline
`1ad6b4a40390696d772e9821bb331d25846d4cad`. The consumer's net diff from that
baseline is empty. The package PR belongs to core-kit.

[cleanup.json](cleanup.json) contains cleanup receipts and restoration hashes.
[manifest.json](manifest.json) identifies the stage and hashes the preserved raw
and curated evidence. Raw paths are stage-relative unless absolute; artifact
archive paths are relative to their owning member. Local verified archives,
test binaries, and raw evidence remain available in the issue stage. Dated
original receipts may describe resources retained at that earlier point; the
final cleanup records supersede that historical retention status.

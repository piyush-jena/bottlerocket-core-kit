# Issue 14: UKI mount drop-in revision, 2026-10-06

Historical result: the 2026-10-07 review supersedes this implementation
with the [systemd image-feature guard](../issue-14-systemd-guard/README.md).
The pass below applies only to the recorded drop-in revision.

## Acceptance follow-up, 23:02 UTC

Both architecture builds at `fa6d388f834b2c370fed8f61c6c7145a20c0bab0`
passed. The verified archives were published as
`17.0.1-symphony.14.build-fa6d388f`; [kit-publication.json](kit-publication.json)
records both platform digests and the combined index.

Five fresh private AMIs passed: base mantle on x86_64 and aarch64, plus
FIPS, NVIDIA and NVIDIA-FIPS on x86_64. Every case passed 38 effective-policy
checks before and 38 after native container execution. Both supplied GPU
smoke workflows reported one NVIDIA L4, deviceQuery PASS and vectorAdd
PASSED. [acceptance-summary.json](acceptance-summary.json) records the
tested source/configuration, AMI and instance IDs, build jobs and SSM IDs.
Full scripts, command output, artifact hashes and source attestations are
in the Bottlerocket member under `docs/evidence/issue-14-uki-dropin`,
prepared for the PR 16 update.

GPU capacity failures required g6.2xlarge in us-east-2c. GPU guest shutdown
reached Power down but EC2 remained in shutting-down; cleanup used force
termination for the two verified issue-owned GPU IDs. Shutdown filesystem
and mdadm warnings are preserved without attributing their cause to this
change. All five instances, five AMIs, ten snapshots and the access stack
were removed; [ec2-cleanup.json](ec2-cleanup.json) records the final result.
Registry cleanup passed in conductor job
`b4f66664-3558-8fb5-ae17-1345a7ac8403`, exit 0.
[registry-cleanup.json](registry-cleanup.json) records removal of the
issue-owned registry at localhost:6539. All cleanup is complete.

The fresh Symphony integration gate passed. The prior passing Twoliter
gate applies to unchanged tool code. No reboot or non-UKI EC2 control
result is claimed for this revision. The record below describes the
implementation milestone before these new builds and live tests.

## Implementation-time record

PR 16 review comment 881 requests moving binfmt mount suppression from the
mantle command line into a core-kit subpackage selected by `uki-image`.
This revision supersedes the command-line-mask acceptance results.

## Implementation

- Core-kit `390b1d3ee5e645c66c86171db6ef353558f52001`,
  `packages/release/release.spec:201,264,385,637`: the base release selects
  `release-uki` only for `image-feature(uki-image)`. The subpackage requires
  that feature and conflicts with `no-uki-image`; it owns the drop-in directory
  and `10-uki.conf`.
- Same revision, `packages/release/binfmt-misc-mount-uki.conf:1`: use
  `[Unit] ConditionPathExists=!/`. Since root exists, boot and explicit start
  requests skip the mount. This is a condition skip, not LoadState=masked.
- Same revision restores `packages/systemd-257/systemd-257.spec` to upstream
  `209347ba321f023c242f48cb1afc590e5f9793bb` ownership. The mask generator is
  no longer a prerequisite. The product diff is confined to release packaging.
- Bottlerocket `3f1f18e74640c671872a6b114bd42dddb612ea68`, the four
  `variants/aws-mantle-1*/Cargo.toml` files: remove `systemd.mask=` and retain
  `module_blacklist=i8042,binfmt_misc`.
- Kernel-kit `cf5b162a0da6a91132d282e32a426851afb29683`, kernel-6.18
  `config-full-bottlerocket-x86_64-on-x86_64:1109` and
  `config-full-bottlerocket-aarch64-on-x86_64:1075`: CONFIG_BINFMT_MISC=m
  remains unchanged.
- Twoliter `aa8a122d1b3f085fa792b2181c599dc5de503467` has the identical tree
  to the previously validated `d2c650ad1489fe545f68c4fcbd481f2163ed5629`.
  Its full integration result fba33ee6-f05c-4a89-958b-4264f17a54fd remains
  applicable. No Twoliter code changed in this turn.

The separate core-kit PR 3 can carry this packaging change in place of the
obsolete generator prerequisite; Bottlerocket PR 16 will link it and Twoliter
PR 3 after validation. No branch push or PR publication was performed here.

## Condition choice and local verification

`upstream-sources.json` pins the downloaded systemd v257 reference files by
hash. Its load-fragment table registers ExecCondition under Service, whereas
ConditionPathExists belongs to Unit. The condition implementation tests path
existence and negates the result when requested. No runtime shell or feature
environment is needed: RPM selection applies the policy to UKI images.

`local-checks.json` records exact commands, stdout, stderr and exit codes.
Host systemd 259 verifies the upstream v257 mount with and without the drop-in.
Without it, unit conditions pass (exit 0). With it, parsing succeeds (exit 0)
and condition evaluation fails (exit 1, expected). These are offline checks;
no host unit was installed or started. RPM parses the rich dependencies and
subpackage definition successfully. Member diff checks pass.

The required `make -C /home/fedora/Repositories/symphony integ` gate passed
with exit 0. `make-integ.log` retains its full output: formatting, clippy,
Rust tests, skill/brain/prose lint, 213 Python tests, and 21 shell syntax
checks all passed. No Symphony source or configuration was edited.

`policy.py` is the revised live probe, syntax-checked but not yet executed.
It checks the command line, module object hash, effective drop-in, boot units,
kernel loading rejection, absent module/filesystem/mount state, and a manual
mount-unit start. Expected unit state is loaded/inactive, ConditionResult=no,
Result=success. The probe is to run before and after container execution.

## Existing artifacts inspected before rebuilding

Both collected OCI kit archives remain available and match their prior receipts:

| Architecture | Prior build | Archive SHA-256 |
| --- | --- | --- |
| x86_64 | 303a1ab6-ec33-4949-95de-9bcda86491bb | ba6fd227646d403059f08d155a461f1ed8716be3d67b6398ed7657ec8eeadcd9 |
| aarch64 | e217a248-a89a-43cb-aeed-fcb1f744784d | c3dcc65dc54b403c2133dc274ad6991e2a808fb851caf871bdeea9b6d50a72b3 |

Their paths are `build/symphony/<build>/kits/bottlerocket-core-kit/` in this
member. Archive filenames contain `v17.0.1-15903b6a-<arch>.tar`. Checks used
`sha256sum` on both files and `tar tf` on the x86_64 archive. These artifacts
contain the previous generator policy, so new core-kit builds are justified
by changed source, not artifact loss. Historical killed/refused jobs do not
establish a current blocker. The previous registry cleanup succeeded.

## Remaining acceptance work

1. Build both core-kit architectures at this milestone; inspect RPM ownership,
   dependencies and installed drop-in, verify archive hashes, then request
   publication of both architectures. Use the returned issue-tagged version.
2. Build fresh base mantle on x86_64 and aarch64, and FIPS, NVIDIA and
   NVIDIA-FIPS on x86_64 using the unchanged Twoliter prerequisite. Preserve
   private release configuration and issue identification in the image inputs.
3. Register private AMIs and repeat boot, native container and GPU smoke
   validation, recording exact sources, digests, commands, SSM IDs and cleanup.
   Confirm package selection excludes the drop-in from non-UKI composition.
4. Update the two affected PRs after validation. The final adversarial review
   already ran in the earlier round; do not request another one. Keep the
   changed revision unqualified until these tests pass.

No new AMIs, instances, registries or cloud test resources were created in
this turn. Previous cleanup remains historical; no new live pass is claimed.

# Issue 14: systemd image-feature guard, 2026-10-07

The new core-kit PR 3 review asks for the binfmt mount to be omitted from
UKI images through `packages/systemd-257/systemd-257.spec`. This replaces
the release-uki package and its drop-in. The earlier successful AMI results
remain historical and do not validate this new packaging revision.

## Implementation

Core-kit `fcaa82729f1f10519ce592ba54a1f710507c19c1`:

- `packages/systemd-257/systemd-257.spec` requires
  `(%{name}-binfmt or %{_cross_os}image-feature(uki-image))`.
- The binfmt subpackage owns only the existing
  `proc-sys-fs-binfmt_misc.mount` and its `sysinit.target.wants` symlink.
  It requires the base package and conflicts with `uki-image`.
- The base systemd package no longer owns those two files. Non-UKI
  composition selects the binfmt subpackage; UKI composition omits it.
- `packages/release/release.spec` is restored byte-for-byte to upstream
  `209347ba321f023c242f48cb1afc590e5f9793bb`. The drop-in source is removed.
  The net product diff from that upstream base is one systemd spec file.

This is an image-composition guard, not a build-time `%if`. Twoliter
`aa8a122d1b3f085fa792b2181c599dc5de503467`,
`tools/buildsys/src/builder.rs:201`, passes package/build identity arguments
to the package builder, not variant image features. In the same revision,
`twoliter/embedded/build.Dockerfile:141` builds the shared package RPMs;
`:262` and `:283` generate the UKI condition and metadata capability later,
for image composition. A `%if` around the shared package's file list
therefore cannot select different contents for different consuming images.
The two-file subpackage gives the guard an installable unit to omit.

The existing `packages/os/os.spec:127` uses the same package-or-UKI
dependency pattern. No new runtime unit, drop-in, environment-variable
test, kernel change or firmware restriction is introduced.

## Checks performed

`verify-guard.py` parses the actual spec through RPM and checks exclusive
ownership of both paths. It loads the parsed feature dependency into a
temporary metadata-only repository and asks the host DNF/Hawkey dependency
solver to resolve three cases on each architecture:

| Case | Expected and observed |
| --- | --- |
| UKI plus base systemd | Solves; binfmt package absent |
| Non-UKI plus base systemd | Solves; binfmt package selected |
| UKI with binfmt explicitly requested | Conflict; resolution fails |

All six cases passed. `local-command.json`, `local-checks.json` and
`local-checks.stderr` record the exact command, exit status, output,
architecture results and spec hash. The host's Hawkey deprecation warnings
are retained. These are dependency-metadata checks; no RPM was built or
installed locally, and this is not yet a built-image pass.

`previous-artifacts.json` records fresh SHA-256 checks of both prior
fa6d388f kit archives. Both remain intact and match their publication
receipts. New builds are required because they contain the superseded
release-uki drop-in, not because artifacts were lost.

The revised `policy.py` is syntax-checked but has not run on a new AMI.
It requires the mount and wants-link to be absent, `LoadState=not-found`,
an unsuccessful explicit start, no failed boot units, and absent module,
filesystem and mount state before and after loading attempts. It retains
the exact blacklist, module-object and image identity checks.

## Built kit checks, 2026-10-07

Both builds succeeded from clean snapshot
`f36d9c2bbc25887c07d7c8b79662b3944eb60ddc`, containing product change
`fcaa82729f1f10519ce592ba54a1f710507c19c1`:

| Architecture | Build request | Verified archive SHA-256 |
| --- | --- | --- |
| x86_64 | b47eda7c-d766-8bee-ac9d-05bfc2ebdae4 | 3b1938f64940fd9c5e65763d71088ddc21344c06685c0734d24a897c446c93ba |
| aarch64 | fbf7b2d1-57c7-8e6d-acd5-a0d0bda1bf19 | 936677ac5398aae118e93ea6732acd67b3cf2820c71b691315ebac97cf31ab83 |

`inspect-kit.py` ran successfully for both archives. The complete arguments,
RPM query commands, exit statuses, stdout/stderr, file ownership, symlink
targets and hashes are in `built-kit-x86_64.json` and
`built-kit-aarch64.json`. It checked the archive bytes against the worker
receipts, the input archive hash, captured spec bytes against the source
commit, and the OCI manifest, config and all 132 layer hashes per architecture.
The x86_64 completion was confirmed through its collected cloud result and
observer exit code 0 before its notification arrived.

The built base RPM contains the expected package-or-UKI requirement.
Only `systemd-257-binfmt` owns the mount and wants-link, whose target is
`../proc-sys-fs-binfmt_misc.mount`. Its dependency on base systemd and conflict
with the UKI feature are present. There is no release-uki RPM, and the release
RPMs contain no binfmt policy or drop-in. The debug generator remains in
the console package.

The first inspection attempt assumed final root paths in the RPM header
and failed its ownership assertion: these cross RPMs use an architecture
sysroot prefix. The inspector was corrected to require that exact prefix
for each architecture. Both subsequent inspections passed without product
source changes or rebuilds. These are artifact checks; actual image
composition and live policy remain unvalidated for this revision.

## Remaining validation

1. Publish both verified archives through the conductor and verify the
   combined registry receipt before changing the image dependency lock.
2. Build fresh private 1.66.1 images with the empty migration entry and
   names containing symphony.14.r10. Preserve the exact temporary config
   patch; keep it out of product configuration commits.
3. Test base mantle on both architectures and FIPS, NVIDIA and NVIDIA-FIPS
   on x86_64. Run the revised policy before and after native containers,
   and the supplied NVIDIA smoke workflow on both GPU cases. Recheck base
   x86_64 after reboot and a negative direct mount attempt.
4. Record full commands, digests, instances and cleanup; update both PRs
   after validation. The single final adversarial review already ran;
   do not request another round.

The four variant blacklists and the tested Twoliter prerequisite are
unchanged. Their previous source identity and integration-gate evidence
remain available. No AMIs or cloud runtime resources were created for
this revision in this turn, and no live pass is claimed.

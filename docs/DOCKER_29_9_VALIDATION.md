# Docker 29.9.0 packaging decisions

Recorded 2026-10-13 for Symphony issue 25.

## Security advisory decision

This package version does **not** establish full remediation of the security
issues advertised in Docker's 29.9.0 release notes. SDK 0.80.0 builds these
binaries with Go 1.26.8, whereas Docker's release used Go 1.26.9. The package
version and SBOM version identify the Docker source release, not the security
patch level of the compiler and standard library.

Decision: do not file a staging BRSA claiming that this build fixes
CVE-2026-97032, CVE-2026-78659, CVE-2026-78663 or CVE-2026-78669 through the Go
runtime update. The Go-runtime remediation is absent from this build and
requires a separately validated SDK/toolchain update. CVE-2026-56857 concerns
Windows and is not applicable to these Linux images. This decision is not a
claim that the other issues are unexploitable.

The engine source does vendor `golang.org/x/net v0.60.0`, which the retained
Docker release notes identify as carrying the corresponding HTTP/2 fixes for
the deprecated gRPC endpoint and BuildKit frontend server paths. That component
update must not be confused with updating Go's standard library. The packaged
daemon uses socket activation on `/run/docker.sock`, with no default TCP
listener; this limits direct HTTP/2 server exposure but does not substitute
for the missing toolchain remediation or an exploitability assessment.

Sources: Docker's 2026-10-08 release notes retained in issue evidence,
Moby `a5b58c94d334674eae3a3207d4f05417643a90d0` `go.mod`, this package's
`docker.socket` and `docker.service`, and Go build information from the
verified kit and live node binaries. The prior validated core-kit revision
was `0639b49c7d3add5ef10a76974b27a5eeebcc4737`; its binary metadata reports
`go1.26.8`.

## Registry-token limitation

Preissued `AuthConfig.RegistryToken` pulls use Moby's upstream local resolver.
That resolver does not read containerd `hosts.toml`. These requests therefore
bypass `settings.container-registry.mirrors` and other hosts.toml-only routing
or trust configuration. The transfer resolver used for ordinary pulls reads
hosts.toml but cannot represent a preissued bearer token.

The live bearer-token test used a direct registry endpoint without a mirror.
It does not validate bearer-token mirror routing. Use the ordinary transfer
credential path when hosts.toml routing is required; do not treat the mirror
configuration as an endpoint restriction for preissued-token requests.

## Platform selection and test coverage

The transfer protocol requires an explicit ordered platform list.
The compatibility adapter is kept in `daemon/containerd/image_pull_transfer.go`,
derived from containerd/platforms v1.0.0-rc.5's Apache-2.0 implementation.
The selected platforms module remains unmodified. Keep the adapter aligned
with `platforms.Only` when refreshing that dependency.

The in-process regression drives containerd client 2.4.1's transfer image
filter. The runtime daemon is containerd 2.2.9 and reconstructs the marshalled
store options independently. The prior live campaign confirmed default
amd64/v4 hosts selecting baseline amd64 images against that actual pair.
Preference among multiple compatible CPU-specific manifests and rejection
of incompatible architectures are unit-test coverage, not end-to-end daemon
coverage.

Package regressions run in RPM `%check`, using the SDK host architecture even
for a cross build. The selected Twoliter 0.25.1 invokes `rpmbuild -bb --clean`
without `--nocheck`. Docker-engine package builds request Cargo's very verbose
output so successful build-script output, including named Go test results,
can be retained in the conductor build log. Full kit builds still execute
the same checks without making all successful build scripts verbose.

## Temporary source-cache exception

The two Docker archive entries currently set `force-upstream = true` because
the exact cache paths returned HTTP 403. Fetches still require the recorded
SHA-512. Removal trigger: once the source cache serves each exact archive
with bytes matching its manifest SHA-512, remove the corresponding flag and
run the affected package build through the default cache path. Until then,
builds depend on GitHub source availability rather than preferring the cache.

## Rapid capability changes can stop Docker and ECS

This is an observed node behavior, not only an assertion-timing issue.
On stock Bottlerocket 1.67.0 `aws-ecs-3` (`BUILD_ID=8db07cb8`, Docker 29.0.4),
a daemon restart followed by three rapid CHOWN setting changes
(`false`, `true`, `false`) caused Docker to fail with `start-limit-hit` and
ECS to fail as well. No start counter was reset during that reproduction.
The stock `docker.service` file is byte-identical to this package's unit:
`StartLimitBurst=3` and `StartLimitIntervalSec=60s`.

The stock test used AMI `ami-037fd4c8f8f62ce45` and issue-owned instance
`i-03b7be384ef26d4d0` in us-west-2. SSM command
`b9d775f0-bb7d-419a-accb-3332271f1870` recorded the reproduction with terminal
Success and response code zero, meaning the expected outage was observed.
It does not mean the services stayed healthy.

The prior 29.9.0 campaign needed manual `systemctl reset-failed` and service
starts to restore both Docker and ECS. The corrected capability test waits
for Docker's changed MainPID and readiness and resets the start counter;
its pass does not demonstrate safe rapid operator changes without that
intervention. The stock reproduction establishes that the outage predates
this package update. This change does not alter the service limits or fix
that pre-existing behavior.

## Delivery and history

ARM runtime validation, GPU/FIPS, full MACIS and Go 1.26.9 remediation are
outside this campaign. Aarch64 package compilation is being added following
review and must be reported separately from runtime coverage.

Earlier commit messages describe validation as pending at the time those
commits were made. Preserve these historical commit IDs because the retained
package, kit, image and runtime evidence names them. The final PR validation
section records the current results and supersedes those historical status
paragraphs.

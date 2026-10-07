"""Verify collected kit bytes and query RPMs without building or installing."""
import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


root = Path(__file__).resolve().parents[3]
job, arch, build_directory = sys.argv[1:]
receipt_dir = Path(build_directory) / job
receipt = json.loads((receipt_dir / "remote-result.json").read_text())
result = json.loads((receipt_dir / "cloud-result.json").read_text())
provenance_path = receipt_dir / "artifacts/source-provenance.json"
provenance = json.loads(provenance_path.read_text())
assert result["status"] == "succeeded" and receipt["exit_code"] == 0
assert (receipt_dir / "observer-exit.code").read_text().strip() == "0"
assert provenance["sources"]["member"] == {
    "revision": "f36d9c2bbc25887c07d7c8b79662b3944eb60ddc",
    "tracked_changes": "",
}
assert provenance["input_sha256"] == receipt["input_sha256"]
assert sha(receipt_dir / "input.tar") == receipt["input_sha256"]
source_hashes = {}
with tarfile.open(receipt_dir / "input.tar") as captured:
    captured_names = {m.name.removeprefix("./"): m for m in captured}
    assert "member/packages/release/binfmt-misc-mount-uki.conf" not in captured_names
    for source in ["packages/systemd-257/systemd-257.spec", "packages/release/release.spec"]:
        data = captured.extractfile(captured_names["member/" + source]).read()
        expected = subprocess.check_output([
            "git", "-C", str(root), "show",
            provenance["sources"]["member"]["revision"] + ":" + source])
        assert data == expected, source
        source_hashes[source] = hashlib.sha256(data).hexdigest()
assert sha(provenance_path) == next(
    f["sha256"] for f in receipt["files"] if f["path"] == "source-provenance.json")
item = next(f for f in receipt["files"] if f["path"].endswith(".tar"))
archive = root / "build/symphony" / job / item["path"].removeprefix("member/build/")
archive_hash = sha(archive)
assert archive_hash == item["sha256"] and archive.stat().st_size == item["size"]
out = root / "build/symphony" / job / "inspection"
out.mkdir(exist_ok=True)
commands = []


def query(path, *args):
    argv = ["rpm", "-qp", *args, str(path)]
    proc = subprocess.run(argv, text=True, capture_output=True, check=False)
    commands.append({"argv": argv, "exit_code": proc.returncode,
                     "stdout": proc.stdout, "stderr": proc.stderr})
    assert proc.returncode == 0, commands[-1]
    return proc.stdout


rpms = {}
all_rpms = []
with tarfile.open(archive) as outer:
    members = {m.name.removeprefix("./"): m for m in outer}

    def blob(descriptor):
        data = outer.extractfile(members["blobs/sha256/" + descriptor["digest"][7:]]).read()
        assert len(data) == descriptor["size"]
        assert "sha256:" + hashlib.sha256(data).hexdigest() == descriptor["digest"]
        return data

    index = json.load(outer.extractfile(members["index.json"]))
    assert len(index["manifests"]) == 1
    descriptor = index["manifests"][0]
    manifest = json.loads(blob(descriptor))
    config = json.loads(blob(manifest["config"]))
    assert config["architecture"] == {"aarch64": "arm64", "x86_64": "amd64"}[arch]
    for layer in manifest["layers"]:
        with tarfile.open(fileobj=io.BytesIO(blob(layer))) as inner:
            for member in inner:
                if not member.isfile() or not member.name.endswith(".rpm"):
                    continue
                all_rpms.append(member.name)
                if not member.name.startswith(("Packages/systemd-257/", "Packages/release/")):
                    continue
                path = out / Path(member.name).name
                path.write_bytes(inner.extractfile(member).read())
                name = query(path, "--qf", "%{NAME}").strip()
                files = query(path, "--qf", "[%{FILENAMES}\\t%{FILELINKTOS}\\n]")
                rpms[name] = {
                    "sha256": sha(path), "files_and_links": files,
                    "requires": query(path, "--requires").splitlines(),
                    "conflicts": query(path, "--conflicts").splitlines(),
                }

base = "bottlerocket-systemd-257"
sysroot = f"/{arch}-bottlerocket-linux-gnu/sys-root"
unit = sysroot + "/usr/lib/systemd/system/proc-sys-fs-binfmt_misc.mount"
link = sysroot + "/usr/lib/systemd/system/sysinit.target.wants/proc-sys-fs-binfmt_misc.mount"
mount_files = dict(line.split("\t") for line in rpms[base + "-binfmt"]["files_and_links"].splitlines())
assert mount_files == {unit: "", link: "../proc-sys-fs-binfmt_misc.mount"}, mount_files
assert f"({base}-binfmt or bottlerocket-image-feature(uki-image))" in rpms[base]["requires"]
assert base in rpms[base + "-binfmt"]["requires"]
assert "bottlerocket-image-feature(uki-image)" in rpms[base + "-binfmt"]["conflicts"]
assert not any("bottlerocket-release-uki-" in n for n in all_rpms)
assert any(n.startswith("bottlerocket-release-") for n in rpms)
for name, package in rpms.items():
    if name != base + "-binfmt":
        assert unit not in package["files_and_links"] and link not in package["files_and_links"]
    if name.startswith("bottlerocket-release"):
        assert "binfmt" not in package["files_and_links"]
assert "systemd-debug-generator" not in rpms[base]["files_and_links"]
assert "systemd-debug-generator" in rpms[base + "-console"]["files_and_links"]
report = {
    "status": "passed", "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "argv": sys.argv, "request_id": job, "arch": arch,
    "archive": str(archive.relative_to(root)), "archive_sha256": archive_hash,
    "archive_size": item["size"], "provenance": provenance,
    "captured_source_sha256": source_hashes,
    "oci_manifest": descriptor, "verified_layer_count": len(manifest["layers"]),
    "all_rpm_paths": all_rpms, "queried_rpms": rpms, "commands": commands,
    "limitation": "Artifact and RPM metadata checks only; image composition and EC2 validation remain pending.",
}
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: report[k] for k in [
    "status", "request_id", "arch", "archive", "archive_sha256", "verified_layer_count"]}, indent=2))

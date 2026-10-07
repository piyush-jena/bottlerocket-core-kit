"""Check parsed RPM ownership and solve the image-feature dependency cases.

No packages are built or installed. A temporary metadata-only repository
isolates the real spec's binfmt dependency from unrelated library dependencies.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import warnings
import xml.etree.ElementTree as ET

import hawkey
import rpm

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "packages/systemd-257/systemd-257.spec"
rpm.addMacro("_cross_os", "bottlerocket-")
packages = {p.header["name"]: p for p in rpm.spec(str(SPEC)).packages}
base = "bottlerocket-systemd-257"
mount = base + "-binfmt"
feature = "bottlerocket-image-feature(uki-image)"
guard = next(n for n in packages[base].header["requirename"] if mount in n)
requires = list(packages[mount].header["requirename"])
conflicts = list(packages[mount].header["conflictname"])
assert guard == f"({mount} or {feature})"
assert requires == [base] and conflicts == [feature]
files = packages[mount].fileList.split()
assert files == [
    "%{_cross_unitdir}/proc-sys-fs-binfmt_misc.mount",
    "%{_cross_unitdir}/sysinit.target.wants/proc-sys-fs-binfmt_misc.mount",
]
for name, package in packages.items():
    if name != mount:
        assert not any(f in package.fileList.split() for f in files), name
release = ROOT / "packages/release/release.spec"
assert release.read_bytes() == subprocess.check_output(
    ["git", "-C", str(ROOT), "show",
     "209347ba321f023c242f48cb1afc590e5f9793bb:packages/release/release.spec"])
assert not (ROOT / "packages/release/binfmt-misc-mount-uki.conf").exists()

COMMON = "http://linux.duke.edu/metadata/common"
RPM = "http://linux.duke.edu/metadata/rpm"
REPO = "http://linux.duke.edu/metadata/repo"
ET.register_namespace("", COMMON)
ET.register_namespace("rpm", RPM)


def primary_xml(uki, arch):
    metadata = ET.Element(f"{{{COMMON}}}metadata", packages="3")
    rows = [
        (base, [guard], [], []),
        (mount, requires, [], conflicts),
        ("metadata", [], [feature if uki else "bottlerocket-image-feature(no-uki-image)"], []),
    ]
    for name, deps, provides, excludes in rows:
        package = ET.SubElement(metadata, f"{{{COMMON}}}package", type="rpm")
        for tag, value in [("name", name), ("arch", arch), ("summary", name),
                           ("description", name), ("packager", ""), ("url", "")]:
            ET.SubElement(package, f"{{{COMMON}}}{tag}").text = value
        ET.SubElement(package, f"{{{COMMON}}}version", epoch="0", ver="1", rel="1")
        ET.SubElement(package, f"{{{COMMON}}}checksum", type="sha256", pkgid="YES").text = hashlib.sha256(name.encode()).hexdigest()
        ET.SubElement(package, f"{{{COMMON}}}time", file="0", build="0")
        ET.SubElement(package, f"{{{COMMON}}}size", package="1", installed="1", archive="1")
        ET.SubElement(package, f"{{{COMMON}}}location", href=name + ".rpm")
        form = ET.SubElement(package, f"{{{COMMON}}}format")
        ET.SubElement(form, f"{{{RPM}}}license").text = "MIT"
        ET.SubElement(form, f"{{{RPM}}}header-range", start="0", end="0")
        for kind, values in [("requires", deps), ("provides", [name, *provides]), ("conflicts", excludes)]:
            if values:
                entries = ET.SubElement(form, f"{{{RPM}}}{kind}")
                for value in values:
                    ET.SubElement(entries, f"{{{RPM}}}entry", name=value)
    return ET.tostring(metadata, encoding="utf-8", xml_declaration=True)


results = []
warnings.filterwarnings("ignore", category=DeprecationWarning)
for arch in ["x86_64", "aarch64"]:
    for uki, forced in [(True, False), (False, False), (True, True)]:
        with tempfile.TemporaryDirectory(prefix="symphony14-guard-") as tmp:
            directory = Path(tmp)
            data = primary_xml(uki, arch)
            (directory / "primary.xml").write_bytes(data)
            repomd = ET.Element(f"{{{REPO}}}repomd")
            entry = ET.SubElement(repomd, f"{{{REPO}}}data", type="primary")
            ET.SubElement(entry, f"{{{REPO}}}checksum", type="sha256").text = hashlib.sha256(data).hexdigest()
            ET.SubElement(entry, f"{{{REPO}}}location", href="primary.xml")
            ET.SubElement(entry, f"{{{REPO}}}timestamp").text = "0"
            ET.SubElement(entry, f"{{{REPO}}}size").text = str(len(data))
            (directory / "repomd.xml").write_bytes(ET.tostring(repomd))
            repo = hawkey.Repo("guard")
            repo.repomd_fn = str(directory / "repomd.xml")
            repo.primary_fn = str(directory / "primary.xml")
            sack = hawkey.Sack(arch=arch, cachedir=str(directory / "cache"))
            sack.load_repo(repo)
            goal = hawkey.Goal(sack)
            for name in [base, "metadata"] + ([mount] if forced else []):
                goal.install(hawkey.Query(sack).filter(name=name)[0])
            success = goal.run()
            selected = sorted(p.name for p in goal.list_installs()) if success else []
            expected = not forced
            assert success == expected, (arch, uki, forced)
            if success:
                assert (mount in selected) == (not uki), selected
            results.append({"arch": arch, "uki": uki, "force_binfmt": forced,
                            "solver_success": success, "expected_success": expected,
                            "selected": selected})
print(json.dumps({
    "status": "passed", "command": "python3 docs/evidence/issue-14-systemd-guard/verify-guard.py",
    "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
    "guard": guard, "mount_requires": requires, "mount_conflicts": conflicts,
    "exclusive_spec_files": files, "release_spec_matches_upstream": True,
    "cases": results,
    "limitation": "Metadata-only dependency checks; actual built RPMs and fresh AMIs still require validation.",
}, indent=2))

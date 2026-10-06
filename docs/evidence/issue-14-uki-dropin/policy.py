"""Read the live R10.2 policy and prove kernel refusal of module loading."""
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys

root = pathlib.Path("/.bottlerocket/rootfs")
report = {"checks": [], "commands": [], "errors": []}
def require(value, description):
    report["checks"].append({"passed": bool(value), "check": description})
    if not value:
        report["errors"].append(description)
def host(*args, denied=False):
    result = subprocess.run(["sheltie", *args], capture_output=True, text=True, timeout=60)
    report["commands"].append({"argv": list(args), "exit_code": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr})
    if denied:
        require(result.returncode != 0 and "Operation not permitted" in result.stderr, "kernel refuses " + " ".join(args))
    else:
        require(result.returncode == 0, "command succeeds: " + " ".join(args))
    return result.stdout
try:
    require(pathlib.Path("/proc/1/comm").read_text().strip() == "systemd", "host PID namespace")
    report["boot_id"] = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    report["os_release"] = (root / "usr/lib/os-release").read_text()
    release = dict(line.split("=", 1) for line in report["os_release"].splitlines() if "=" in line)
    require(release.get("VERSION_ID") == "1.66.1", "numeric release version")
    require(release.get("BUILD_ID") == sys.argv[1], "exact tested source build identity")
    report["cmdline"] = pathlib.Path("/proc/cmdline").read_text().strip()
    tokens = shlex.split(report["cmdline"])
    require([x for x in tokens if x.startswith("module_blacklist=")] == ["module_blacklist=i8042,binfmt_misc"],
        "one effective blacklist, including i8042 and binfmt_misc")
    require("systemd.mask=proc-sys-fs-binfmt_misc.mount" not in tokens, "mount suppression does not depend on command-line mask")
    report["features"] = (root / "usr/share/bottlerocket/image-features.env").read_text()
    require("IN_PLACE_UPDATES=false" in report["features"].splitlines(), "mantle update feature unchanged")
    for old in ["usr/lib/modprobe.d/00-binfmt-misc.conf",
                "usr/lib/systemd/system/proc-sys-fs-binfmt_misc.mount.d/10-encrypted-storage.conf"]:
        require(not (root / old).exists(), "superseded modprobe policy is absent: " + old)
    dropin = root / "usr/lib/systemd/system/proc-sys-fs-binfmt_misc.mount.d/10-uki.conf"
    report["dropin"] = dropin.read_text()
    require("ConditionPathExists=!/" in report["dropin"].splitlines(), "UKI drop-in carries always-false condition")
    report["image_format"] = (root / "usr/share/bottlerocket/image-format.env").read_text()
    require("UKI_IMAGE=true" in report["image_format"].splitlines(), "UKI image format enabled")
    kernel = host("uname", "-r").strip()
    modules = list((root / "usr/lib/modules" / kernel).rglob("binfmt_misc.ko*"))
    require(len(modules) == 1, "binfmt_misc remains a shipped loadable module")
    report["module_objects"] = [{"path": str(p.relative_to(root)),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in modules]
    def state(label):
        value = {"module_lines": [l for l in pathlib.Path("/proc/modules").read_text().splitlines() if l.startswith("binfmt_misc ")],
            "filesystem_lines": [l for l in pathlib.Path("/proc/filesystems").read_text().splitlines() if l.split()[-1] == "binfmt_misc"],
            "mount_lines": [l for l in pathlib.Path("/proc/1/mountinfo").read_text().splitlines() if " - binfmt_misc " in l],
            "sys_module": pathlib.Path("/sys/module/binfmt_misc").exists()}
        for name, observed in value.items():
            require(not observed, label + ": absent " + name)
        report[label] = value
    state("before")
    host("modprobe", "binfmt_misc", denied=True)
    host("modprobe", "fs-binfmt_misc", denied=True)
    if len(modules) == 1:
        host("insmod", "/" + str(modules[0].relative_to(root)), denied=True)
    state("after")
    host("systemctl", "start", "proc-sys-fs-binfmt_misc.mount")
    unit = host("systemctl", "show", "proc-sys-fs-binfmt_misc.mount",
        "-p", "LoadState", "-p", "ActiveState", "-p", "FragmentPath",
        "-p", "ConditionResult", "-p", "DropInPaths", "-p", "Result")
    report["mount_unit"] = dict(line.split("=", 1) for line in unit.splitlines() if "=" in line)
    require(report["mount_unit"].get("LoadState") == "loaded", "mount unit loads normally")
    require(report["mount_unit"].get("ConditionResult") == "no", "mount start skipped by condition")
    require("/proc-sys-fs-binfmt_misc.mount.d/10-uki.conf" in report["mount_unit"].get("DropInPaths", ""), "packaged drop-in is effective")
    require(report["mount_unit"].get("ActiveState") == "inactive", "skipped mount remains inactive")
    require(report["mount_unit"].get("Result") == "success", "skipped mount is not failed")
    state("after_mount_start")
    report["failed_units"] = host("systemctl", "--failed", "--no-pager", "--no-legend")
    require(not report["failed_units"].strip(), "no failed boot units")
    journal = host("journalctl", "-k", "-b", "--no-pager", "-o", "cat")
    report["blacklist_journal"] = [line for line in journal.splitlines() if "binfmt_misc" in line]
    require(any("blacklisted" in line for line in report["blacklist_journal"]), "kernel log confirms blacklist denial")
except Exception as error:
    report["errors"].append(repr(error))
report["status"] = "passed" if not report["errors"] else "failed"
print(json.dumps(report, indent=2))
sys.exit(0 if report["status"] == "passed" else 1)

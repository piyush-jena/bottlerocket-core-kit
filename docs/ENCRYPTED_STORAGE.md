# Encrypted Storage Implementation

This document describes how Bottlerocket implements encrypted storage for both the data partition (`/local`) and the datastore directory (`/.bottlerocket/datastore`).

## Overview

Bottlerocket's encrypted storage feature provides:

- **Block device encryption** using LUKS2 for the `/local` partition
- **Directory encryption** using fscrypt for `/.bottlerocket/datastore`
- **TPM2-based key management** with automatic unlocking
- **Boot phase measurements** for attestation and policy enforcement
- **Transparent operation** with no user intervention required

All encryption keys are sealed to TPM2 PCRs, ensuring data can only be decrypted when the system boots in a trusted state.

> **Two block-encryption modes.** The description above (LUKS2 + TPM2-sealed keys) applies to variants with the `encrypted-storage` image feature.
> Variants that additionally enable the `ephemeral-encryption-keys` image feature use a different scheme for **block devices** (the `BOTTLEROCKET-DATA` and `BOTTLEROCKET-PRIVATE` partitions, and the `EPHEMERAL-DATA` device from `apiclient ephemeral-storage init`): **plain-mode dm-crypt with a per-boot random key, no TPM sealing, and no persisted key**.
> See [Ephemeral Encryption Keys (plain-mode)](#ephemeral-encryption-keys-plain-mode) below.
> The fscrypt datastore directory encryption is unchanged in both modes (still TPM2-sealed).

## Architecture

### Components

1. **rottweiler** - Unified storage encryption helper (Rust binary)
2. **systemd services** - Orchestrate encryption, unlocking, and measurements
3. **TPM2** - Hardware security module for key sealing and measurements
4. **systemd-creds** - Encrypts keys with TPM2 PCR binding
5. **cryptsetup** - LUKS2 block device encryption
6. **fscrypt** - Directory-level encryption

### Key Storage

Encryption keys are stored in `/.bottlerocket/keystore/` as TPM2-sealed credentials:
- `datastore` - Key for `/.bottlerocket/datastore` (fscrypt)
- `bottlerocket-data` - Key for `/dev/disk/by-partlabel/BOTTLEROCKET-DATA` (LUKS2)

Keys are:
- 64 bytes of random data from `/dev/random`
- Encrypted with `systemd-creds` using TPM2 PCR binding
- Automatically decrypted during boot when PCR values match

### TPM2 PCR Binding

Keys are bound to specific TPM2 Platform Configuration Registers (PCRs):

| PCR | Purpose | Why Included |
|-----|---------|--------------|
| 4 | Boot loader code (shim, grub, kernel) | Ensures kernel hasn't changed (if updates disabled) |
| 7 | Secure Boot policy | Prevents boot of unsigned code |
| 9 | Kernel command line (includes dm-verity root hash) | Ensures userspace hasn't changed (if updates disabled) |
| 11 | Boot phase | Tracks boot progression |
| 14 | Machine-owner keys (MOK) | Validates custom certificates |

Additional PCRs used for measurements (not bound to keys):

| PCR | Purpose | Usage |
|-----|---------|-------|
| 8 | OS settings | Measured after configuration completes |
| 10 | (Reserved) | Reserved for future use |

**PCR selection logic:**
- **With in-place updates enabled**: PCRs 7+11+14 (allows kernel/userspace updates)
- **With in-place updates disabled**: PCRs 4+7+9+11+14 (locks to specific kernel/userspace)

This ensures encrypted data can only be accessed when booting a trusted configuration.

## Boot Flow

### First Boot

```
1. tpm2.target
   ↓
2. encrypt-local-fs.service
   - Checks if /dev/disk/by-partlabel/BOTTLEROCKET-DATA is unencrypted
   - Generates random key → /.bottlerocket/keystore/bottlerocket-data
   - Encrypts key with systemd-creds (TPM2 PCRs 7+11+14 or 4+7+9+11+14)
   - Formats partition with LUKS2 using the key
   ↓
3. unlock-local-fs.service
   - Decrypts key from /.bottlerocket/keystore/bottlerocket-data
   - Attaches LUKS device as /dev/mapper/BOTTLEROCKET-DATA
   ↓
4. prepare-local-fs.service
   - Creates filesystem on /dev/mapper/BOTTLEROCKET-DATA if needed
   ↓
5. local.mount
   - Mounts /dev/mapper/BOTTLEROCKET-DATA to /local
   ↓
6. encrypt-datastore.service
   - Enables encrypt feature on BOTTLEROCKET-PRIVATE filesystem
   - Generates random key → /.bottlerocket/keystore/datastore
   - Encrypts key with systemd-creds (TPM2 PCRs)
   - Sets fscrypt policy on /.bottlerocket/datastore
   ↓
7. unlock-datastore.service
   - Decrypts key from /.bottlerocket/keystore/datastore
   - Adds key to kernel keyring, unlocking directory
```

### Subsequent Boots

```
1. tpm2.target
   ↓
2. encrypt-local-fs.service
   - Checks if already encrypted → skips (ExecCondition fails)
   ↓
3. unlock-local-fs.service
   - Decrypts key and attaches LUKS device
   ↓
4. prepare-local-fs.service
   - Filesystem already exists → skips
   ↓
5. local.mount
   - Mounts encrypted partition
   ↓
6. encrypt-datastore.service
   - Directory already encrypted → skips (ExecCondition fails)
   ↓
7. unlock-datastore.service
   - Decrypts key and unlocks directory
```

## Service Details

### Block Device Encryption (/local)

#### encrypt-local-fs.service

**Purpose:** One-time setup to encrypt the BOTTLEROCKET-DATA partition.

**Key behaviors:**
- Only runs if TPM2 is available (`ConditionSecurity=tpm2`)
- Only runs if partition is unencrypted (`ExecCondition`)
- Generates 64-byte random key
- Encrypts key with TPM2 PCR binding
- Formats partition with LUKS2 (PBKDF2, 1000 iterations)

**Dependencies:**
- After: `tpm2.target`, `dev-disk-by-partlabel-BOTTLEROCKET-DATA.device`
- Before: `unlock-local-fs.service`
- Required by: `unlock-local-fs.service`

#### unlock-local-fs.service

**Purpose:** Decrypt and attach the LUKS device on every boot.

**Key behaviors:**
- Decrypts key from keystore using TPM2
- Attaches LUKS device as `/dev/mapper/BOTTLEROCKET-DATA`
- Detaches on shutdown (`ExecStop`)

**Dependencies:**
- After: `cryptsetup-pre.target`, `systemd-udevd-kernel.socket`
- Before: `cryptsetup.target`, `local-fs.target`
- Required by: `local-fs.target`

#### prepare-local-fs.service (modified)

**Drop-in:** `prepare-local-fs-encrypted.conf`

**Changes:**
- Operates on `/dev/mapper/BOTTLEROCKET-DATA` instead of raw partition
- Depends on `unlock-local-fs.service`

#### local.mount (modified)

**Drop-in:** `local-mount-encrypted.conf`

**Changes:**
- Mounts `/dev/mapper/BOTTLEROCKET-DATA` instead of raw partition

#### repart-local.service (modified)

**Drop-in:** `repart-local-encrypted.conf`

**Changes:**
- Resizes LUKS container after partition resize
- Uses `rottweiler resize block-device` command

### Directory Encryption (/.bottlerocket/datastore)

#### encrypt-datastore.service

**Purpose:** One-time setup to encrypt the datastore directory.

**Key behaviors:**
- Only runs if TPM2 is available
- Only runs if directory is unencrypted (`ExecCondition`)
- Enables encrypt feature on ext4 filesystem (`tune2fs -O encrypt`)
- Generates 64-byte random key
- Encrypts key with TPM2 PCR binding
- Sets fscrypt policy on directory

**Dependencies:**
- After: `tpm2.target`
- Before: `unlock-datastore.service`
- Required by: `unlock-datastore.service`

#### unlock-datastore.service

**Purpose:** Unlock the encrypted directory on every boot.

**Key behaviors:**
- Decrypts key from keystore using TPM2
- Adds key to kernel keyring
- Directory becomes accessible

**Dependencies:**
- Before: `migrator.service`, `storewolf.service`
- Required by: `migrator.service`, `storewolf.service`

## Ephemeral Encryption Keys (plain-mode)

**Keywords:** ephemeral-encryption-keys, plain-mode, dm-crypt, aes-xts-plain64, per-boot key, no-tpm, encrypt-and-attach

Variants with the `ephemeral-encryption-keys` image feature (in addition to `encrypted-storage`) encrypt their block devices with **plain-mode dm-crypt using a fresh random key generated on every boot**, instead of the LUKS2 + TPM2-sealed-key flow described above.

### Rationale

Plain mode has no on-disk header, and the key is fed directly into the kernel device-mapper table:

- **Nothing persisted.** The key is 64 bytes read from `/dev/random` into a zeroized in-memory buffer, passed once to `cryptsetup` on stdin, and thereafter exists only inside the kernel dm-crypt mapping. It is never written to disk or tmpfs, never placed in the kernel keyring, and never TPM-sealed — no keystore file is written for `bottlerocket-data`/`bottlerocket-private`, so there is nothing to read back and nothing to delete.
- **No TPM dependency.** Because the key is never TPM-sealed, the new units carry **no `ConditionSecurity=tpm2`**. This removes the boot hang that occurred when a warm reboot left the TPM hierarchy in an unowned/changed state.
- **Single step, no format→attach window.** Plain mode writes no header, so encryption and attach are a single `cryptsetup open --type plain` operation.
- **Automatic rotation.** A fresh per-boot key makes any prior on-disk contents unreadable, so `/local`, `/.bottlerocket`, and any initialized ephemeral storage are effectively wiped and reformatted on every boot — the intended ephemeral behavior.

### Cipher

The plain-mode mapper is opened with a fixed cipher for the 64-byte key. Every site — DATA, PRIVATE, and `ephemeral-storage init` — uses the same single-step open, driven by `rottweiler encrypt-and-attach block-device <device>` (which generates the key and derives the mapper name from the device file name). The key is passed once to `cryptsetup` on stdin and then exists only inside the kernel dm-crypt mapping:

```bash
cryptsetup open --type plain \
  --cipher aes-xts-plain64 \
  --key-size 512 \
  --hash plain \
  <device> <mapper-name>   # key on stdin
```

`--hash plain` means no key derivation: the 64 raw bytes are used verbatim as the volume key.

No mapper is ever resized, so the volume key never has to be handed to a second process. That is deliberate — see [DATA grow-before-open](#data-grow-before-open). A keyring-backed volume key (`--volume-key-keyring`) would work, but it would require publishing the key in the root user keyring and granting `KeyringMode=shared` to every unit that touches the mapper.

### Services

Two combined single-step services (shipped only in the `ephemeral-crypt` subpackage) replace the legacy `encrypt-*` + `unlock-*` pair for their partition:

#### encrypt-unlock-local-fs.service

**Purpose:** Open `/dev/mapper/BOTTLEROCKET-DATA` as a plain-mode mapper (per-boot key).

**Key behaviors:**
- Single `ExecStart=/usr/bin/rottweiler encrypt-and-attach block-device ${BOTTLEROCKET_DATA}` (no `generate-key`, no `delete-key`, no `ConditionSecurity=tpm2`).
- Grows the raw DATA partition **before** opening the mapper, so the mapper is full size from the start and never needs a keyed online resize: `ExecStartPre=-/usr/bin/systemd-repart --dry-run=no ${BOTTLEROCKET_DATA}`.
- Between the grow and the open, waits on a **device-scoped** barrier: `ExecStartPre=/usr/bin/udevadm wait --settle --timeout=30 ${BOTTLEROCKET_DATA}`. See [DATA grow-before-open](#data-grow-before-open).
- Detaches on shutdown (`ExecStop`).

**Dependencies:**
- After: `dev-disk-by-partlabel-BOTTLEROCKET-DATA.device`, `cryptsetup-pre.target`, `systemd-udevd-kernel.socket`
- Before: `cryptsetup.target`, `blockdev@dev-mapper-BOTTLEROCKET-DATA.target`
- Required by: `local-fs.target`

#### encrypt-unlock-private-fs.service

**Purpose:** Open `/dev/mapper/BOTTLEROCKET-PRIVATE` as a plain-mode mapper (per-boot key).

**Key behaviors:**
- Single `ExecStart=/usr/bin/rottweiler encrypt-and-attach block-device ${BOTTLEROCKET_PRIVATE}` (no TPM condition).
- `prepare-private-fs.service` (mkfs.ext4 `-O encrypt`) and the `bottlerocket-mount-encrypted.conf` mount still run after this unit.

**Dependencies:**
- After: `dev-disk-by-partlabel-BOTTLEROCKET-PRIVATE.device`, `cryptsetup-pre.target`
- Before: `cryptsetup.target`, `blockdev@dev-mapper-BOTTLEROCKET-PRIVATE.target`
- Required by: `prepare-private-fs.service`

### Packaging: each feature ships its own block-device set

The three variant classes get their block-device units from **separate subpackages** rather than sharing one unit set and neutralizing it with drop-ins. `release.spec` splits the crypto units three ways:

| Subpackage | Installed when | Contents |
|------------|----------------|----------|
| `release-crypt` (shared base) | `image-feature(encrypted-storage)` | datastore fscrypt units (`encrypt-datastore` / `unlock-datastore`) + their datastore drop-ins, and the measure/pcrphase TPM units |
| `release-crypt-luks` (LUKS block-device set) | `encrypted-storage` **and NOT** `ephemeral-encryption-keys` | `encrypt-local-fs.service`, `unlock-local-fs.service`, and the `10-encrypted` `prepare-local-fs` / `local.mount` / `repart-local` drop-ins |
| `release-ephemeral-crypt` (plain-mode set) | `image-feature(ephemeral-encryption-keys)` | `encrypt-unlock-local-fs.service`, `encrypt-unlock-private-fs.service`, `prepare-private-fs.service`, `bottlerocket-mount-encrypted.conf`, `run-rottweiler.mount`, the ephemeral datastore drop-ins, and the plain-mode DATA wiring drop-ins (`10-plain` ×2) |

The "NOT ephemeral" gate on the LUKS set uses the RPM rich-dependency idiom
`((%{name}-crypt-luks or %{_cross_os}image-feature(ephemeral-encryption-keys)) if %{_cross_os}image-feature(encrypted-storage))`,
which pulls in `crypt-luks` exactly when `encrypted-storage` is present and `ephemeral-encryption-keys` is absent. `crypt-luks` also carries `Conflicts: %{_cross_os}image-feature(ephemeral-encryption-keys)` as a belt-and-suspenders guard.

**Consequences of the split:**
- On an **ephemeral image** the LUKS block-device units (`encrypt-local-fs.service`, `unlock-local-fs.service`) and the `10-encrypted` drop-ins are **absent** — not installed-then-skipped. There is nothing to neutralize, so the four legacy neutralizing drop-ins are gone (see below).
- On an **encrypted-storage-only image** the LUKS block-device units and their `10-encrypted` drop-ins are present and run exactly as before; the plain-mode units are absent.
- Both encrypted-storage-only and ephemeral images still get the shared `release-crypt` base (datastore fscrypt + TPM units), so datastore behavior is unchanged.

The legacy LUKS **PRIVATE** units (`encrypt-private-fs.service` / `unlock-private-fs.service`) are **retired** — no image ships them. On encrypted-storage-only images `BOTTLEROCKET-PRIVATE` is left **unencrypted** (raw partition mounted at `/.bottlerocket`); PRIVATE encryption is exclusively an ephemeral, plain-mode feature.

**Removed neutralizing drop-ins.** Because the ephemeral image no longer installs the LUKS units, the four drop-ins that used to skip / no-op / rebind them are **deleted**:
`encrypt-local-fs.service.d/20-ephemeral.conf`, `unlock-local-fs.service.d/20-ephemeral.conf`, `prepare-local-fs.service.d/20-ephemeral.conf`, and `repart-local.service.d/20-ephemeral.conf`. The datastore drop-ins (`encrypt-datastore.service.d/20-ephemeral.conf`, `unlock-datastore.service.d/20-ephemeral.conf`, `encrypt-datastore.service.d/30-private-luks.conf`) are **kept** — they express genuinely different datastore behavior, not LUKS-unit neutralization.

### DATA boot chain (ephemeral plain-mode wiring drop-ins)

Since the ephemeral set no longer inherits the `10-encrypted` drop-ins, it ships its own self-contained wiring drop-ins that point the DATA chain at the plain mapper. Because the LUKS units are absent, these set their values **directly** (no `BindsTo`/`ExecStart` reset needed):

- `prepare-local-fs.service.d/10-plain.conf` — `BindsTo=`/`After=encrypt-unlock-local-fs.service` and `Environment=DATA_PARTITION_BLOCK_DEVICE=/dev/mapper/BOTTLEROCKET-DATA` (mkfs the mapper). Mirrors `prepare-local-fs-encrypted.conf`, bound to the plain open unit instead of `unlock-local-fs.service`.
- `local.mount.d/10-plain.conf` — `What=/dev/mapper/BOTTLEROCKET-DATA` (mount the mapper, not the raw partition). Mirrors `local-mount-encrypted.conf`.

There is **no** ephemeral `repart-local` drop-in: the partition is already full size by the time `repart-local` runs, so the base unit's `systemd-repart` grow and `systemd-growfs /local` are idempotent no-ops.

The resulting ephemeral DATA chain:

```
encrypt-unlock-local-fs   (grow partition → udevadm wait → open plain mapper full-size)
  → prepare-local-fs      (+10-plain: BindsTo/After encrypt-unlock; mkfs the mapper)
  → local.mount           (+10-plain: What=/dev/mapper/BOTTLEROCKET-DATA)
  → repart-local          (base systemd-repart grow + growfs /local; both no-ops)
```

### DATA grow-before-open

The DATA partition has to be grown to fill its disk (the EBS volume is usually larger than the image), and plain-mode dm-crypt has no on-disk header — so a mapper opened over a small partition could only be extended later by a `cryptsetup resize` that **re-supplies the volume key**. In a systemd boot that resize necessarily runs in a different service (`repart-local`, which carries `RequiresMountsFor=/local` and therefore runs post-mount), which would mean publishing the per-boot key in the root user keyring as a `logon` key and giving both units `KeyringMode=shared` — systemd's default `KeyringMode=private` gives each service a fresh session keyring with `@u` unlinked, so the kernel `request_key()` behind `--volume-key-keyring` fails with `Required key not available` (ENOKEY).

The design avoids that entirely: **grow the partition first, then open the mapper over it.** The mapper is full size from the start, nothing is ever resized, and the volume key never leaves the process that generated it — so every unit keeps systemd's default `KeyringMode=private`.

The cost is that the open follows a GPT rewrite. Rewriting the partition table makes udev briefly delete and recreate `/dev/disk/by-partlabel/BOTTLEROCKET-DATA`, and resolving the symlink inside that window fails the open with `does not exist or access denied` (Bottlerocket #845). The barrier between the two steps is therefore **device-scoped**:

```
ExecStartPre=-/usr/bin/systemd-repart --dry-run=no ${BOTTLEROCKET_DATA}
ExecStartPre=/usr/bin/udevadm wait --settle --timeout=30 ${BOTTLEROCKET_DATA}
ExecStart=/usr/bin/rottweiler encrypt-and-attach block-device ${BOTTLEROCKET_DATA}
```

`udevadm wait <device>` waits for that specific symlink to exist **and** for its device to be udev-initialized; `--settle` additionally drains the udev event queue. A bare `udevadm settle` does neither — it drains the global queue and returns successfully while the symlink is still missing. `--timeout=30` bounds the wait so a genuinely absent device fails the unit instead of hanging the boot.

This is a barrier, not the removal of a race class: systemd #40499 shows udevd can perform a userspace partition-table reread that recreates the node *after* a barrier returns, so no post-write barrier is absolute. The warm-reboot endurance loop is the arbiter.

Also note the grow must stay in this unit rather than being ordered `After=repart-local.service`: `repart-local` has `RequiresMountsFor=/local`, which would close the cycle `local.mount → prepare-local-fs → encrypt-unlock-local-fs → repart-local → local.mount`, and systemd 257 cannot reset a dependency directive from a drop-in.

### apiclient ephemeral-storage init

When `ephemeral-encryption-keys` is enabled, `encrypt_ephemeral_device` (apiserver) runs a single `rottweiler encrypt-and-attach block-device /dev/disk/EPHEMERAL-DATA` (plain, per-boot key) instead of the LUKS `generate-key`/`encrypt`/`attach` sequence, still returning `/dev/mapper/EPHEMERAL-DATA`. When only `encrypted-storage` is enabled, the unchanged LUKS sequence runs.

### Datastore fscrypt is unchanged

The `/.bottlerocket/datastore` fscrypt directory encryption keeps its TPM2-sealed-key flow (`encrypt-datastore.service` / `unlock-datastore.service`) on ephemeral images as well. This is redundant with the plain-mode PRIVATE partition underneath it but is intentionally left as-is.

## TPM2 Measurements

Bottlerocket extends TPM2 PCRs at various boot stages to establish a cryptographic chain of trust.

### PCR 8: OS Settings

**Service:** `measure-settings.service`

**What's measured:** Canonicalized OS settings from the API

**When:** After `settings-applier.service` and `apiserver.service`, before `bootstrap-commands.service`

**Purpose:** Detect unauthorized configuration changes

**Note:** This measurement occurs before any external configuration can be applied and before any external code can run.

### PCR 9: Kernel Command Line

**Service:** `measure-cmdline.service`

**What's measured:** Contents of `/proc/cmdline` (includes dm-verity root hash)

**When:** Early boot, before `sysinit.target`

**Purpose:** Verify boot parameters and userspace integrity

**Note:** While the kernel normally performs this measurement, Bottlerocket measures from userspace to capture the final command line after bootconfig customization is applied.

### PCR 11: Boot Phases

**Services:**
- `systemd-pcrphase-sysinit.service`
- `systemd-pcrphase-preconfigured.service`
- `systemd-pcrphase-configured.service`
- `systemd-pcrphase-multi-user.service`

**What's measured:** Boot phase strings (`sysinit`, `preconfigured`, `configured`, `ready`, `shutdown`, `final`)

**Purpose:** Track boot progression and establish different trust levels at different stages

**Phase progression:**
```
sysinit → preconfigured → configured → ready → shutdown → final
```

Each phase extends PCR 11 with the phase name as raw bytes (no newline, no null terminator).

**Security model:** Keys sealed to PCR 11 can only be unsealed if the system has not advanced beyond the boot phase during which they were generated. For example:

- **Local storage and datastore keys** are generated before `sysinit` completes, so they can never be unsealed after the `sysinit` phase on any boot
- This provides time-limited access: keys are only accessible during early boot when needed
- After initial setup, keys become permanently inaccessible, reducing attack surface
- Ephemeral storage keys are sealed to the phase when first initialized (preconfigured, configured, or multi-user), preventing initialization from being moved to an earlier phase

## Implementation Details

### Key Generation

Keys are generated using `/dev/random` (blocking, cryptographically secure):

```rust
let mut random_bytes = vec![0u8; 64];
fs::File::open("/dev/random")?.read_exact(&mut random_bytes)?;
```

### Key Encryption

Keys are encrypted using `systemd-creds` with TPM2 PCR binding:

```bash
systemd-creds encrypt - - \
  --name <key-id> \
  --with-key=tpm2 \
  --tpm2-pcrs=7+11+14  # or 4+7+9+11+14 if updates disabled
```

### LUKS2 Formatting

Block devices are formatted with minimal PBKDF2 iterations (1000) since keys are already high-entropy:

```bash
cryptsetup luksFormat \
  --type luks2 \
  --pbkdf pbkdf2 \
  --pbkdf-force-iterations 1000 \
  --batch-mode \
  <device> -
```

This matches systemd's behavior and avoids unnecessary key stretching.

### fscrypt Configuration

Directories are encrypted with:
- **Contents encryption:** AES-256-XTS
- **Filenames encryption:** AES-256-CTS
- **Padding:** 32 bytes
- **Policy version:** v2

Key identifiers are derived using HKDF-SHA512:

```rust
let hkdf = Hkdf::<Sha512>::new(None, key);
hkdf.expand(b"fscrypt\x00\x01", &mut identifier)?;
```

### PCR Extension

PCRs are extended with SHA256, SHA384, and SHA512 hashes:

```bash
tpm2_pcrextend <pcr>:sha256=<hash>,sha384=<hash>,sha512=<hash>
```

## Security Considerations

### Threat Model

**Protects against:**
- Data theft from powered-off systems (disk removal)
- Unauthorized boot configurations (via PCR binding)
- Tampering with kernel, userspace, or boot parameters
- Unauthorized configuration changes (via PCR 8)

**Does not protect against:**
- Physical attacks on running systems (keys in memory)
- Firmware-level compromises before measurements
- Physical TPM attacks (requires specialized equipment)
- Cold boot attacks (DRAM remanence)

### Key Security

- Keys never touch disk in plaintext
- Keys are zeroized after use (Rust `ZeroizeOnDrop`)
- Keys are only accessible when TPM PCRs match expected values
- Keystore directory has restrictive permissions (UMask=0077)

### Update Handling

**With in-place updates enabled:**
- Keys bound to PCRs 7+11+14 (excludes kernel/userspace measurements)
- Kernel and userspace updates work without re-encryption
- Still protected by Secure Boot (PCR 7) and boot phases (PCR 11)

**With in-place updates disabled:**
- Keys bound to PCRs 4+7+9+11+14 (includes kernel/userspace measurements)
- Any kernel or userspace change breaks decryption
- Provides strongest security but requires re-encryption for updates

### Recovery

If TPM PCR values change unexpectedly (e.g., firmware update, hardware change):
- Encrypted data becomes inaccessible
- No built-in recovery mechanism
- Requires backup/restore or data loss

**Mitigation strategies:**
- Test updates in non-production environments
- Maintain backups of critical data
- Consider using PCR policies that allow updates (7+11+14)

## Verification

### Check Encryption Status

```bash
# Check if block device is encrypted
rottweiler check block-device /dev/disk/by-partlabel/BOTTLEROCKET-DATA encrypted

# Check if directory is encrypted
rottweiler check directory /.bottlerocket/datastore encrypted
```

### Verify TPM PCR Values

```bash
# Read current PCR values
tpm2_pcrread sha256:4,7,9,11,14

# View PCR event log
tpm2_eventlog /sys/kernel/security/tpm0/binary_bios_measurements
```

### Check Key Binding

```bash
# List keys in keystore
ls -la /.bottlerocket/keystore/

# Attempt to decrypt a key (requires matching PCR values)
systemd-creds decrypt /.bottlerocket/keystore/bottlerocket-data - --name bottlerocket-data
```

### Verify LUKS Configuration

```bash
# Show LUKS header information
cryptsetup luksDump /dev/disk/by-partlabel/BOTTLEROCKET-DATA

# Check LUKS status
cryptsetup status BOTTLEROCKET-DATA
```

### Verify fscrypt Configuration

```bash
# Check filesystem encryption support
tune2fs -l /dev/disk/by-partlabel/BOTTLEROCKET-PRIVATE | grep encrypt

# Show directory encryption policy
rottweiler check directory /.bottlerocket/datastore encrypted
```

## References

### External Documentation

- [systemd-creds(1)](https://www.freedesktop.org/software/systemd/man/systemd-creds.html) - Credential encryption
- [cryptsetup(8)](https://man7.org/linux/man-pages/man8/cryptsetup.8.html) - LUKS management
- [TPM2 Tools](https://github.com/tpm2-software/tpm2-tools) - TPM2 utilities
- [Linux TPM PCR Registry](https://uapi-group.org/specifications/specs/linux_tpm_pcr_registry/) - PCR definitions

### Source Code

- `sources/rottweiler/` - Storage encryption helper implementation
- `packages/release/release.spec` - Service packaging
- `packages/release/encrypt-*.service` - Encryption services
- `packages/release/unlock-*.service` - Unlocking services
- `packages/release/measure-*.service` - Measurement services
- `packages/release/systemd-pcrphase-*.service` - Boot phase measurements

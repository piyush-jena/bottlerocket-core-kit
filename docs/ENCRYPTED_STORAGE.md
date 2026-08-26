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

> **Two block-encryption modes.** The description above (LUKS2 with TPM2-sealed keys) applies to variants with the `encrypted-storage` image feature. Variants that also enable `ephemeral-encryption-keys` encrypt their block devices with plain-mode dm-crypt using a per-boot key that is generated, used, and deleted within a single service, so nothing is persisted across a reboot. The key is briefly TPM2-sealed into the `/run/rottweiler` tmpfs keystore while the service runs and then deleted, rather than being written to disk. The fscrypt datastore encryption is the same in both modes. See [Ephemeral Encryption Keys](#ephemeral-encryption-keys).

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

## Ephemeral Encryption Keys

Variants with the `ephemeral-encryption-keys` image feature (in addition to `encrypted-storage`) encrypt block devices with plain-mode dm-crypt using a fresh key generated on every boot, instead of the LUKS2 flow described above. This covers `BOTTLEROCKET-DATA`, `BOTTLEROCKET-PRIVATE`, and the `EPHEMERAL-DATA` device from `apiclient ephemeral-storage init`.

Each key is generated, used to open the mapper, and deleted within the single service that opens the device, so nothing survives that service. While it exists the key is TPM2-sealed into the `/run/rottweiler` tmpfs keystore (never written to disk), matching the datastore key handling below.

Directory encryption for `/.bottlerocket/datastore` still uses fscrypt with a TPM2-sealed key, and it runs as a single service (`encrypt-unlock-datastore.service`) instead of the `encrypt-datastore.service` / `unlock-datastore.service` pair, with the sealed key living in the `/run/rottweiler` tmpfs keystore rather than on `/.bottlerocket`.

### Keys

No key survives the service that uses it. Each key is 64 bytes read from `/dev/random`, TPM2-sealed with `systemd-creds`, and written to the `/run/rottweiler` tmpfs keystore only for the duration of the service that opens the device:

- The `bottlerocket-data`, `bottlerocket-private`, and ephemeral-storage keys are generated, used, and deleted within a single service (an `ExecStartPost=rottweiler delete-key` removes each one)
- Keys live in the `/run/rottweiler` tmpfs keystore, never on `/.bottlerocket` and never on a persistent disk
- Data does not survive a reboot: the next boot generates a new key, so prior contents are unreadable and the filesystem is recreated

### Boot Flow (Every Boot)

Nothing is conditional on prior state, so every boot follows the same path.

```
1. encrypt-unlock-local-fs.service
   - Generates a per-boot key, opens BOTTLEROCKET-DATA as /dev/mapper/BOTTLEROCKET-DATA with it,
     creates the filesystem, then grows the partition and resizes the mapper, and finally deletes
     the key. Opening the mapper before growing the partition avoids waiting on the by-partlabel
     device node to be recreated.
   ↓
2. prepare-local-fs.service
   - Filesystem already created by encrypt-unlock-local-fs.service; this is a no-op
   ↓
3. local.mount
   - Mounts /dev/mapper/BOTTLEROCKET-DATA to /local
   ↓
4. repart-local.service
   - The base ExecStart (systemd-repart) is masked by a drop-in, since the grow and resize already
     happened in encrypt-unlock-local-fs.service; systemd-growfs /local grows the filesystem to
     fill the resized mapper
```

```
1. encrypt-unlock-private-fs.service
   - Generates a per-boot key, opens BOTTLEROCKET-PRIVATE as /dev/mapper/BOTTLEROCKET-PRIVATE with
     it, then deletes the key
   ↓
2. prepare-private-fs.service
   - Creates filesystem on /dev/mapper/BOTTLEROCKET-PRIVATE (mkfs.ext4 -O encrypt)
   ↓
3. .bottlerocket.mount
   - Mounts /dev/mapper/BOTTLEROCKET-PRIVATE to /.bottlerocket
   ↓
4. encrypt-unlock-datastore.service
   - Generates a TPM2-sealed key into the /run/rottweiler tmpfs keystore
   - Sets the fscrypt policy on /.bottlerocket/datastore and unlocks it
   - Deletes the sealed key from the keystore (ExecStartPost)
```

### Services

#### encrypt-unlock-local-fs.service

**Purpose:** Encrypt and open the BOTTLEROCKET-DATA partition in one step.

**Key behaviors:**
- Generates a per-boot key (`rottweiler generate-key bottlerocket-data`), opens the plain-mode mapper with it (`rottweiler encrypt-and-attach block-device`), and creates the filesystem — all before growing the partition, so the mapper is opened against the current device node and the unit does not have to wait for the by-partlabel symlink to be recreated
- Grows the partition (`systemd-repart`) and then the mapper (`rottweiler resize block-device`)
- `ExecStartPost=rottweiler delete-key bottlerocket-data` removes the sealed key, so nothing outlives the unit
- Detaches on shutdown (`ExecStop`)

**Dependencies:**
- After: `dev-disk-by-partlabel-BOTTLEROCKET-DATA.device`, `cryptsetup-pre.target`, `systemd-udevd-kernel.socket`
- Before: `cryptsetup.target`, `blockdev@dev-mapper-BOTTLEROCKET-DATA.target`
- Required by: `local-fs.target`

#### encrypt-unlock-private-fs.service

**Purpose:** Encrypt and open the BOTTLEROCKET-PRIVATE partition in one step.

**Key behaviors:**
- Generates a per-boot key (`rottweiler generate-key bottlerocket-private`) and opens the plain-mode mapper with it (`rottweiler encrypt-and-attach block-device`)
- `ExecStartPost=rottweiler delete-key bottlerocket-private` removes the sealed key, so nothing outlives the unit
- `RequiresMountsFor=/run/rottweiler` pulls in the tmpfs keystore, so the key cannot land on a plain `/run` directory
- Detaches on shutdown (`ExecStop`)

**Dependencies:**
- After: `dev-disk-by-partlabel-BOTTLEROCKET-PRIVATE.device`, `cryptsetup-pre.target`, `systemd-udevd-kernel.socket`
- Before: `cryptsetup.target`, `blockdev@dev-mapper-BOTTLEROCKET-PRIVATE.target`
- Required by: `prepare-private-fs.service`

#### encrypt-unlock-datastore.service

**Purpose:** Encrypt and unlock `/.bottlerocket/datastore` in one step, replacing the
`encrypt-datastore.service` / `unlock-datastore.service` pair used in LUKS mode.

**Key behaviors:**
- `rottweiler generate-key datastore` seals a fresh key into the `/run/rottweiler` tmpfs keystore
- `rottweiler encrypt directory` sets the fscrypt policy, `rottweiler unlock directory` adds the key
- `ExecStartPost=rottweiler delete-key datastore` removes the sealed key, so nothing outlives the unit
- `RequiresMountsFor=/.bottlerocket /run/rottweiler` pulls in both the PRIVATE filesystem and the
  tmpfs keystore, so the key cannot land on a plain `/run` directory

**Dependencies:**
- Before: `migrator.service`, `storewolf.service`
- Required by: `migrator.service`, `storewolf.service`
- Requires mounts for: `/.bottlerocket`, `/run/rottweiler`

#### prepare-local-fs.service (modified)

**Drop-in:** `prepare-local-fs-plain.conf`

**Changes:**
- Operates on `/dev/mapper/BOTTLEROCKET-DATA` instead of the raw partition
- Depends on `encrypt-unlock-local-fs.service`

#### local.mount (modified)

**Drop-in:** `local.mount.d/10-plain.conf`, installed from the same `local-mount-encrypted.conf`
source as the LUKS mode's `10-encrypted.conf` — the mapper name is identical in both modes, so the
drop-in only needs to override `What=`.

**Changes:**
- Mounts `/dev/mapper/BOTTLEROCKET-DATA` instead of the raw partition

#### .bottlerocket.mount (modified)

**Drop-in:** `bottlerocket-mount-ephemeral.conf`, installed as
`.bottlerocket.mount.d/20-ephemeral.conf`

**Changes:**
- Mounts `/dev/mapper/BOTTLEROCKET-PRIVATE` instead of the raw partition
- Requires `prepare-private-fs.service`

`repart-local.service` gets a `10-plain.conf` drop-in in this mode that masks the base unit's `ExecStart` (the `systemd-repart` grow), because the partition grow and mapper resize are performed inside `encrypt-unlock-local-fs.service` so the key can be generated and deleted within that single service. The base unit's `systemd-growfs /local` still runs to grow the filesystem to fill the resized mapper.

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

### Plain Mode Formatting

With `ephemeral-encryption-keys`, block devices are opened in plain mode instead. No header is written, so encryption and attach are a single operation, and `--hash plain` uses the random bytes verbatim:

```bash
cryptsetup open \
  --type plain \
  --cipher aes-xts-plain64 \
  --key-size 512 \
  --hash plain \
  --key-file=- \
  --keyfile-size=64 \
  <device> <name>
```

`--key-file=-` and `--keyfile-size=64` are required, not decoration.
The 64 random key bytes are written to cryptsetup's stdin, and without those two arguments cryptsetup treats stdin as an interactive *passphrase* and stops reading at the first newline; `--hash plain` then zero-pads whatever it read out to `--key-size`.
A random key contains a `0x0A` byte about 22% of the time, so the mapper would silently be opened with a truncated key — and a key whose *first* byte is `0x0A` yields an all-zero dm-crypt key, which a non-FIPS kernel accepts and a FIPS kernel rejects with `crypt: Error decoding and setting key (-EINVAL)`.

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
- `sources/bottlerocket-image-features/` - Parses the `encrypted-storage` and `ephemeral-encryption-keys` image features
- `sources/api/apiserver/src/server/ephemeral_storage.rs` - Encryption of the `EPHEMERAL-DATA` device for `apiclient ephemeral-storage init`
- `packages/release/release.spec` - Service packaging. The units split across three subpackages:
  `release-crypt` (measure/pcrphase, mode-independent), `release-crypt-luks` (LUKS block-device and
  two-service datastore units plus the `10-encrypted` drop-ins), and `release-ephemeral-crypt`
  (plain-mode `encrypt-unlock-*` units, `prepare-private-fs.service`, `run-rottweiler.mount`, and the
  `10-plain` / `20-ephemeral` drop-ins)
- `packages/release/encrypt-*.service` - Encryption services
- `packages/release/unlock-*.service` - Unlocking services
- `packages/release/encrypt-unlock-*.service` - Combined plain-mode encryption and unlock services
- `packages/release/prepare-local-fs.service`, `packages/release/prepare-private-fs.service` - Filesystem creation
- `packages/release/*-encrypted.conf` - LUKS mode drop-ins
- `packages/release/*-plain.conf` - Plain mode drop-ins
- `packages/release/bottlerocket-mount-ephemeral.conf` - Plain-mode `.bottlerocket.mount` drop-in
- `packages/release/run-rottweiler.mount` - tmpfs keystore directory at `/run/rottweiler`
- `packages/release/measure-*.service` - Measurement services
- `packages/release/systemd-pcrphase-*.service` - Boot phase measurements

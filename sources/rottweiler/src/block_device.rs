use snafu::prelude::*;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::{key, system};

type Result<T> = std::result::Result<T, snafu::Whatever>;

/// Encrypt a block device with LUKS2 using the specified key
pub fn encrypt(path: PathBuf, key_id: String) -> Result<()> {
    let device = path
        .to_str()
        .with_whatever_context(|| format!("path is not valid UTF-8: '{}'", path.display()))?;

    let key_bytes = key::load(key_id)?;

    system::cryptsetup_luks_format(device, &key_bytes)
}

/// Encrypt and attach a block device in a single step using plain-mode dm-crypt.
///
/// Generates 64 random bytes as a per-boot key (never persisted, never TPM-sealed), then opens the
/// device as a **headerless** plain-mode dm-crypt mapper named after the device file name. No LUKS
/// header is written and there is no separate format step.
///
/// The key is fed once on stdin and thereafter lives only in the kernel dm-crypt table; the
/// process buffer is zeroized on drop. Callers that need the device to be full size (the DATA
/// partition) grow the partition *before* this runs, so no plain-mode mapper ever needs a keyed
/// online resize and the key never has to be shared with another process.
pub fn encrypt_and_attach(path: PathBuf) -> Result<()> {
    let volume_name = filename(&path)?;

    let device = path
        .to_str()
        .with_whatever_context(|| format!("path is not valid UTF-8: '{}'", path.display()))?;

    let key_bytes = key::random_bytes()?;

    system::cryptsetup_plain_format(volume_name, device, &key_bytes)
}

/// Attach (unlock) an encrypted block device, creating a device mapper entry
pub fn attach(path: PathBuf, key_id: String) -> Result<()> {
    let volume_name = filename(&path)?;

    let source_device = path
        .to_str()
        .with_whatever_context(|| format!("path is not valid UTF-8: '{}'", path.display()))?;

    let key_bytes = key::load(key_id)?;

    system::systemd_cryptsetup_attach(volume_name, source_device, &key_bytes)
}

/// Detach (lock) an encrypted block device, removing the device mapper entry
pub fn detach(path: PathBuf) -> Result<()> {
    let volume_name = filename(&path)?;

    system::systemd_cryptsetup_detach(volume_name)
}

/// Resize a LUKS2 block-device mapper to match the underlying (grown) device size, loading the
/// TPM-sealed key from the keystore.
///
/// This is the encrypted-storage flow only. Plain-mode (ephemeral) mappers are never resized: the
/// DATA partition is grown before the mapper is opened, so the mapper is full size from the start
/// and its per-boot volume key never has to leave the process that generated it.
pub fn resize(path: PathBuf, key_id: String) -> Result<()> {
    let volume_name = filename(&path)?;

    let key_bytes = key::load(key_id)?;

    system::cryptsetup_resize(volume_name, &key_bytes)
}

const LUKS2_MAGIC: &[u8; 6] = b"LUKS\xba\xbe";
const LUKS2_VERSION: u16 = 2;

/// Check if a block device is LUKS2 encrypted by reading its header
pub fn is_encrypted(path: PathBuf) -> Result<bool> {
    let mut file = std::fs::File::open(&path)
        .with_whatever_context(|_| format!("failed to open '{}'", path.display()))?;

    let mut header = [0u8; 8];
    file.read_exact(&mut header)
        .with_whatever_context(|_| format!("failed to read header from '{}'", path.display()))?;

    if &header[..6] == LUKS2_MAGIC {
        let version = u16::from_be_bytes([header[6], header[7]]);
        if version == LUKS2_VERSION {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Extract filename from path as UTF-8 string
fn filename(path: &Path) -> Result<&str> {
    path.file_name()
        .with_whatever_context(|| format!("failed to extract filename from '{}'", path.display()))?
        .to_str()
        .with_whatever_context(|| format!("filename is not valid UTF-8: '{}'", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filename_derives_mapper_name_from_partlabel_path() {
        // The mapper name that encrypt_and_attach / attach derive is the device file name.
        let path = PathBuf::from("/dev/disk/by-partlabel/BOTTLEROCKET-DATA");
        assert_eq!(filename(&path).unwrap(), "BOTTLEROCKET-DATA");
    }

    #[test]
    fn filename_derives_mapper_name_from_ephemeral_path() {
        let path = PathBuf::from("/dev/disk/EPHEMERAL-DATA");
        assert_eq!(filename(&path).unwrap(), "EPHEMERAL-DATA");
    }

    #[test]
    fn filename_handles_plain_device_node() {
        let path = PathBuf::from("/dev/nvme1n1");
        assert_eq!(filename(&path).unwrap(), "nvme1n1");
    }

    #[test]
    fn filename_errors_without_a_file_name() {
        let path = PathBuf::from("/");
        assert!(filename(&path).is_err());
    }
}

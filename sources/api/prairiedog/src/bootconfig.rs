use crate::error;
use crate::error::Result;
use crate::initrd::generate_initrd;
use bottlerocket_modeled_types::{BootConfigKey, BootConfigValue};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use snafu::{ensure, ResultExt};
use std::convert::TryInto;
use std::io::ErrorKind;
use std::path::Path;
use std::{fs, io};

// Boot config related consts
const BOOTCONFIG_INITRD_PATH: &str = "/var/lib/bottlerocket/bootconfig.data";
const PROC_BOOTCONFIG: &str = "/proc/bootconfig";
const DEFAULT_BOOTCONFIG_STR: &str = r#"
    kernel = ""
    init = ""
"#;
const DEFAULT_BOOT_SETTINGS: BootSettings = BootSettings {
    reboot_to_reconcile: None,
    kernel_parameters: None,
    init_parameters: None,
};

#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "kebab-case")]
struct BootSettings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reboot_to_reconcile: Option<bool>,
    #[serde(
        alias = "kernel",
        rename(serialize = "kernel"),
        default,
        skip_serializing_if = "Option::is_none"
    )]
    kernel_parameters: Option<IndexMap<BootConfigKey, Vec<BootConfigValue>>>,
    #[serde(
        alias = "init",
        rename(serialize = "init"),
        default,
        skip_serializing_if = "Option::is_none"
    )]
    init_parameters: Option<IndexMap<BootConfigKey, Vec<BootConfigValue>>>,
}

fn append_boot_config_value_list(values: &[BootConfigValue], output: &mut String) {
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            output.push(',');
        }
        // If the value itself has double quotes in it, then we wrap the value with single-quotes
        if v.contains('\"') {
            output.push_str(&format!(" \'{v}\'"));
        } else {
            output.push_str(&format!(" \"{v}\""));
        }
    }
}

/// Serializes `BootSettings` out to a multi-line string representation of the boot config that can be
/// loaded by the kernel. Uses brace-grouped syntax so all keys under a section are siblings added
/// in a single tree-building pass, preserving insertion order in the kernel's bootconfig parser.
fn serialize_boot_settings_to_boot_config(boot_settings: &BootSettings) -> Result<String> {
    let mut output = String::with_capacity(128);
    // Output init parameters first since "init" comes before "kernel" alphabetically
    if let Some(init_param) = &boot_settings.init_parameters {
        write_params(&mut output, init_param, "init");
    }
    if let Some(kernel_param) = &boot_settings.kernel_parameters {
        write_params(&mut output, kernel_param, "kernel");
    }
    Ok(output)
}

/// Writes parameters to output in insertion order using brace-grouped syntax.
/// Emits `prefix { key = "val"\n ... }` instead of `prefix.key = "val"` per line,
/// so the kernel bootconfig parser adds all keys as siblings in one pass.
fn write_params(
    output: &mut String,
    params: &IndexMap<BootConfigKey, Vec<BootConfigValue>>,
    prefix: &str,
) {
    if params.is_empty() {
        return;
    }
    output.push_str(&format!("{prefix} {{\n"));
    for (key, values) in params {
        output.push_str(&format!("  {key}"));
        if !values.is_empty() {
            output.push_str(" =");
            append_boot_config_value_list(values, output);
        }
        output.push('\n');
    }
    output.push_str("}\n");
}

/// Queries Bottlerocket boot settings and generates initrd image file with boot config as the only data
pub(crate) fn generate_boot_config<P>(config_path: P) -> Result<()>
where
    P: AsRef<Path>,
{
    let bootconfig_bytes = match get_boot_config_settings(config_path)? {
        Some(boot_settings) => {
            info!("Generating initrd boot config from boot settings");
            trace!("Boot settings: {boot_settings:?}");
            let bootconfig = serialize_boot_settings_to_boot_config(&boot_settings)?;
            trace!("Serializing boot config string: {bootconfig}");
            bootconfig.into_bytes()
        }
        None => {
            // If we don't have any boot settings, write out an initrd with default boot config contents
            trace!("Serializing boot config string: {DEFAULT_BOOTCONFIG_STR}");
            DEFAULT_BOOTCONFIG_STR.to_string().into_bytes()
        }
    };
    let initrd = generate_initrd(&bootconfig_bytes)?;
    trace!("Writing initrd image file: {initrd:?}");
    fs::write(BOOTCONFIG_INITRD_PATH, &initrd).context(error::WriteInitrdSnafu)?;
    Ok(())
}

/// Retrieves boot config related Bottlerocket settings. If they don't exist in the settings model,
/// we return `None` instead.
fn get_boot_config_settings<P>(config_path: P) -> Result<Option<BootSettings>>
where
    P: AsRef<Path>,
{
    let config_path = config_path.as_ref();
    match fs::read_to_string(config_path) {
        Ok(config_str) => toml::from_str(config_str.as_str()).context(error::InputTomlSnafu),
        Err(e) => {
            if e.kind() == ErrorKind::NotFound {
                Ok(None)
            } else {
                Err(error::Error::ReadFile {
                    source: e,
                    path: config_path.to_path_buf(),
                })
            }
        }
    }
}

/// Reads `/proc/bootconfig`. Not having any boot config is ignored.
fn read_proc_bootconfig() -> Result<Option<String>> {
    match fs::read_to_string(PROC_BOOTCONFIG) {
        Ok(s) => Ok(Some(s)),
        Err(e) => {
            // If there's no `/proc/bootconfig`, then the user hasn't provisioned any kernel boot configuration.
            if e.kind() == io::ErrorKind::NotFound {
                Ok(None)
            } else {
                Err(e).context(error::ReadFileSnafu {
                    path: PROC_BOOTCONFIG,
                })
            }
        }
    }
}

/// Reads `/proc/bootconfig` and populates the Bottlerocket boot settings based on the existing boot config data
pub(crate) fn generate_boot_settings() -> Result<()> {
    if let Some(proc_bootconfig) = read_proc_bootconfig()? {
        debug!("Generating kernel boot config settings from `{PROC_BOOTCONFIG}`");
        println!("{}", boot_config_to_boot_settings_json(&proc_bootconfig)?);
    }
    Ok(())
}

/// Parses out a valid boot config value
fn parse_value(input: &str) -> Result<BootConfigValue> {
    let input = input.trim();
    let quoted = (input.starts_with('"') && input.ends_with('"'))
        || (input.starts_with('\'') && input.ends_with('\''));
    let chars_that_require_quotes = ['\'', '"', '\n', ',', ';', '#', '}'];
    let valid_value = input
        .chars()
        .all(|c| c.is_ascii() && (quoted || !chars_that_require_quotes.contains(&c)));
    ensure!(valid_value, error::InvalidBootConfigValueSnafu { input });
    // We want the value without the quotes
    let s = if quoted {
        &input[1..input.len() - 1]
    } else {
        input
    };
    s.try_into().context(error::ParseBootConfigValueSnafu)
}

/// Takes a string and parse it into a list of valid bootconfig values
fn parse_boot_config_values(input: &str) -> Result<Vec<BootConfigValue>> {
    // Sequences of elements can mix quoted and unquoted values
    // We also don't want to separate on a quoted comma
    let mut elements = Vec::new();
    let mut quote = None;
    let mut expect_delimiter = false;
    let mut start_index = 0;
    for (i, c) in input.trim().chars().enumerate() {
        if expect_delimiter && !c.is_whitespace() && c != ',' {
            return error::ExpectedArrayCommaSnafu { input }.fail();
        }
        if c == '\'' || c == '\"' {
            if let Some(q) = quote {
                // If the quote-types match, we're expecting a delimiter next
                if q == c {
                    quote = None;
                    expect_delimiter = true;
                }
            } else {
                quote = Some(c);
            }
        } else if c == ',' && quote.is_none() {
            // We've encountered the delimiter, and if it's outside quotes, we have a new element
            elements.push(parse_value(&input[start_index..i])?);
            start_index = i + 1;
            expect_delimiter = false;
        }
    }
    ensure!(quote.is_none(), error::UnbalancedQuotesSnafu { input });
    // Push last element
    let last_ele = if &input[start_index..] == "," {
        // If it's just a comma, assume it's an empty value at the end
        ""
    } else {
        &input[start_index..]
    };
    // Value-less bootconfig keys are allowed
    if !last_ele.is_empty() {
        elements.push(parse_value(last_ele)?);
    }
    Ok(elements)
}

/// Takes a string representation of a bootconfig file and parse it into `BootSettings`
fn parse_boot_config_to_boot_settings(bootconfig: &str) -> Result<BootSettings> {
    let mut kernel_params: IndexMap<BootConfigKey, Vec<BootConfigValue>> = IndexMap::new();
    let mut init_params: IndexMap<BootConfigKey, Vec<BootConfigValue>> = IndexMap::new();
    for line in bootconfig.trim().lines() {
        // ignore comment lines
        if line.trim_start().starts_with("#") {
            continue;
        }
        let mut kv = line.trim().splitn(2, '=').map(|kv| kv.trim());
        // Ensure the key is a valid boot config key
        let key: BootConfigKey = kv
            .next()
            .ok_or(error::Error::InvalidBootConfig)?
            .try_into()
            .context(error::ParseBootConfigKeySnafu)?;
        // Value-less boot config keys are acceptable, i.e. 'key =' or 'key'
        // We represent the absence of a value with as an empty list
        let values = match kv.next() {
            Some(value) => parse_boot_config_values(value)?,
            None => Vec::new(),
        };

        if key != "kernel" && key.starts_with("kernel") {
            kernel_params.insert(
                key["kernel.".len()..]
                    .try_into()
                    .context(error::ParseBootConfigKeySnafu)?,
                values,
            );
        } else if key != "init" && key.starts_with("init") {
            init_params.insert(
                key["init.".len()..]
                    .try_into()
                    .context(error::ParseBootConfigKeySnafu)?,
                values,
            );
        } else if key == "kernel" || key == "init" {
            let empty_value_list: Vec<BootConfigValue> =
                vec!["".try_into().context(error::ParseBootConfigValueSnafu)?];
            // `BootSettings` does not support `kernel` or `init` as parent keys to non-null values.
            if values != empty_value_list {
                return error::ParentBootConfigKeySnafu.fail();
            }
        } else {
            return error::UnsupportedBootConfigKeySnafu { key }.fail();
        }
    }

    Ok(BootSettings {
        reboot_to_reconcile: None,
        kernel_parameters: if kernel_params.is_empty() {
            None
        } else {
            Some(kernel_params)
        },
        init_parameters: if init_params.is_empty() {
            None
        } else {
            Some(init_params)
        },
    })
}

/// Given a boot config string, deserialize it to `BootSettings` and then serialize it back
/// out as a JSON string for sundog consumption
fn boot_config_to_boot_settings_json(bootconfig_str: &str) -> Result<String> {
    // We'll only send the setting if the existing boot config file fits our settings model
    let boot_settings = parse_boot_config_to_boot_settings(bootconfig_str)?;
    // sundog expects JSON-serialized output
    serde_json::to_string(&boot_settings).context(error::OutputJsonSnafu)
}

/// Decides whether the host should be rebooted to have its boot settings take effect
pub(crate) fn is_reboot_required<P>(config_path: P) -> Result<bool>
where
    P: AsRef<Path>,
{
    let old_boot_settings = match read_proc_bootconfig()? {
        Some(proc_bootconfig) => parse_boot_config_to_boot_settings(&proc_bootconfig)?,
        None => DEFAULT_BOOT_SETTINGS,
    };

    let new_boot_settings = get_boot_config_settings(config_path)?.unwrap_or(DEFAULT_BOOT_SETTINGS);

    let reboot_required = if new_boot_settings.reboot_to_reconcile.unwrap_or(false) {
        boot_settings_change_requires_reboot(&old_boot_settings, &new_boot_settings)
    } else {
        false
    };

    Ok(reboot_required)
}

/// Check whether `BootSettings` changed in a way to warrant a reboot
fn boot_settings_change_requires_reboot(
    old_boot_settings: &BootSettings,
    new_boot_settings: &BootSettings,
) -> bool {
    fn parameters_changed_materially(
        old_params: &Option<IndexMap<BootConfigKey, Vec<BootConfigValue>>>,
        new_params: &Option<IndexMap<BootConfigKey, Vec<BootConfigValue>>>,
    ) -> bool {
        // Consider a missing hash map equal to an empty one: There is no configuration in either case.
        match (old_params, new_params) {
            (None, None) => false,
            (None, Some(new)) => !new.is_empty(),
            (Some(old), None) => !old.is_empty(),
            (Some(old), Some(new)) => old != new,
        }
    }

    // Only reboot for changes actually requiring a reboot. Changing a Bottlerocket setting
    // like boot.reboot-to-reconcile does not qualify as a reason to reboot.
    parameters_changed_materially(
        &old_boot_settings.kernel_parameters,
        &new_boot_settings.kernel_parameters,
    ) || parameters_changed_materially(
        &old_boot_settings.init_parameters,
        &new_boot_settings.init_parameters,
    )
}

#[cfg(test)]
mod boot_settings_tests {
    use super::BootSettings;
    use crate::bootconfig::{
        boot_config_to_boot_settings_json, boot_settings_change_requires_reboot,
        serialize_boot_settings_to_boot_config, DEFAULT_BOOTCONFIG_STR,
    };
    use bottlerocket_modeled_types::{BootConfigKey, BootConfigValue};
    use indexmap::IndexMap;
    use serde_json::json;
    use serde_json::value::Value;
    use std::convert::TryInto;

    /// Build an ordered IndexMap of boot settings parameters from a list of (key, values) pairs.
    /// Insertion order is preserved, which is the whole point of the ordering fix.
    fn to_boot_settings_params(
        params: Vec<(&str, Vec<&str>)>,
    ) -> Option<IndexMap<BootConfigKey, Vec<BootConfigValue>>> {
        Some(
            params
                .into_iter()
                .map(|(k, v)| {
                    (
                        k.try_into().unwrap(),
                        v.into_iter().map(|s| s.try_into().unwrap()).collect(),
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn boot_settings_to_string() {
        let boot_settings = BootSettings {
            reboot_to_reconcile: None,
            kernel_parameters: to_boot_settings_params(vec![
                ("console", vec!["ttyS1,115200n8", "tty0"]),
            ]),
            init_parameters: to_boot_settings_params(vec![
                ("systemd.log_level", vec!["debug"]),
                ("splash", vec![]),
                ("weird", vec!["'single'quotes'", "\"double\"quotes\""]),
            ]),
        };
        let output = serialize_boot_settings_to_boot_config(&boot_settings).unwrap();
        // Assert insertion order is preserved — no sorting, brace-grouped syntax
        assert_eq!(
            output,
            r#"init {
  systemd.log_level = "debug"
  splash
  weird = "'single'quotes'", '"double"quotes"'
}
kernel {
  console = "ttyS1,115200n8", "tty0"
}
"#
        );
    }

    #[test]
    fn none_boot_settings_to_string() {
        let boot_settings = BootSettings {
            reboot_to_reconcile: None,
            kernel_parameters: None,
            init_parameters: None,
        };
        assert_eq!(
            serialize_boot_settings_to_boot_config(&boot_settings).unwrap(),
            r#""#
        );

        let init_none_boot_settings = BootSettings {
            reboot_to_reconcile: None,
            kernel_parameters: to_boot_settings_params(vec![
                ("console", vec!["ttyS1,115200n8", "tty0"]),
                ("usbcore.quirks", vec!["0781:5580:bk", "0a5c:5834:gij"]),
            ]),
            init_parameters: None,
        };
        let output = serialize_boot_settings_to_boot_config(&init_none_boot_settings).unwrap();
        // Assert insertion order is preserved — no sorting, brace-grouped syntax
        assert_eq!(
            output,
            r#"kernel {
  console = "ttyS1,115200n8", "tty0"
  usbcore.quirks = "0781:5580:bk", "0a5c:5834:gij"
}
"#
        );
    }

    #[test]
    fn empty_map_boot_settings_to_string() {
        let boot_settings = BootSettings {
            reboot_to_reconcile: None,
            kernel_parameters: Some(IndexMap::new()),
            init_parameters: None,
        };
        assert_eq!(
            serialize_boot_settings_to_boot_config(&boot_settings).unwrap(),
            r#""#
        );
    }

    static STANDARD_BOOTCONFIG: &str = r#"
        kernel.console = "ttyS1,115200n8", "tty0"
        init.splash
        init.splash2 =
        init.systemd.log_level = "debug"
        "#;

    #[test]
    fn standard_boot_config_to_boot_settings_json() {
        assert_eq!(
            json!({"kernel":{"console":["ttyS1,115200n8","tty0"]},"init":{"systemd.log_level":["debug"],"splash":[],"splash2":[]}}),
            serde_json::from_str::<Value>(
                &boot_config_to_boot_settings_json(STANDARD_BOOTCONFIG).unwrap()
            )
            .unwrap()
        );
    }

    static SPECIAL_BOOTCONFIG: &str = r#"
        kernel = ""
        kernel.console = "ttyS1,115200n8", "tty0"
        init = ""
        init.systemd.log_level = "debug"
        "#;

    #[test]
    fn special_boot_config_to_boot_settings_json() {
        assert_eq!(
            json!({"kernel":{"console":["ttyS1,115200n8","tty0"]},"init":{"systemd.log_level":["debug"]}}),
            serde_json::from_str::<Value>(
                &boot_config_to_boot_settings_json(SPECIAL_BOOTCONFIG).unwrap()
            )
            .unwrap()
        );
    }

    static EQUALS_BOOTCONFIG: &str = r#"
        kernel.dm-mod.create = "root,,,ro,0 0 delay PARTUUID=00000000-0000-0000-0000-000000000000/PARTNROFF=1 0 500"
        "#;

    #[test]
    fn equals_boot_config_to_boot_settings_json() {
        assert_eq!(
            json!({"kernel":{"dm-mod.create":[
                "root,,,ro,0 0 delay PARTUUID=00000000-0000-0000-0000-000000000000/PARTNROFF=1 0 500"]
            }}),
            serde_json::from_str::<Value>(
                &boot_config_to_boot_settings_json(EQUALS_BOOTCONFIG).unwrap()
            )
            .unwrap()
        );
    }

    static UNSUPPORTED_BOOTCONFIG: &str = r#"
        do.androids.dream.of.electric.sheep = "?"
        kernel.console = "ttyS1,115200n8", "tty0"
        init.systemd.log_level = "debug"
        "#;

    #[test]
    fn unsupported_boot_config_to_boot_settings_json() {
        assert!(&boot_config_to_boot_settings_json(UNSUPPORTED_BOOTCONFIG).is_err());
    }

    static MISSING_COMMA: &str = r#"
        kernel = "?" "???"
        "#;

    #[test]
    fn missing_comma_boot_config_to_boot_settings_json() {
        assert!(&boot_config_to_boot_settings_json(MISSING_COMMA).is_err());
    }

    static BAD_UNQUOTED_VALUE: &str = r#"
        kernel = #bang
        "#;

    #[test]
    fn bad_unquoted_value_boot_config_to_boot_settings_json() {
        assert!(&boot_config_to_boot_settings_json(BAD_UNQUOTED_VALUE).is_err());
    }

    static KERNEL_INIT_PARENT_KEY: &str = r#"
        kernel = "foo"
        init = "bar"
        "#;

    #[test]
    fn kernel_init_parent_key_boot_config_to_boot_settings_json() {
        assert!(&boot_config_to_boot_settings_json(KERNEL_INIT_PARENT_KEY).is_err());
    }

    #[test]
    fn test_default_boot_config_to_boot_settings_json() {
        assert_eq!(
            // We expect null with a bootconfig with empty keys
            serde_json::from_str::<Value>(r#"{}"#).unwrap(),
            serde_json::from_str::<Value>(
                &boot_config_to_boot_settings_json(DEFAULT_BOOTCONFIG_STR).unwrap()
            )
            .unwrap()
        );
    }

    #[test]
    fn test_bootconfig_output_preserves_insertion_order() {
        // Keys are intentionally NOT in alphabetical order to verify insertion order is preserved
        let boot_settings = BootSettings {
            reboot_to_reconcile: None,
            kernel_parameters: to_boot_settings_params(vec![
                ("zebra", vec!["last"]),
                ("apple", vec!["first"]),
                ("middle", vec!["middle"]),
            ]),
            init_parameters: to_boot_settings_params(vec![
                ("zoo", vec!["last"]),
                ("aardvark", vec!["first"]),
            ]),
        };
        let output = serialize_boot_settings_to_boot_config(&boot_settings).unwrap();
        let lines: Vec<&str> = output.lines().collect();

        // Init parameters come first (serialize_boot_settings_to_boot_config emits init before kernel)
        // Within each group, insertion order is preserved — NOT alphabetical
        // Brace-grouped syntax: prefix { key = "val" ... }
        assert_eq!(lines[0], "init {");
        assert_eq!(lines[1], "  zoo = \"last\"");
        assert_eq!(lines[2], "  aardvark = \"first\"");
        assert_eq!(lines[3], "}");
        assert_eq!(lines[4], "kernel {");
        assert_eq!(lines[5], "  zebra = \"last\"");
        assert_eq!(lines[6], "  apple = \"first\"");
        assert_eq!(lines[7], "  middle = \"middle\"");
        assert_eq!(lines[8], "}");
    }

    /// Verifies the core use case from issue #3647: hugepagesz must come before hugepages
    /// in the kernel command line when the user specifies them in that order.
    #[test]
    fn test_insertion_order_preserved_through_serialization() {
        let toml_input = r#"
            [kernel]
            hugepagesz = ["1G"]
            hugepages = ["4"]
            transparent_hugepage = ["never"]
        "#;
        let boot_settings: BootSettings = toml::from_str(toml_input).unwrap();
        let output = serialize_boot_settings_to_boot_config(&boot_settings).unwrap();
        let lines: Vec<&str> = output.lines().collect();

        assert_eq!(lines[0], "kernel {");
        assert_eq!(lines[1], "  hugepagesz = \"1G\"");
        assert_eq!(lines[2], "  hugepages = \"4\"");
        assert_eq!(lines[3], "  transparent_hugepage = \"never\"");
        assert_eq!(lines[4], "}");
    }

    #[test]
    fn test_unchanged_boot_settings_require_no_reboot() {
        let a = BootSettings {
            reboot_to_reconcile: None,
            kernel_parameters: None,
            init_parameters: to_boot_settings_params(vec![
                ("systemd.log_level", vec!["debug"]),
            ]),
        };
        let b = BootSettings {
            reboot_to_reconcile: None,
            kernel_parameters: None,
            init_parameters: to_boot_settings_params(vec![
                ("systemd.log_level", vec!["debug"]),
            ]),
        };
        assert!(!boot_settings_change_requires_reboot(&a, &b));
    }

    #[test]
    fn test_changed_boot_settings_require_a_reboot() {
        let a = BootSettings {
            reboot_to_reconcile: None,
            kernel_parameters: None,
            init_parameters: to_boot_settings_params(vec![
                ("systemd.log_level", vec!["debug"]),
            ]),
        };
        let b = BootSettings {
            reboot_to_reconcile: None,
            kernel_parameters: to_boot_settings_params(vec![
                ("debug", vec![""]),
            ]),
            init_parameters: to_boot_settings_params(vec![
                ("systemd.log_level", vec!["debug"]),
            ]),
        };
        assert!(boot_settings_change_requires_reboot(&a, &b));
    }

    #[test]
    fn test_missing_boot_settings_require_no_reboot() {
        let a = BootSettings {
            reboot_to_reconcile: None,
            kernel_parameters: None,
            init_parameters: to_boot_settings_params(vec![]),
        };
        let b = BootSettings {
            reboot_to_reconcile: None,
            kernel_parameters: to_boot_settings_params(vec![]),
            init_parameters: None,
        };
        assert!(!boot_settings_change_requires_reboot(&a, &b));
    }

    #[test]
    fn test_changed_bottlerocket_boot_settings_require_no_reboot() {
        let a = BootSettings {
            reboot_to_reconcile: None,
            kernel_parameters: None,
            init_parameters: None,
        };
        let b = BootSettings {
            reboot_to_reconcile: Some(true),
            kernel_parameters: None,
            init_parameters: None,
        };
        assert!(!boot_settings_change_requires_reboot(&a, &b));
    }
}

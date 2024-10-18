use log::{info, trace};
use std::collections::HashSet;
use simplelog::{Config as LogConfig, LevelFilter, SimpleLogger};
use snafu::{ensure, OptionExt, ResultExt};
use std::ffi::OsStr;
use std::env;
use std::process::{self, Command};
use std::str::FromStr;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;

const DEFAULT_CONFIG_PATH: &str = "/etc/bootstrap-commands/bootstrap-commands.toml";

/*
./nvidia-smi --query-gpu=gpu_name,mig.mode.current,mig.mode.pending --format=csv,noheader
NVIDIA A100-SXM4-40GB, Disabled, Enabled
NVIDIA A100-SXM4-40GB, Disabled, Enabled
NVIDIA A100-SXM4-40GB, Disabled, Enabled
NVIDIA A100-SXM4-40GB, Disabled, Enabled
NVIDIA A100-SXM4-40GB, Disabled, Enabled
NVIDIA A100-SXM4-40GB, Disabled, Enabled
NVIDIA A100-SXM4-40GB, Disabled, Enabled
NVIDIA A100-SXM4-40GB, Disabled, Enabled
*/

/*
Capacity:
  cpu:                96
  ephemeral-storage:  418812624Ki
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             1176277924Ki
  nvidia.com/gpu:     16
  pods:               737
Allocatable:
  cpu:                95690m
  ephemeral-storage:  384903971816
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             1167612836Ki
  nvidia.com/gpu:     16
  pods:               737
*/

/// Stores user-supplied global arguments
struct Args {
    log_level: LevelFilter,
    config_path: PathBuf,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            log_level: LevelFilter::Info,
            config_path: PathBuf::from_str(DEFAULT_CONFIG_PATH).unwrap(),
        }
    }
}

/// Wrapper around process::Command that adds error checking.
fn command<I, S>(bin_path: &str, args: I) -> Result<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut command = Command::new(bin_path);
    command.args(args);
    let output = command
        .output()
        .context(error::ExecutionFailureSnafu { command })?;

    trace!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    trace!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    ensure!(
        output.status.success(),
        error::CommandFailureSnafu { bin_path, output }
    );

    let output_str = String::from_utf8_lossy(&output.stdout);

    Ok(output_str.to_string())
}


#[derive(Hash, Debug, Clone, PartialEq, Eq)]
enum MigState {
    Enabled,
    Disabled,
    Transition,
    Unknown,
}

fn is_mig_capable(modes: &[(String, MigState, MigState)]) -> bool {
    for (_, current, pending) in modes {
        // Early exit because all the GPUs are of same make and model
        if *current != MigState::Unknown || *pending != MigState::Unknown {
            return true;
        }
    }

    return false
}


fn get_mig_mode(modes: &[(String, MigState, MigState)]) -> MigState {
    let mut current_states = HashSet::new();
    let mut pending_states = HashSet::new();

    for (_, current, pending) in modes {
        current_states.insert(current.clone());
        pending_states.insert(pending.clone());
    }

    if current_states.len() == 1 && pending_states.len() == 1 {
        let current_state = current_states.iter().next().unwrap();
        let pending_state = pending_states.iter().next().unwrap();

        if *current_state == MigState::Enabled && *pending_state == MigState::Enabled {
            return MigState::Enabled
        } else if *current_state == MigState::Disabled && *pending_state == MigState::Disabled {
            return MigState::Disabled
        } else if *current_state == MigState::Disabled && *pending_state == MigState::Enabled {
            return MigState::Transition
        } else {
            return MigState::Unknown
        }
    } else {
        return MigState::Unknown
    }
}

fn set_mig_mode() -> Result<()> {
    let _ = command("/usr/libexec/nvidia/tesla/bin/nvidia-smi", ["-mig", "1"])?;
    Ok(())
}

fn analyze_mig_status(mig_modes: &Vec<(String, MigState, MigState)>) -> Result<(bool, MigState)> {
    info!("entered analyze_mig_status function here");

    let is_mig_capable = is_mig_capable(&mig_modes);
    info!("is_mig_capable: {:?}", is_mig_capable);

    let overall_mig_mode = get_mig_mode(&mig_modes);
    info!("overall_mig_mode: {:?}", overall_mig_mode);

    Ok((is_mig_capable, overall_mig_mode))
}

fn set_mig_profile() -> Result<()> {
    let _ = command("/usr/libexec/nvidia/tesla/bin/nvidia-smi", ["mig", "-cgi", "9,9", "-C"])?;
    Ok(())
}

fn run_gpu_query() -> Result<Vec<(String, MigState, MigState)>> {
    let output = command("/usr/libexec/nvidia/tesla/bin/nvidia-smi", ["--query-gpu=gpu_name,mig.mode.current,mig.mode.pending", "--format=csv,noheader"])?;
    let mut modes = Vec::new();

    for line in output.lines() {
        let parts: Vec<_> = line.split(", ").collect();
        info!("{:?}", parts);

        if parts.len() == 3 {
            let current = match parts[1] {
                "Enabled" => MigState::Enabled,
                "Disabled" => MigState::Disabled,
                _ => MigState::Unknown,
            };

            let pending = match parts[2] {
                "Enabled" => MigState::Enabled,
                "Disabled" => MigState::Disabled,
                _ => MigState::Unknown,
            };

            modes.push((parts[0].to_string(), current, pending));
        }
    }

    Ok(modes)
}

/// Parse the args to the program and return an Args struct
fn parse_args(args: env::Args) -> Result<Args> {
    let mut global_args = Args::default();
    let mut iter = args.skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_ref() {
            // Global args
            "--log-level" => {
                let log_level = iter.next().context(error::UsageSnafu {
                    message: "Did not give argument to --log-level",
                })?;
                global_args.log_level = LevelFilter::from_str(&log_level)
                    .context(error::LogLevelSnafu { log_level })?;
            }

            "-c" | "--config-path" => {
                let config_str = iter.next().context(error::UsageSnafu {
                    message: "Did not give argument to --config-path",
                })?;
                global_args.config_path = PathBuf::from(config_str.as_str());
            }

            _ => (),
        }
    }

    Ok(global_args)
}

fn run() -> Result<()> {
    println!("entered run function here");

    let args = parse_args(env::args())?;

    // SimpleLogger will send errors to stderr and anything less to stdout.
    SimpleLogger::init(LevelFilter::Info, LogConfig::default()).context(error::LoggerSnafu)?;

    let mut modes = run_gpu_query()?;
    let (is_mig_capable, overall_mig_mode) = analyze_mig_status(&modes)?;

    if is_mig_capable {
        if overall_mig_mode == MigState::Disabled {
            let _ = set_mig_mode()?;
            modes = run_gpu_query()?;
            let _ = analyze_mig_status(&modes)?;
            let _ = command("apiclient", ["reboot"])?;
        } else if overall_mig_mode == MigState::Enabled {
            let _ = set_mig_profile()?;
        }
    }

    Ok(())
}

fn main() {
    println!("entered main function here");

    if let Err(e) = run() {
        eprintln!("{}", e);
        process::exit(1);
    }
}

// =^..^=   =^..^=   =^..^=   =^..^=   =^..^=   =^..^=   =^..^=   =^..^=   =^..^=   =^..^=   =^..^=

mod error {
    use snafu::Snafu;
    use std::process::{Command, Output};

    #[derive(Debug, Snafu)]
    #[snafu(visibility(pub(super)))]
    pub(super) enum Error {
        #[snafu(display("'{}' failed - stderr: {}",
                        bin_path, String::from_utf8_lossy(&output.stderr)))]
        CommandFailure { bin_path: String, output: Output },

        #[snafu(display("Failed to execute '{:?}': {}", command, source))]
        ExecutionFailure {
            command: Command,
            source: std::io::Error,
        },

        #[snafu(display("Logger setup error: {}", source))]
        Logger { source: log::SetLoggerError },

        #[snafu(display("Invalid log level '{}'", log_level))]
        LogLevel {
            log_level: String,
            source: log::ParseLevelError,
        },

        #[snafu(display("{}", message))]
        Usage { message: String },
    }
}

type Result<T> = std::result::Result<T, error::Error>;

/*
Oct 18 21:18:27 ip-192-168-113-68.us-west-2.compute.internal systemd[1]: Starting NVIDIA MIG manager service...
Oct 18 21:18:27 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: entered main function here0
Oct 18 21:18:27 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: entered run function here
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] ["NVIDIA A100-SXM4-40GB", "Disabled", "Disabled"]
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] ["NVIDIA A100-SXM4-40GB", "Disabled", "Disabled"]
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] ["NVIDIA A100-SXM4-40GB", "Disabled", "Disabled"]
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] ["NVIDIA A100-SXM4-40GB", "Disabled", "Disabled"]
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] ["NVIDIA A100-SXM4-40GB", "Disabled", "Disabled"]
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] ["NVIDIA A100-SXM4-40GB", "Disabled", "Disabled"]
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] ["NVIDIA A100-SXM4-40GB", "Disabled", "Disabled"]
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] ["NVIDIA A100-SXM4-40GB", "Disabled", "Disabled"]
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] entered analyze_mig_status function here
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] is_mig_capable: true
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] overall_mig_mode: Disabled
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] ["NVIDIA A100-SXM4-40GB", "Disabled", "Enabled"]
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] ["NVIDIA A100-SXM4-40GB", "Disabled", "Enabled"]
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] ["NVIDIA A100-SXM4-40GB", "Disabled", "Enabled"]
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] ["NVIDIA A100-SXM4-40GB", "Disabled", "Enabled"]
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] ["NVIDIA A100-SXM4-40GB", "Disabled", "Enabled"]
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] ["NVIDIA A100-SXM4-40GB", "Disabled", "Enabled"]
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] ["NVIDIA A100-SXM4-40GB", "Disabled", "Enabled"]
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] ["NVIDIA A100-SXM4-40GB", "Disabled", "Enabled"]
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] entered analyze_mig_status function here
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] is_mig_capable: true
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal nvidia-migmanager[15644]: 21:18:51 [INFO] overall_mig_mode: Transition
Oct 18 21:18:51 ip-192-168-113-68.us-west-2.compute.internal systemd[1]: Finished NVIDIA MIG manager service.
*/

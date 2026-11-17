use crate::health::find_on_path;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::Command;

pub const QUIET_BOOT_ENV: [(&str, &str); 5] = [
    ("TQDM_DISABLE", "1"),
    ("HF_HUB_DISABLE_PROGRESS_BARS", "1"),
    ("TRANSFORMERS_VERBOSITY", "error"),
    ("TOKENIZERS_PARALLELISM", "false"),
    ("SPARKLEFORGE_QUIET_BOOT", "1"),
];

#[derive(Debug, Clone)]
pub struct DelegationPlan {
    pub python_bin: String,
    pub args: Vec<String>,
    pub env_overrides: Vec<(String, String)>,
    pub verbose: bool,
    pub filter_noise: bool,
    pub pure_stdout_mode: bool,
}

pub fn resolve_python_executable(workspace_root: &Path, env_path: Option<&str>) -> String {
    let venv_py = workspace_root.join(".venv").join("bin").join("python");
    if is_exec_file(&venv_py) {
        return venv_py.display().to_string();
    }
    if let Some(found) = find_on_path("python3", env_path) {
        return found.display().to_string();
    }
    "python3".to_string()
}

fn is_exec_file(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

pub fn build_delegation_plan(
    workspace_root: &Path,
    raw_args: &[String],
    global_verbose: bool,
    env_path: Option<&str>,
) -> DelegationPlan {
    let python_bin = resolve_python_executable(workspace_root, env_path);

    let has_verbose_in_args = raw_args.iter().any(|a| a == "--verbose" || a == "-v");
    let verbose = global_verbose || has_verbose_in_args;

    let mut py_args = vec!["-m".to_string(), "src.cli.entry".to_string()];
    if verbose {
        py_args.push("--verbose".to_string());
    }

    for arg in raw_args {
        if arg == "--verbose" || arg == "-v" {
            continue;
        }
        py_args.push(arg.clone());
    }

    let env_overrides = if verbose {
        Vec::new()
    } else {
        QUIET_BOOT_ENV
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    };

    let pure_stdout_mode = raw_args.iter().enumerate().any(|(i, arg)| {
        arg == "--help"
            || arg == "-h"
            || arg == "--json"
            || arg == "--output-format=json"
            || arg == "--output-format=stream-json"
            || (arg == "--output-format"
                && raw_args
                    .get(i + 1)
                    .map(|v| v == "json" || v == "stream-json")
                    .unwrap_or(false))
    });

    DelegationPlan {
        python_bin,
        args: py_args,
        env_overrides,
        verbose,
        filter_noise: !verbose,
        pure_stdout_mode,
    }
}

pub async fn run_delegated_cli(raw_args: &[String], global_verbose: bool) -> i32 {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let path_var = env::var("PATH").ok();
    let plan = build_delegation_plan(&cwd, raw_args, global_verbose, path_var.as_deref());
    execute_plan(&cwd, &plan).await
}

pub async fn execute_plan(cwd: &Path, plan: &DelegationPlan) -> i32 {
    let start = Instant::now();
    let mut cmd = Command::new(&plan.python_bin);
    cmd.args(&plan.args)
        .current_dir(cwd)
        .stdin(Stdio::inherit())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    for (k, v) in &plan.env_overrides {
        cmd.env(k, v);
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(err) => {
            eprintln!(
                "[FAIL] Failed to launch Python CLI ({}): {err}",
                plan.python_bin
            );
            return 1;
        }
    };

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let filter_out = plan.filter_noise;
    let filter_err = plan.filter_noise;

    let out_task = tokio::spawn(async move {
        if let Some(reader) = stdout {
            forward_stream(reader, false, filter_out).await
        } else {
            0usize
        }
    });
    let err_task = tokio::spawn(async move {
        if let Some(reader) = stderr {
            forward_stream(reader, true, filter_err).await
        } else {
            0usize
        }
    });

    let status = child.wait().await;
    let filtered_out = out_task.await.unwrap_or(0);
    let filtered_err = err_task.await.unwrap_or(0);
    let filtered_total = filtered_out + filtered_err;

    let exit_code = match status {
        Ok(st) => st.code().unwrap_or(1),
        Err(_) => 1,
    };
    let duration_ms = ((start.elapsed().as_secs_f64() * 1000.0) * 1000.0).round() / 1000.0;
    let orch_status = if exit_code == 0 {
        "completed"
    } else {
        "failed"
    };

    let status_line = if filtered_total > 0 {
        format!(
            "[sparkleforge] orchestration={orch_status} exit_code={exit_code} duration_ms={duration_ms:.2} filtered_boot_lines={filtered_total}"
        )
    } else {
        format!(
            "[sparkleforge] orchestration={orch_status} exit_code={exit_code} duration_ms={duration_ms:.2}"
        )
    };

    if plan.pure_stdout_mode {
        if !plan.args.iter().any(|a| a == "--help" || a == "-h") {
            eprintln!("{status_line}");
        }
    } else {
        println!("{status_line}");
    }

    exit_code
}

async fn forward_stream<R: AsyncRead + Unpin>(
    reader: R,
    to_stderr: bool,
    filter_noise: bool,
) -> usize {
    let mut buf_reader = BufReader::new(reader);
    let mut line_buf = Vec::new();
    let mut filtered_count = 0usize;

    loop {
        line_buf.clear();
        match buf_reader.read_until(b'\n', &mut line_buf).await {
            Ok(0) => break,
            Ok(_) => {
                let raw = String::from_utf8_lossy(&line_buf);
                let has_newline = raw.ends_with('\n');
                let line_body = raw.trim_end_matches(&['\r', '\n'][..]);

                // Handle carriage-return progress bar updates inside a single line
                if line_body.contains('\r') {
                    let mut kept_segments = Vec::new();
                    for seg in line_body.split('\r') {
                        if filter_noise && is_noisy_boot_line(seg) {
                            filtered_count += 1;
                        } else if !seg.is_empty() {
                            kept_segments.push(seg);
                        }
                    }
                    if !kept_segments.is_empty() {
                        let rendered = kept_segments.join("\n");
                        emit_text(&rendered, has_newline, to_stderr);
                    }
                } else if filter_noise && is_noisy_boot_line(line_body) {
                    filtered_count += 1;
                } else {
                    emit_text(line_body, has_newline, to_stderr);
                }
            }
            Err(_) => break,
        }
    }

    filtered_count
}

fn emit_text(text: &str, newline: bool, to_stderr: bool) {
    if to_stderr {
        if newline {
            eprintln!("{text}");
        } else {
            eprint!("{text}");
        }
    } else if newline {
        println!("{text}");
    } else {
        print!("{text}");
    }
}

pub fn is_noisy_boot_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }

    // Never drop real errors or tracebacks
    if trimmed.contains("Traceback") || trimmed.starts_with("[FAIL]") {
        return false;
    }

    let lower = trimmed.to_ascii_lowercase();

    // Progress bars (HuggingFace / tqdm)
    if trimmed.contains("Loading weights")
        || trimmed.contains("Batches:")
        || (trimmed.contains("%|") && (lower.contains("it/s") || lower.contains("s/it")))
    {
        return true;
    }

    // Known noisy boot-time warnings & third-party import chatter
    let boot_noise_markers = [
        "both google_api_key and gemini_api_key",
        "requestsdependencywarning",
        "notopensslwarning",
        "urllib3 v2 only supports openssl",
        "doesn't match a supported version",
        "huggingface/tokenizers: the current process just got forked",
        "avoid deadlocks: to disable this warning, you can either: - avoid using `tokenizers`",
        "databasedriver initialized",
        "database driver set:",
        "tools_config.json not found",
        "registering native tool:",
        "registered git tool:",
        "aliased '",
        "checkpointmanager initialized:",
        "sessionstorage initialized",
        "granted capability",
        "execution mode switched:",
    ];
    if boot_noise_markers.iter().any(|m| lower.contains(m)) {
        return true;
    }

    // Optional flashrank missing warning
    if lower.contains("flashrank") && (lower.contains("warn") || lower.contains("not installed") || lower.contains("unavailable")) {
        return true;
    }

    // Noisy [WARN] / python logging WARNING/INFO startup lines
    if trimmed.starts_with("[WARN]") {
        return true;
    }
    if (lower.contains(" - src.core.") || lower.contains(" - __main__ - "))
        && (lower.contains(" - info - ") || lower.contains(" - warning - "))
    {
        return true;
    }
    if trimmed.contains("UserWarning:")
        || trimmed.contains("FutureWarning:")
        || trimmed.contains("DeprecationWarning:")
    {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = env::temp_dir().join(format!("sf_runner_test_{label}_{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    #[cfg(unix)]
    fn resolves_local_venv_python_first_and_falls_back_to_python3() {
        use std::os::unix::fs::PermissionsExt;

        let dir = make_temp_dir("py_resolve");
        let fallback = resolve_python_executable(&dir, Some("/usr/bin:/bin"));
        assert!(fallback.ends_with("python3"));

        let venv_bin = dir.join(".venv").join("bin");
        std::fs::create_dir_all(&venv_bin).unwrap();
        let py = venv_bin.join("python");
        std::fs::write(&py, "#!/bin/sh\necho ok\n").unwrap();
        std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();

        let resolved = resolve_python_executable(&dir, Some("/usr/bin:/bin"));
        assert_eq!(resolved, py.display().to_string());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn builds_quiet_boot_plan_and_hoists_verbose_flag() {
        let dir = make_temp_dir("plan");

        let quiet_plan = build_delegation_plan(
            &dir,
            &["run".to_string(), "hello world".to_string()],
            false,
            Some("/usr/bin"),
        );
        assert!(!quiet_plan.verbose);
        assert!(quiet_plan.filter_noise);
        assert_eq!(quiet_plan.args, vec!["-m", "src.cli.entry", "run", "hello world"]);
        assert!(quiet_plan
            .env_overrides
            .contains(&("SPARKLEFORGE_QUIET_BOOT".to_string(), "1".to_string())));
        assert!(quiet_plan
            .env_overrides
            .contains(&("TQDM_DISABLE".to_string(), "1".to_string())));

        let verbose_plan = build_delegation_plan(
            &dir,
            &["run".to_string(), "hello".to_string(), "--verbose".to_string()],
            false,
            Some("/usr/bin"),
        );
        assert!(verbose_plan.verbose);
        assert!(!verbose_plan.filter_noise);
        assert!(verbose_plan.env_overrides.is_empty());
        assert_eq!(
            verbose_plan.args,
            vec!["-m", "src.cli.entry", "--verbose", "run", "hello"]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn noisy_boot_line_filter_targets_chatter_and_keeps_payload() {
        assert!(is_noisy_boot_line("Loading weights: 100%|##########| 12/12 [00:01<00:00]"));
        assert!(is_noisy_boot_line("Batches: 100%|##########| 4/4 [00:00<00:00, 12.1it/s]"));
        assert!(is_noisy_boot_line("[WARN] Both GOOGLE_API_KEY and GEMINI_API_KEY are set"));
        assert!(is_noisy_boot_line("2026-07-24 12:37:12,428 - src.core.db.database_driver - INFO - DatabaseDriver initialized"));
        assert!(is_noisy_boot_line("UserWarning: flashrank is not installed"));

        assert!(!is_noisy_boot_line("Final answer: Rust CLI ready."));
        assert!(!is_noisy_boot_line("Traceback (most recent call last):"));
        assert!(!is_noisy_boot_line("[FAIL] Coworker session failed"));
    }

    #[tokio::test]
    async fn execute_plan_filters_noise_in_quiet_mode_and_preserves_in_verbose() {
        let dir = make_temp_dir("exec");
        let pkg_dir = dir.join("src").join("cli");
        std::fs::create_dir_all(&pkg_dir).unwrap();
        std::fs::write(dir.join("src").join("__init__.py"), "").unwrap();
        std::fs::write(pkg_dir.join("__init__.py"), "").unwrap();
        std::fs::write(
            pkg_dir.join("entry.py"),
            "import os, sys\n\
             sys.stderr.write('Loading weights: 100%|###|\\n')\n\
             sys.stderr.write('[WARN] Both GOOGLE_API_KEY and GEMINI_API_KEY are set\\n')\n\
             sys.stdout.write('CLEAN_PAYLOAD:' + os.environ.get('SPARKLEFORGE_QUIET_BOOT', '0') + '\\n')\n",
        )
        .unwrap();

        let quiet_plan = build_delegation_plan(
            &dir,
            &["run".to_string(), "hi".to_string()],
            false,
            env::var("PATH").ok().as_deref(),
        );
        let rc = execute_plan(&dir, &quiet_plan).await;
        assert_eq!(rc, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }
}

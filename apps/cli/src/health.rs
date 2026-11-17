use serde::Serialize;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Serialize)]
pub struct BinaryHealth {
    pub ready: bool,
    pub name: String,
    pub version: String,
    pub executable: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PythonHealth {
    pub ready: bool,
    pub venv_python: String,
    pub venv_exists: bool,
    pub active_python: Option<String>,
    pub python_version: Option<String>,
    pub entrypoint_exists: bool,
    pub main_py_exists: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiKeysHealth {
    pub any_configured: bool,
    pub google_configured: bool,
    pub openrouter_configured: bool,
    pub openai_configured: bool,
    pub anthropic_configured: bool,
    pub nvidia_configured: bool,
    pub providers: BTreeMap<String, String>,
    pub dotenv_present: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceState {
    pub root: String,
    pub sparkleforge_dir: String,
    pub exists: bool,
    pub state: String,
    pub writable: bool,
    pub entry_count: usize,
    pub has_session_db: bool,
    pub json_files: usize,
    pub trace_files: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct HealthReport {
    pub status: String,
    pub healthy: bool,
    pub elapsed_ms: f64,
    pub binary: BinaryHealth,
    pub python: PythonHealth,
    pub api_keys: ApiKeysHealth,
    pub workspace: WorkspaceState,
    pub notes: Vec<String>,
}

pub fn run_health(workspace_root: Option<&Path>, json: bool, strict: bool) -> i32 {
    let start = Instant::now();
    let root = workspace_root
        .map(PathBuf::from)
        .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let env_map: BTreeMap<String, String> = env::vars().collect();
    let mut report = inspect_health(&root, &env_map);
    report.elapsed_ms = (start.elapsed().as_secs_f64() * 1000.0 * 1000.0).round() / 1000.0;

    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{text}"),
            Err(err) => {
                eprintln!("{{\"status\":\"error\",\"error\":{:?}}}", err.to_string());
                return 1;
            }
        }
    } else {
        print_human_report(&report);
    }

    if strict && (!report.python.ready || !report.api_keys.any_configured) {
        return 1;
    }
    if !report.binary.ready {
        return 1;
    }
    0
}

pub fn inspect_health(root: &Path, env_map: &BTreeMap<String, String>) -> HealthReport {
    let binary = check_binary();
    let python = check_python(root, env_map);
    let api_keys = check_api_keys(root, env_map);
    let workspace = check_workspace_state(root);

    let mut notes = Vec::new();
    if !python.venv_exists {
        if python.active_python.is_some() {
            notes.push("Local .venv/bin/python not found; using fallback python3.".to_string());
        } else {
            notes.push("Neither .venv/bin/python nor python3 found on PATH.".to_string());
        }
    }
    if !api_keys.any_configured {
        notes.push(
            "No LLM API key detected (set GOOGLE_API_KEY or OPENROUTER_API_KEY for agent runs)."
                .to_string(),
        );
    }
    if !workspace.exists {
        notes.push(".sparkleforge workspace state not initialized yet (will be created lazily).".to_string());
    }

    let healthy = binary.ready && python.ready;
    let status = if !healthy {
        "unhealthy"
    } else if api_keys.any_configured {
        "ok"
    } else {
        "degraded"
    };

    HealthReport {
        status: status.to_string(),
        healthy,
        elapsed_ms: 0.0,
        binary,
        python,
        api_keys,
        workspace,
        notes,
    }
}

fn check_binary() -> BinaryHealth {
    let exe = env::current_exe()
        .ok()
        .map(|p| p.display().to_string());
    BinaryHealth {
        ready: true,
        name: "sparkleforge".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        executable: exe,
    }
}

fn check_python(root: &Path, env_map: &BTreeMap<String, String>) -> PythonHealth {
    let venv_rel = root.join(".venv").join("bin").join("python");
    let venv_exists = is_executable_file(&venv_rel);
    let fallback_python = find_on_path("python3", env_map.get("PATH").map(String::as_str));

    let active_python = if venv_exists {
        Some(venv_rel.display().to_string())
    } else {
        fallback_python.map(|p| p.display().to_string())
    };

    let python_version = active_python.as_deref().and_then(probe_python_version);
    let entrypoint_exists = root.join("src").join("cli").join("entry.py").is_file();
    let main_py_exists = root.join("main.py").is_file();
    let ready = active_python.is_some() && python_version.is_some();

    PythonHealth {
        ready,
        venv_python: venv_rel.display().to_string(),
        venv_exists,
        active_python,
        python_version,
        entrypoint_exists,
        main_py_exists,
    }
}

fn probe_python_version(python_bin: &str) -> Option<String> {
    let output = Command::new(python_bin).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let text = if !stdout.is_empty() { stdout } else { stderr };
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

pub fn find_on_path(binary: &str, path_var: Option<&str>) -> Option<PathBuf> {
    let path_str = path_var?;
    for dir in env::split_paths(path_str) {
        let candidate = dir.join(binary);
        if is_executable_file(&candidate) {
            return Some(candidate);
        }
    }
    None
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(meta) = fs::metadata(path) else {
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

fn check_api_keys(root: &Path, env_map: &BTreeMap<String, String>) -> ApiKeysHealth {
    let dotenv_path = root.join(".env");
    let dotenv_present = dotenv_path.is_file();
    let dotenv_map = if dotenv_present {
        parse_dotenv_file(&dotenv_path)
    } else {
        BTreeMap::new()
    };

    let key_source = |name: &str| -> Option<&'static str> {
        if env_map
            .get(name)
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
        {
            Some("env")
        } else if dotenv_map
            .get(name)
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
        {
            Some("dotenv")
        } else {
            None
        }
    };

    // Unify GOOGLE_API_KEY and GEMINI_API_KEY cleanly without duplicate warnings.
    let google_src = key_source("GOOGLE_API_KEY").or_else(|| key_source("GEMINI_API_KEY"));
    let openrouter_src = key_source("OPENROUTER_API_KEY");
    let openai_src = key_source("OPENAI_API_KEY");
    let anthropic_src = key_source("ANTHROPIC_API_KEY");
    let nvidia_src = key_source("NVIDIA_API_KEY");

    let mut providers = BTreeMap::new();
    providers.insert(
        "google".to_string(),
        google_src.unwrap_or("unset").to_string(),
    );
    providers.insert(
        "openrouter".to_string(),
        openrouter_src.unwrap_or("unset").to_string(),
    );
    providers.insert(
        "openai".to_string(),
        openai_src.unwrap_or("unset").to_string(),
    );
    providers.insert(
        "anthropic".to_string(),
        anthropic_src.unwrap_or("unset").to_string(),
    );
    providers.insert(
        "nvidia".to_string(),
        nvidia_src.unwrap_or("unset").to_string(),
    );

    let any_configured = [
        google_src,
        openrouter_src,
        openai_src,
        anthropic_src,
        nvidia_src,
    ]
    .iter()
    .any(Option::is_some);

    ApiKeysHealth {
        any_configured,
        google_configured: google_src.is_some(),
        openrouter_configured: openrouter_src.is_some(),
        openai_configured: openai_src.is_some(),
        anthropic_configured: anthropic_src.is_some(),
        nvidia_configured: nvidia_src.is_some(),
        providers,
        dotenv_present,
    }
}

pub fn parse_dotenv_file(path: &Path) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let Ok(content) = fs::read_to_string(path) else {
        return map;
    };
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line).trim();
        let Some((key, val)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let mut val = val.trim();
        if (val.starts_with('"') && val.ends_with('"') && val.len() >= 2)
            || (val.starts_with('\'') && val.ends_with('\'') && val.len() >= 2)
        {
            val = &val[1..val.len() - 1];
        }
        if !key.is_empty() && !val.is_empty() {
            map.insert(key.to_string(), val.to_string());
        }
    }
    map
}

fn check_workspace_state(root: &Path) -> WorkspaceState {
    let sf_dir = root.join(".sparkleforge");
    let exists = sf_dir.is_dir();

    if !exists {
        return WorkspaceState {
            root: root.display().to_string(),
            sparkleforge_dir: sf_dir.display().to_string(),
            exists: false,
            state: "not_initialized".to_string(),
            writable: is_dir_writable(root),
            entry_count: 0,
            has_session_db: false,
            json_files: 0,
            trace_files: 0,
        };
    }

    let mut entry_count = 0usize;
    let mut json_files = 0usize;
    let mut trace_files = 0usize;
    let mut has_session_db = sf_dir.join("session.db").is_file() || sf_dir.join("sessions.db").is_file();

    if let Ok(entries) = fs::read_dir(&sf_dir) {
        for entry in entries.flatten() {
            entry_count += 1;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    match ext.to_ascii_lowercase().as_str() {
                        "json" => json_files += 1,
                        "jsonl" | "trace" => trace_files += 1,
                        "db" | "sqlite" | "sqlite3" => has_session_db = true,
                        _ => {}
                    }
                }
            }
        }
    }

    WorkspaceState {
        root: root.display().to_string(),
        sparkleforge_dir: sf_dir.display().to_string(),
        exists: true,
        state: if entry_count == 0 {
            "empty".to_string()
        } else {
            "active".to_string()
        },
        writable: is_dir_writable(&sf_dir),
        entry_count,
        has_session_db,
        json_files,
        trace_files,
    }
}

fn is_dir_writable(dir: &Path) -> bool {
    let Ok(meta) = fs::metadata(dir) else {
        return false;
    };
    !meta.permissions().readonly()
}

fn print_human_report(report: &HealthReport) {
    println!(
        "SparkleForge Health: {} ({:.2}ms)",
        report.status.to_ascii_uppercase(),
        report.elapsed_ms
    );
    println!(
        "[OK] Binary: {} v{} (ready={})",
        report.binary.name, report.binary.version, report.binary.ready
    );

    if report.python.ready {
        let ver = report
            .python
            .python_version
            .as_deref()
            .unwrap_or("unknown");
        let source = if report.python.venv_exists {
            "venv"
        } else {
            "fallback"
        };
        println!(
            "[OK] Python ({source}): {} ({ver})",
            report.python.active_python.as_deref().unwrap_or("none")
        );
    } else {
        println!("[FAIL] Python: not found (.venv/bin/python or python3)");
    }

    let configured_providers: Vec<&str> = report
        .api_keys
        .providers
        .iter()
        .filter(|(_, src)| src.as_str() != "unset")
        .map(|(name, _)| name.as_str())
        .collect();
    if configured_providers.is_empty() {
        println!("[INFO] API Keys: none configured (GOOGLE_API_KEY/OPENROUTER_API_KEY unset)");
    } else {
        println!(
            "[OK] API Keys configured: {}",
            configured_providers.join(", ")
        );
    }

    println!(
        "[OK] Workspace .sparkleforge: {} (entries={}, session_db={}, json={}, trace={})",
        report.workspace.state,
        report.workspace.entry_count,
        report.workspace.has_session_db,
        report.workspace.json_files,
        report.workspace.trace_files
    );

    for note in &report.notes {
        println!("[INFO] {note}");
    }
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
        let dir = env::temp_dir().join(format!("sf_health_test_{label}_{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn health_unifies_google_and_gemini_keys_without_duplicate_warnings() {
        let root = make_temp_dir("keys");
        let mut env_map = BTreeMap::new();
        env_map.insert("GOOGLE_API_KEY".to_string(), "g-key".to_string());
        env_map.insert("GEMINI_API_KEY".to_string(), "gem-key".to_string());
        env_map.insert("OPENROUTER_API_KEY".to_string(), "or-key".to_string());

        let report = inspect_health(&root, &env_map);
        assert!(report.api_keys.any_configured);
        assert!(report.api_keys.google_configured);
        assert!(report.api_keys.openrouter_configured);
        assert!(report
            .notes
            .iter()
            .all(|n| !n.contains("Both GOOGLE_API_KEY and GEMINI_API_KEY")));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn health_reads_dotenv_and_workspace_state() {
        let root = make_temp_dir("dotenv");
        fs::write(
            root.join(".env"),
            "# comment\nOPENROUTER_API_KEY=\"sk-or-test\"\n",
        )
        .unwrap();
        let sf_dir = root.join(".sparkleforge");
        fs::create_dir_all(&sf_dir).unwrap();
        fs::write(sf_dir.join("session.db"), b"SQLite format 3\0").unwrap();
        fs::write(sf_dir.join("state.json"), b"{}").unwrap();

        let env_map = BTreeMap::new();
        let report = inspect_health(&root, &env_map);
        assert!(report.api_keys.dotenv_present);
        assert!(report.api_keys.openrouter_configured);
        assert_eq!(report.workspace.state, "active");
        assert!(report.workspace.has_session_db);
        assert_eq!(report.workspace.json_files, 1);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn health_detects_missing_sparkleforge_gracefully() {
        let root = make_temp_dir("empty_ws");
        let env_map = BTreeMap::new();
        let report = inspect_health(&root, &env_map);
        assert!(!report.workspace.exists);
        assert_eq!(report.workspace.state, "not_initialized");
        assert!(report.binary.ready);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    #[cfg(unix)]
    fn health_detects_local_venv_python() {
        use std::os::unix::fs::PermissionsExt;

        let root = make_temp_dir("venv");
        let bin_dir = root.join(".venv").join("bin");
        fs::create_dir_all(&bin_dir).unwrap();
        let py_path = bin_dir.join("python");
        fs::write(&py_path, "#!/bin/sh\necho 'Python 3.12.0'\n").unwrap();
        fs::set_permissions(&py_path, fs::Permissions::from_mode(0o755)).unwrap();

        let report = inspect_health(&root, &BTreeMap::new());
        assert!(report.python.venv_exists);
        assert!(report.python.ready);
        assert_eq!(
            report.python.python_version.as_deref(),
            Some("Python 3.12.0")
        );

        let _ = fs::remove_dir_all(&root);
    }
}

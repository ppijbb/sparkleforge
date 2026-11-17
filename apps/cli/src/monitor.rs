use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Serialize)]
pub struct TaskMetrics {
    pub active: usize,
    pub completed: usize,
    pub failed: usize,
    pub pending: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SubagentMetrics {
    pub active: usize,
    pub total_unique: usize,
    pub max_concurrency: usize,
    pub avg_concurrency: f64,
    pub subagents: Vec<SubagentSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SubagentSummary {
    pub name: String,
    pub status: String,
    pub tasks_total: usize,
    pub avg_duration_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StepDurationSample {
    pub step: String,
    pub duration_ms: f64,
    pub status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StepDurationMetrics {
    pub count: usize,
    pub total_ms: f64,
    pub avg_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub p95_ms: f64,
    pub recent: Vec<StepDurationSample>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorMetrics {
    pub count: usize,
    pub warning_count: usize,
    pub recent: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SqliteFileSummary {
    pub path: String,
    pub size_bytes: u64,
    pub valid_header: bool,
    pub page_size: Option<u32>,
    pub tables: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MonitorReport {
    pub session_dir: String,
    pub exists: bool,
    pub empty: bool,
    pub elapsed_ms: f64,
    pub sessions_count: usize,
    pub active_sessions: usize,
    pub tasks: TaskMetrics,
    pub subagents: SubagentMetrics,
    pub step_durations: StepDurationMetrics,
    pub errors: ErrorMetrics,
    pub sqlite_files: Vec<SqliteFileSummary>,
    pub files_inspected: Vec<String>,
}

#[derive(Default)]
struct Aggregator {
    sessions_count: usize,
    active_sessions: usize,
    tasks_active: usize,
    tasks_completed: usize,
    tasks_failed: usize,
    tasks_pending: usize,
    subagent_map: BTreeMap<String, SubagentAcc>,
    concurrency_samples: Vec<usize>,
    max_concurrency_hint: usize,
    step_samples: Vec<StepDurationSample>,
    raw_durations_ms: Vec<f64>,
    error_count: usize,
    warning_count: usize,
    recent_errors: Vec<String>,
    sqlite_files: Vec<SqliteFileSummary>,
    files_inspected: Vec<String>,
}

#[derive(Default)]
struct SubagentAcc {
    status: String,
    tasks_total: usize,
    duration_sum_ms: f64,
    duration_count: usize,
}

pub fn run_monitor(session_dir: Option<&Path>, json: bool) -> i32 {
    let start = Instant::now();
    let dir = session_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(".sparkleforge")
        });

    let mut report = inspect_session_dir(&dir);
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
        print_human_monitor(&report);
    }

    0
}

pub fn inspect_session_dir(dir: &Path) -> MonitorReport {
    let exists = dir.is_dir();
    if !exists {
        return empty_report(dir, false);
    }

    let mut files = Vec::new();
    collect_files(dir, 0, 3, &mut files);
    files.sort();

    if files.is_empty() {
        return empty_report(dir, true);
    }

    let mut agg = Aggregator::default();
    for file in &files {
        let ext = file
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();

        match ext.as_str() {
            "json" => {
                agg.files_inspected.push(file.display().to_string());
                if let Ok(content) = fs::read_to_string(file) {
                    if let Ok(val) = serde_json::from_str::<Value>(&content) {
                        ingest_json_value(&val, &mut agg);
                    }
                }
            }
            "jsonl" | "trace" => {
                agg.files_inspected.push(file.display().to_string());
                ingest_jsonl_file(file, &mut agg);
            }
            "db" | "sqlite" | "sqlite3" => {
                agg.files_inspected.push(file.display().to_string());
                ingest_sqlite_file(file, &mut agg);
            }
            _ => {}
        }
    }

    if agg.files_inspected.is_empty() {
        return empty_report(dir, true);
    }

    finalize_report(dir, true, agg)
}

fn empty_report(dir: &Path, exists: bool) -> MonitorReport {
    MonitorReport {
        session_dir: dir.display().to_string(),
        exists,
        empty: true,
        elapsed_ms: 0.0,
        sessions_count: 0,
        active_sessions: 0,
        tasks: TaskMetrics {
            active: 0,
            completed: 0,
            failed: 0,
            pending: 0,
            total: 0,
        },
        subagents: SubagentMetrics {
            active: 0,
            total_unique: 0,
            max_concurrency: 0,
            avg_concurrency: 0.0,
            subagents: Vec::new(),
        },
        step_durations: StepDurationMetrics {
            count: 0,
            total_ms: 0.0,
            avg_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            p95_ms: 0.0,
            recent: Vec::new(),
        },
        errors: ErrorMetrics {
            count: 0,
            warning_count: 0,
            recent: Vec::new(),
        },
        sqlite_files: Vec::new(),
        files_inspected: Vec::new(),
    }
}

fn collect_files(dir: &Path, depth: usize, max_depth: usize, out: &mut Vec<PathBuf>) {
    if depth > max_depth {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, depth + 1, max_depth, out);
        } else if path.is_file() {
            out.push(path);
        }
    }
}

fn ingest_jsonl_file(path: &Path, agg: &mut Aggregator) {
    let Ok(file) = fs::File::open(path) else {
        return;
    };
    let reader = BufReader::new(file);
    for line in reader.lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
            ingest_event_object(&val, agg);
        }
    }
}

fn ingest_json_value(val: &Value, agg: &mut Aggregator) {
    match val {
        Value::Array(items) => {
            for item in items {
                ingest_json_value(item, agg);
            }
        }
        Value::Object(map) => {
            // Check if this is a session/job map like sdk_jobs.json { "job-1": { ... }, ... }
            let looks_like_id_map = !map.is_empty()
                && map.values().all(|v| v.is_object())
                && !map.contains_key("session_id")
                && !map.contains_key("state")
                && !map.contains_key("tasks")
                && !map.contains_key("records")
                && !map.contains_key("subagents")
                && !map.contains_key("steps")
                && !map.contains_key("status");

            if looks_like_id_map {
                for child in map.values() {
                    ingest_json_value(child, agg);
                }
                return;
            }

            if map.contains_key("session_id") || map.contains_key("saved_at") {
                agg.sessions_count += 1;
            }

            if let Some(status) = map.get("status").and_then(Value::as_str) {
                let norm = normalize_status(status);
                if map.contains_key("session_id") || map.contains_key("job_id") {
                    if norm == "active" {
                        agg.active_sessions += 1;
                    }
                }
                if !map.contains_key("tasks") && !map.contains_key("steps") {
                    record_task_status(norm, agg);
                }
            }

            if let Some(err_cnt) = map.get("error_count").and_then(Value::as_u64) {
                agg.error_count += err_cnt as usize;
            }
            if let Some(warn_cnt) = map.get("warning_count").and_then(Value::as_u64) {
                agg.warning_count += warn_cnt as usize;
            }
            if let Some(err_str) = map.get("error").and_then(Value::as_str) {
                if !err_str.trim().is_empty() {
                    push_error(agg, err_str.to_string());
                }
            }
            if let Some(errors_arr) = map.get("errors").and_then(Value::as_array) {
                for err in errors_arr {
                    if let Some(msg) = err.as_str() {
                        push_error(agg, msg.to_string());
                    } else if let Some(msg) = err.get("message").and_then(Value::as_str) {
                        push_error(agg, msg.to_string());
                    } else {
                        agg.error_count += 1;
                    }
                }
            }

            if let Some(max_c) = map.get("max_concurrency").and_then(Value::as_u64) {
                agg.max_concurrency_hint = agg.max_concurrency_hint.max(max_c as usize);
            }
            if let Some(conc) = map.get("concurrency").and_then(Value::as_u64) {
                agg.concurrency_samples.push(conc as usize);
            }
            if let Some(active_sa) = map.get("active_subagents").and_then(Value::as_u64) {
                agg.concurrency_samples.push(active_sa as usize);
            }

            // Explicit summary counters if present without task list
            if !map.contains_key("tasks") {
                if let Some(v) = map.get("active_tasks").and_then(Value::as_u64) {
                    agg.tasks_active += v as usize;
                }
                if let Some(v) = map.get("completed_tasks").and_then(Value::as_u64) {
                    agg.tasks_completed += v as usize;
                }
                if let Some(v) = map.get("failed_tasks").and_then(Value::as_u64) {
                    agg.tasks_failed += v as usize;
                }
            }

            if let Some(tasks) = map.get("tasks").and_then(Value::as_array) {
                for t in tasks {
                    ingest_task_entry(t, agg);
                }
            }

            for key in ["subagents", "agents", "workers"] {
                if let Some(arr) = map.get(key).and_then(Value::as_array) {
                    for item in arr {
                        ingest_subagent_entry(item, agg);
                    }
                } else if let Some(obj) = map.get(key).and_then(Value::as_object) {
                    for (name, item) in obj {
                        ingest_named_subagent(name, item, agg);
                    }
                }
            }

            if let Some(records) = map.get("records").and_then(Value::as_array) {
                for rec in records {
                    if rec.get("agent_name").is_some() {
                        ingest_subagent_entry(rec, agg);
                    } else if rec.get("stage").is_some() {
                        ingest_event_object(rec, agg);
                    }
                }
            }

            for key in ["steps", "stage_history", "traces"] {
                if let Some(arr) = map.get(key).and_then(Value::as_array) {
                    for step in arr {
                        ingest_step_entry(step, agg);
                    }
                }
            }

            if let Some(durations) = map.get("step_durations_ms").and_then(Value::as_array) {
                for (idx, d) in durations.iter().enumerate() {
                    if let Some(ms) = d.as_f64() {
                        record_step_duration(format!("step_{idx}"), ms, "completed", agg);
                    }
                }
            }

            if let Some(state_val) = map.get("state") {
                if state_val.is_object() {
                    ingest_json_value(state_val, agg);
                }
            }
        }
        _ => {}
    }
}

fn ingest_task_entry(val: &Value, agg: &mut Aggregator) {
    let status = val
        .get("status")
        .and_then(Value::as_str)
        .map(normalize_status)
        .unwrap_or("pending");
    record_task_status(status, agg);
    if status == "failed" {
        if let Some(err) = val.get("error").and_then(Value::as_str) {
            push_error_without_double_count(agg, err.to_string());
        }
    }
    if let Some(ms) = extract_duration_ms(val) {
        let name = val
            .get("name")
            .or_else(|| val.get("task_id"))
            .or_else(|| val.get("id"))
            .and_then(Value::as_str)
            .unwrap_or("task")
            .to_string();
        record_step_duration(name, ms, status, agg);
    }
    if let Some(agent) = extract_subagent_name(val) {
        update_subagent(&agent, status, extract_duration_ms(val), agg);
    }
}

fn ingest_subagent_entry(val: &Value, agg: &mut Aggregator) {
    let name = extract_subagent_name(val).unwrap_or_else(|| "subagent".to_string());
    ingest_named_subagent(&name, val, agg);
}

fn ingest_named_subagent(name: &str, val: &Value, agg: &mut Aggregator) {
    let status = val
        .get("status")
        .and_then(Value::as_str)
        .map(normalize_status)
        .unwrap_or("completed");
    let dur = extract_duration_ms(val);
    let tasks_total = val
        .get("total_tasks")
        .and_then(Value::as_u64)
        .map(|n| n as usize);

    let entry = agg
        .subagent_map
        .entry(name.to_string())
        .or_insert_with(|| SubagentAcc {
            status: status.to_string(),
            tasks_total: 0,
            duration_sum_ms: 0.0,
            duration_count: 0,
        });
    entry.status = status.to_string();
    entry.tasks_total += tasks_total.unwrap_or(1);
    if let Some(ms) = dur {
        entry.duration_sum_ms += ms;
        entry.duration_count += 1;
    }
}

fn ingest_step_entry(val: &Value, agg: &mut Aggregator) {
    let name = val
        .get("step")
        .or_else(|| val.get("stage"))
        .or_else(|| val.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("step")
        .to_string();
    let status = val
        .get("status")
        .and_then(Value::as_str)
        .map(normalize_status)
        .unwrap_or("completed");
    if let Some(ms) = extract_duration_ms(val) {
        record_step_duration(name, ms, status, agg);
    }
    if status == "failed" {
        if let Some(err) = val.get("error").and_then(Value::as_str) {
            push_error(agg, err.to_string());
        } else {
            agg.error_count += 1;
        }
    }
}

fn ingest_event_object(val: &Value, agg: &mut Aggregator) {
    let event_type = val
        .get("event")
        .or_else(|| val.get("type"))
        .or_else(|| val.get("event_type"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();

    let status = val
        .get("status")
        .and_then(Value::as_str)
        .map(normalize_status);

    let level = val
        .get("level")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_uppercase();

    let is_step_event = event_type.starts_with("step")
        || (val.get("step").is_some() && !event_type.contains("task") && val.get("task_id").is_none());
    let is_task_event = event_type.contains("task")
        || val.get("task_id").is_some()
        || (!is_step_event
            && status.is_some()
            && level != "ERROR"
            && level != "WARNING"
            && level != "WARN"
            && val.get("stage").is_none());

    if is_task_event {
        let st = status.unwrap_or_else(|| {
            if event_type.contains("start") {
                "active"
            } else if event_type.contains("fail") || event_type.contains("error") {
                "failed"
            } else if event_type.contains("complete") || event_type.contains("done") {
                "completed"
            } else {
                "pending"
            }
        });
        record_task_status(st, agg);
    }

    if let Some(agent) = extract_subagent_name(val) {
        let st = status.unwrap_or_else(|| {
            if event_type.contains("start") || event_type.contains("spawn") {
                "active"
            } else if event_type.contains("fail") {
                "failed"
            } else {
                "completed"
            }
        });
        update_subagent(&agent, st, extract_duration_ms(val), agg);
    }

    if let Some(conc) = val
        .get("active_subagents")
        .or_else(|| val.get("concurrency"))
        .and_then(Value::as_u64)
    {
        agg.concurrency_samples.push(conc as usize);
    }

    if let Some(ms) = extract_duration_ms(val) {
        let name = val
            .get("step")
            .or_else(|| val.get("stage"))
            .or_else(|| val.get("name"))
            .or_else(|| val.get("event"))
            .and_then(Value::as_str)
            .unwrap_or("step")
            .to_string();
        record_step_duration(name, ms, status.unwrap_or("completed"), agg);
    }

    if level == "ERROR" || event_type.contains("error") || status == Some("failed") {
        let msg = val
            .get("error")
            .or_else(|| val.get("message"))
            .and_then(Value::as_str)
            .unwrap_or("trace error");
        push_error(agg, msg.to_string());
    } else if level == "WARNING" || level == "WARN" {
        agg.warning_count += 1;
    } else if let Some(err_str) = val.get("error").and_then(Value::as_str) {
        if !err_str.trim().is_empty() {
            push_error(agg, err_str.to_string());
        }
    }
}

fn ingest_sqlite_file(path: &Path, agg: &mut Aggregator) {
    let size_bytes = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let mut header = [0u8; 20];
    let valid_header = if let Ok(mut file) = fs::File::open(path) {
        use std::io::Read;
        file.read(&mut header)
            .map(|n| n >= 16 && &header[..16] == b"SQLite format 3\0")
            .unwrap_or(false)
    } else {
        false
    };

    let page_size = if valid_header {
        let raw = u16::from_be_bytes([header[16], header[17]]);
        Some(if raw == 1 { 65536 } else { raw as u32 })
    } else {
        None
    };

    let mut tables = Vec::new();
    if valid_header {
        if let Some(summary) = query_sqlite_via_python(path) {
            tables = summary.tables;
            agg.sessions_count += summary.sessions_count;
            agg.active_sessions += summary.active_sessions;
            agg.tasks_active += summary.tasks_active;
            agg.tasks_completed += summary.tasks_completed;
            agg.tasks_failed += summary.tasks_failed;
            agg.tasks_pending += summary.tasks_pending;
            agg.error_count += summary.error_count;
            for dur in summary.step_durations_ms {
                record_step_duration(dur.step, dur.duration_ms, &dur.status, agg);
            }
            for sa in summary.subagents {
                update_subagent(&sa.name, &sa.status, sa.duration_ms, agg);
            }
            for err in summary.errors {
                push_error_without_double_count(agg, err);
            }
        }
    }

    agg.sqlite_files.push(SqliteFileSummary {
        path: path.display().to_string(),
        size_bytes,
        valid_header,
        page_size,
        tables,
    });
}

#[derive(serde::Deserialize)]
struct SqliteProbeOutput {
    #[serde(default)]
    tables: Vec<String>,
    #[serde(default)]
    sessions_count: usize,
    #[serde(default)]
    active_sessions: usize,
    #[serde(default)]
    tasks_active: usize,
    #[serde(default)]
    tasks_completed: usize,
    #[serde(default)]
    tasks_failed: usize,
    #[serde(default)]
    tasks_pending: usize,
    #[serde(default)]
    error_count: usize,
    #[serde(default)]
    step_durations_ms: Vec<SqliteStepRow>,
    #[serde(default)]
    subagents: Vec<SqliteSubagentRow>,
    #[serde(default)]
    errors: Vec<String>,
}

#[derive(serde::Deserialize)]
struct SqliteStepRow {
    step: String,
    duration_ms: f64,
    status: String,
}

#[derive(serde::Deserialize)]
struct SqliteSubagentRow {
    name: String,
    status: String,
    duration_ms: Option<f64>,
}

fn query_sqlite_via_python(db_path: &Path) -> Option<SqliteProbeOutput> {
    let py_script = r#"
import json, sqlite3, sys
path = sys.argv[1]
out = {
    "tables": [], "sessions_count": 0, "active_sessions": 0,
    "tasks_active": 0, "tasks_completed": 0, "tasks_failed": 0, "tasks_pending": 0,
    "error_count": 0, "step_durations_ms": [], "subagents": [], "errors": []
}
try:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    out["tables"] = tables
    def norm(s):
        v = str(s or "").strip().lower()
        if v in ("active", "running", "in_progress"): return "active"
        if v in ("completed", "done", "success", "succeeded"): return "completed"
        if v in ("failed", "error", "errored"): return "failed"
        return "pending"
    for tbl in tables:
        low = tbl.lower()
        if low in ("sessions", "session_metadata"):
            rows = cur.execute(f'SELECT * FROM "{tbl}"').fetchall()
            out["sessions_count"] += len(rows)
            for r in rows:
                d = dict(r)
                if "status" in d and norm(d["status"]) == "active":
                    out["active_sessions"] += 1
                if "error_count" in d and isinstance(d["error_count"], int):
                    out["error_count"] += d["error_count"]
        elif low in ("tasks", "subagents", "steps", "trace_events", "traces", "events"):
            rows = cur.execute(f'SELECT * FROM "{tbl}"').fetchall()
            for r in rows:
                d = dict(r)
                st = norm(d.get("status"))
                if low in ("tasks", "events"):
                    out[f"tasks_{st}"] += 1
                if st == "failed":
                    out["error_count"] += 1
                    if d.get("error"):
                        out["errors"].append(str(d["error"]))
                dur = d.get("duration_ms") or d.get("latency_ms") or d.get("elapsed_ms")
                if dur is None and d.get("duration_sec") is not None:
                    dur = float(d["duration_sec"]) * 1000.0
                if dur is not None:
                    out["step_durations_ms"].append({
                        "step": str(d.get("step") or d.get("name") or d.get("stage") or tbl),
                        "duration_ms": float(dur),
                        "status": st,
                    })
                agent = d.get("agent_id") or d.get("subagent") or d.get("agent_name")
                if agent or low == "subagents":
                    out["subagents"].append({
                        "name": str(agent or d.get("name") or "subagent"),
                        "status": st,
                        "duration_ms": float(dur) if dur is not None else None,
                    })
    conn.close()
except Exception:
    pass
print(json.dumps(out))
"#;

    let venv_py = Path::new(".venv/bin/python");
    let python_bin = if venv_py.is_file() {
        venv_py.to_str().unwrap_or("python3")
    } else {
        "python3"
    };

    let output = Command::new(python_bin)
        .arg("-c")
        .arg(py_script)
        .arg(db_path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    serde_json::from_slice::<SqliteProbeOutput>(&output.stdout).ok()
}

fn extract_subagent_name(val: &Value) -> Option<String> {
    for key in ["agent_id", "agent_name", "subagent", "worker", "role"] {
        if let Some(s) = val.get(key).and_then(Value::as_str) {
            if !s.trim().is_empty() {
                return Some(s.trim().to_string());
            }
        }
    }
    None
}

fn extract_duration_ms(val: &Value) -> Option<f64> {
    for key in ["duration_ms", "elapsed_ms", "latency_ms"] {
        if let Some(v) = val.get(key).and_then(Value::as_f64) {
            if v >= 0.0 {
                return Some(v);
            }
        }
    }
    for key in [
        "duration_sec",
        "duration_s",
        "elapsed_sec",
        "latency_sec",
        "avg_execution_time",
    ] {
        if let Some(v) = val.get(key).and_then(Value::as_f64) {
            if v >= 0.0 {
                return Some(v * 1000.0);
            }
        }
    }
    None
}

fn normalize_status(raw: &str) -> &'static str {
    match raw.trim().to_ascii_lowercase().as_str() {
        "active" | "running" | "in_progress" | "executing" => "active",
        "completed" | "done" | "success" | "succeeded" => "completed",
        "failed" | "error" | "errored" | "cancelled" | "canceled" => "failed",
        _ => "pending",
    }
}

fn record_task_status(status: &str, agg: &mut Aggregator) {
    match status {
        "active" => agg.tasks_active += 1,
        "completed" => agg.tasks_completed += 1,
        "failed" => agg.tasks_failed += 1,
        _ => agg.tasks_pending += 1,
    }
}

fn update_subagent(name: &str, status: &str, dur_ms: Option<f64>, agg: &mut Aggregator) {
    let acc = agg
        .subagent_map
        .entry(name.to_string())
        .or_insert_with(|| SubagentAcc {
            status: status.to_string(),
            tasks_total: 0,
            duration_sum_ms: 0.0,
            duration_count: 0,
        });
    acc.status = status.to_string();
    acc.tasks_total += 1;
    if let Some(ms) = dur_ms {
        acc.duration_sum_ms += ms;
        acc.duration_count += 1;
    }
}

fn record_step_duration(step: String, duration_ms: f64, status: &str, agg: &mut Aggregator) {
    agg.raw_durations_ms.push(duration_ms);
    if agg.step_samples.len() < 20 {
        agg.step_samples.push(StepDurationSample {
            step,
            duration_ms: round_3(duration_ms),
            status: status.to_string(),
        });
    }
}

fn push_error(agg: &mut Aggregator, msg: String) {
    agg.error_count += 1;
    push_error_without_double_count(agg, msg);
}

fn push_error_without_double_count(agg: &mut Aggregator, msg: String) {
    if agg.recent_errors.len() < 10 && !msg.trim().is_empty() {
        agg.recent_errors.push(msg);
    }
}

fn finalize_report(dir: &Path, exists: bool, mut agg: Aggregator) -> MonitorReport {
    let tasks_total = agg.tasks_active + agg.tasks_completed + agg.tasks_failed + agg.tasks_pending;

    let active_subagents = agg
        .subagent_map
        .values()
        .filter(|s| s.status == "active")
        .count();

    let mut max_concurrency = agg
        .concurrency_samples
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(agg.max_concurrency_hint)
        .max(active_subagents);
    if max_concurrency == 0 && !agg.subagent_map.is_empty() {
        max_concurrency = 1;
    }

    let avg_concurrency = if !agg.concurrency_samples.is_empty() {
        let sum: usize = agg.concurrency_samples.iter().sum();
        round_3(sum as f64 / agg.concurrency_samples.len() as f64)
    } else {
        active_subagents as f64
    };

    let subagents: Vec<SubagentSummary> = agg
        .subagent_map
        .into_iter()
        .map(|(name, acc)| SubagentSummary {
            name,
            status: acc.status,
            tasks_total: acc.tasks_total,
            avg_duration_ms: if acc.duration_count > 0 {
                Some(round_3(acc.duration_sum_ms / acc.duration_count as f64))
            } else {
                None
            },
        })
        .collect();

    let step_durations = compute_step_duration_metrics(&mut agg.raw_durations_ms, agg.step_samples);

    let has_data = agg.sessions_count > 0
        || tasks_total > 0
        || !subagents.is_empty()
        || step_durations.count > 0
        || agg.error_count > 0
        || !agg.sqlite_files.is_empty();

    MonitorReport {
        session_dir: dir.display().to_string(),
        exists,
        empty: !has_data,
        elapsed_ms: 0.0,
        sessions_count: agg.sessions_count,
        active_sessions: agg.active_sessions,
        tasks: TaskMetrics {
            active: agg.tasks_active,
            completed: agg.tasks_completed,
            failed: agg.tasks_failed,
            pending: agg.tasks_pending,
            total: tasks_total,
        },
        subagents: SubagentMetrics {
            active: active_subagents,
            total_unique: subagents.len(),
            max_concurrency,
            avg_concurrency,
            subagents,
        },
        step_durations,
        errors: ErrorMetrics {
            count: agg.error_count,
            warning_count: agg.warning_count,
            recent: agg.recent_errors,
        },
        sqlite_files: agg.sqlite_files,
        files_inspected: agg.files_inspected,
    }
}

fn compute_step_duration_metrics(
    durations: &mut [f64],
    recent: Vec<StepDurationSample>,
) -> StepDurationMetrics {
    if durations.is_empty() {
        return StepDurationMetrics {
            count: 0,
            total_ms: 0.0,
            avg_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            p95_ms: 0.0,
            recent,
        };
    }

    durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let count = durations.len();
    let total_ms: f64 = durations.iter().sum();
    let avg_ms = total_ms / count as f64;
    let min_ms = durations[0];
    let max_ms = durations[count - 1];
    let p95_idx = ((count as f64) * 0.95).ceil() as usize;
    let p95_ms = durations[p95_idx.saturating_sub(1).min(count - 1)];

    StepDurationMetrics {
        count,
        total_ms: round_3(total_ms),
        avg_ms: round_3(avg_ms),
        min_ms: round_3(min_ms),
        max_ms: round_3(max_ms),
        p95_ms: round_3(p95_ms),
        recent,
    }
}

fn round_3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn print_human_monitor(report: &MonitorReport) {
    println!(
        "SparkleForge Orchestration Monitor ({:.2}ms)",
        report.elapsed_ms
    );
    println!("Session dir: {}", report.session_dir);

    if report.empty {
        println!(
            "State: empty (exists={}) - no active or recorded sessions found.",
            report.exists
        );
        println!(
            "Tasks: active=0 completed=0 failed=0 pending=0 | Subagents: active=0 max_concurrency=0 | Errors: 0"
        );
        return;
    }

    println!(
        "Sessions: {} total ({} active) | Files inspected: {}",
        report.sessions_count,
        report.active_sessions,
        report.files_inspected.len()
    );
    println!(
        "Tasks: active={} completed={} failed={} pending={} total={}",
        report.tasks.active,
        report.tasks.completed,
        report.tasks.failed,
        report.tasks.pending,
        report.tasks.total
    );
    println!(
        "Subagent Concurrency: active={} unique={} max_concurrency={} avg_concurrency={:.2}",
        report.subagents.active,
        report.subagents.total_unique,
        report.subagents.max_concurrency,
        report.subagents.avg_concurrency
    );
    if report.step_durations.count > 0 {
        println!(
            "Step Durations ({} steps): avg={:.1}ms min={:.1}ms max={:.1}ms p95={:.1}ms",
            report.step_durations.count,
            report.step_durations.avg_ms,
            report.step_durations.min_ms,
            report.step_durations.max_ms,
            report.step_durations.p95_ms
        );
    } else {
        println!("Step Durations: no step timing samples recorded");
    }
    println!(
        "Errors: {} (warnings={})",
        report.errors.count, report.errors.warning_count
    );

    if !report.errors.recent.is_empty() {
        for err in report.errors.recent.iter().take(5) {
            println!("  - error: {err}");
        }
    }

    // Check unique table list summary if sqlite files were inspected
    let total_tables: BTreeSet<&str> = report
        .sqlite_files
        .iter()
        .flat_map(|s| s.tables.iter().map(String::as_str))
        .collect();
    if !total_tables.is_empty() {
        let joined: Vec<&str> = total_tables.into_iter().collect();
        println!("SQLite tables: {}", joined.join(", "));
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
        let dir = env::temp_dir().join(format!("sf_monitor_test_{label}_{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn monitor_handles_missing_and_empty_dirs_gracefully() {
        let missing = PathBuf::from("/tmp/definitely_missing_sparkleforge_dir_123456");
        let rep = inspect_session_dir(&missing);
        assert!(!rep.exists);
        assert!(rep.empty);
        assert_eq!(rep.tasks.total, 0);
        assert_eq!(rep.errors.count, 0);

        let empty_dir = make_temp_dir("empty");
        let rep2 = inspect_session_dir(&empty_dir);
        assert!(rep2.exists);
        assert!(rep2.empty);
        assert_eq!(rep2.subagents.max_concurrency, 0);
        let _ = fs::remove_dir_all(&empty_dir);
    }

    #[test]
    fn monitor_aggregates_session_json_and_trace_jsonl() {
        let dir = make_temp_dir("data");
        let session_json = r#"{
            "session_id": "sess-001",
            "status": "active",
            "error_count": 1,
            "tasks": [
                {"id": "t1", "status": "completed", "duration_ms": 120.5, "agent_id": "researcher"},
                {"id": "t2", "status": "running", "agent_id": "verifier"},
                {"id": "t3", "status": "failed", "error": "timeout"}
            ],
            "subagents": [
                {"agent_id": "researcher", "status": "completed", "duration_ms": 120.5},
                {"agent_id": "verifier", "status": "active"}
            ],
            "max_concurrency": 2
        }"#;
        fs::write(dir.join("session_001.json"), session_json).unwrap();

        let trace_jsonl = concat!(
            "{\"event\":\"step_end\",\"step\":\"plan\",\"duration_ms\":45.0,\"status\":\"completed\",\"concurrency\":2}\n",
            "{\"event\":\"step_end\",\"step\":\"synth\",\"duration_ms\":200.0,\"status\":\"completed\",\"concurrency\":1}\n",
            "{\"level\":\"ERROR\",\"error\":\"rate limit\"}\n"
        );
        fs::write(dir.join("trace.jsonl"), trace_jsonl).unwrap();

        let report = inspect_session_dir(&dir);
        assert!(!report.empty);
        assert_eq!(report.sessions_count, 1);
        assert_eq!(report.active_sessions, 1);
        assert_eq!(report.tasks.active, 1);
        assert_eq!(report.tasks.completed, 1);
        assert_eq!(report.tasks.failed, 1);
        assert_eq!(report.subagents.total_unique, 2);
        assert_eq!(report.subagents.max_concurrency, 2);
        assert_eq!(report.step_durations.count, 3);
        assert!(report.step_durations.avg_ms > 100.0);
        assert_eq!(report.errors.count, 2);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn monitor_reads_sqlite_database_metrics() {
        let dir = make_temp_dir("sqlite");
        let db_path = dir.join("session.db");
        let status = Command::new("python3")
            .arg("-c")
            .arg(
                "import sqlite3, sys; c = sqlite3.connect(sys.argv[1]); \
                 c.execute('CREATE TABLE sessions (session_id TEXT, status TEXT, error_count INT)'); \
                 c.execute('INSERT INTO sessions VALUES (\"s1\", \"active\", 1)'); \
                 c.execute('CREATE TABLE tasks (task_id TEXT, status TEXT, duration_ms REAL, agent_id TEXT)'); \
                 c.execute('INSERT INTO tasks VALUES (\"t1\", \"completed\", 80.0, \"worker-a\")'); \
                 c.execute('INSERT INTO tasks VALUES (\"t2\", \"active\", 10.0, \"worker-b\")'); \
                 c.commit(); c.close()",
            )
            .arg(&db_path)
            .status();

        if let Ok(st) = status {
            if st.success() {
                let report = inspect_session_dir(&dir);
                assert!(!report.empty);
                assert_eq!(report.sqlite_files.len(), 1);
                assert!(report.sqlite_files[0].valid_header);
                assert_eq!(report.sessions_count, 1);
                assert_eq!(report.tasks.completed, 1);
                assert_eq!(report.tasks.active, 1);
                assert_eq!(report.subagents.total_unique, 2);
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }
}

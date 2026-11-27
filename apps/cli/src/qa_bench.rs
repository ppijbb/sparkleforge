use serde::Serialize;
use std::env;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::process::Command;
use tokio::time::timeout;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AntiPatternHit {
    pub kind: String,
    pub count: usize,
    pub sample: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DimensionScores {
    pub latency: f64,
    pub hygiene: f64,
    pub exit_code: f64,
    pub output_non_empty: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CommandBenchResult {
    pub command: String,
    pub exit_code: i32,
    pub timed_out: bool,
    pub latency_ms: f64,
    pub stdout_bytes: usize,
    pub stderr_bytes: usize,
    pub output_non_empty: bool,
    pub anti_patterns: Vec<AntiPatternHit>,
    pub anti_pattern_count: usize,
    pub dimensions: DimensionScores,
    pub ux_score: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct QaBenchReport {
    pub benchmark_suite: String,
    pub commands_run: usize,
    pub passed_commands: usize,
    pub avg_latency_ms: f64,
    pub total_anti_patterns: usize,
    pub overall_score: f64,
    pub dimensions: DimensionScores,
    pub passed: bool,
    pub results: Vec<CommandBenchResult>,
}

pub async fn run_qa_bench(
    commands: Vec<String>,
    max_latency_ms: u64,
    timeout_ms: u64,
    fail_under: Option<f64>,
    json: bool,
) -> i32 {
    let target_cmds = if commands.is_empty() {
        default_smoke_commands()
    } else {
        commands
    };

    let mut results = Vec::with_capacity(target_cmds.len());
    for cmd in &target_cmds {
        results.push(execute_and_score_command(cmd, max_latency_ms, timeout_ms).await);
    }

    let report = aggregate_report(results, fail_under.unwrap_or(0.7));

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

    if let Some(min_score) = fail_under {
        if report.overall_score < min_score {
            return 1;
        }
    }
    0
}

fn default_smoke_commands() -> Vec<String> {
    let exe = env::current_exe()
        .ok()
        .map(|p| shell_quote(&p.display().to_string()))
        .unwrap_or_else(|| "sparkleforge".to_string());
    vec![
        format!("{exe} ping"),
        format!("{exe} version"),
        format!("{exe} health --json"),
        format!("{exe} monitor --json"),
    ]
}

fn shell_quote(s: &str) -> String {
    if s.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '.' | '_' | '-'))
    {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\"'\"'"))
    }
}

pub async fn execute_and_score_command(
    cmd_str: &str,
    max_latency_ms: u64,
    timeout_ms: u64,
) -> CommandBenchResult {
    let start = Instant::now();
    let mut child_cmd = Command::new("sh");
    child_cmd
        .arg("-c")
        .arg(cmd_str)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let run_future = child_cmd.output();
    let timeout_dur = Duration::from_millis(timeout_ms.max(100));

    let (exit_code, timed_out, stdout_bytes, stderr_bytes, combined_output, stdout_non_empty) =
        match timeout(timeout_dur, run_future).await {
            Ok(Ok(output)) => {
                let code = output.status.code().unwrap_or(-1);
                let out_str = String::from_utf8_lossy(&output.stdout).into_owned();
                let err_str = String::from_utf8_lossy(&output.stderr).into_owned();
                let non_empty = !out_str.trim().is_empty();
                let mut combined = String::with_capacity(out_str.len() + err_str.len() + 1);
                combined.push_str(&out_str);
                if !out_str.is_empty() && !err_str.is_empty() {
                    combined.push('\n');
                }
                combined.push_str(&err_str);
                (
                    code,
                    false,
                    output.stdout.len(),
                    output.stderr.len(),
                    combined,
                    non_empty,
                )
            }
            Ok(Err(err)) => (
                -1,
                false,
                0,
                err.to_string().len(),
                err.to_string(),
                false,
            ),
            Err(_) => (
                -1,
                true,
                0,
                0,
                format!("Command timed out after {timeout_ms}ms"),
                false,
            ),
        };

    let latency_ms = round_3(start.elapsed().as_secs_f64() * 1000.0);
    score_execution(
        cmd_str,
        exit_code,
        timed_out,
        latency_ms,
        stdout_bytes,
        stderr_bytes,
        stdout_non_empty,
        &combined_output,
        max_latency_ms,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn score_execution(
    command: &str,
    exit_code: i32,
    timed_out: bool,
    latency_ms: f64,
    stdout_bytes: usize,
    stderr_bytes: usize,
    output_non_empty: bool,
    combined_output: &str,
    max_latency_ms: u64,
) -> CommandBenchResult {
    let anti_patterns = detect_anti_patterns(combined_output);
    let anti_pattern_count: usize = anti_patterns.iter().map(|a| a.count).sum();

    let latency_score = compute_latency_score(latency_ms, max_latency_ms, timed_out);
    let hygiene_score = compute_hygiene_score(&anti_patterns);
    let exit_code_score = if !timed_out && exit_code == 0 { 1.0 } else { 0.0 };
    let output_score = if output_non_empty {
        1.0
    } else if stderr_bytes > 0 {
        0.25
    } else {
        0.0
    };

    let raw_score = 0.25 * latency_score
        + 0.35 * hygiene_score
        + 0.25 * exit_code_score
        + 0.15 * output_score;
    let ux_score = round_3(raw_score.clamp(0.0, 1.0));
    let passed = !timed_out && exit_code == 0 && output_non_empty && anti_patterns.is_empty();

    CommandBenchResult {
        command: command.to_string(),
        exit_code,
        timed_out,
        latency_ms,
        stdout_bytes,
        stderr_bytes,
        output_non_empty,
        anti_patterns,
        anti_pattern_count,
        dimensions: DimensionScores {
            latency: round_3(latency_score),
            hygiene: round_3(hygiene_score),
            exit_code: round_3(exit_code_score),
            output_non_empty: round_3(output_score),
        },
        ux_score,
        passed,
    }
}

pub fn detect_anti_patterns(text: &str) -> Vec<AntiPatternHit> {
    let mut hits = Vec::new();

    let literal_checks: [(&str, &[&str]); 4] = [
        ("loading_weights", &["Loading weights"]),
        ("batches_progress", &["Batches:"]),
        ("warn_leak", &["[WARN]"]),
        ("traceback", &["Traceback (most recent call last):", "Traceback"]),
    ];

    for (kind, patterns) in literal_checks {
        let mut count = 0usize;
        let mut sample = String::new();
        for line in text.lines() {
            if patterns.iter().any(|pat| line.contains(pat)) {
                count += 1;
                if sample.is_empty() {
                    sample = truncate_sample(line, 120);
                }
            }
        }
        if count > 0 {
            hits.push(AntiPatternHit {
                kind: kind.to_string(),
                count,
                sample,
            });
        }
    }

    // Provider fallback / rate limit spam (402 / 404 / 429) without false-positiving on plain timestamps/latencies.
    let mut http_spam_count = 0usize;
    let mut http_sample = String::new();
    for line in text.lines() {
        if is_provider_status_spam_line(line) {
            http_spam_count += 1;
            if http_sample.is_empty() {
                http_sample = truncate_sample(line, 120);
            }
        }
    }
    if http_spam_count > 0 {
        hits.push(AntiPatternHit {
            kind: "provider_fallback_spam".to_string(),
            count: http_spam_count,
            sample: http_sample,
        });
    }

    // Garbled Unicode replacement character \u{fffd}
    let replacement_count = text.chars().filter(|&c| c == '\u{fffd}').count();
    if replacement_count > 0 {
        let sample_line = text
            .lines()
            .find(|l| l.contains('\u{fffd}'))
            .map(|l| truncate_sample(l, 120))
            .unwrap_or_else(|| "\\u{fffd}".to_string());
        hits.push(AntiPatternHit {
            kind: "replacement_char".to_string(),
            count: replacement_count,
            sample: sample_line,
        });
    }

    hits
}

pub fn is_provider_status_spam_line(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    if lower.contains("all fallback models failed") {
        return true;
    }

    let codes = ["402", "404", "429"];
    let matched_codes: Vec<&str> = codes
        .iter()
        .copied()
        .filter(|code| contains_code_token(&lower, code))
        .collect();
    if matched_codes.is_empty() {
        return false;
    }

    for code in &matched_codes {
        let direct_phrases = [
            format!("({code})"),
            format!("http {code}"),
            format!("http/1.1 {code}"),
            format!("http/2 {code}"),
            format!("status {code}"),
            format!("status: {code}"),
            format!("status={code}"),
            format!("status_code={code}"),
            format!("status_code: {code}"),
            format!("code {code}"),
            format!("code: {code}"),
            format!("code={code}"),
            format!("error {code}"),
            format!("error: {code}"),
            format!("{code} payment required"),
            format!("{code} not found"),
            format!("{code} too many requests"),
        ];
        if direct_phrases.iter().any(|p| lower.contains(p)) {
            return true;
        }
    }

    let spam_context = [
        "fallback",
        "rate limit",
        "rate-limit",
        "rate_limit",
        "too many requests",
        "payment required",
        "trying next",
        "openrouter",
        "nvidia nim",
    ];
    spam_context.iter().any(|m| lower.contains(m))
}

fn contains_code_token(lower: &str, code: &str) -> bool {
    for (idx, _) in lower.match_indices(code) {
        let before_ok = idx == 0
            || !lower.as_bytes()[idx - 1].is_ascii_alphanumeric();
        let after_idx = idx + code.len();
        let after_ok = after_idx >= lower.len()
            || !lower.as_bytes()[after_idx].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

fn compute_latency_score(latency_ms: f64, max_latency_ms: u64, timed_out: bool) -> f64 {
    if timed_out {
        return 0.0;
    }
    let budget = (max_latency_ms.max(1)) as f64;
    if latency_ms <= budget * 0.5 {
        1.0
    } else if latency_ms <= budget {
        // Gentle decay from 1.0 to 0.75 inside budget
        let frac = (latency_ms - budget * 0.5) / (budget * 0.5);
        1.0 - 0.25 * frac
    } else {
        // Decay toward 0.0 past budget
        let over = (latency_ms - budget) / budget;
        (0.75 / (1.0 + over * 1.5)).max(0.0)
    }
}

fn compute_hygiene_score(hits: &[AntiPatternHit]) -> f64 {
    if hits.is_empty() {
        return 1.0;
    }
    let mut penalty = 0.0f64;
    for hit in hits {
        let base = match hit.kind.as_str() {
            "traceback" | "replacement_char" => 0.45,
            "provider_fallback_spam" | "loading_weights" => 0.35,
            _ => 0.25,
        };
        let extra = ((hit.count.saturating_sub(1)) as f64) * 0.08;
        penalty += base + extra;
    }
    (1.0 - penalty).clamp(0.0, 1.0)
}

fn aggregate_report(results: Vec<CommandBenchResult>, min_score: f64) -> QaBenchReport {
    let n = results.len();
    if n == 0 {
        return QaBenchReport {
            benchmark_suite: "rust-cli-qa-bench".to_string(),
            commands_run: 0,
            passed_commands: 0,
            avg_latency_ms: 0.0,
            total_anti_patterns: 0,
            overall_score: 0.0,
            dimensions: DimensionScores {
                latency: 0.0,
                hygiene: 0.0,
                exit_code: 0.0,
                output_non_empty: 0.0,
            },
            passed: false,
            results,
        };
    }

    let nf = n as f64;
    let passed_commands = results.iter().filter(|r| r.passed).count();
    let avg_latency_ms = round_3(results.iter().map(|r| r.latency_ms).sum::<f64>() / nf);
    let total_anti_patterns = results.iter().map(|r| r.anti_pattern_count).sum();
    let overall_score = round_3(results.iter().map(|r| r.ux_score).sum::<f64>() / nf);

    let dimensions = DimensionScores {
        latency: round_3(results.iter().map(|r| r.dimensions.latency).sum::<f64>() / nf),
        hygiene: round_3(results.iter().map(|r| r.dimensions.hygiene).sum::<f64>() / nf),
        exit_code: round_3(results.iter().map(|r| r.dimensions.exit_code).sum::<f64>() / nf),
        output_non_empty: round_3(
            results
                .iter()
                .map(|r| r.dimensions.output_non_empty)
                .sum::<f64>()
                / nf,
        ),
    };

    QaBenchReport {
        benchmark_suite: "rust-cli-qa-bench".to_string(),
        commands_run: n,
        passed_commands,
        avg_latency_ms,
        total_anti_patterns,
        overall_score,
        dimensions,
        passed: overall_score >= min_score && passed_commands == n,
        results,
    }
}

fn truncate_sample(line: &str, max_len: usize) -> String {
    let trimmed = line.trim();
    if trimmed.chars().count() <= max_len {
        trimmed.to_string()
    } else {
        let mut s: String = trimmed.chars().take(max_len).collect();
        s.push_str("...");
        s
    }
}

fn round_3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn print_human_report(report: &QaBenchReport) {
    println!(
        "SparkleForge QA Bench Scorecard: {:.3} / 1.000 (passed={}/{}, avg_latency={:.2}ms)",
        report.overall_score,
        report.passed_commands,
        report.commands_run,
        report.avg_latency_ms
    );
    println!(
        "Dimensions: latency={:.3} hygiene={:.3} exit_code={:.3} output={:.3}",
        report.dimensions.latency,
        report.dimensions.hygiene,
        report.dimensions.exit_code,
        report.dimensions.output_non_empty
    );

    for res in &report.results {
        let status = if res.passed { "PASS" } else { "FAIL" };
        println!(
            "  [{status}] score={:.3} latency={:.2}ms rc={} non_empty={} anti_patterns={} cmd=\"{}\"",
            res.ux_score,
            res.latency_ms,
            res.exit_code,
            res.output_non_empty,
            res.anti_pattern_count,
            res.command
        );
        for hit in &res.anti_patterns {
            println!("      - anti-pattern {}: count={} ({})", hit.kind, hit.count, hit.sample);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anti_pattern_detector_catches_all_ux_issues() {
        let noisy = concat!(
            "Loading weights: 100%|##########|\n",
            "Batches: 100%|##########|\n",
            "[WARN] Duplicate GOOGLE_API_KEY\n",
            "Traceback (most recent call last):\n",
            "  File \"x.py\", line 1\n",
            "Fallback model failed with HTTP 429 rate limit\n",
            "OpenRouter error 402 payment required\n",
            "Model 404 not found fallback\n",
            "Garbled: \u{fffd} glyph\n"
        );
        let hits = detect_anti_patterns(noisy);
        let kinds: Vec<&str> = hits.iter().map(|h| h.kind.as_str()).collect();
        assert!(kinds.contains(&"loading_weights"));
        assert!(kinds.contains(&"batches_progress"));
        assert!(kinds.contains(&"warn_leak"));
        assert!(kinds.contains(&"traceback"));
        assert!(kinds.contains(&"provider_fallback_spam"));
        assert!(kinds.contains(&"replacement_char"));
    }

    #[test]
    fn clean_output_with_numeric_404_ms_does_not_false_positive() {
        let clean = "{\"status\":\"ok\",\"elapsed_ms\":404,\"items\":429}\n";
        let hits = detect_anti_patterns(clean);
        assert!(hits.is_empty(), "unexpected hits: {:?}", hits);

        let scored = score_execution("echo ok", 0, false, 12.0, clean.len(), 0, true, clean, 2000);
        assert!(scored.passed);
        assert_eq!(scored.ux_score, 1.0);
    }

    #[test]
    fn failed_command_and_empty_stdout_penalized_accurately() {
        let scored = score_execution("false", 1, false, 5.0, 0, 12, false, "Traceback", 2000);
        assert!(!scored.passed);
        assert!(scored.ux_score < 0.5);
        assert_eq!(scored.dimensions.exit_code, 0.0);
    }
}

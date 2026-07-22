# SparkleForge Community Launch Posts

This document contains the publication-ready launch posts for the developer
community outreach campaign (issue #866). Each post is benchmark-backed and
technically rigorous. Copy the relevant section verbatim into the target
community, then monitor the thread for technical critiques and feature
requests.

## Shared Benchmark Metrics

- **MTTM (Mean Time To Merge):** 141.08 min
- **Unattended merge rate:** 66.7%
- **Cost savings:** 92%
- **Autonomous self-repair:** 24/7 Deep Research loop

---

## 1. Hacker News (Show HN)

**Title:** Show HN: SparkleForge – An Open-Source Generic Agent OS with 24/7 Deep Research & Autonomous Self-Repair

**Body:**

Hi HN,

We open-sourced SparkleForge, a generic Agent OS that runs a 24/7 autonomous
deep-research and self-repair loop on top of your own repository. The two
architecture decisions we want to highlight are:

1. **Two-Tier Constant-Size Memory** — a working-set lane plus a compacted
   semantic tier keeps the agent's effective context bounded regardless of how
   long the session runs, so long-running research sessions don't blow up the
   prompt budget.
2. **Zero-Cost Reactive Scheduler** — instead of polling on a fixed cadence,
   the scheduler wakes on events (issue opened, PR review submitted, check
   failed) and defers all idle work, which is what lets the 24/7 loop run
   without a constant compute bill.

We measured it against our own CI/CD backlog:

- MTTM: 141.08 min
- Unattended merge rate: 66.7%
- Cost savings: ~92% vs. the human-reviewed baseline

The Nightwelding pipeline reproduces a failing test first, implements against
the red test, and only opens a Draft PR once the test goes green — it never
merges on its own.

Repo: https://github.com/ppijbb/sparkleforge

Happy to answer architecture, benchmarking, or agent-safety questions.

---

## 2. Reddit — r/LocalLLM & r/MachineLearning

**Title:** [Project] SparkleForge: Open-Source Generic Agent OS with 24/7 Autonomous Deep Research & Empirical Benchmarks

**Body:**

We open-sourced SparkleForge, a generic Agent OS that runs autonomous deep
research and self-repair against a real repository, with empirical CI/CD
benchmarks rather than toy tasks.

**What it does**

- 24/7 autonomous deep-research loop with a Two-Tier Constant-Size Memory
  (working-set lane + compacted semantic tier) so context stays bounded.
- Zero-Cost Reactive Scheduler: work is triggered by repo events (issues, PR
  reviews, failed checks), not a fixed poll cadence.
- Nightwelding: reproduce-first — a failing test must go red before the agent
  implements, and green before a Draft PR opens. It never auto-merges.

**Empirical benchmarks (our CI/CD backlog)**

| Metric | Value |
| --- | --- |
| Mean Time To Merge (MTTM) | 141.08 min |
| Unattended merge rate | 66.7% |
| Cost savings vs. human baseline | 92% |

**Why this matters for local-LLM / ML practitioners**

- The memory tiering is model-agnostic and designed to keep local-LLM context
  windows usable over multi-hour sessions.
- The reactive scheduler means you can run the loop on a single local GPU
  without a constant inference cost — it only spends tokens when there's real
  work.

Repo: https://github.com/ppijbb/sparkleforge

We'd love feedback on the benchmark methodology and the memory-tiering
design. Feature requests welcome.

---

## 3. Community Q&A Management

After publishing, monitor each thread and:

1. **Respond to technical critiques first** — especially questions about the
   Two-Tier Memory bounds, the reactive scheduler's correctness under event
   storms, and the benchmark methodology (141.08 min MTTM, 66.7% unattended
   merge rate, 92% cost savings).
2. **Collect feature requests** — log them as GitHub issues with the
   `community-feedback` label so they're traceable back to this campaign.
3. **Be honest about scope** — Nightwelding never auto-merges; the 66.7%
   unattended merge rate reflects the daytime auto-fix pipeline, not
   Nightwelding. Don't conflate the two in replies.
4. **Link back to the repo**, not to a landing page, so the community can read
   the code.

---

## Publication Checklist

- [ ] Post Show HN to Hacker News.
- [ ] Post to r/LocalLLM.
- [ ] Post to r/MachineLearning.
- [ ] Monitor threads for 72 hours after posting.
- [ ] File actionable feature requests as `community-feedback` issues.

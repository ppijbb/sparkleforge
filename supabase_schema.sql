-- Supabase schema for SparkleForge forge_jobs and agent_logs tables.
--
-- Security notes:
--   Row Level Security (RLS) is enabled on every table. Write operations
--   (INSERT/UPDATE) are restricted to authenticated principals only so that
--   unauthenticated clients cannot inject or mutate rows through the public
--   REST API. Anonymous reads are permitted for the public dashboard surface.
--   Rotate keys via environment-aware build tooling; never commit credentials.

create table if not exists public.forge_jobs (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    status text not null default 'pending',
    payload jsonb not null default '{}'::jsonb
);

create table if not exists public.agent_logs (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    job_id uuid references public.forge_jobs(id) on delete cascade,
    level text not null default 'info',
    message text not null,
    metadata jsonb not null default '{}'::jsonb
);

alter table public.forge_jobs enable row level security;
alter table public.agent_logs enable row level security;

-- forge_jobs: public read, authenticated-only writes.
drop policy if exists "Public can read forge jobs" on public.forge_jobs;
create policy "Public can read forge jobs"
  on public.forge_jobs for select
  to anon, authenticated
  using (true);

drop policy if exists "Authenticated users can insert forge jobs" on public.forge_jobs;
create policy "Authenticated users can insert forge jobs"
  on public.forge_jobs for insert
  to authenticated
  with check (true);

drop policy if exists "Authenticated users can update forge jobs" on public.forge_jobs;
create policy "Authenticated users can update forge jobs"
  on public.forge_jobs for update
  to authenticated
  using (true)
  with check (true);

-- agent_logs: public read, authenticated-only writes.
drop policy if exists "Public can read agent logs" on public.agent_logs;
create policy "Public can read agent logs"
  on public.agent_logs for select
  to anon, authenticated
  using (true);

drop policy if exists "Authenticated users can insert agent logs" on public.agent_logs;
create policy "Authenticated users can insert agent logs"
  on public.agent_logs for insert
  to authenticated
  with check (true);

drop policy if exists "Authenticated users can update agent logs" on public.agent_logs;
create policy "Authenticated users can update agent logs"
  on public.agent_logs for update
  to authenticated
  using (true)
  with check (true);

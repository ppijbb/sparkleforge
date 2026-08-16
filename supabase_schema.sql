-- SQL Schema for SparkleForge Web Terminal & Live Browser Preview integration.
-- Copy-paste this script into the Supabase SQL editor to create the necessary tables and set up Row Level Security (RLS).
--
-- Security notes:
--   Row Level Security (RLS) is enabled on every table. Write operations
--   (INSERT/UPDATE) are restricted to authenticated principals only so that
--   unauthenticated clients cannot inject or mutate rows through the public
--   REST API. Anonymous reads are permitted for the public dashboard surface.

-- 1. Create a table for completed research reports (Gallery)
CREATE TABLE IF NOT EXISTS public.reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic TEXT NOT NULL,                     -- Research topic/query
    summary TEXT,                            -- Executive summary of the report
    full_report TEXT NOT NULL,               -- Complete report in Markdown
    confidence_score FLOAT DEFAULT 0.0,      -- Confidence evaluation score (0.0 to 1.0)
    source_count INTEGER DEFAULT 0,          -- Count of verified sources/links
    sources JSONB DEFAULT '[]'::jsonb,       -- List of sources used: [{title, url, reliability}]
    keywords TEXT[] DEFAULT '{}'::text[],    -- Tags/keywords associated with the topic
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,  -- Reference to the requesting user (optional)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 2. Create a table for research queue jobs (Request)
CREATE TABLE IF NOT EXISTS public.forge_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    topic TEXT,                              -- Topic requested for research (legacy)
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed
    priority INTEGER DEFAULT 0,
    worker_id TEXT,
    error_message TEXT,
    payload JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Create a table for persistent agent logs (Live Broadcast backup)
CREATE TABLE IF NOT EXISTS public.agent_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    job_id UUID REFERENCES public.forge_jobs(id) ON DELETE CASCADE,
    session_id TEXT,                         -- maps to objective_id
    agent_name TEXT,                         -- planner, executor, verifier, generator, terminal
    level TEXT NOT NULL DEFAULT 'info',
    message TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMP WITH TIME ZONE
);

-- Enable RLS on all tables
ALTER TABLE public.reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.forge_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_logs ENABLE ROW LEVEL SECURITY;

-- #1106/#1105: these three agent_logs policies were dropped without
-- replacement when agent_error_contexts was added, which would strip all
-- SELECT/INSERT/UPDATE access on an existing database and break any
-- application code reading/writing agent_logs. Restored as they were.
DROP POLICY IF EXISTS "Public can read agent logs" ON public.agent_logs;
CREATE POLICY "Public can read agent logs"
    ON public.agent_logs FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "Authenticated users can insert agent logs" ON public.agent_logs;
CREATE POLICY "Authenticated users can insert agent logs"
    ON public.agent_logs FOR INSERT TO authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "Authenticated users can update agent logs" ON public.agent_logs;
CREATE POLICY "Authenticated users can update agent logs"
    ON public.agent_logs FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

-- reports policies: public read, authenticated-only writes.
DROP POLICY IF EXISTS "Allow public read access to reports" ON public.reports;
CREATE POLICY "Allow public read access to reports" 
    ON public.reports FOR SELECT USING (true);

DROP POLICY IF EXISTS "Allow authenticated insert to reports" ON public.reports;
CREATE POLICY "Allow authenticated insert to reports" 
    ON public.reports FOR INSERT TO authenticated WITH CHECK (true);

-- forge_jobs policies: public read, authenticated-only writes.
DROP POLICY IF EXISTS "Public can read forge jobs" ON public.forge_jobs;
CREATE POLICY "Public can read forge jobs"
    ON public.forge_jobs FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "Authenticated users can insert forge jobs" ON public.forge_jobs;
CREATE POLICY "Authenticated users can insert forge jobs"
    ON public.forge_jobs FOR INSERT TO authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "Authenticated users can update forge jobs" ON public.forge_jobs;
CREATE POLICY "Authenticated users can update forge jobs"
    ON public.forge_jobs FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

-- 4. Create a table for historical agent error contexts & failure analysis (Root Cause Engine)
CREATE TABLE IF NOT EXISTS public.agent_error_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    session_id TEXT,                         -- objective_id or run_id
    scenario_name TEXT,                      -- e.g., scheduled_summary, security_scan, swebench
    error_type TEXT NOT NULL,                -- e.g., ToolBindingGap, LoopStagnation, ModelTimeout, TestFailure
    user_query TEXT,                         -- initial goal / query
    failed_tool_name TEXT,                   -- name of tool that failed
    error_message TEXT NOT NULL,             -- raw error message
    stack_trace TEXT,                        -- full un-truncated traceback
    execution_context JSONB DEFAULT '{}'::jsonb, -- active context, parameters, workspace state
    root_cause_analysis TEXT,                -- system architect / LLM diagnostic analysis
    remediation_status TEXT DEFAULT 'pending', -- pending, analyzing, resolved, ignored
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Enable RLS on agent_error_contexts
ALTER TABLE public.agent_error_contexts ENABLE ROW LEVEL SECURITY;

-- #1106/#1105: agent_error_contexts stores full un-truncated stack traces,
-- raw error messages, and workspace state -- anon read access here is a
-- data exfiltration risk, unlike the intentionally-public reports/forge_jobs
-- tables above.
DROP POLICY IF EXISTS "Public can read agent error contexts" ON public.agent_error_contexts;
DROP POLICY IF EXISTS "Authenticated can read agent error contexts" ON public.agent_error_contexts;
CREATE POLICY "Authenticated can read agent error contexts"
    ON public.agent_error_contexts FOR SELECT TO authenticated USING (true);

DROP POLICY IF EXISTS "Authenticated users can insert agent error contexts" ON public.agent_error_contexts;
CREATE POLICY "Authenticated users can insert agent error contexts"
    ON public.agent_error_contexts FOR INSERT TO authenticated WITH CHECK (true);

DROP POLICY IF EXISTS "Authenticated users can update agent error contexts" ON public.agent_error_contexts;
CREATE POLICY "Authenticated users can update agent error contexts"
    ON public.agent_error_contexts FOR UPDATE TO authenticated USING (true) WITH CHECK (true);


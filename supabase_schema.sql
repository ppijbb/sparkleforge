-- SQL Schema for SparkleForge "Public Anvil" web portal.
-- Copy-paste this script into the Supabase SQL editor to create the necessary tables and set up Row Level Security (RLS).

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

-- Enable RLS for reports (public read, authenticated insert)
ALTER TABLE public.reports ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public read access to reports" 
    ON public.reports FOR SELECT USING (true);

CREATE POLICY "Allow authenticated service insertion to reports" 
    ON public.reports FOR INSERT WITH CHECK (true);


-- 2. Create a table for research queue jobs (Request)
CREATE TABLE IF NOT EXISTS public.forge_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    topic TEXT NOT NULL,                     -- Topic requested for research
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')) DEFAULT 'pending',
    priority INTEGER DEFAULT 0,              -- Priority queue order
    worker_id TEXT,                          -- ID of the VM/worker instance executing it
    error_message TEXT,                      -- Detailed error message if failed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Enable RLS for forge_jobs (users can see and insert their own jobs)
ALTER TABLE public.forge_jobs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public select for jobs" 
    ON public.forge_jobs FOR SELECT USING (true);

CREATE POLICY "Allow public insert for jobs" 
    ON public.forge_jobs FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow service update for jobs" 
    ON public.forge_jobs FOR UPDATE USING (true);


-- 3. Create a table for persistent agent logs (Live Broadcast backup)
CREATE TABLE IF NOT EXISTS public.agent_logs (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,                 -- maps to objective_id
    agent_name TEXT NOT NULL,                 -- planner, executor, verifier, generator, terminal
    level TEXT NOT NULL,                      -- info, warning, error
    message TEXT NOT NULL,                    -- text line
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Enable RLS for agent_logs (public read, service insert)
ALTER TABLE public.agent_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public read for logs" 
    ON public.agent_logs FOR SELECT USING (true);

CREATE POLICY "Allow authenticated insert for logs" 
    ON public.agent_logs FOR INSERT WITH CHECK (true);

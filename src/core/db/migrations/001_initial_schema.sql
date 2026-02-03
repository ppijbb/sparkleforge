-- Initial Schema for Sparkleforge
-- PostgreSQL migration script

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    state_version INTEGER DEFAULT 1,
    context_size INTEGER DEFAULT 0,
    memory_size INTEGER DEFAULT 0,
    tags JSONB,
    description TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_last_accessed ON sessions(last_accessed);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    message_id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES sessions(session_id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);

-- Compaction events table (Phase 3)
CREATE TABLE IF NOT EXISTS compaction_events (
    event_id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES sessions(session_id) ON DELETE CASCADE,
    original_message_count INTEGER NOT NULL,
    compressed_message_count INTEGER NOT NULL,
    original_tokens INTEGER NOT NULL,
    compressed_tokens INTEGER NOT NULL,
    tokens_saved INTEGER NOT NULL,
    strategy_used VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_compaction_events_session_id ON compaction_events(session_id);
CREATE INDEX IF NOT EXISTS idx_compaction_events_created_at ON compaction_events(created_at);

-- Memory storage table (Phase 1)
CREATE TABLE IF NOT EXISTS memories (
    memory_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    memory_type VARCHAR(50) NOT NULL,  -- semantic, episodic, procedural
    content TEXT NOT NULL,
    importance FLOAT DEFAULT 0.5,
    tags JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed);

-- Message archive table (for rollback support)
CREATE TABLE IF NOT EXISTS message_archive (
    archive_id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES sessions(session_id) ON DELETE CASCADE,
    compaction_event_id VARCHAR(255) REFERENCES compaction_events(event_id) ON DELETE CASCADE,
    original_message_id VARCHAR(255) NOT NULL,
    message_data JSONB NOT NULL,
    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_message_archive_session_id ON message_archive(session_id);
CREATE INDEX IF NOT EXISTS idx_message_archive_compaction_event_id ON message_archive(compaction_event_id);


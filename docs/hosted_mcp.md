# Hosted MCP Server Setup & Authentication

SparkleForge supports running as a hosted Model Context Protocol (MCP) server so that external clients (such as ChatGPT, Claude, Codex, or remote MCP inspector tools) can securely connect to SparkleForge's toolsets and research capabilities over the network.

---

## 1. Server Entrypoint & Transport

SparkleForge provides a dedicated CLI command and runner script to start the MCP server using Standard I/O (stdio) or Server-Sent Events (SSE) / HTTP transports depending on your deployment architecture.

### Starting the Server

To launch the MCP server locally or within a container:



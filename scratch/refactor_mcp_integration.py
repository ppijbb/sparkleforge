import sys

def refactor_mcp_integration(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    start_line = -1
    end_line = -1
    
    for idx, line in enumerate(lines):
        if 'class ToolCategory' in line and (idx > 290):
            start_line = idx
        if 'return self.mcp_tool_mapping.get(tool_name)' in line and idx > start_line and start_line != -1:
            end_line = idx
            break
            
    if start_line == -1 or end_line == -1:
        print(f"Failed to find the block to replace. start={start_line}, end={end_line}")
        return

    new_content = [
        "\n",
        "# Centralized Tool Registry imports\n",
        "from src.core.tools.registry import (\n",
        "    ToolCategory, \n",
        "    ToolInfo, \n",
        "    ToolMetadata, \n",
        "    ToolResult, \n",
        "    ToolRegistry,\n",
        "    registry as global_registry\n",
        ")\n",
        "\n",
        "# Alias for backward compatibility\n",
        "SEP986ToolResult = ToolResult\n",
        "\n"
    ]

    final_lines = lines[:start_line] + new_content + lines[end_line+1:]

    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(final_lines)
    print(f"Successfully refactored {filename}")

if __name__ == "__main__":
    refactor_mcp_integration(sys.argv[1])

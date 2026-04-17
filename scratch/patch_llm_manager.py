import sys

def patch_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    while i < len(lines):
        # Target: Groq model message construction
        if i + 3 < len(lines) and 'messages = []' in lines[i] and 'if system_message:' in lines[i+1] and 'messages.append({"role": "system", "content": system_message})' in lines[i+2] and 'messages.append({"role": "user", "content": prompt})' in lines[i+3]:
            indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            new_lines.append(f"{indent}history = kwargs.pop('history_messages', [])\n")
            new_lines.append(f"{indent}messages = []\n")
            new_lines.append(f"{indent}if system_message:\n")
            new_lines.append(f"{indent}    messages.append({{\"role\": \"system\", \"content\": system_message}})\n")
            new_lines.append(f"{indent}if history:\n")
            new_lines.append(f"{indent}    messages.extend(history)\n")
            new_lines.append(f"{indent}if not history or (history and history[-1].get('content') != prompt):\n")
            new_lines.append(f"{indent}    messages.append({{\"role\": \"user\", \"content\": prompt}})\n")
            i += 4
        else:
            new_lines.append(lines[i])
            i += 1

    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    patch_file(sys.argv[1])

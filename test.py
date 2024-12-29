import json
from datetime import datetime

with open("agent.py", "r", encoding="utf-8") as f:
    content = f.read()
    # Escape special characters while preserving newlines
    content = content.replace("\\", "\\\\")
    content = content.replace('"', "\\")
    content = content.replace("\n", "\\n")


print(content)

memory_entry = {
    "type": "file_learning",
    "source": "agent.py",
    "content": content,
    "timestamp": datetime.now().isoformat(),
}

with open("agent_history/goals.json", "w", encoding="utf-8") as f:
    json.dump(memory_entry, f, indent=2)

import re

# Read the file
with open('test_complete_dataset.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace emojis in logging messages
replacements = [
    ('🎮', ''),
    ('🚀', ''),
    ('📦', ''),
    ('✅', '[OK]'),
    ('❌', '[ERROR]'),
    ('⚠️', '[WARNING]'),
    ('📂', ''),
    ('📊', ''),
    ('🧪', ''),
    ('🔧', ''),
    ('🔮', ''),
]

for emoji, replacement in replacements:
    content = content.replace(emoji, replacement)

# Write back
with open('test_complete_dataset.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed emoji logging issues")

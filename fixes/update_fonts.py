import re

# Read the main_app.py file
with open('main_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all SF Pro Display with DM Sans
content = re.sub(r"font-family: 'SF Pro Display'", "font-family: 'DM Sans'", content)

# Replace all SF Pro Text with Plus Jakarta Sans  
content = re.sub(r"font-family: 'SF Pro Text'", "font-family: 'Plus Jakarta Sans'", content)

# Write back to file
with open('main_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Updated all font references to Filson Soft-inspired fonts:")
print("   • SF Pro Display → DM Sans (for headings)")
print("   • SF Pro Text → Plus Jakarta Sans (for body text)")
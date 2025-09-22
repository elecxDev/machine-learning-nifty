import re

# Read the main_app.py file
with open('main_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the font import URL with Inter only
content = re.sub(
    r"@import url\('https://fonts\.googleapis\.com/css2\?family=[^']+'\);",
    "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;600;700;800;900&display=swap');",
    content
)

# Replace all font-family declarations with Inter
font_replacements = [
    (r"font-family: '[^']*', '[^']*', -apple-system, BlinkMacSystemFont, sans-serif", "font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif"),
    (r"font-family: '[^']*', '[^']*', -apple-system, BlinkMacSystemFont, sans-serif !important", "font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important"),
    (r"font-family: '[^']*', '[^']*', sans-serif !important", "font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important"),
    (r"font-family: '[^']*', '[^']*', sans-serif", "font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif"),
    (r"font-family: '[^']*', sans-serif !important", "font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important"),
    (r"font-family: '[^']*', sans-serif", "font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif"),
    (r"font-family: 'DM Sans'", "font-family: 'Inter'"),
    (r"font-family: 'Plus Jakarta Sans'", "font-family: 'Inter'"),
    (r"font-family: 'Fizon Soft'", "font-family: 'Inter'"),
    # Handle the complex CSS variables section
    (r"'Inter', 'Fizon Soft', -apple-system", "'Inter', -apple-system"),
    (r"'Inter', 'Fizon Soft', sans-serif", "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"),
]

for pattern, replacement in font_replacements:
    content = re.sub(pattern, replacement, content)

# Write back to file
with open('main_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Updated all fonts to Inter consistently:")
print("   • Unified font import to Inter only")
print("   • Replaced all font-family declarations with Inter")
print("   • Added proper fallbacks for better compatibility")
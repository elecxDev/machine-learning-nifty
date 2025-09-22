with open('main_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all remaining Plus Jakarta Sans
content = content.replace("'Plus Jakarta Sans', sans-serif", "'Inter', -apple-system, BlinkMacSystemFont, sans-serif")
content = content.replace("'Plus Jakarta Sans', sans-serif !", "'Inter', -apple-system, BlinkMacSystemFont, sans-serif !")

# Replace any DM Sans references too
content = content.replace("'DM Sans', sans-serif", "'Inter', -apple-system, BlinkMacSystemFont, sans-serif")

with open('main_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ All fonts updated to Inter consistently!")
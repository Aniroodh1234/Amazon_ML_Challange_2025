
import re

def update_config(settings):
    """Update config.py with new settings"""
    config_file = '/content/Amazon_ML_Challange_2025/src/config.py'
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    for key, value in settings.items():
        # Handle different value types
        if isinstance(value, str):
            pattern = f'{key} = ["\'].*?["\']'
            replacement = f'{key} = "{value}"'
        elif isinstance(value, bool):
            pattern = f'{key} = (True|False)'
            replacement = f'{key} = {value}'
        else:
            pattern = f'{key} = [\\d.e-]+'
            replacement = f'{key} = {value}'
        
        content = re.sub(pattern, replacement, content)
    
    with open(config_file, 'w') as f:
        f.write(content)
    
    print(f"Config updated: {settings}")

# Save this function for use
if __name__ == "__main__":
    print("Config updater ready!")
import json

file_path = 'scalable_lab(1).ipynb'

with open(file_path, 'r') as f:
    notebook = json.load(f)

cells = notebook['cells']

# Remove standardize_sharegpt call
for cell in cells:
    source = "".join(cell.get('source', []))
    
    if 'dataset = standardize_sharegpt(dataset)' in source:
        new_source = []
        for line in cell['source']:
            if 'dataset = standardize_sharegpt(dataset)' in line:
                # Comment it out or remove it
                new_source.append("# dataset = standardize_sharegpt(dataset)  # Removed as requested\n")
            elif 'from unsloth.chat_templates import standardize_sharegpt' in line:
                new_source.append("# from unsloth.chat_templates import standardize_sharegpt\n")
            else:
                new_source.append(line)
        
        cell['source'] = new_source

with open(file_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print("✅ Removed standardize_sharegpt!")
print("\nThe dataset will now be processed in its original format.")

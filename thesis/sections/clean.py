import re

def clean_superscripts(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # This replaces \textsuperscript{10} with [10]
    # If you prefer just the number without brackets, change to r' \1'
    content = re.sub(r'\\textsuperscript\{(\d+)\}', r'[\1]', content)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

# Run the script
clean_superscripts('./thesis/sections/theory.tex', './thesis/sections/theory.tex')
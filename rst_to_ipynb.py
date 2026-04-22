import json
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else 'convolution.rst'
DST = sys.argv[2] if len(sys.argv) > 2 else 'convolution.ipynb'

with open(SRC, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

cells = []
prose = []
code = []
in_code = False
code_indent = None

def flush_prose():
    if prose:
        md = '\n'.join(prose).strip('\n')
        if md.strip():
            cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': md})
        prose.clear()

def flush_code():
    if code:
        src = '\n'.join(code).strip('\n')
        if src.strip():
            cells.append({'cell_type': 'code', 'execution_count': None,
                          'metadata': {}, 'outputs': [], 'source': src})
        code.clear()

def inline_rst(s):
    s = re.sub(r':math:`([^`]+)`', r'$\1$', s)
    s = re.sub(r'``([^`]+)``', r'`\1`', s)
    s = re.sub(r':ref:`([^`]+)`', r'*\1*', s)
    return s

i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()

    # Enter a code block
    if stripped == '.. jupyter-execute::' or stripped.startswith('.. code-block::'):
        flush_prose()
        in_code = True
        code_indent = None
        i += 1
        while i < len(lines) and lines[i].strip() == '':
            i += 1
        continue

    # Inside a code block, read indented lines
    if in_code:
        if line.strip() == '':
            code.append('')
            i += 1
            continue
        indent = len(line) - len(line.lstrip(' '))
        if code_indent is None:
            code_indent = indent
        if indent >= code_indent and line.startswith(' ' * code_indent):
            code.append(line[code_indent:])
            i += 1
            continue
        flush_code()
        in_code = False
        # fall through to handle this line as prose

    # .. math:: block -> $$ ... $$ markdown cell
    if stripped == '.. math::':
        flush_prose()
        i += 1
        while i < len(lines) and lines[i].strip() == '':
            i += 1
        mlines = []
        mindent = None
        while i < len(lines):
            ml = lines[i]
            if ml.strip() == '':
                mlines.append('')
                i += 1
                continue
            ind = len(ml) - len(ml.lstrip(' '))
            if mindent is None:
                mindent = ind
            if ind >= mindent and ml.startswith(' ' * mindent):
                mlines.append(ml[mindent:])
                i += 1
            else:
                break
        math = '\n'.join(mlines).strip('\n')
        cells.append({'cell_type': 'markdown', 'metadata': {},
                      'source': f'$$\n{math}\n$$'})
        continue

    # Skip anchors and link targets:  .. _foo:  or  .. _foo: https://...
    if re.match(r'^\.\. _[^:]+:\s*(https?://.*)?$', stripped):
        i += 1
        continue

    # Heading: underline on next line
    if (i + 1 < len(lines)
            and stripped
            and lines[i + 1].strip()
            and re.fullmatch(r'[=\-~^]+', lines[i + 1].strip())
            and len(lines[i + 1].strip()) >= len(stripped) - 2):
        ch = lines[i + 1].strip()[0]
        level = {'=': 1, '-': 2, '~': 3, '^': 4}.get(ch, 2)
        prose.append('#' * level + ' ' + inline_rst(stripped))
        i += 2
        continue

    prose.append(inline_rst(line))
    i += 1

flush_prose()
flush_code()

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

with open(DST, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

code_count = sum(1 for c in cells if c['cell_type'] == 'code')
md_count = sum(1 for c in cells if c['cell_type'] == 'markdown')
print(f"Wrote {DST}: {code_count} code cells, {md_count} markdown cells")

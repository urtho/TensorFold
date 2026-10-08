#!/usr/bin/env python3
"""The token ranking behind the drafter's cached Markov bias rows (``deepseek_v41/cuda/markov_tokens.py``): token
frequencies of this tree's Python source and as many characters of English (notes/dsv41/long_doc*.txt, docs, README,
RUNBOOK), most frequent first. Which tokens are cached changes the drafter's time, never a draft.

  uv run --with tokenizers python tools/dsv41_markov_tokens.py > src/tensorfold/families/deepseek_v41/cuda/markov_tokens.py
"""

import collections
import glob
import sys

from tokenizers import Tokenizer

tok = Tokenizer.from_file("notes/dsv41/tokenizer.json")
code = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob("src/tensorfold/**/*.py", recursive=True)))
english = "".join(open(f, errors="ignore").read() for f in ["notes/dsv41/long_doc.txt", "notes/dsv41/long_doc2.txt",
                                                             *sorted(glob.glob("docs/**/*.md", recursive=True)),
                                                             "README.md", "RUNBOOK.md"])
n = min(len(code), len(english))
count = collections.Counter()
for text in (code[:n], english[:n]):
    for i in range(0, len(text), 200000):
        count.update(tok.encode(text[i:i + 200000], add_special_tokens=False).ids)
top = [t for t, _ in count.most_common(1024)]
total = sum(count.values())
share = sum(count[t] for t in top[:256]) / total, sum(count[t] for t in top) / total
print(f"{n} characters of each, {total} tokens; top 256 {share[0]:.1%}, top 1024 {share[1]:.1%}", file=sys.stderr)
print(f'"""Most frequent tokens first (tools/dsv41_markov_tokens.py: code + English; the first 256 are {share[0]:.0%} of '
      f'the sample\'s tokens, all 1024 {share[1]:.0%})."""\n\nTOKENS = {top!r}')

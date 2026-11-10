#!/usr/bin/env python3
"""Strip real credentials from the dialogue corpus before it leaves this machine.

The corpus is real conversation logs, so it carries a small number of genuine
secrets: password hashes, an IMEI, a phone number. Renting a GPU means copying
it onto someone else's disk, and "the platform probably wipes it" is not a
reason to hand over a bcrypt hash.

Two properties matter beyond removing the secrets:

  * length is preserved, so token boundaries move as little as possible;
  * the mapping is deterministic, so a value appearing twice is replaced by the
    same fake twice. diagnose.py groups target tokens by how often they occur in
    the context, and a random-per-occurrence substitution would quietly destroy
    exactly that statistic.

Replacements are derived by hashing, keyed with a constant, and are not
recoverable to the original without the original.
"""
from __future__ import annotations

import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "data"
DST = HERE / "data_clean"
KEY = "memzip-scrub-v1"

# Two things went wrong in the first version and both are worth keeping in view:
#
#   * an upper bound on the digit run. A 19-digit secret sat inside a longer
#     digit string; the longer string matched nothing, so the secret survived as
#     a substring of it. Ranges must be open at the top.
#   * \b as the boundary. \b does not fire between a digit and an underscore,
#     because _ is a word character, so `x_123456789012345` slipped through.
#     A negative lookaround on the same character class is the right guard.
PATTERNS = [
    # bcrypt / sha-crypt / md5-crypt, e.g. $2a$.. $6$..
    ("cryptfmt", re.compile(r"\$(?:2[aby]|5|6|1)\$[./A-Za-z0-9$]{12,}")),
    # bare long hex: md5/sha digests. Also catches git object ids, which are
    # harmless to replace.
    ("hexdigest", re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])")),
    # IMEI / IMSI / card-like digit runs, no upper bound. The floor is 11: below
    # that, digit runs are ordinary content (ports, counts, line numbers) and
    # replacing them would damage the corpus for no gain. Four values of length
    # 11-14 here were neither millisecond timestamps nor phone-shaped, so they
    # are replaced rather than argued about.
    ("longdigits", re.compile(r"(?<!\d)\d{11,}(?!\d)")),
    # CN mobile numbers
    ("mobile", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
]


def _stream(seed: str) -> str:
    """Deterministic hex stream, long enough for anything we replace."""
    out, h = "", seed
    while len(out) < 256:
        h = hashlib.sha256((KEY + h).encode()).hexdigest()
        out += h
    return out


def fake(value: str, kind: str) -> str:
    s = _stream(value)
    if kind == "hexdigest":
        return s[: len(value)]
    if kind in ("longdigits", "mobile"):
        digits = "".join(str(int(c, 16) % 10) for c in s)
        # keep a leading 1 on mobiles so the shape still reads as a phone number
        return ("1" + digits)[: len(value)] if kind == "mobile" else digits[: len(value)]
    if kind == "cryptfmt":
        # keep the $id$ prefix so the string still looks structurally like a hash
        m = re.match(r"(\$(?:2[aby]|5|6|1)\$)", value)
        pre = m.group(1) if m else "$6$"
        body = s[: max(0, len(value) - len(pre))]
        return pre + body
    return s[: len(value)]


def scrub(text: str, tally: Counter, mapping: dict) -> str:
    for kind, rx in PATTERNS:
        def sub(m):
            v = m.group(0)
            if v not in mapping:
                mapping[v] = fake(v, kind)
                tally[kind] += 1
            return mapping[v]
        text = rx.sub(sub, text)
    return text


def main() -> int:
    if not SRC.is_dir():
        print(f"no corpus at {SRC}", file=sys.stderr)
        return 1
    DST.mkdir(exist_ok=True)
    tally, mapping = Counter(), {}
    n_files = 0
    for f in sorted(SRC.glob("*.txt")):
        t = f.read_text(errors="replace")
        out = scrub(t, tally, mapping)
        # length drift would shift every downstream token index
        if len(out) != len(t):
            print(f"  WARNING {f.name}: length changed {len(t)} -> {len(out)}")
        (DST / f.name).write_text(out)
        n_files += 1

    print(f"{n_files} files -> {DST}")
    print(f"distinct secrets replaced: {len(mapping)}")
    for k, v in sorted(tally.items()):
        print(f"  {k:<12} {v}")

    # Verify by rescanning the SOURCE with the substitution patterns and
    # checking that not one matched value survives anywhere in the output. This
    # catches the failure the first version had -- a secret nested inside a
    # longer run, so that the scrubber never matched it on its own.
    #
    # Verifying with *wider* patterns than were substituted is the wrong test:
    # it reports everything outside the chosen boundary as a leak forever. Those
    # are listed separately, as information for a human, not as failures.
    cleaned = {p.name: p.read_text(errors="replace") for p in DST.glob("*.txt")}
    intended = set()
    for f in SRC.glob("*.txt"):
        t = f.read_text(errors="replace")
        for _, rx in PATTERNS:
            intended.update(rx.findall(t))
    intended = {m if isinstance(m, str) else m[0] for m in intended}
    leaks = [(c, n) for c in intended for n, t in cleaned.items() if c in t]

    print(f"\nverification: {len(intended)} values matched in the source")
    for c, n in leaks:
        print(f"  LEAK len={len(c)} in {n}")
    print(f"  surviving in the cleaned copy: {len(leaks)}")

    # Just below the boundary: reported so the choice of floor stays visible.
    NEAR = re.compile(r"(?<!\d)\d{8,10}(?!\d)")
    near = set()
    for f in SRC.glob("*.txt"):
        near.update(NEAR.findall(f.read_text(errors="replace")))
    print(f"  (not touched, below the floor: {len(near)} runs of 8-10 digits)")

    if leaks:
        print("\n  -> DO NOT UPLOAD")
    else:
        print(f"\n  clean. upload {DST.name}/ , not data/")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())

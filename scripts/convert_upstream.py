"""Convert upstream enamel.csv and enamel-references.json to the cache format."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from enamel_ext.data.schema import GeneratedLevel, MaterializedLevel, Problem, ProblemSet, Provenance
from enamel_ext.data.sources import problem_set_to_json

UPSTREAM_DIR = _ROOT / "enamel_ext" / "data" / "cache" / "upstream"
OUTPUT = _ROOT / "enamel_ext" / "data" / "cache" / "problems.json"


def make_generator_adapter(upstream_generator: str, problem_id: int) -> str:
    """Wrap upstream generator to provide make_input(seed, scale) interface."""
    needs_rand_parens = 'rand_parens' in upstream_generator
    needs_miller_rabin = 'miller_rabin' in upstream_generator or 'rand_probably_prime' in upstream_generator or 'rand_primality' in upstream_generator
    needs_encode_shift = 'encode_shift' in upstream_generator
    
    imports = 'import random\nimport string\nimport math\n'
    if needs_encode_shift:
        imports += '''
def encode_shift(s: str):
    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])

def decode_shift(s: str):
    return "".join([chr(((ord(ch) - 5 - ord("a")) % 26) + ord("a")) for ch in s])
'''
    if needs_rand_parens:
        imports += '''
def rand_parens(size: int, valid: bool = True, par: str = '()', sep = ''):
    if valid:
        size += size % 2
        seq = []
        cnt = 0
        for i in range(size):
            if cnt == 0 or (cnt < size - i and random.randint(0, 1)):
                if cnt == 0:
                    seq.append([])
                seq[-1].append(par[0])
                cnt += 1
            else:
                seq[-1].append(par[1])
                cnt -= 1
        seq = [''.join(subseq) for subseq in seq]
        if sep is not None:
            seq = sep.join(seq)
    else:
        assert sep == ''
        seq = rand_parens(size = (size + 1) // 2 - 1, valid = True, par = par) + par[1] + par[0] + rand_parens(size = (size + 1) // 2 - 1, valid = True, par = par)
    return seq
'''
    
    if needs_miller_rabin:
        imports += '''
def miller_rabin(n, k = 5):
    if n == 2 or n == 3:
        return True
    if n == 1 or n % 2 == 0:
        return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def rand_probably_prime(lb, ub = None):
    if ub is None:
        lb = max(lb, 3)
        lb, ub = lb // 2, lb
    else:
        lb = max(lb, 0)
        ub = max(lb * 2, ub, 3)
    p = random.randint(lb, ub)
    while not miller_rabin(p):
        p = random.randint(lb, ub)
    return p

def rand_primality(size: int, lid: int, cid: int):
    size = max(size, 4)
    if lid == 0 and cid == 0:
        n = 2 * 3 * 5 * 7 * 11 * 13 + 1
    elif lid == 0 and cid == 1:
        n = 561
    elif lid == 0 and cid == 2:
        n = 2
    elif lid == 0 and cid == 3:
        n = 125
    elif cid % 3 == 0:
        n = rand_probably_prime(size)
    elif cid % 3 == 1:
        n = rand_probably_prime(int(size ** 0.5 + 1)) ** 2
    else:
        m = int(size ** 0.5 + 1)
        n = rand_probably_prime(m) * rand_probably_prime(m)
    return n
'''
    
    return f'''{imports}
{upstream_generator}

def make_input(seed, scale):
    random.seed(seed)
    lid = scale // 1000
    cid = scale % 1000
    result = generate_input(scale, lid, cid)
    return result if isinstance(result, tuple) else (result,)
'''


def parse_levels(input_levels: str, input_generator: str, problem_id: int) -> tuple:
    """Parse input_levels string into level specs."""
    scales = [int(x) for x in input_levels.split()]
    levels = []
    for level_idx, scale in enumerate(scales):
        n_cases = 8 if level_idx == 0 else 4
        seeds = []
        for case_idx in range(n_cases):
            seed = level_idx * 1000 + case_idx
            seeds.append(seed)
        if level_idx > 0 and scale <= levels[-1].scale:
            scale = levels[-1].scale + 1
        levels.append(GeneratedLevel(level=level_idx, scale=scale, seeds=tuple(seeds)))
    return tuple(levels)


def convert():
    csv_path = UPSTREAM_DIR / "dataset" / "enamel.csv"
    refs_path = UPSTREAM_DIR / "samples" / "enamel-references.json"

    with open(refs_path, "r") as f:
        references = json.load(f)

    problems = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            task_id = row["task_id"]
            problem_id = int(task_id.split("/")[-1])
            entry_point = row["entry_point"]
            prompt = row["prompt"]
            reference_solution = references[idx][0]
            input_generator = row["input_generator"]
            input_levels = row["input_levels"]

            levels = parse_levels(input_levels, input_generator, problem_id)
            adapted_generator = make_generator_adapter(input_generator, problem_id)

            problems.append(
                Problem(
                    problem_id=problem_id,
                    entry_point=entry_point,
                    prompt=prompt,
                    reference_solution=reference_solution,
                    input_generator=adapted_generator,
                    levels=levels,
                )
            )

    pset = ProblemSet(
        provenance=Provenance(
            name="ENAMEL (q-rz/enamel)",
            url="https://github.com/q-rz/enamel",
            license="unknown",
            retrieved=datetime.now(timezone.utc).isoformat(),
        ),
        problems=tuple(problems),
    )

    OUTPUT.write_text(problem_set_to_json(pset))
    print(f"wrote {len(problems)} problems to {OUTPUT}")


if __name__ == "__main__":
    convert()

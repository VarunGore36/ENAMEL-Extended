"""The paper's published numbers, as data.

Tables 3, 6, 7, 9, 10, 11 and 12 of arXiv:2406.06647v4 plus the Appendix C.1
setting, so parity has one source rather than a constant per call site. Rationale
in docs/decisions/0007-parity-gate.md.
"""

from __future__ import annotations

from typing import NamedTuple

#: The edition every number here was read from.
PAPER = "arXiv:2406.06647v4 (ICLR 2025)"


class Scores(NamedTuple):
    """One row of a results table. ``None`` is the paper's em-dash."""

    eff1: float
    pass1: float
    eff10: float | None = None
    pass10: float | None = None
    eff100: float | None = None
    pass100: float | None = None


#: Column order used throughout, matching Table 3's sampling block.
COLUMNS = ("eff1", "pass1", "eff10", "pass10", "eff100", "pass100")

# Appendix C.1, verbatim: alpha = 2, R = 6, h1 = h2 = 3, h3 = 4,
# M0 = 8, M1 = M2 = M3 = 4.
ALPHA = 2.0
REPEATS = 6
LEVEL_WEIGHTS = (3, 3, 4)
CASES_PER_LEVEL = (8, 4, 4, 4)
PROBLEMS = 142

#: The machine the published times were measured on (Appendix C.1). Recorded to
#: be compared against, not to be matched.
ENVIRONMENT = (
    "virtualized Google Cloud server, Ubuntu 20.04.6 LTS, "
    "Intel Xeon CPU @ 2.20GHz, Python 3.10.12"
)

#: Appendix C.1 gives 200 samples for "relatively smaller models" and 100 for
#: "larger models" without saying which are which, so ``n`` is not recoverable for
#: the sampling rows. Its third clause is not a bucket: the largest commercial
#: models are greedy-only, which is ``GREEDY_ONLY``. Counting the released samples
#: is the open question this leaves; see decision 0007.
SAMPLE_SIZE_BUCKETS = {
    "relatively smaller models": 200,
    "larger models": 100,
}

#: The one model the text pins: Appendix C.7 computes Table 11 "from the 100
#: generated samples" of this one.
KNOWN_SAMPLE_SIZE = {"Llama 3 70B Instruct": 100}

#: Appendix C.1 names the checkpoint behind five display names and no others.
#: The rest are models whose samples the paper re-uses from Liu et al. (2023a),
#: so their naming follows that release rather than anything stated here.
MODEL_IDENTIFIERS = {
    "Claude 3 Opus": "claude-3-opus-20240229",
    "Claude 3 Sonnet": "claude-3-sonnet-20240229",
    "Claude 3 Haiku": "claude-3-haiku-20240307",
    "GPT-4 Turbo": "gpt-4-1106-preview",
    "GPT-4": "gpt-4-0613",
}

#: Appendix C.1: "For models that are included in Liu et al. (2023a), we re-use
#: their generated code samples." Which models those are is not listed, so which
#: rows carry EvalPlus's sampling settings rather than the paper's own is not
#: recoverable from the text.
REUSED_SAMPLES_SOURCE = "Liu et al. (2023a), HumanEval+ / EvalPlus"

#: Table 3, greedy decoding, in the paper's row order. The parity target.
TABLE3_GREEDY: dict[str, Scores] = {
    "GPT-4 Turbo": Scores(0.470, 0.796),
    "GPT-4": Scores(0.454, 0.831),
    "Llama 3 70B Instruct": Scores(0.421, 0.746),
    "Llama 3 8B Instruct": Scores(0.344, 0.592),
    "Mixtral 8x22B Instruct": Scores(0.408, 0.746),
    "Mixtral 8x7B Instruct": Scores(0.266, 0.444),
    "Claude 3 Opus": Scores(0.401, 0.789),
    "Claude 3 Sonnet": Scores(0.345, 0.662),
    "Claude 3 Haiku": Scores(0.386, 0.739),
    "Phind Code Llama V2": Scores(0.394, 0.683),
    "ChatGPT": Scores(0.364, 0.683),
    "Code Llama 70B Python": Scores(0.264, 0.500),
    "Code Llama 34B Python": Scores(0.268, 0.458),
    "Code Llama 13B Python": Scores(0.216, 0.408),
    "Code Llama 7B Python": Scores(0.247, 0.373),
    "StarCoder": Scores(0.195, 0.352),
    "CodeGen 16B": Scores(0.169, 0.310),
    "CodeGen 6B": Scores(0.193, 0.296),
    "CodeGen 2B": Scores(0.153, 0.254),
    "CodeT5+ 16B": Scores(0.160, 0.317),
    "Mistral 7B": Scores(0.152, 0.275),
    "Vicuna 13B": Scores(0.123, 0.176),
    "Vicuna 7B": Scores(0.061, 0.099),
    "SantaCoder": Scores(0.100, 0.141),
    "Incoder 6B": Scores(0.091, 0.127),
    "Incoder 1B": Scores(0.066, 0.092),
    "GPT-J": Scores(0.083, 0.106),
    "GPT-Neo 2B": Scores(0.043, 0.056),
    "PolyCoder": Scores(0.037, 0.049),
    "StableLM 7B": Scores(0.020, 0.021),
}

#: Models with no sampling row: greedy decoding only, per Appendix C.1.
GREEDY_ONLY = ("GPT-4 Turbo", "GPT-4", "Claude 3 Opus")

#: Table 3, sampling. ``n`` differs across these models and is not published.
TABLE3_SAMPLING: dict[str, Scores] = {
    "Llama 3 70B Instruct": Scores(0.438, 0.747, 0.526, 0.836, 0.575, 0.880),
    "Llama 3 8B Instruct": Scores(0.345, 0.564, 0.500, 0.770, 0.595, 0.874),
    "Mixtral 8x22B Instruct": Scores(0.407, 0.721, 0.575, 0.870, 0.704, 0.923),
    "Mixtral 8x7B Instruct": Scores(0.279, 0.456, 0.436, 0.689, 0.542, 0.810),
    "Claude 3 Sonnet": Scores(0.365, 0.677, 0.498, 0.814, 0.594, 0.887),
    "Claude 3 Haiku": Scores(0.382, 0.730, 0.478, 0.831, 0.529, 0.861),
    "Phind Code Llama V2": Scores(0.372, 0.638, 0.584, 0.862, 0.723, 0.935),
    "ChatGPT": Scores(0.374, 0.673, 0.557, 0.847, 0.690, 0.937),
    "Code Llama 70B Python": Scores(0.082, 0.177, 0.326, 0.610, 0.614, 0.908),
    "Code Llama 34B Python": Scores(0.226, 0.405, 0.511, 0.786, 0.711, 0.934),
    "Code Llama 13B Python": Scores(0.204, 0.372, 0.487, 0.732, 0.714, 0.899),
    "Code Llama 7B Python": Scores(0.180, 0.320, 0.432, 0.663, 0.643, 0.837),
    "StarCoder": Scores(0.134, 0.236, 0.355, 0.557, 0.542, 0.787),
    "CodeGen 16B": Scores(0.122, 0.219, 0.326, 0.512, 0.536, 0.761),
    "CodeGen 6B": Scores(0.111, 0.188, 0.298, 0.455, 0.491, 0.694),
    "CodeGen 2B": Scores(0.098, 0.168, 0.264, 0.389, 0.421, 0.602),
    "CodeT5+ 16B": Scores(0.130, 0.250, 0.343, 0.551, 0.551, 0.785),
    "Mistral 7B": Scores(0.116, 0.222, 0.335, 0.541, 0.557, 0.791),
    "Vicuna 13B": Scores(0.080, 0.125, 0.188, 0.310, 0.319, 0.537),
    "Vicuna 7B": Scores(0.054, 0.081, 0.149, 0.231, 0.283, 0.423),
    "SantaCoder": Scores(0.088, 0.126, 0.204, 0.298, 0.349, 0.470),
    "Incoder 6B": Scores(0.054, 0.078, 0.164, 0.242, 0.319, 0.439),
    "Incoder 1B": Scores(0.031, 0.043, 0.100, 0.139, 0.191, 0.241),
    "GPT-J": Scores(0.039, 0.058, 0.119, 0.166, 0.221, 0.331),
    "GPT-Neo 2B": Scores(0.019, 0.027, 0.069, 0.096, 0.127, 0.181),
    "PolyCoder": Scores(0.021, 0.029, 0.067, 0.084, 0.121, 0.155),
    "StableLM 7B": Scores(0.007, 0.010, 0.039, 0.048, 0.097, 0.123),
}

#: Section 4.2: 20 hard problems needing an advanced algorithm, and 75 seemingly
#: easy ones needing implementation optimization. The paper does not say the two
#: are disjoint, though "seemingly easy" against "hard" implies it, and it names
#: no problem in either. Disjointness is what makes ``REMAINDER_PROBLEMS`` a
#: quantity rather than a guess; decision 0007 records the check.
ALGORITHM_PROBLEMS = 20
IMPLEMENTATION_PROBLEMS = 75
REMAINDER_PROBLEMS = PROBLEMS - ALGORITHM_PROBLEMS - IMPLEMENTATION_PROBLEMS

#: Table 6, algorithm-design subset, sampling. Table 4 is its first twelve rows.
TABLE6_ALGORITHM: dict[str, Scores] = {
    "Llama 3 70B Instruct": Scores(0.246, 0.660, 0.306, 0.749, 0.359, 0.750),
    "Llama 3 8B Instruct": Scores(0.201, 0.518, 0.303, 0.724, 0.367, 0.849),
    "Mixtral 8x22B Instruct": Scores(0.225, 0.635, 0.363, 0.837, 0.470, 0.900),
    "Mixtral 8x7B Instruct": Scores(0.124, 0.391, 0.244, 0.681, 0.344, 0.850),
    "Claude 3 Sonnet": Scores(0.184, 0.577, 0.328, 0.804, 0.450, 0.950),
    "Claude 3 Haiku": Scores(0.149, 0.692, 0.208, 0.752, 0.266, 0.775),
    "Phind Code Llama V2": Scores(0.185, 0.554, 0.353, 0.789, 0.401, 0.849),
    "ChatGPT": Scores(0.120, 0.488, 0.304, 0.799, 0.483, 0.950),
    "Code Llama 70B Python": Scores(0.018, 0.100, 0.129, 0.519, 0.402, 0.950),
    "Code Llama 34B Python": Scores(0.071, 0.293, 0.271, 0.713, 0.425, 0.881),
    "Code Llama 13B Python": Scores(0.058, 0.212, 0.276, 0.665, 0.478, 0.844),
    "Code Llama 7B Python": Scores(0.068, 0.202, 0.231, 0.589, 0.393, 0.761),
    "StarCoder": Scores(0.047, 0.161, 0.156, 0.485, 0.257, 0.709),
    "CodeGen 16B": Scores(0.031, 0.133, 0.146, 0.451, 0.292, 0.684),
    "CodeGen 6B": Scores(0.023, 0.091, 0.106, 0.372, 0.235, 0.612),
    "CodeGen 2B": Scores(0.036, 0.131, 0.121, 0.387, 0.193, 0.644),
    "CodeT5+ 16B": Scores(0.043, 0.192, 0.173, 0.509, 0.321, 0.673),
    "Mistral 7B": Scores(0.030, 0.152, 0.157, 0.516, 0.319, 0.737),
    "Vicuna 13B": Scores(0.008, 0.072, 0.033, 0.269, 0.076, 0.449),
    "Vicuna 7B": Scores(0.019, 0.071, 0.083, 0.241, 0.113, 0.300),
    "SantaCoder": Scores(0.037, 0.102, 0.101, 0.316, 0.203, 0.493),
    "Incoder 6B": Scores(0.010, 0.050, 0.062, 0.203, 0.112, 0.325),
    "Incoder 1B": Scores(0.003, 0.023, 0.021, 0.110, 0.071, 0.200),
    "GPT-J": Scores(0.021, 0.051, 0.063, 0.146, 0.081, 0.243),
    "GPT-Neo 2B": Scores(0.003, 0.019, 0.015, 0.098, 0.032, 0.172),
    "PolyCoder": Scores(0.002, 0.010, 0.018, 0.070, 0.050, 0.163),
    "StableLM 7B": Scores(0.001, 0.005, 0.010, 0.039, 0.033, 0.099),
}

#: Table 6, implementation-optimization subset, sampling.
TABLE6_IMPLEMENTATION: dict[str, Scores] = {
    "Llama 3 70B Instruct": Scores(0.404, 0.791, 0.497, 0.869, 0.551, 0.920),
    "Llama 3 8B Instruct": Scores(0.313, 0.582, 0.468, 0.806, 0.571, 0.906),
    "Mixtral 8x22B Instruct": Scores(0.376, 0.783, 0.556, 0.914, 0.686, 0.947),
    "Mixtral 8x7B Instruct": Scores(0.248, 0.473, 0.411, 0.699, 0.515, 0.827),
    "Claude 3 Sonnet": Scores(0.358, 0.723, 0.475, 0.846, 0.548, 0.893),
    "Claude 3 Haiku": Scores(0.360, 0.772, 0.465, 0.889, 0.513, 0.923),
    "Phind Code Llama V2": Scores(0.351, 0.712, 0.567, 0.901, 0.732, 0.968),
    "ChatGPT": Scores(0.337, 0.715, 0.508, 0.864, 0.633, 0.949),
    "Code Llama 70B Python": Scores(0.076, 0.181, 0.294, 0.627, 0.589, 0.920),
    "Code Llama 34B Python": Scores(0.197, 0.415, 0.473, 0.804, 0.687, 0.949),
    "Code Llama 13B Python": Scores(0.176, 0.405, 0.476, 0.784, 0.715, 0.928),
    "Code Llama 7B Python": Scores(0.165, 0.349, 0.417, 0.703, 0.620, 0.863),
    "StarCoder": Scores(0.112, 0.247, 0.332, 0.598, 0.514, 0.802),
    "CodeGen 16B": Scores(0.099, 0.220, 0.303, 0.541, 0.531, 0.801),
    "CodeGen 6B": Scores(0.090, 0.188, 0.285, 0.478, 0.483, 0.731),
    "CodeGen 2B": Scores(0.081, 0.160, 0.256, 0.400, 0.410, 0.610),
    "CodeT5+ 16B": Scores(0.106, 0.257, 0.313, 0.581, 0.536, 0.845),
    "Mistral 7B": Scores(0.100, 0.227, 0.327, 0.574, 0.565, 0.821),
    "Vicuna 13B": Scores(0.056, 0.096, 0.168, 0.288, 0.316, 0.569),
    "Vicuna 7B": Scores(0.031, 0.061, 0.121, 0.215, 0.260, 0.439),
    "SantaCoder": Scores(0.069, 0.114, 0.203, 0.308, 0.357, 0.488),
    "Incoder 6B": Scores(0.037, 0.062, 0.152, 0.252, 0.320, 0.477),
    "Incoder 1B": Scores(0.018, 0.030, 0.080, 0.129, 0.172, 0.232),
    "GPT-J": Scores(0.025, 0.043, 0.110, 0.167, 0.221, 0.354),
    "GPT-Neo 2B": Scores(0.007, 0.014, 0.050, 0.084, 0.113, 0.184),
    "PolyCoder": Scores(0.004, 0.007, 0.034, 0.051, 0.092, 0.122),
    "StableLM 7B": Scores(0.002, 0.003, 0.016, 0.025, 0.074, 0.099),
}

#: Table 7, Appendix C.3: the top 12 greedy models ranked by ``eff@1`` and by the
#: classic speedup metric. The ``eff@1`` column is an independent publication of
#: Table 3's greedy ordering and is used as a cross-check on that transcription.
TABLE7_EFF1_RANKING = (
    "GPT-4 Turbo",
    "GPT-4",
    "Llama 3 70B Instruct",
    "Mixtral 8x22B Instruct",
    "Claude 3 Opus",
    "Phind Code Llama V2",
    "Claude 3 Haiku",
    "ChatGPT",
    "Claude 3 Sonnet",
    "Llama 3 8B Instruct",
    "Code Llama 34B Python",
    "Mixtral 8x7B Instruct",
)

#: Table 7's other column. The paper calls the two rankings "very different" and
#: argues from the difference that speedup is unreasonable under censoring, which
#: makes this pair a published statement about how much disagreement matters.
TABLE7_SPEEDUP_RANKING = (
    "GPT-4 Turbo",
    "Mixtral 8x22B Instruct",
    "Llama 3 70B Instruct",
    "GPT-4",
    "Claude 3 Opus",
    "Phind Code Llama V2",
    "ChatGPT",
    "Claude 3 Haiku",
    "Claude 3 Sonnet",
    "Llama 3 8B Instruct",
    "Mixtral 8x7B Instruct",
    "Code Llama 34B Python",
)

#: Table 9, Appendix C.5: ENAMEL against two other efficiency benchmarks, on Code
#: Llama 34B Python because Mercury did not evaluate GPT-4. The metrics are not
#: comparable to each other; the ENAMEL entry repeats Table 3's greedy ``eff@1``
#: for that model, which is why it is here.
TABLE9_CROSS_BENCHMARK = {
    "EffiBench": ("1/NET", 0.336),
    "Mercury": ("Beyond", 0.424),
    "ENAMEL (ours)": ("eff@1", 0.268),
}
TABLE9_MODEL = "Code Llama 34B Python"

#: Table 10 (a): GPT-4 Turbo greedy ``eff@1`` as alpha varies, h at defaults.
TABLE10_ALPHA: dict[float, float] = {
    1.5: 0.421,
    2.0: 0.470,
    2.5: 0.502,
    3.0: 0.525,
    3.5: 0.541,
}

#: Table 10 (b), (c), (d): the same, sweeping one hardness with the others at
#: their defaults. Keyed by level, then by that level's hardness.
TABLE10_HARDNESS: dict[int, dict[int, float]] = {
    1: {1: 0.428, 2: 0.451, 3: 0.470, 4: 0.486, 5: 0.498},
    2: {1: 0.474, 2: 0.472, 3: 0.470, 4: 0.469, 5: 0.467},
    3: {1: 0.520, 2: 0.499, 3: 0.483, 4: 0.470, 5: 0.460},
}

#: Table 11: standard deviations of the two estimators on Llama 3 70B Instruct,
#: keyed by k. The vanilla row's protocol is stated (1000 random k-subsets of
#: the model's 100 samples); the Rao-Blackwellized row's is not, and cannot be
#: the same one, since that estimator is a deterministic function of the samples.
TABLE11_VANILLA_STD: dict[int, float] = {1: 0.20, 10: 0.25}
TABLE11_RAO_BLACKWELLIZED_STD: dict[int, float] = {1: 0.02, 10: 0.08}
TABLE11_SAMPLES = 100

#: Table 12, Appendix C.8: greedy ``eff@1`` and ``pass@1`` under the basic prompt
#: and under one asking for "the most efficient algorithm". The basic rows repeat
#: Table 3's greedy entries, which is a check on both transcriptions.
TABLE12_BASIC: dict[str, Scores] = {
    "Llama 3 70B Instruct": Scores(0.421, 0.746),
    "Mixtral 8x22B Instruct": Scores(0.408, 0.746),
}
TABLE12_ENCOURAGING: dict[str, Scores] = {
    "Llama 3 70B Instruct": Scores(0.418, 0.746),
    "Mixtral 8x22B Instruct": Scores(0.426, 0.732),
}


def remainder_scores(model: str) -> Scores:
    """Implied means over the problems in neither Table 6 subset.

    Subtraction of Table 6 from Table 3, valid only if the two subsets are
    disjoint. A ``pass@k`` outside [0, 1] would show that they are not, or that
    a number here is mistranscribed.
    """
    full = TABLE3_SAMPLING[model]
    algorithm = TABLE6_ALGORITHM[model]
    implementation = TABLE6_IMPLEMENTATION[model]
    out = []
    for index in range(len(COLUMNS)):
        total = PROBLEMS * full[index]
        total -= ALGORITHM_PROBLEMS * algorithm[index]
        total -= IMPLEMENTATION_PROBLEMS * implementation[index]
        out.append(total / REMAINDER_PROBLEMS)
    return Scores(*out)


def rb_std_bound(k: int, n: int, vanilla_std: float) -> float:
    """Eq. (8) as a standard deviation: ``vanilla * sqrt(k / n)``.

    Eq. (8) bounds the Rao-Blackwellized estimator's variance by ``k/n`` times
    the variance of the max over ``k`` samples, which is what the vanilla
    estimator's variance is.
    """
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    return vanilla_std * (k / n) ** 0.5


def benchmark_std(per_problem_std: float, problems: int = PROBLEMS) -> float:
    """A per-problem standard deviation carried to the mean over problems.

    ``eff@k`` averages independent per-problem estimates, so the mean's noise is
    smaller by ``sqrt(problems)``. Independence across problems is the paper's
    own framing in Theorem 1.
    """
    if problems < 1:
        raise ValueError(f"problems must be >= 1, got {problems}")
    return per_problem_std / problems**0.5


#: Largest score Eq. (1) and (2) admit: ``alpha / (alpha - q)`` at ``q = 1``,
#: which is the level that sets ``T_i``. Infinitely fast code at every level
#: reaches it; nothing exceeds it.
MAX_SAMPLE_SCORE = ALPHA / (ALPHA - 1.0)


def table(name: str = "greedy") -> dict[str, Scores]:
    """One published results table.

    ``greedy`` and ``sampling`` are Table 3's two blocks over all 142 problems;
    ``algorithm`` and ``implementation`` are Table 6's subsets, which the paper
    reports under sampling only. Flat names because the other combinations do
    not exist.
    """
    tables = {
        "greedy": TABLE3_GREEDY,
        "sampling": TABLE3_SAMPLING,
        "algorithm": TABLE6_ALGORITHM,
        "implementation": TABLE6_IMPLEMENTATION,
    }
    if name not in tables:
        raise ValueError(f"unknown table {name!r}, expected one of {sorted(tables)}")
    return tables[name]


def leaderboard(column: str = "eff1", name: str = "greedy") -> tuple[tuple[str, float], ...]:
    """Models and their published values, best first, ties in table order."""
    if column not in COLUMNS:
        raise ValueError(f"unknown column {column!r}, expected one of {COLUMNS}")
    index = COLUMNS.index(column)
    rows = [
        (model, scores[index])
        for model, scores in table(name).items()
        if scores[index] is not None
    ]
    return tuple(sorted(rows, key=lambda row: -row[1]))

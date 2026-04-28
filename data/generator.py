"""
Synthetic semiconductor wafer process parameter generator.

Generates realistic process sequences with physics-inspired causal rules
linking manufacturing parameters to defect outcomes.
"""

import json
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

DEFECT_TYPES = {
    0: "none",
    1: "particle_contamination",
    2: "scratch",
    3: "pit",
    4: "oxide_defect",
    5: "metal_contamination",
}

PROCESS_STEPS = [
    "oxidation",
    "lithography",
    "etching",
    "deposition",
    "doping",
    "cmp",
    "cleaning",
    "annealing",
]

# Parameter ranges for each process step (realistic fab values)
PARAMETER_RANGES = {
    "oxidation":    {"temperature": (800, 1200), "pressure": (0.1, 10.0), "duration": (10, 120)},
    "lithography":  {"exposure_dose": (10, 50),  "focus_offset": (-0.2, 0.2), "wavelength": (193, 248)},
    "etching":      {"etch_rate": (50, 500),     "selectivity": (5, 50),      "duration": (30, 300)},
    "deposition":   {"temperature": (300, 700),  "rate": (0.1, 5.0),          "thickness": (10, 500)},
    "doping":       {"concentration": (1e14, 1e18), "energy": (10, 200),      "dose": (1e12, 1e16)},
    "cmp":          {"pressure": (1, 10),        "velocity": (10, 100),       "slurry_conc": (0.01, 0.5)},
    "cleaning":     {"chemical_conc": (0.1, 5.0),"temperature": (20, 80),     "duration": (5, 30)},
    "annealing":    {"temperature": (400, 1100), "duration": (10, 180),       "atmosphere": (0, 1)},
}


@dataclass
class ProcessStep:
    step_name: str
    params: dict


@dataclass
class DefectLabel:
    defect_type: int       # 0-5
    defect_type_name: str
    location_x: float      # normalized [0, 1]
    location_y: float      # normalized [0, 1]
    severity: float        # [0, 1]
    has_defect: bool


@dataclass
class WaferSample:
    wafer_id: str
    process_steps: list
    defect: dict
    node_features: list    # flattened per-step feature vectors
    adjacency: list        # edge list (step i → step i+1)


def _sample_step_params(step_name: str, rng: np.random.Generator) -> dict:
    ranges = PARAMETER_RANGES[step_name]
    params = {}
    for key, (lo, hi) in ranges.items():
        if key == "atmosphere":
            params[key] = float(rng.integers(0, 2))  # 0=N2, 1=H2
        elif key in ("concentration", "dose"):
            # log-uniform sampling for wide-range params
            params[key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            params[key] = float(rng.uniform(lo, hi))
    return params


def _compute_defect(steps: list[ProcessStep], rng: np.random.Generator) -> DefectLabel:
    """
    Physics-inspired causal rules:
    - High oxidation temperature + long duration → oxide defect
    - High CMP pressure + low slurry → scratch
    - Poor cleaning chemical concentration → particle contamination
    - Extreme doping concentration → pit
    - Deposition rate too high → metal contamination
    - All parameters nominal → no defect
    """
    defect_scores = np.zeros(6)  # one score per defect type

    step_map = {s.step_name: s.params for s in steps}

    ox = step_map.get("oxidation", {})
    temp_ox = ox.get("temperature", 1000)
    dur_ox = ox.get("duration", 60)
    if temp_ox > 1100 and dur_ox > 90:
        defect_scores[4] += 0.6 + rng.uniform(0, 0.3)  # oxide_defect

    cmp = step_map.get("cmp", {})
    cmp_p = cmp.get("pressure", 5)
    slurry = cmp.get("slurry_conc", 0.25)
    if cmp_p > 8 and slurry < 0.1:
        defect_scores[2] += 0.5 + rng.uniform(0, 0.4)  # scratch

    clean = step_map.get("cleaning", {})
    chem = clean.get("chemical_conc", 2.5)
    if chem < 0.5:
        defect_scores[1] += 0.4 + rng.uniform(0, 0.4)  # particle_contamination

    dop = step_map.get("doping", {})
    conc = dop.get("concentration", 1e16)
    if conc > 5e17:
        defect_scores[3] += 0.45 + rng.uniform(0, 0.35)  # pit

    dep = step_map.get("deposition", {})
    rate = dep.get("rate", 2.5)
    if rate > 4.0:
        defect_scores[5] += 0.4 + rng.uniform(0, 0.4)  # metal_contamination

    # Add noise so not every rule triggers a defect deterministically
    defect_scores += rng.uniform(0, 0.15, size=6)

    best_type = int(np.argmax(defect_scores))
    best_score = defect_scores[best_type]

    # Threshold: only call it a defect if score is high enough
    if best_score < 0.55 or best_type == 0:
        return DefectLabel(
            defect_type=0, defect_type_name="none",
            location_x=0.0, location_y=0.0, severity=0.0, has_defect=False
        )

    return DefectLabel(
        defect_type=best_type,
        defect_type_name=DEFECT_TYPES[best_type],
        location_x=float(rng.uniform(0.1, 0.9)),
        location_y=float(rng.uniform(0.1, 0.9)),
        severity=float(np.clip(best_score - 0.55, 0, 1)),
        has_defect=True,
    )


def _step_to_feature_vector(step: ProcessStep) -> list[float]:
    """Normalize each parameter to [0, 1] and return a fixed-length vector."""
    ranges = PARAMETER_RANGES[step.step_name]
    vec = []
    for key, (lo, hi) in ranges.items():
        raw = step.params.get(key, (lo + hi) / 2)
        if key in ("concentration", "dose"):
            normalized = (np.log(raw) - np.log(lo)) / (np.log(hi) - np.log(lo))
        else:
            normalized = (raw - lo) / (hi - lo)
        vec.append(float(np.clip(normalized, 0.0, 1.0)))
    return vec


def _build_adjacency(n_steps: int) -> list[list[int]]:
    """Linear chain: step i → step i+1 (directed)."""
    edges = []
    for i in range(n_steps - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])  # undirected for message passing
    return edges


class WaferDataGenerator:
    def __init__(self, n_wafers: int = 2000, seed: int = 42):
        self.n_wafers = n_wafers
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self) -> list[dict]:
        samples = []
        for i in range(self.n_wafers):
            steps = [
                ProcessStep(
                    step_name=name,
                    params=_sample_step_params(name, self.rng)
                )
                for name in PROCESS_STEPS
            ]
            defect = _compute_defect(steps, self.rng)
            node_features = [_step_to_feature_vector(s) for s in steps]
            adjacency = _build_adjacency(len(steps))

            sample = WaferSample(
                wafer_id=f"W_{i:06d}",
                process_steps=[{"step": s.step_name, "params": s.params} for s in steps],
                defect=asdict(defect),
                node_features=node_features,
                adjacency=adjacency,
            )
            samples.append(asdict(sample))

        return samples

    def save(self, output_path: str = "data/raw/synthetic_wafers.json") -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        samples = self.generate()
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"Generated {len(samples)} wafer samples → {output_path}")
        self._print_stats(samples)
        return output_path

    def _print_stats(self, samples: list[dict]) -> None:
        from collections import Counter
        types = [s["defect"]["defect_type_name"] for s in samples]
        counts = Counter(types)
        print("\nDefect distribution:")
        for name, count in sorted(counts.items()):
            pct = 100 * count / len(samples)
            print(f"  {name:<25} {count:>5}  ({pct:.1f}%)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n",    type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out",  default="data/raw/synthetic_wafers.json")
    args = p.parse_args()
    gen = WaferDataGenerator(n_wafers=args.n, seed=args.seed)
    gen.save(args.out)

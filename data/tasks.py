"""Natural-language instructions for the pick-and-place task.

Training demos use TRAIN_TEMPLATES. EVAL_TEMPLATES are held out (never recorded) so Phase 7 can test
whether the policy generalizes to unseen wording rather than memorizing exact sentences.
"""

import numpy as np

COLOR_OF = {"red_cube": "red", "green_cube": "green", "blue_cube": "blue"}

TRAIN_TEMPLATES = [
    "Put the {color} cube in the container",
    "Place the {color} cube into the container",
    "Pick up the {color} cube and put it in the container",
    "Move the {color} cube into the box",
    "Put the {color} block in the box",
    "Drop the {color} cube into the container",
]

EVAL_TEMPLATES = [
    "Grab the {color} cube and place it inside the bin",
    "Could you put the {color} block into the container?",
    "Transfer the {color} cube to the box",
]


def target_from_instruction(text: str) -> str | None:
    """Which cube an instruction asks for, from its colour word (e.g. "...the red block..." -> "red_cube").

    Used only to score the episode; the policy itself only ever sees the text. Returns None if the
    instruction names no colour, or more than one.
    """
    words = set(text.lower().replace("?", " ").replace(",", " ").replace(".", " ").split())
    found = [cube for cube, color in COLOR_OF.items() if color in words]
    return found[0] if len(found) == 1 else None


def make_instruction(target: str, rng: np.random.Generator | None = None, split: str = "train") -> str:
    """Instruction for `target` (e.g. "red_cube"). With no rng, the first (canonical) template is used."""
    templates = TRAIN_TEMPLATES if split == "train" else EVAL_TEMPLATES
    template = templates[0] if rng is None else templates[rng.integers(len(templates))]
    return template.format(color=COLOR_OF[target])

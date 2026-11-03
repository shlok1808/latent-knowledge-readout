"""Shared data types. All builders import from here; do not redefine."""
from dataclasses import dataclass, field

LETTERS = "ABCDEFGH"


@dataclass(frozen=True)
class Item:
    item_id: str            # "<dataset>:<split>:<native id>", unique and stable
    dataset: str            # "csqa" | "arc_challenge" | "obqa"
    split: str
    question: str
    options: tuple          # tuple of (letter, text), letters "A".."E" in order, contiguous from A
    answer: str             # gold letter

    @property
    def letters(self):
        return tuple(letter for letter, _ in self.options)


@dataclass(frozen=True)
class Cue:
    """One cue cell. kind: 'neutral' | 'answer_again' | 'suggestion' | 'doubt'."""
    kind: str
    source: str             # template key, e.g. "user"; "none" for neutral/answer_again
    turn: str               # "single" | "multi"
    target: str | None      # letter the cue names (suggested option, or doubted option)
    prior: str | None       # "correct" | "wrong" (from that checkpoint's neutral pass); None for neutral


@dataclass(frozen=True)
class Prompt:
    item_id: str
    cue: Cue
    messages: tuple         # tuple of {"role": "system"|"user"|"assistant", "content": str} dicts
    text: str               # fully rendered string; the next token is the answer letter
    fmt: str                # "plain" | "native"
    letters: tuple          # option letters for this item


@dataclass
class Score:
    item_id: str
    cue: Cue
    fmt: str
    checkpoint: str
    probs: dict             # letter -> renormalized probability over the item's letters (sums to 1)
    letter_mass: float      # total unnormalized probability on the letter tokens (0..1)
    logprobs: dict          # letter -> raw (unnormalized) log-probability, summed over token variants
    extra: dict = field(default_factory=dict)

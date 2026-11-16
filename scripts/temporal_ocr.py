"""Small OCR scheduler; consensus describes repeated agreement, not correctness."""
from collections import Counter, deque
from dataclasses import dataclass, field


@dataclass
class ReadingState:
    last_seen: int
    last_attempt: int = -1000000
    attempts: int = 0
    latest: str = ""
    text: str = ""
    consensus: bool = False
    layout: str = ""
    error: str = ""
    votes: deque = field(default_factory=lambda: deque(maxlen=5))


class TemporalOCR:
    def __init__(self, interval=10, stable_interval=30, retention=60):
        self.interval = interval
        self.stable_interval = stable_interval
        self.retention = retention
        self.states = {}

    def observe(self, identities, frame):
        for identity in identities:
            self.states.setdefault(identity, ReadingState(frame)).last_seen = frame
        for identity in list(self.states):
            if frame - self.states[identity].last_seen > self.retention:
                del self.states[identity]

    def due(self, identity, frame, every_frame=False):
        state = self.states[identity]
        interval = self.stable_interval if state.consensus else self.interval
        return every_frame or frame - state.last_attempt >= interval

    def attempted(self, identity, frame):
        state = self.states[identity]
        state.last_attempt = frame
        state.attempts += 1

    def accept(self, identity, reading):
        state = self.states[identity]
        state.latest, state.layout, state.error = reading.text, reading.layout, reading.error
        # Failed/uncertain reads consume a slot too: old agreement must not
        # remain marked as consensus after repeated unreadable crops.
        valid = bool(reading.text and reading.split_ok and not reading.error)
        state.votes.append(reading.text if valid else "")
        counts = Counter(text for text in state.votes if text)
        if not counts:
            state.text, state.consensus = "", False
            return
        # Prefer the newest candidate in a tie; never permanently lock text.
        best = max(counts, key=lambda text: (counts[text], list(state.votes)[::-1].index(text) * -1))
        state.text = best
        state.consensus = counts[best] >= 3 and counts[best] / len(state.votes) >= .6

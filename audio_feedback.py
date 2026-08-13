"""Spoken feedback during a run, so you can hear what was detected while walking.

Steps are announced as "walking" with no direction. Live forward/backward can still
be corrected at the end of the run, and the voice must never contradict the map.

The speech runs on its own thread, and if pyttsx3 is missing everything turns into
a no-op, so a run can never stall or fail because of the audio.
"""

import queue
import threading

LIVE_PHRASE = {
    "step_forward": "walking",
    "step_backward": "walking",
    "left_turn": "left turn",
    "right_turn": "right turn",
    "left_crab": "crab left",
    "right_crab": "crab right",
}

ALWAYS_ANNOUNCE = {"left_turn", "right_turn"}


class Speaker:
    def __init__(self, enabled=True):
        self._q = queue.Queue(maxsize=8)
        self._ok = False
        self._thread = None
        if not enabled:
            return
        try:
            import pyttsx3  # noqa: F401
        except Exception as e:
            print(f"[audio] voice OFF, pyttsx3 not available ({e}). pip install pyttsx3 to enable.")
            return
        self._ok = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @property
    def active(self):
        return self._ok

    def _worker(self):
        import pyttsx3
        try:
            pyttsx3.init().stop()
        except Exception as e:
            self._ok = False
            print(f"[audio] voice OFF, engine failed to start ({e}).")
            return
        while True:
            text = self._q.get()
            if text is None:
                return
            # A fresh engine per phrase. Reusing one engine across calls locks up
            # on Windows after the first few phrases.
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception:
                pass

    def say(self, text):
        if not self._ok:
            return
        try:
            self._q.put_nowait(text)
        except queue.Full:
            # A late announcement is worse than a missing one.
            pass

    def close(self):
        if not self._ok:
            return
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)


class MovementAnnouncer:
    """Collapses repeats: eight steps in a row say "walking" once, not eight times."""

    def __init__(self, speaker):
        self._speaker = speaker
        self._last_phrase = None

    def movement(self, movement_type):
        phrase = LIVE_PHRASE.get(movement_type)
        if phrase is None:
            return None
        if movement_type in ALWAYS_ANNOUNCE or phrase != self._last_phrase:
            self._speaker.say(phrase)
            self._last_phrase = phrase
            return phrase
        return None

    def stairs(self, phrase):
        self._speaker.say(phrase)
        self._last_phrase = phrase
        return phrase


class StairVoiceCue:
    """Says "up stairs" or "down stairs" once per flight.

    One stair is inside the barometer noise, but a whole climb is not, so the cue
    watches the altitude over the last few seconds instead of step by step. Steps
    have to be happening at the same time, otherwise it is a lift and it stays quiet.
    """

    def __init__(self, min_rise, min_steps=2, rearm_flat=0.25):
        self.min_rise = float(min_rise)
        self.min_steps = int(min_steps)
        self.rearm_flat = float(rearm_flat)
        self._armed = True

    def update(self, rise, steps_in_window):
        if rise is None:
            return None
        if self._armed:
            if steps_in_window >= self.min_steps:
                if rise >= self.min_rise:
                    self._armed = False
                    return "up stairs"
                if rise <= -self.min_rise:
                    self._armed = False
                    return "down stairs"
        elif abs(rise) <= self.rearm_flat:
            # Re-arms only once the altitude is flat again, so one climb is announced once.
            self._armed = True
        return None

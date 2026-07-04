"""本地雙緩衝區：暫存一批已算好的 NumPy 陣列，滿 1000 tick 或滿 250ms 就整批出貨。"""


class LocalDoubleBuffer:
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.timestamps = None
        self.current = None
        self.vibration = None
        self.size = 0
        self.pos = 0
        self._temp_timestamps = []
        self._temp_values = []

    def load_batch(self, timestamps, current_batch, vibration_batch):
        self.timestamps = timestamps
        self.current = current_batch
        self.vibration = vibration_batch
        self.size = len(timestamps)
        self.pos = 0

    def advance(self):
        self.pos += 1

    @property
    def full(self) -> bool:
        return self.pos >= self.size

    def add_temperature(self, timestamp, value):
        self._temp_timestamps.append(timestamp)
        self._temp_values.append(value)

    def flush(self) -> dict:
        n = self.pos
        batch = {
            "device_id": self.device_id,
            "timestamps": self.timestamps[:n].copy(),
            "current": {phase: arr[:n].copy() for phase, arr in self.current.items()},
            "vibration": {axis: arr[:n].copy() for axis, arr in self.vibration.items()},
        }
        if self._temp_timestamps:
            batch["temperature"] = {
                "timestamps": list(self._temp_timestamps),
                "values": list(self._temp_values),
            }
            self._temp_timestamps = []
            self._temp_values = []
        return batch

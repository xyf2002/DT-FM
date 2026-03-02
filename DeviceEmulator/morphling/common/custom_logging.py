import time
from typing import Any, Dict, List, Optional

UL_COMM = "ul_comm"
DL_COMM = "dl_comm"
DEV_COMP = "dev_comp"


class EventTimeLogger:
    def __init__(self, id, events: List[str] = None):
        self.event_timers: Dict[str, float] = {}

        self.total_time: float = 0.0
        self.id = id

        for event in events:
            self.event_timers[event] = 0.0

        self.real_time = time.time()

    def max_time(self) -> float:
        return max(self.event_timers.values())

    def sync_time(self):
        max_time = self.max_time()
        for event in self.event_timers:
            self.event_timers[event] = max_time

        print(f"[{self.id}] Synced time to {max_time}")
        return max_time

    def set_time(self, timestamp: float):
        # for event in self.event_timers:
        #     self.event_timers[event] = timestamp

        self.total_time = timestamp
        self.real_elapsed = time.time() - self.real_time
        print(
            f"[{self.id}] Set time to {timestamp}, real elapsed {self.real_elapsed}"
        )

    def record(self, elapsed, event: str, last_events: List[str] = None):
        if last_events is not None:
            # dependent_events = [
            #     event for event in last_events if event not in self.event_timers
            # ]
            max_time = max([self.event_timers[event] for event in last_events])
            max_event = max(last_events, key=lambda x: self.event_timers[x])
            max_time = max(max_time, self.total_time)
            self.event_timers[event] = max_time + elapsed
            start_time = max_time
        else:
            self.event_timers[event] = (
                max(self.total_time, self.max_time()) + elapsed
            )
            start_time = self.event_timers[event] - elapsed
            max_event = event

        # print(
        #     f"[{self.id}] {event} took {elapsed} seconds, started at {start_time}, depend event {max_event}"
        # )

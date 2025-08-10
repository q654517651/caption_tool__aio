# services/event_bus.py
from collections import defaultdict
from typing import Any, Callable, Dict, List

class EventBus:
    def __init__(self) -> None:
        self._subs: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)

    def on(self, topic: str, handler: Callable[[Any], None]) -> None:
        self._subs[topic].append(handler)

    def emit(self, topic: str, payload: Any) -> None:
        for h in list(self._subs.get(topic, [])):
            try:
                h(payload)
            except Exception:
                # 不让单个订阅者的异常拖垮总线
                pass

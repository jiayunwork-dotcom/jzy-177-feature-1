"""内存事件总线：估计结果产生后向各会话订阅者推送。

仅负责内存中的发布/订阅；持久化由存储层负责。SSE 端点为每个订阅
创建有界队列，慢消费者只丢自己的消息（丢弃最旧的待发结果），不阻塞
追加/估计路径。
"""

from __future__ import annotations

import asyncio
import itertools
from collections import deque


class EventBroker:
    def __init__(self, queue_maxsize: int = 64):
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._queue_maxsize = queue_maxsize
        self._seq = itertools.count(1)

    def subscribe(self, session_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subscribers.setdefault(session_id, set()).add(q)
        return q

    def unsubscribe(self, session_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(session_id)
        if subs:
            subs.discard(q)
            if not subs:
                self._subscribers.pop(session_id, None)

    def publish(self, session_id: str, result: dict) -> None:
        """向某会话的全部订阅者投递；订阅者队列满时为其丢弃最旧消息。"""
        payload = {"seq": next(self._seq), "session_id": session_id, "result": result}
        for q in list(self._subscribers.get(session_id, ())):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass

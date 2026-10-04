"""Chat history in Redis: one list per session, trimmed and expiring.

Only user and assistant turns are kept; tool traffic stays inside a single agent run.
A session belongs to the user who created it, and nobody else can read or extend it.
"""

import json

from redis.asyncio import Redis

from app.llm.base import Message


class SessionForbidden(Exception):
    pass


class SessionStore:
    def __init__(self, redis: Redis, ttl_s: int, max_messages: int):
        self._redis = redis
        self._ttl_s = ttl_s
        self._max_messages = max_messages

    async def load(self, session_id: str, user_id: str) -> list[Message]:
        owner = await self._redis.get(f"session:{session_id}:owner")
        if owner is None:
            return []
        if owner.decode() != user_id:
            raise SessionForbidden(session_id)
        raw = await self._redis.lrange(f"session:{session_id}:messages", 0, -1)
        return [Message(**json.loads(item)) for item in raw]

    async def append(self, session_id: str, user_id: str, user_text: str, assistant_text: str) -> None:
        owner_key, messages_key = f"session:{session_id}:owner", f"session:{session_id}:messages"
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.set(owner_key, user_id, nx=True)
            pipe.rpush(
                messages_key,
                json.dumps({"role": "user", "content": user_text}, ensure_ascii=False),
                json.dumps({"role": "assistant", "content": assistant_text}, ensure_ascii=False),
            )
            pipe.ltrim(messages_key, -self._max_messages, -1)
            pipe.expire(owner_key, self._ttl_s)
            pipe.expire(messages_key, self._ttl_s)
            await pipe.execute()

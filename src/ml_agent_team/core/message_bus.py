"""Async publish-subscribe message bus for agent communication."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Awaitable, Callable

from .messages import AgentMessage
from .types import MessageType

Subscriber = Callable[[AgentMessage], Awaitable[None]]

logger = logging.getLogger(__name__)


class MessageBus:
    """In-process async pub-sub message bus for inter-agent coordination.

    Agents subscribe by name (for targeted messages) or by topic/message type
    (for broadcast monitoring). The bus maintains a full audit trail of all
    published messages.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._topic_subscribers: dict[MessageType, list[Subscriber]] = defaultdict(list)
        self._global_subscribers: list[Subscriber] = []
        self._history: list[AgentMessage] = []
        self._lock = asyncio.Lock()

    async def publish(self, message: AgentMessage) -> None:
        """Publish a message. Delivers to targeted agent or broadcasts."""
        async with self._lock:
            self._history.append(message)

        # Deliver to targeted agent
        if message.target_agent and message.target_agent in self._subscribers:
            for handler in self._subscribers[message.target_agent]:
                try:
                    await handler(message)
                except Exception:
                    logger.exception(
                        "Handler error for agent %s", message.target_agent
                    )

        # Deliver to topic subscribers
        if message.type in self._topic_subscribers:
            for handler in self._topic_subscribers[message.type]:
                try:
                    await handler(message)
                except Exception:
                    logger.exception("Topic handler error for %s", message.type)

        # Deliver to global subscribers
        for handler in self._global_subscribers:
            try:
                await handler(message)
            except Exception:
                logger.exception("Global handler error")

    def subscribe(self, agent_name: str, handler: Subscriber) -> None:
        """Subscribe an agent to receive messages addressed to it."""
        self._subscribers[agent_name].append(handler)

    def subscribe_topic(self, topic: MessageType, handler: Subscriber) -> None:
        """Subscribe to all messages of a given type."""
        self._topic_subscribers[topic].append(handler)

    def subscribe_all(self, handler: Subscriber) -> None:
        """Subscribe to all messages (for logging/monitoring)."""
        self._global_subscribers.append(handler)

    def get_history(
        self,
        agent_name: str | None = None,
        message_type: MessageType | None = None,
        limit: int | None = None,
    ) -> list[AgentMessage]:
        """Query message history with optional filters."""
        messages = self._history

        if agent_name:
            messages = [
                m
                for m in messages
                if m.source_agent == agent_name or m.target_agent == agent_name
            ]

        if message_type:
            messages = [m for m in messages if m.type == message_type]

        if limit:
            messages = messages[-limit:]

        return messages

    def clear_history(self) -> None:
        """Clear the message history."""
        self._history.clear()

    @property
    def message_count(self) -> int:
        """Total number of messages published."""
        return len(self._history)

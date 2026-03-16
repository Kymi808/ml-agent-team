"""Tests for the async message bus."""

import pytest

from ml_agent_team.core.message_bus import MessageBus
from ml_agent_team.core.messages import AgentMessage
from ml_agent_team.core.types import MessageType


@pytest.mark.asyncio
async def test_publish_to_subscriber():
    bus = MessageBus()
    received = []

    async def handler(msg: AgentMessage) -> None:
        received.append(msg)

    bus.subscribe("agent_a", handler)

    msg = AgentMessage(
        type=MessageType.COMMAND,
        source_agent="orchestrator",
        target_agent="agent_a",
    )
    await bus.publish(msg)

    assert len(received) == 1
    assert received[0].source_agent == "orchestrator"


@pytest.mark.asyncio
async def test_topic_subscription():
    bus = MessageBus()
    errors = []

    async def error_handler(msg: AgentMessage) -> None:
        errors.append(msg)

    bus.subscribe_topic(MessageType.ERROR, error_handler)

    await bus.publish(AgentMessage(type=MessageType.ERROR, source_agent="test"))
    await bus.publish(AgentMessage(type=MessageType.LOG, source_agent="test"))

    assert len(errors) == 1


@pytest.mark.asyncio
async def test_message_history():
    bus = MessageBus()

    await bus.publish(AgentMessage(type=MessageType.LOG, source_agent="a"))
    await bus.publish(AgentMessage(type=MessageType.ERROR, source_agent="b"))
    await bus.publish(AgentMessage(type=MessageType.LOG, source_agent="a"))

    assert bus.message_count == 3
    assert len(bus.get_history(agent_name="a")) == 2
    assert len(bus.get_history(message_type=MessageType.ERROR)) == 1


@pytest.mark.asyncio
async def test_global_subscriber():
    bus = MessageBus()
    all_msgs = []

    async def handler(msg: AgentMessage) -> None:
        all_msgs.append(msg)

    bus.subscribe_all(handler)

    await bus.publish(AgentMessage(type=MessageType.LOG, source_agent="x"))
    await bus.publish(AgentMessage(type=MessageType.ERROR, source_agent="y"))

    assert len(all_msgs) == 2

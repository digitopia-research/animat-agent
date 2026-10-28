"""pytest 配置和共享 fixtures"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from animat.nervous import NervousSystem
from animat.memory import MemoryStore


@pytest.fixture
def mock_nervous():
    """Mock 神经系统"""
    nervous = MagicMock(spec=NervousSystem)
    nervous.fire = AsyncMock()
    nervous.listen = AsyncMock()
    nervous.reflex_fire = AsyncMock()
    return nervous


@pytest.fixture
def mock_memory():
    """Mock 记忆存储"""
    memory = MagicMock(spec=MemoryStore)
    memory.store_episode = AsyncMock()
    memory.recall_episodes = AsyncMock(return_value=[])
    memory.save_state = AsyncMock()
    memory.load_state = AsyncMock(return_value={})
    memory.get_recent_feelings = AsyncMock(return_value=[])
    memory.append_feeling = AsyncMock()
    memory.healthy = True
    return memory


@pytest.fixture(scope="session")
def event_loop():
    """创建 session 级事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

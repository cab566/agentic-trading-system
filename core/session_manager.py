#!/usr/bin/env python3
"""
Session Manager for HTTP Clients

Manages aiohttp client sessions to prevent resource leaks and ensure proper cleanup.
Provides singleton pattern for session reuse and automatic cleanup on application shutdown.
"""

import asyncio
import logging
import weakref
from typing import Dict, Optional, Any
from contextlib import asynccontextmanager
import aiohttp
import atexit
from datetime import datetime, timedelta


class SessionManager:
    """Manages aiohttp client sessions with proper cleanup"""
    
    _instance = None
    _sessions: Dict[str, aiohttp.ClientSession] = {}
    _session_configs: Dict[str, Dict[str, Any]] = {}
    _cleanup_tasks: Dict[str, asyncio.Task] = {}
    _logger = logging.getLogger(__name__)
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Register cleanup on exit
            atexit.register(cls._cleanup_all_sessions)
        return cls._instance
    
    @classmethod
    def get_session(cls, 
                   session_name: str = "default",
                   timeout: Optional[aiohttp.ClientTimeout] = None,
                   connector: Optional[aiohttp.BaseConnector] = None,
                   **kwargs) -> aiohttp.ClientSession:
        """
        Get or create a named session with specified configuration
        
        Args:
            session_name: Unique name for the session
            timeout: Client timeout configuration
            connector: Custom connector (e.g., for SSL settings)
            **kwargs: Additional ClientSession arguments
        
        Returns:
            aiohttp.ClientSession instance
        """
        
        if session_name in cls._sessions:
            session = cls._sessions[session_name]
            if not session.closed:
                return session
            else:
                # Session was closed, remove it
                del cls._sessions[session_name]
        
        # Create new session
        session_config = {
            'timeout': timeout or aiohttp.ClientTimeout(total=30),
            'connector': connector,
            **kwargs
        }
        
        # Remove None values
        session_config = {k: v for k, v in session_config.items() if v is not None}
        
        try:
            session = aiohttp.ClientSession(**session_config)
            cls._sessions[session_name] = session
            cls._session_configs[session_name] = session_config
            
            cls._logger.info(f"Created new aiohttp session: {session_name}")
            
            # Schedule cleanup after inactivity
            cls._schedule_cleanup(session_name)
            
            return session
            
        except Exception as e:
            cls._logger.error(f"Failed to create session {session_name}: {e}")
            raise
    
    @classmethod
    async def close_session(cls, session_name: str):
        """Close a specific session"""
        
        if session_name in cls._sessions:
            session = cls._sessions[session_name]
            if not session.closed:
                await session.close()
                cls._logger.info(f"Closed session: {session_name}")
            
            del cls._sessions[session_name]
            
            # Cancel cleanup task if exists
            if session_name in cls._cleanup_tasks:
                cls._cleanup_tasks[session_name].cancel()
                del cls._cleanup_tasks[session_name]
    
    @classmethod
    async def close_all_sessions(cls):
        """Close all active sessions"""
        
        session_names = list(cls._sessions.keys())
        for session_name in session_names:
            await cls.close_session(session_name)
        
        cls._logger.info("All sessions closed")
    
    @classmethod
    def _cleanup_all_sessions(cls):
        """Synchronous cleanup for atexit handler"""
        
        if cls._sessions:
            try:
                # Try to get current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule cleanup
                    asyncio.create_task(cls.close_all_sessions())
                else:
                    # If no loop, run cleanup
                    loop.run_until_complete(cls.close_all_sessions())
            except RuntimeError:
                # No event loop, create one
                asyncio.run(cls.close_all_sessions())
    
    @classmethod
    def _schedule_cleanup(cls, session_name: str, timeout_minutes: int = 30):
        """Schedule automatic cleanup of inactive session"""
        
        async def cleanup_after_timeout():
            await asyncio.sleep(timeout_minutes * 60)
            if session_name in cls._sessions:
                cls._logger.info(f"Auto-closing inactive session: {session_name}")
                await cls.close_session(session_name)
        
        # Cancel existing cleanup task
        if session_name in cls._cleanup_tasks:
            cls._cleanup_tasks[session_name].cancel()
        
        # Schedule new cleanup
        cls._cleanup_tasks[session_name] = asyncio.create_task(cleanup_after_timeout())
    
    @classmethod
    @asynccontextmanager
    async def managed_session(cls, 
                            session_name: str = None,
                            auto_close: bool = True,
                            **session_kwargs):
        """
        Context manager for temporary sessions that are automatically closed
        
        Args:
            session_name: Optional session name (generates unique if None)
            auto_close: Whether to close session after use
            **session_kwargs: Arguments for ClientSession
        
        Usage:
            async with SessionManager.managed_session() as session:
                async with session.get('https://api.example.com') as response:
                    data = await response.json()
        """
        
        if session_name is None:
            session_name = f"temp_{datetime.now().timestamp()}"
        
        session = cls.get_session(session_name, **session_kwargs)
        
        try:
            yield session
        finally:
            if auto_close:
                await cls.close_session(session_name)
    
    @classmethod
    def get_session_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all active sessions"""
        
        info = {}
        for name, session in cls._sessions.items():
            info[name] = {
                'closed': session.closed,
                'connector_type': type(session.connector).__name__,
                'timeout': str(session.timeout),
                'config': cls._session_configs.get(name, {})
            }
        
        return info


class ResilientHttpClient:
    """HTTP client with automatic retries and session management"""
    
    def __init__(self, 
                 session_name: str = "resilient_client",
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 timeout: int = 30):
        
        self.session_name = session_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.logger = logging.getLogger(__name__)
    
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """GET request with retries"""
        return await self._request('GET', url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """POST request with retries"""
        return await self._request('POST', url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """PUT request with retries"""
        return await self._request('PUT', url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """DELETE request with retries"""
        return await self._request('DELETE', url, **kwargs)
    
    async def _request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request with retries and proper session management"""
        
        session = SessionManager.get_session(
            self.session_name,
            timeout=self.timeout
        )
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                async with session.request(method, url, **kwargs) as response:
                    # Check if we should retry based on status code
                    if response.status >= 500 and attempt < self.max_retries:
                        self.logger.warning(f"Server error {response.status} for {url}, retrying...")
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    
                    return response
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}, retrying...")
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    self.logger.error(f"All retry attempts failed for {url}")
                    break
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise aiohttp.ClientError(f"Request failed after {self.max_retries} retries")


# Convenience functions for common use cases
async def get_json(url: str, session_name: str = "default", **kwargs) -> Dict[str, Any]:
    """Convenience function to get JSON data"""
    
    session = SessionManager.get_session(session_name)
    async with session.get(url, **kwargs) as response:
        response.raise_for_status()
        return await response.json()


async def post_json(url: str, data: Dict[str, Any], session_name: str = "default", **kwargs) -> Dict[str, Any]:
    """Convenience function to post JSON data"""
    
    session = SessionManager.get_session(session_name)
    async with session.post(url, json=data, **kwargs) as response:
        response.raise_for_status()
        return await response.json()


# Context manager for temporary sessions
@asynccontextmanager
async def temporary_session(**session_kwargs):
    """Create a temporary session that's automatically closed"""
    async with SessionManager.managed_session(**session_kwargs) as session:
        yield session


if __name__ == "__main__":
    import asyncio
    
    async def test_session_manager():
        """Test the session manager functionality"""
        
        print("Testing SessionManager...")
        
        # Test basic session creation
        session1 = SessionManager.get_session("test1")
        session2 = SessionManager.get_session("test2")
        
        print(f"Created sessions: {list(SessionManager._sessions.keys())}")
        
        # Test session reuse
        session1_reused = SessionManager.get_session("test1")
        assert session1 is session1_reused, "Session should be reused"
        
        # Test managed session
        async with SessionManager.managed_session("temp") as temp_session:
            print(f"Using temporary session: {temp_session}")
        
        # Test resilient client
        client = ResilientHttpClient()
        try:
            # This will likely fail, but tests the retry mechanism
            response = await client.get("https://httpbin.org/status/500")
        except Exception as e:
            print(f"Expected failure: {e}")
        
        # Get session info
        info = SessionManager.get_session_info()
        print(f"Session info: {info}")
        
        # Cleanup
        await SessionManager.close_all_sessions()
        print("All sessions closed")
    
    asyncio.run(test_session_manager())
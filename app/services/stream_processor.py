"""Stream processing for HLS and Redis streams"""

import asyncio
import numpy as np
from typing import Iterator, Optional, List, Union
import logging
import re
from urllib.parse import urlparse

import ffmpeg
import redis.asyncio as aioredis
import cv2

from app.config import settings
from app.core.exceptions import (
    StreamException,
    HLSStreamException,
    RedisStreamException,
    StreamConnectionException,
    StreamTimeoutException,
    FrameExtractionException
)

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Process HLS and Redis video streams"""
    
    def __init__(self):
        """Initialize stream processor"""
        self._redis_client: Optional[aioredis.Redis] = None
        self._redis_connected = False
    
    async def initialize_redis(self) -> None:
        """Initialize Redis connection"""
        try:
            self._redis_client = aioredis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                socket_timeout=settings.redis_socket_timeout,
                decode_responses=False  # We'll handle binary frame data
            )
            # Test connection
            await self._redis_client.ping()
            self._redis_connected = True
            logger.info(f"Connected to Redis at {settings.redis_host}:{settings.redis_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._redis_connected = False
            raise StreamConnectionException(f"Redis connection failed: {e}") from e
    
    async def close_redis(self) -> None:
        """Close Redis connection"""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_connected = False
            logger.info("Redis connection closed")
    
    def detect_stream_type(self, stream_ref: str) -> str:
        """Detect stream type from reference
        
        Args:
            stream_ref: Stream reference (URL or Redis key)
            
        Returns:
            'hls' or 'redis'
        """
        # Check if it's a URL (HLS)
        if stream_ref.startswith(('http://', 'https://', 'rtmp://', 'rtsp://')):
            return 'hls'
        
        # Check if it contains .m3u8 (HLS playlist)
        if '.m3u8' in stream_ref.lower():
            return 'hls'
        
        # Otherwise assume Redis
        return 'redis'
    
    async def get_frames_from_hls(
        self,
        hls_url: str,
        fps: float = 2.0,
        max_frames: Optional[int] = None,
        timeout: float = 30.0
    ) -> Iterator[np.ndarray]:
        """Extract frames from HLS stream using ffmpeg-python
        
        Args:
            hls_url: HLS stream URL
            fps: Target frames per second to extract
            max_frames: Maximum number of frames to extract (None for unlimited)
            timeout: Timeout in seconds
            
        Yields:
            Frames as numpy arrays (H, W, C) in RGB format
        """
        try:
            logger.info(f"Starting HLS stream processing: {hls_url}")
            
            # Use ffmpeg to extract frames
            # -r sets output frame rate (fps)
            # -f image2pipe outputs to pipe
            # -vcodec rawvideo uses raw video codec
            # -pix_fmt rgb24 ensures RGB format
            
            process = (
                ffmpeg
                .input(hls_url, f='hls')
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', r=fps)
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
            
            frame_count = 0
            width = None
            height = None
            
            # Try to get video properties from probe
            try:
                probe = ffmpeg.probe(hls_url)
                video_stream = next(
                    (s for s in probe['streams'] if s['codec_type'] == 'video'),
                    None
                )
                if video_stream:
                    width = int(video_stream.get('width', 640))
                    height = int(video_stream.get('height', 480))
                    logger.info(f"Detected video dimensions: {width}x{height}")
            except Exception as e:
                logger.warning(f"Could not probe video properties: {e}, using defaults")
                width = 640
                height = 480
            
            # Read frames from pipe
            frame_size = width * height * 3  # RGB = 3 bytes per pixel
            
            while True:
                if max_frames is not None and frame_count >= max_frames:
                    break
                
                # Read frame data
                raw_frame = await asyncio.wait_for(
                    asyncio.to_thread(process.stdout.read, frame_size),
                    timeout=timeout
                )
                
                if len(raw_frame) != frame_size:
                    # End of stream or incomplete frame
                    break
                
                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((height, width, 3))
                
                yield frame
                frame_count += 1
                
                if frame_count % 10 == 0:
                    logger.debug(f"Extracted {frame_count} frames from HLS stream")
            
            # Cleanup
            process.stdout.close()
            process.wait()
            
            logger.info(f"Completed HLS stream processing: {frame_count} frames extracted")
            
        except asyncio.TimeoutError:
            raise StreamTimeoutException(f"HLS frame extraction timed out after {timeout}s")
        except Exception as e:
            logger.error(f"HLS stream processing failed: {e}")
            raise HLSStreamException(f"Failed to process HLS stream: {e}") from e
    
    async def get_frames_from_redis(
        self,
        stream_key: str,
        consumer_group: Optional[str] = None,
        consumer_name: Optional[str] = None,
        timeout: float = 5.0,
        max_frames: Optional[int] = None
    ) -> Iterator[np.ndarray]:
        """Extract frames from Redis stream
        
        Args:
            stream_key: Redis stream key
            consumer_group: Optional consumer group name
            consumer_name: Optional consumer name
            timeout: Timeout for reading from stream
            max_frames: Maximum number of frames to read (None for unlimited)
            
        Yields:
            Frames as numpy arrays (H, W, C) in RGB format
        """
        if not self._redis_connected:
            await self.initialize_redis()
        
        if not self._redis_client:
            raise RedisStreamException("Redis client not initialized")
        
        try:
            logger.info(f"Starting Redis stream processing: {stream_key}")
            
            # Create consumer group if provided
            if consumer_group:
                try:
                    await self._redis_client.xgroup_create(
                        name=stream_key,
                        groupname=consumer_group,
                        id='0',
                        mkstream=True
                    )
                except aioredis.ResponseError as e:
                    if "BUSYGROUP" not in str(e):
                        raise
            
            frame_count = 0
            
            while True:
                if max_frames is not None and frame_count >= max_frames:
                    break
                
                # Read from stream
                if consumer_group and consumer_name:
                    # Use consumer group
                    messages = await asyncio.wait_for(
                        self._redis_client.xreadgroup(
                            groupname=consumer_group,
                            consumername=consumer_name,
                            streams={stream_key: '>'},
                            count=1,
                            block=int(timeout * 1000)  # Convert to milliseconds
                        ),
                        timeout=timeout + 1.0
                    )
                else:
                    # Simple read
                    messages = await asyncio.wait_for(
                        self._redis_client.xread(
                            streams={stream_key: '$'},
                            count=1,
                            block=int(timeout * 1000)
                        ),
                        timeout=timeout + 1.0
                    )
                
                if not messages:
                    # No more messages
                    break
                
                # Parse messages
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        # Extract frame data from message
                        # Assuming frame is stored as binary in a field
                        frame_data = None
                        for key, value in fields.items():
                            if isinstance(key, bytes):
                                key = key.decode('utf-8')
                            if 'frame' in key.lower() or 'image' in key.lower():
                                frame_data = value
                                break
                        
                        if frame_data is None and fields:
                            # Use first field as frame data
                            frame_data = list(fields.values())[0]
                        
                        if frame_data:
                            # Decode frame (assuming it's stored as bytes)
                            if isinstance(frame_data, bytes):
                                # Try to decode as image
                                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                                if frame is not None:
                                    # Convert BGR to RGB
                                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    yield frame
                                    frame_count += 1
                                    
                                    if frame_count % 10 == 0:
                                        logger.debug(f"Extracted {frame_count} frames from Redis stream")
                
                # Small delay to avoid busy loop
                await asyncio.sleep(0.01)
            
            logger.info(f"Completed Redis stream processing: {frame_count} frames extracted")
            
        except asyncio.TimeoutError:
            logger.debug("Redis stream read timeout (no new messages)")
            # This is normal, just stop yielding
        except Exception as e:
            logger.error(f"Redis stream processing failed: {e}")
            raise RedisStreamException(f"Failed to process Redis stream: {e}") from e
    
    async def get_frames(
        self,
        stream_ref: str,
        fps: float = 2.0,
        max_frames: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> Iterator[np.ndarray]:
        """Get frames from stream (auto-detect type)
        
        Args:
            stream_ref: Stream reference (HLS URL or Redis key)
            fps: Target frames per second
            max_frames: Maximum frames to extract
            timeout: Timeout in seconds
            
        Yields:
            Frames as numpy arrays
        """
        stream_type = self.detect_stream_type(stream_ref)
        
        if timeout is None:
            timeout = settings.frame_timeout
        
        logger.info(f"Processing {stream_type} stream: {stream_ref}")
        
        if stream_type == 'hls':
            async for frame in self.get_frames_from_hls(
                stream_ref, fps=fps, max_frames=max_frames, timeout=timeout
            ):
                yield frame
        else:
            async for frame in self.get_frames_from_redis(
                stream_ref, timeout=timeout, max_frames=max_frames
            ):
                yield frame


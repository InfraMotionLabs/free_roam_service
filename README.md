# Free-Roam Inference Service

A production-ready FastAPI service for running Vision-Language Model (VLM) inference on video streams. Supports both HLS and Redis streams with configurable frame sampling, prompt management, and comprehensive monitoring.

## Features

- **Multi-Stream Support**: Process HLS streams and Redis streams with automatic detection
- **Configurable Frame Sampling**: Adjustable FPS for frame extraction (default: 2 FPS)
- **Prompt Management**: Update prompts dynamically with versioning and history
- **Async Processing**: Fully asynchronous architecture for high throughput
- **Model Abstraction**: Pluggable VLM interface - inject any vision-language model
- **Health Monitoring**: Comprehensive health, readiness, and liveness checks
- **Error Handling**: Robust error handling with graceful degradation
- **Memory Management**: Efficient memory usage with automatic cleanup
- **Rate Limiting**: Built-in rate limiting for API protection
- **CORS Support**: Configurable CORS for frontend integration

## Architecture

```
┌─────────────┐
│   FastAPI   │
│   Service   │
└──────┬──────┘
       │
       ├─── Inference Engine ─── Model (VLM)
       │
       ├─── Stream Processor ─── HLS / Redis
       │
       ├─── Frame Sampler ─── Configurable FPS
       │
       └─── State Manager ─── Prompt Management
```

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for HLS stream processing)
- Redis (optional, for Redis stream support)

### Setup

1. **Clone and navigate to the service directory:**
```bash
cd free_roam_service
```

2. **Create virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install FFmpeg:**
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

5. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings
```

## Configuration

All configuration is done via environment variables. See `.env.example` for all available options.

### Key Configuration Options

- `MODEL_PATH`: Path to your VLM model directory
- `MODEL_TYPE`: Model type identifier (default: `example_vlm`)
- `DEVICE`: Device to use (`auto`, `cuda`, or `cpu`)
- `DEFAULT_FPS`: Default frames per second for sampling (default: `2.0`)
- `MAX_BATCH_SIZE`: Maximum batch size for inference (default: `8`)
- `REDIS_HOST`: Redis server host (default: `localhost`)
- `REDIS_PORT`: Redis server port (default: `6379`)

## Usage

### Starting the Service

```bash
# Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or using Python
python -m app.main
```

The service will be available at `http://localhost:8000`

### API Documentation

Once the service is running, access:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### Health Checks

- `GET /health` - Overall health check
- `GET /health/ready` - Readiness check (for orchestration)
- `GET /health/live` - Liveness check

### Inference

- `POST /api/v1/inference/start` - Start inference on a stream
- `GET /api/v1/inference/status/{job_id}` - Get job status
- `GET /api/v1/inference/results/{job_id}` - Get inference results
- `DELETE /api/v1/inference/cancel/{job_id}` - Cancel a job
- `GET /api/v1/inference/jobs` - List all jobs

### Prompt Management

- `POST /api/v1/prompt/update` - Update the active prompt
- `GET /api/v1/prompt/current` - Get current prompt
- `GET /api/v1/prompt/history` - Get prompt history

## Example Usage

### Start Inference on HLS Stream

```bash
curl -X POST "http://localhost:8000/api/v1/inference/start" \
  -H "Content-Type: application/json" \
  -d '{
    "stream_ref": "https://example.com/stream.m3u8",
    "prompt": "Detect all people and vehicles",
    "fps": 2.0
  }'
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Inference job started"
}
```

### Check Job Status

```bash
curl "http://localhost:8000/api/v1/inference/status/550e8400-e29b-41d4-a716-446655440000"
```

### Get Results

```bash
curl "http://localhost:8000/api/v1/inference/results/550e8400-e29b-41d4-a716-446655440000"
```

### Update Prompt

```bash
curl -X POST "http://localhost:8000/api/v1/prompt/update" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Detect only vehicles",
    "preserve_previous": true
  }'
```

## Integrating Your VLM Model

The service uses an abstract model interface. To integrate your VLM:

1. **Create your model class** inheriting from `BaseVLM`:

```python
from app.models.base import BaseVLM
import numpy as np

class MyVLM(BaseVLM):
    async def initialize(self, model_path: str, device: str = "auto"):
        # Load your model here
        pass
    
    async def predict(self, frames: List[np.ndarray], prompt: str):
        # Run inference here
        return {"predictions": [...]}
    
    def get_model_info(self):
        return {"model_type": "my_vlm", ...}
```

2. **Update `app/main.py`** to use your model:

```python
from app.models.my_vlm import MyVLM

# In lifespan function:
model = MyVLM()
await model.initialize(model_path=settings.model_path, device=settings.device)
```

## Stream Types

### HLS Streams

Provide an HLS URL (`.m3u8` file):
```json
{
  "stream_ref": "https://example.com/live/stream.m3u8"
}
```

### Redis Streams

Provide a Redis stream key:
```json
{
  "stream_ref": "video_frames"
}
```

The service auto-detects the stream type based on the reference format.

## Error Handling

The service includes comprehensive error handling:

- **Stream Errors**: Automatic retry and graceful degradation
- **Inference Errors**: Timeout handling and error reporting
- **Model Errors**: Clear error messages and logging
- **Validation Errors**: Detailed validation error responses

## Monitoring

### Health Checks

- `/health`: Overall service health
- `/health/ready`: Service readiness (model loaded, etc.)
- `/health/live`: Service liveness

### Logging

Logging is configured via `LOG_LEVEL` and `LOG_FORMAT`:
- `LOG_LEVEL`: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- `LOG_FORMAT`: `json` or `text`

## Performance Considerations

- **Concurrent Streams**: Configured via `MAX_CONCURRENT_STREAMS` (default: 10)
- **Batch Processing**: Configured via `MAX_BATCH_SIZE` (default: 8)
- **Memory Management**: Automatic cleanup after each batch
- **Frame Sampling**: Adjustable FPS to balance accuracy vs. performance

## Development

### Project Structure

```
free_roam_service/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── models/              # Model abstraction
│   ├── services/            # Business logic
│   ├── api/                 # API routes and schemas
│   ├── core/                # Core utilities
│   └── utils/               # Utilities
├── tests/                   # Test suite
├── requirements.txt         # Dependencies
├── .env.example             # Environment template
└── README.md                # This file
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

## Troubleshooting

### Model Not Loading

- Check `MODEL_PATH` is correct
- Verify model files are accessible
- Check device availability (`cuda` vs `cpu`)

### Stream Connection Issues

- For HLS: Verify URL is accessible and stream is live
- For Redis: Check Redis connection settings and server status

### Memory Issues

- Reduce `MAX_BATCH_SIZE`
- Reduce `MAX_CONCURRENT_STREAMS`
- Lower `DEFAULT_FPS`

## License

[Your License Here]

## Contributing

[Contributing Guidelines]

## Support

[Support Information]


# Free-Roam Inference Service (Ray Serve)

A production-ready Ray Serve service for running Vision-Language Model (VLM) inference on video streams. Supports both HLS and Redis streams with configurable frame sampling, prompt management, and comprehensive monitoring.

## Features

- **Multi-Stream Support**: Process HLS streams and Redis streams with automatic detection
- **Configurable Frame Sampling**: Adjustable FPS for frame extraction (default: 2 FPS)
- **Prompt Management**: Update prompts dynamically with versioning and history
- **Ray Serve Architecture**: Distributed, scalable inference with Ray Serve
- **Model Abstraction**: Pluggable VLM interface - inject any vision-language model
- **Health Monitoring**: Comprehensive health, readiness, and liveness checks
- **Error Handling**: Robust error handling with graceful degradation
- **Memory Management**: Efficient memory usage with automatic cleanup

## Architecture

```
┌─────────────┐
│  Ray Serve  │
│   Service   │
└──────┬──────┘
       │
       ├─── Model Deployment ─── VLM Model (scalable)
       │
       ├─── Inference Service ─── Job Tracking & State
       │
       ├─── API Deployment ─── HTTP Endpoints
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
- Ray cluster (single node or distributed)

### Setup

1. **Clone and navigate to the service directory:**
```bash
cd rayserve_service
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
- `NUM_REPLICAS`: Number of model deployment replicas (default: `1`)

## Usage

### Starting Ray Cluster

```bash
# Start Ray head node
ray start --head

# Or start with specific resources
ray start --head --num-cpus=4 --num-gpus=1
```

### Deploying the Service

```bash
# Deploy the service
serve run app.main:app

# Or with specific host/port
serve run app.main:app --host 0.0.0.0 --port 8000
```

The service will be available at `http://localhost:8000`

**Note**: Ray Serve does not provide automatic API documentation endpoints. Use the API endpoints directly or refer to this README for endpoint documentation.

### Stopping the Service

```bash
# Stop Ray Serve
serve shutdown

# Stop Ray cluster
ray stop
```

## API Endpoints

All endpoints are handled by Ray Serve deployments using Ray Serve's native HTTP handling. All responses are JSON format.

### Health Checks

#### `GET /health`

Overall health check endpoint.

**Response** (200 OK):
```json
{
  "status": "healthy",
  "uptime_seconds": 3600.5,
  "uptime_formatted": "1h 0m 0s",
  "timestamp": "2024-01-19T12:00:00Z",
  "model_loaded": true
}
```

#### `GET /health/ready`

Readiness check for orchestration systems (Kubernetes, etc.). Returns 503 if not ready.

**Response** (200 OK if ready, 503 if not ready):
```json
{
  "ready": true,
  "status": "ready",
  "checks": {
    "model_loaded": true
  },
  "timestamp": "2024-01-19T12:00:00Z"
}
```

#### `GET /health/live`

Liveness check endpoint. Always returns 200 if service is running.

**Response** (200 OK):
```json
{
  "alive": true,
  "timestamp": "2024-01-19T12:00:00Z"
}
```

### Inference Endpoints

#### `POST /api/v1/inference/start`

Start inference on a video stream.

**Request Body**:
```json
{
  "stream_ref": "https://example.com/stream.m3u8",  // Required: HLS URL or Redis key
  "prompt": "Detect all people and vehicles",        // Optional: Uses current prompt if not provided
  "fps": 2.0,                                        // Optional: Default from config
  "job_id": "custom-job-id"                          // Optional: Auto-generated if not provided
}
```

**Response** (202 Accepted):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Inference job 550e8400-e29b-41d4-a716-446655440000 started"
}
```

**Error Responses**:
- `400 Bad Request`: Invalid request body or validation error
- `429 Too Many Requests`: Maximum concurrent streams reached
- `500 Internal Server Error`: Server error

#### `GET /api/v1/inference/status/{job_id}`

Get the status of an inference job.

**Path Parameters**:
- `job_id` (string, required): Job identifier

**Response** (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "stream_ref": "https://example.com/stream.m3u8",
  "prompt": "Detect all people and vehicles",
  "fps": 2.0,
  "created_at": 1705665600.0,
  "started_at": 1705665601.0,
  "completed_at": null,
  "frames_processed": 120,
  "results_count": 15,
  "error": null
}
```

**Status Values**: `pending`, `running`, `completed`, `failed`, `cancelled`

**Error Responses**:
- `404 Not Found`: Job not found

#### `GET /api/v1/inference/results/{job_id}`

Get inference results for a completed job.

**Path Parameters**:
- `job_id` (string, required): Job identifier

**Query Parameters**:
- `limit` (integer, optional): Maximum number of results to return

**Response** (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "results": [
    {
      "predictions": [
        {
          "label": "detected_object_0",
          "confidence": 0.85,
          "bbox": [10, 10, 100, 100]
        }
      ],
      "confidence": [0.85],
      "metadata": {
        "num_frames": 16,
        "frame_shape": [224, 224, 3],
        "prompt": "Detect all people and vehicles",
        "model_type": "example_vlm",
        "device": "cuda",
        "timestamp": 1705665605.0
      }
    }
  ],
  "total_results": 15
}
```

**Error Responses**:
- `404 Not Found`: Job not found

#### `DELETE /api/v1/inference/cancel/{job_id}`

Cancel a running or pending inference job.

**Path Parameters**:
- `job_id` (string, required): Job identifier

**Response** (200 OK):
```json
{
  "message": "Job 550e8400-e29b-41d4-a716-446655440000 cancelled",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Error Responses**:
- `404 Not Found`: Job not found or cannot be cancelled

#### `GET /api/v1/inference/jobs`

List all inference jobs.

**Query Parameters**:
- `status` (string, optional): Filter by status (`pending`, `running`, `completed`, `failed`, `cancelled`)

**Response** (200 OK):
```json
{
  "jobs": [
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "stream_ref": "https://example.com/stream.m3u8",
      "prompt": "Detect all people and vehicles",
      "fps": 2.0,
      "created_at": 1705665600.0,
      "frames_processed": 120,
      "results_count": 15
    }
  ],
  "total": 1
}
```

### Prompt Management Endpoints

#### `POST /api/v1/prompt/update`

Update the active prompt (global or job-specific).

**Request Body**:
```json
{
  "prompt": "Detect only vehicles",  // Required: New prompt text
  "job_id": "550e8400-...",          // Optional: Job-specific prompt
  "preserve_previous": true           // Optional: Keep in history (default: true)
}
```

**Response** (200 OK):
```json
{
  "prompt": "Detect only vehicles",
  "version": 2,
  "timestamp": 1705665600.0,
  "created_at": "2024-01-19T10:00:00Z",
  "updated_at": "2024-01-19T12:00:00Z",
  "job_id": null
}
```

**Error Responses**:
- `400 Bad Request`: Empty prompt or validation error

#### `GET /api/v1/prompt/current`

Get the current active prompt.

**Query Parameters**:
- `job_id` (string, optional): Get job-specific prompt

**Response** (200 OK):
```json
{
  "prompt": "Detect only vehicles",
  "version": 2,
  "timestamp": 1705665600.0,
  "created_at": "2024-01-19T10:00:00Z",
  "updated_at": "2024-01-19T12:00:00Z",
  "job_id": null
}
```

**Error Responses**:
- `404 Not Found`: No prompt set

#### `GET /api/v1/prompt/history`

Get prompt history.

**Query Parameters**:
- `limit` (integer, optional): Maximum history entries (default: 10)
- `job_id` (string, optional): Job-specific history (not yet implemented)

**Response** (200 OK):
```json
{
  "history": [
    {
      "prompt": "Detect only vehicles",
      "version": 2,
      "timestamp": 1705665600.0,
      "created_at": "2024-01-19T10:00:00Z",
      "updated_at": "2024-01-19T12:00:00Z"
    },
    {
      "prompt": "Detect all people and vehicles",
      "version": 1,
      "timestamp": 1705665500.0,
      "created_at": "2024-01-19T10:00:00Z",
      "updated_at": "2024-01-19T10:00:00Z"
    }
  ],
  "total": 2
}
```

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

**Response**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Inference job 550e8400-e29b-41d4-a716-446655440000 started"
}
```

### Start Inference on Redis Stream

```bash
curl -X POST "http://localhost:8000/api/v1/inference/start" \
  -H "Content-Type: application/json" \
  -d '{
    "stream_ref": "video_frames",
    "prompt": "Detect all people and vehicles",
    "fps": 2.0
  }'
```

### Check Job Status

```bash
curl "http://localhost:8000/api/v1/inference/status/550e8400-e29b-41d4-a716-446655440000"
```

**Response**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "stream_ref": "https://example.com/stream.m3u8",
  "prompt": "Detect all people and vehicles",
  "fps": 2.0,
  "created_at": 1705665600.0,
  "started_at": 1705665601.0,
  "completed_at": null,
  "frames_processed": 120,
  "results_count": 15,
  "error": null
}
```

### Get Results

```bash
curl "http://localhost:8000/api/v1/inference/results/550e8400-e29b-41d4-a716-446655440000?limit=10"
```

**Response**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "results": [
    {
      "predictions": [
        {
          "label": "detected_object_0",
          "confidence": 0.85,
          "bbox": [10, 10, 100, 100]
        }
      ],
      "confidence": [0.85],
      "metadata": {
        "num_frames": 16,
        "prompt": "Detect all people and vehicles",
        "timestamp": 1705665605.0
      }
    }
  ],
  "total_results": 15
}
```

### Cancel a Job

```bash
curl -X DELETE "http://localhost:8000/api/v1/inference/cancel/550e8400-e29b-41d4-a716-446655440000"
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

**Response**:
```json
{
  "prompt": "Detect only vehicles",
  "version": 2,
  "timestamp": 1705665600.0,
  "created_at": "2024-01-19T10:00:00Z",
  "updated_at": "2024-01-19T12:00:00Z"
}
```

### Get Current Prompt

```bash
curl "http://localhost:8000/api/v1/prompt/current"
```

### Get Prompt History

```bash
curl "http://localhost:8000/api/v1/prompt/history?limit=5"
```

### List All Jobs

```bash
# List all jobs
curl "http://localhost:8000/api/v1/inference/jobs"

# List only running jobs
curl "http://localhost:8000/api/v1/inference/jobs?status=running"
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

# In ModelDeployment class:
self.model = MyVLM()
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

## Ray Serve Scaling

Ray Serve allows you to scale deployments independently:

```python
# Scale model deployment
@serve.deployment(
    name="model",
    num_replicas=4,  # Scale to 4 replicas
    ray_actor_options={"num_gpus": 1}  # One GPU per replica
)
```

You can also use Ray Serve's autoscaling:

```python
from ray import serve

@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 10
    }
)
```

## Error Handling

The service includes comprehensive error handling with standardized error responses.

### Error Response Format

All errors follow this format:
```json
{
  "error": "Error message",
  "detail": "Additional error details (optional)"
}
```

### HTTP Status Codes

- `200 OK`: Successful request
- `202 Accepted`: Request accepted for processing (inference start)
- `400 Bad Request`: Invalid request parameters or validation error
- `404 Not Found`: Resource not found (job, prompt, etc.)
- `429 Too Many Requests`: Rate limit exceeded or max concurrent streams reached
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service not ready (health check)

### Common Error Scenarios

**Invalid Stream Reference**:
```json
{
  "error": "stream_ref cannot be empty",
  "detail": null
}
```

**Job Not Found**:
```json
{
  "error": "Job 550e8400-e29b-41d4-a716-446655440000 not found",
  "detail": null
}
```

**Maximum Concurrent Streams Reached**:
```json
{
  "error": "Maximum concurrent streams (10) reached",
  "detail": null
}
```

**Stream Connection Error**:
```json
{
  "error": "Failed to process HLS stream: Connection timeout",
  "detail": null
}
```

### Error Handling Features

- **Stream Errors**: Automatic retry and graceful degradation
- **Inference Errors**: Timeout handling and error reporting
- **Model Errors**: Clear error messages and logging
- **Validation Errors**: Detailed validation error responses
- **Timeout Handling**: Configurable timeouts for all operations

## Monitoring

### Health Checks

- `/health`: Overall service health
- `/health/ready`: Service readiness (model loaded, etc.)
- `/health/live`: Service liveness

### Logging

Logging is configured via `LOG_LEVEL` and `LOG_FORMAT`:
- `LOG_LEVEL`: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- `LOG_FORMAT`: `json` or `text`

### Ray Dashboard

Ray provides a built-in dashboard for monitoring:
```bash
# Access dashboard at http://localhost:8265
# (automatically available when Ray cluster is running)
```

## Performance Considerations

### Configuration Tuning

- **Concurrent Streams**: Configured via `MAX_CONCURRENT_STREAMS` (default: 10)
  - Increase for more parallel processing
  - Decrease if experiencing memory pressure
  
- **Batch Processing**: Configured via `MAX_BATCH_SIZE` (default: 8)
  - Larger batches = better GPU utilization but more memory
  - Smaller batches = less memory but potentially slower
  
- **Frame Sampling**: Adjustable FPS to balance accuracy vs. performance
  - Lower FPS (1-2) = faster processing, less accurate
  - Higher FPS (5-10) = more accurate, slower processing
  
- **Replicas**: Scale model deployment with `NUM_REPLICAS` for higher throughput
  - Each replica can handle requests independently
  - Requires sufficient GPU/CPU resources

### Expected Performance

**Typical Latency** (per frame batch):
- Frame extraction: 10-100ms
- Model inference: 50-500ms (depends on model)
- Total per batch: 60-600ms

**Throughput** (with default settings):
- ~2 FPS sampling = ~2 frames/second processed
- With 8-frame batches = ~1 batch every 4 seconds
- With 1 replica = ~15 batches/minute
- With 4 replicas = ~60 batches/minute

### Performance Optimization Tips

1. **Use GPU**: Set `DEVICE=cuda` for 10-100x speedup
2. **Increase Batch Size**: If memory allows, increase `MAX_BATCH_SIZE`
3. **Scale Replicas**: Add more model replicas for parallel processing
4. **Lower FPS**: Reduce `DEFAULT_FPS` for faster processing
5. **Use Autoscaling**: Enable Ray Serve autoscaling for variable load

## Development

### Project Structure

```
rayserve_service/
├── app/
│   ├── main.py              # Ray Serve application
│   ├── config.py            # Configuration
│   ├── models/              # Model abstraction
│   ├── services/            # Business logic
│   ├── api/                 # API handlers
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

## Production Deployment

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Ray Serve port
EXPOSE 8000

# Start Ray and deploy service
CMD ["sh", "-c", "ray start --head && serve run app.main:app --host 0.0.0.0 --port 8000"]
```

### Kubernetes Deployment

Example Kubernetes deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: free-roam-service
spec:
  replicas: 1
  selector:
    matchLabels:
      app: free-roam-service
  template:
    metadata:
      labels:
        app: free-roam-service
    spec:
      containers:
      - name: service
        image: your-registry/free-roam-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models/vlm"
        - name: DEVICE
          value: "cuda"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: free-roam-service
spec:
  selector:
    app: free-roam-service
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

### Production Checklist

- [ ] Set `LOG_LEVEL=INFO` or `WARNING` (not DEBUG)
- [ ] Configure `MODEL_PATH` to production model location
- [ ] Set appropriate `MAX_CONCURRENT_STREAMS` based on resources
- [ ] Configure Redis connection if using Redis streams
- [ ] Set up monitoring and alerting
- [ ] Configure health check endpoints for orchestration
- [ ] Set up log aggregation
- [ ] Configure resource limits (CPU, memory, GPU)
- [ ] Enable Ray Serve autoscaling if needed
- [ ] Set up backup and recovery procedures
- [ ] Configure CORS origins for production frontend
- [ ] Review and set security settings (API keys, etc.)

## Security Considerations

### Authentication

The service supports optional API key authentication via `API_KEY` environment variable. When set, clients should include the API key in requests (implementation depends on your security requirements).

### Network Security

- Use HTTPS in production (configure reverse proxy/load balancer)
- Restrict CORS origins to known frontend domains
- Use firewall rules to restrict access
- Consider using VPN or private networks for internal services

### Data Privacy

- Video streams may contain sensitive data
- Ensure compliance with data protection regulations
- Consider encryption for data in transit and at rest
- Implement data retention policies

### Best Practices

- Run service with least privilege user
- Keep dependencies updated
- Regularly audit logs for suspicious activity
- Use secrets management for sensitive configuration
- Implement rate limiting to prevent abuse

## Troubleshooting

### Model Not Loading

**Symptoms**: Service starts but model not initialized, health checks fail

**Solutions**:
- Check `MODEL_PATH` is correct and accessible
- Verify model files are readable
- Check device availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Review logs: `serve logs` or check Ray dashboard
- Ensure sufficient memory/GPU memory available

### Stream Connection Issues

**HLS Streams**:
- Verify URL is accessible: `curl -I https://example.com/stream.m3u8`
- Check stream is live (not ended)
- Verify FFmpeg is installed: `ffmpeg -version`
- Check network connectivity and firewall rules
- Review HLS stream format compatibility

**Redis Streams**:
- Verify Redis is running: `redis-cli ping`
- Check connection settings (`REDIS_HOST`, `REDIS_PORT`)
- Verify Redis password if required
- Check Redis stream key exists and has data
- Review Redis logs for connection errors

### Ray Serve Issues

**Service Not Starting**:
```bash
# Check Ray cluster status
ray status

# Check deployment status
serve status

# View detailed logs
serve logs
```

**Deployment Failures**:
- Check Ray cluster has sufficient resources
- Verify all dependencies are installed
- Review Ray dashboard at http://localhost:8265
- Check for port conflicts (default: 8000)

**Performance Issues**:
- Check Ray dashboard for resource utilization
- Review deployment replica counts
- Monitor GPU/CPU usage
- Check for memory leaks in logs

### Memory Issues

**Symptoms**: Out of memory errors, service crashes

**Solutions**:
- Reduce `MAX_BATCH_SIZE` (try 4 or 2)
- Reduce `MAX_CONCURRENT_STREAMS` (try 5)
- Lower `DEFAULT_FPS` (try 1.0)
- Reduce `NUM_REPLICAS` (if using multiple)
- Increase system memory or use smaller model
- Enable memory cleanup: check `MEMORY_CLEANUP_INTERVAL`

### Job Stuck in "Running" State

**Possible Causes**:
- Stream connection lost
- Model inference timeout
- Service crash during processing

**Solutions**:
- Check job status: `GET /api/v1/inference/status/{job_id}`
- Review service logs for errors
- Cancel and restart job if needed
- Check stream is still active

### High Latency

**Possible Causes**:
- Model too large for hardware
- Batch size too small
- Network latency to stream
- Insufficient GPU resources

**Solutions**:
- Use GPU if available: `DEVICE=cuda`
- Increase batch size if memory allows
- Reduce FPS for faster processing
- Scale with more replicas
- Use model quantization if supported

## Known Limitations

- **No Built-in Authentication**: API key support is optional and must be implemented in middleware
- **No API Documentation UI**: Unlike FastAPI, Ray Serve doesn't provide `/docs` endpoint
- **Job History**: Old jobs are kept in memory (consider implementing persistence)
- **Stream Reconnection**: Limited automatic reconnection for dropped streams
- **Model Hot-Swapping**: Model changes require service restart

## Roadmap

Potential future improvements:

- [ ] Persistent job storage (database backend)
- [ ] WebSocket support for real-time results
- [ ] Model hot-swapping without restart
- [ ] Enhanced stream reconnection logic
- [ ] Built-in authentication middleware
- [ ] API rate limiting per endpoint
- [ ] Metrics export (Prometheus format)
- [ ] Distributed tracing support
- [ ] Model versioning and A/B testing

## License

[Your License Here]

## Contributing

[Contributing Guidelines]

## Support

[Support Information]

## Changelog

### Version 1.0.0

- Initial release
- Ray Serve implementation
- HLS and Redis stream support
- Configurable frame sampling
- Prompt management with versioning
- Health monitoring endpoints
- Comprehensive error handling



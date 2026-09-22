"""Request logging and Prometheus metrics for the wafer classifier service.

Every request gets a request ID (returned as the X-Request-ID header and echoed
in the log line), and its method, path, status and latency are logged as one
JSON object per request — the shape most log aggregators (CloudWatch, Loki,
Datadog) expect without extra parsing rules.

Metrics are counters and histograms in the Prometheus text format, served at
/metrics. http_requests_total and http_request_duration_seconds are generic;
predictions_total and auto_accept_total are specific to this service, so a
dashboard can show prediction volume by pattern and the review workload
alongside plain HTTP health.
"""
import json
import logging
import sys
import time
import uuid

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("wafer_service")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.propagate = False

HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP requests", ["method", "path", "status"]
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "path"]
)
PREDICTIONS = Counter("predictions_total", "Wafers classified", ["pattern"])
AUTO_ACCEPT = Counter(
    "auto_accept_total", "Predictions by review decision", ["decision"]
)


def record_predictions(results):
    """Call once per response with a list of describe() dicts, to update business metrics."""
    for r in results:
        PREDICTIONS.labels(pattern=r["pattern"]).inc()
        AUTO_ACCEPT.labels(decision="accept" if r["auto_accept"] else "review").inc()


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            raise
        finally:
            elapsed = time.perf_counter() - t0
            path = request.url.path
            HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            HTTP_LATENCY.labels(request.method, path).observe(elapsed)
            logger.info(
                json.dumps(
                    {
                        "request_id": request_id,
                        "method": request.method,
                        "path": path,
                        "status": status,
                        "latency_ms": round(elapsed * 1000, 3),
                    }
                )
            )
        response.headers["X-Request-ID"] = request_id
        return response


def metrics_response():
    return generate_latest(), CONTENT_TYPE_LATEST

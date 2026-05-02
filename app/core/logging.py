import logging
import logging.config

from prometheus_client import REGISTRY, Counter, Histogram, start_http_server

from app.config.settings import settings


class Telemetry:
    """Orchestrates Prometheus metrics for the RAG engine."""

    def __init__(self, registry=REGISTRY):
        # 1. Latency Histograms
        self.query_latency = Histogram(
            "rag_query_latency_seconds",
            "Latency of RAG /ask queries in seconds",
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")),
            registry=registry,
        )

        # 2. Token Usage Counters
        self.token_usage = Counter(
            "rag_token_usage_total",
            "Total tokens consumed from Gemini API",
            ["type"],  # prompt or completion
            registry=registry,
        )

        # 3. Error Counters
        self.storage_errors = Counter(
            "rag_storage_errors_total",
            "Total storage service failures",
            ["service"],  # minio or postgres
            registry=registry,
        )

        self.llm_errors = Counter(
            "rag_llm_errors_total",
            "Total LLM service failures",
            ["model"],  # embed or generate
            registry=registry,
        )

        self.processing_errors = Counter(
            "rag_processing_errors_total",
            "Total document processing failures",
            ["stage"],  # parse or chunk
            registry=registry,
        )

    def track_query(self, duration: float):
        self.query_latency.observe(duration)

    def track_tokens(self, prompt: int, completion: int):
        logging.info(f"[Telemetry] Tokens: p={prompt}, c={completion}")
        self.token_usage.labels(type="prompt").inc(prompt)
        self.token_usage.labels(type="completion").inc(completion)

    def start_telemetry_server(self, port: int = 8001):
        """Starts a standalone HTTP server for Prometheus scraping (used in workers)."""
        start_http_server(port)
        logging.info(f"Telemetry server started on port {port}")


# Singleton instance
telemetry = Telemetry()


def setup_logging():
    """Configures structured logging based on environment."""
    json_log_format = (
        '{"time": "%(asctime)s", "name": "%(name)s", '
        '"level": "%(levelname)s", "message": "%(message)s"}'
    )

    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        if settings.ENVIRONMENT != "production"
        else json_log_format
    )

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": log_format},
            "json": {
                "format": (
                    '{"time": "%(asctime)s", "level": "%(levelname)s", '
                    '"message": "%(message)s"}'
                )
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json"
                if settings.ENVIRONMENT == "production"
                else "standard",
            }
        },
        "root": {"handlers": ["console"], "level": "INFO"},
    }
    logging.config.dictConfig(logging_config)

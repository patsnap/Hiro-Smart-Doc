import os

from gunicorn.arbiter import Arbiter  # type: ignore

from hiro_smart_doc.common.log import service_logconfig_dict
# from hiro_smart_doc.common.modelhub_init import modelhub_init

# ==================== Gunicorn basic configs ====================


bind = f"0.0.0.0:{os.getenv('RD_INTERNAL_PORT', 8000)}"
timeout = int(os.getenv("RD_STARTUP_TIMEOUT", 30))
workers = int(os.getenv("RD_WORKERS", 1))
loglevel = os.getenv("RD_LOG_LEVEL", "info")
logconfig_dict = service_logconfig_dict
worker_class = "uvicorn.workers.UvicornWorker"
preload_app = False  # No preload_app, because CUDA ctx need to init in each process
max_requests = 1_000_000
max_requests_jitter = 10_000


# ==================== Startup hook ====================


def on_starting(server: Arbiter) -> None:
    server.log.info("Executing Arbiter `on_starting` script...")
    server.log.info(f"{os.environ=}")
    # modelhub_init()

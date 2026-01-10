import os
import logging
import modal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Modal app and image
# dependencies found in pyproject.toml
image = (
            modal.Image.debian_slim(python_version="3.12")
            .apt_install("ffmpeg", "libsm6", "libxext6") # for video processing
            .uv_sync(extra_options="--no-dev")  # exclude dev dependencies to avoid package conflicts
            .add_local_python_source(  # add all local modules here
                "preprocessing",
                "embeddings",
                "models",
                "database",
                "search",
                "api",
                "background_tasks",
            )
        )

# Environment: "dev" (default) or "prod" (set via ENVIRONMENT variable)
env = os.environ.get("ENVIRONMENT", "dev")
if env not in ["dev", "prod", "staging"]:
    raise ValueError(f"Invalid ENVIRONMENT value: {env}. Must be one of: dev, prod, staging")
logger.info(f"Starting Modal app in '{env}' environment")

IS_INTERNAL_ENV = env in ["dev", "staging"]

# Create Modal app
app = modal.App(
    name=env,
    image=image,
    secrets=[modal.Secret.from_name(env)]
)


@app.cls(cpu=4.0, timeout=600, min_containers=1)
class Server:

    @modal.enter()
    def startup(self):
        """
            Startup logic. This runs once when the container starts.
            Here is where you would instantiate classes and load models that are
            reused across multiple requests to avoid reloading them each time.
        """

        # Import local module inside class
        import os
        from datetime import datetime, timezone

        # Import classes here
        from database.pinecone_connector import PineconeConnector
        from database.job_store_connector import JobStoreConnector
        from search.searcher import Searcher
        from database.r2_connector import R2Connector
        from api.fastapi_router import FastAPIRouter
        from fastapi import FastAPI
        from background_tasks.tasks import BackgroundTasks

        logger.info(f"Container starting up! Environment = {env}")
        self.start_time = datetime.now(timezone.utc)

        # Get environment variables (TODO: abstract to config module)
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
        if not R2_ACCOUNT_ID:
            raise ValueError("R2_ACCOUNT_ID not found in environment variables")
        
        R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
        if not R2_ACCESS_KEY_ID:
            raise ValueError("R2_ACCESS_KEY_ID not found in environment variables")
        
        R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
        if not R2_SECRET_ACCESS_KEY:
            raise ValueError("R2_SECRET_ACCESS_KEY not found in environment variables")
        
        ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
        if ENVIRONMENT not in ["dev", "prod", "staging"]:
            raise ValueError(f"Invalid ENVIRONMENT value: {ENVIRONMENT}. Must be one of: dev, prod, staging")
        logger.info(f"Running in environment: {ENVIRONMENT}")

        # Select Pinecone index based on environment
        pinecone_index = f"{ENVIRONMENT}-chunks"
        logger.info(f"Using Pinecone index: {pinecone_index}")

        # Instantiate classes
        self.pinecone_connector = PineconeConnector(api_key=PINECONE_API_KEY, index_name=pinecone_index)
        self.job_store = JobStoreConnector(dict_name="clipabit-jobs")

        self.r2_connector = R2Connector(
            account_id=R2_ACCOUNT_ID,
            access_key_id=R2_ACCESS_KEY_ID,
            secret_access_key=R2_SECRET_ACCESS_KEY,
            environment=ENVIRONMENT
        )

        self.searcher = Searcher(
            api_key=PINECONE_API_KEY,
            index_name=pinecone_index,
            r2_connector=self.r2_connector
        )

        # Create the FastAPI app
        self.fastapi_app = FastAPI()
        self.background_tasks = BackgroundTasks()
        self.api = FastAPIRouter(self, self.background_tasks, IS_INTERNAL_ENV)
        self.fastapi_app.include_router(self.api.router)

        logger.info("Container modules initialized and ready!")

        print(f"[Container] Started at {self.start_time.isoformat()}")

    @modal.asgi_app()
    def asgi_app(self):
        """
        Exposes the FastAPI app as a web endpoint.
        """
        return self.fastapi_app


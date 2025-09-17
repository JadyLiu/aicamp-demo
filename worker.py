import os
import asyncio
import logging
import abraxas
from dotenv import load_dotenv

from workflow.workflows import ALL_WORKFLOWS

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logging.getLogger("docai.workflow").setLevel(logging.DEBUG)

if __name__ == "__main__":
    asyncio.run(
        abraxas.run_worker(
            workflows=ALL_WORKFLOWS,
            api_key=os.getenv("ABRAXAS_API_KEY"),
            namespace=os.getenv("TEMPORAL_NAMESPACE"),
        )
    )
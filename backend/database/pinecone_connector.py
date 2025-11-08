from pinecone import Pinecone
import os


class PineconeConnector:
    def __init__(self, index_name: str, environment: str = "us-east1-aws"):
        api_key = os.getenv("PINECONE_API_KEY")
        self.client = Pinecone(api_key=api_key, environment=environment)
        self.index_name = index_name

    def connect_to_pinecone(self):
        """
        Initialize and connect to Pinecone.

        Returns:
            Pinecone: Initialized Pinecone client
        """
        pc = Pinecone(api_key=self.client.api_key, environment=self.client.environment)
        return pc
    
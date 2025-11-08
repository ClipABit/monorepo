from pinecone import Pinecone
import os
import logging


class PineconeConnector:
    """
    Pinecone Connector Class for managing Pinecone interactions. 
    """
    def __init__(self, api_key: str, index_name: str):
        self.client = Pinecone(api_key=api_key)
        self.index_name = index_name
    
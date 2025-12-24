"""
Video Deletion Service - Core orchestrator for video deletion operations.

This service coordinates deletion across multiple storage systems (R2 and Pinecone)
while enforcing environment-based access controls and providing comprehensive logging.
"""

import logging
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from database.r2_connector import R2Connector, R2DeletionResult
from database.pinecone_connector import PineconeConnector, PineconeDeletionResult

logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    """Result of deletion verification operations."""
    r2_verified: bool
    pinecone_verified: bool
    verification_errors: list[str]
    timestamp: datetime

@dataclass
class DeletionResult:
    """Complete result of video deletion operation."""
    success: bool
    hashed_identifier: str
    namespace: str
    r2_result: R2DeletionResult
    pinecone_result: PineconeDeletionResult
    verification_result: Optional[VerificationResult] = None
    error_message: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class VideoDeletionService:
    """
    Core orchestrator for video deletion operations.
    
    Coordinates deletion across R2 storage and Pinecone database while enforcing
    environment-based access controls and providing comprehensive logging.
    """

    def __init__(
        self,
        r2_connector: R2Connector,
        pinecone_connector: PineconeConnector,
        environment: str
    ):
        """
        Initialize the video deletion service.

        Args:
            r2_connector: Configured R2 connector instance
            pinecone_connector: Configured Pinecone connector instance
            environment: Current environment ("dev" or "prod")
        """
        self.r2_connector = r2_connector
        self.pinecone_connector = pinecone_connector
        self.environment = environment
        
        logger.info(f"VideoDeletionService initialized for environment: {environment}")

    def _validate_environment(self) -> bool:
        """
        Validate that deletion is allowed in the current environment.
        
        Returns:
            bool: True if deletion is allowed, False otherwise
        """
        is_dev = self.environment == "dev"
        logger.info(f"Environment validation: {self.environment} -> deletion_allowed={is_dev}")
        
        # Log security event for production environment access attempts
        if not is_dev:
            logger.warning("Production environment access attempt blocked for deletion operations")
            
        return is_dev

    def _decode_identifier(self, hashed_identifier: str) -> Tuple[str, str, str]:
        """
        Decode hashed identifier into its components.
        
        Args:
            hashed_identifier: Base64-encoded identifier
            
        Returns:
            Tuple[str, str, str]: (bucket_name, namespace, filename)
            
        Raises:
            ValueError: If identifier format is invalid
        """
        try:
            return self.r2_connector._decode_path(hashed_identifier)
        except Exception as e:
            logger.error(f"Failed to decode identifier {hashed_identifier}: {e}")
            raise ValueError(f"Invalid hashed identifier format: {e}")

    async def _delete_from_r2(self, hashed_identifier: str) -> R2DeletionResult:
        """
        Delete video file from R2 storage.
        
        Args:
            hashed_identifier: Base64-encoded identifier
            
        Returns:
            R2DeletionResult: Result of R2 deletion operation
        """
        logger.info(f"Starting R2 deletion for identifier: {hashed_identifier}")
        
        try:
            result = self.r2_connector.delete_video_file(hashed_identifier)
            
            if result.success:
                if result.file_existed:
                    logger.info(f"R2 deletion successful: {result.key} ({result.bytes_deleted} bytes)")
                else:
                    logger.info(f"R2 deletion completed: file {result.key} did not exist")
            else:
                logger.error(f"R2 deletion failed: {result.error_message}")
                
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error during R2 deletion: {e}")
            return R2DeletionResult(
                success=False,
                bucket="",
                key="",
                file_existed=False,
                error_message=f"Unexpected R2 deletion error: {str(e)}"
            )

    async def _delete_from_pinecone(self, hashed_identifier: str, namespace: str) -> PineconeDeletionResult:
        """
        Delete video chunks from Pinecone database.
        
        Args:
            hashed_identifier: Base64-encoded identifier
            namespace: Pinecone namespace
            
        Returns:
            PineconeDeletionResult: Result of Pinecone deletion operation
        """
        logger.info(f"Starting Pinecone deletion for identifier: {hashed_identifier}, namespace: {namespace}")
        
        try:
            video_metadata = {"hashed_identifier": hashed_identifier}
            result = self.pinecone_connector.delete_by_metadata(video_metadata, namespace)
            
            if result.success:
                logger.info(f"Pinecone deletion successful: {result.chunks_deleted} chunks deleted")
            else:
                logger.error(f"Pinecone deletion failed: {result.error_message}")
                
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error during Pinecone deletion: {e}")
            return PineconeDeletionResult(
                success=False,
                chunks_found=0,
                chunks_deleted=0,
                chunk_ids=[],
                namespace=namespace,
                error_message=f"Unexpected Pinecone deletion error: {str(e)}"
            )

    async def _verify_deletion(self, hashed_identifier: str, namespace: str) -> VerificationResult:
        """
        Verify that deletion was successful in both storage systems.
        
        Args:
            hashed_identifier: Base64-encoded identifier
            namespace: Pinecone namespace
            
        Returns:
            VerificationResult: Result of verification checks
        """
        logger.info(f"Starting deletion verification for identifier: {hashed_identifier}")
        
        verification_errors = []
        
        # Verify R2 deletion
        try:
            r2_verified = self.r2_connector.verify_deletion(hashed_identifier)
            if not r2_verified:
                verification_errors.append("R2 verification failed: file still exists")
        except Exception as e:
            r2_verified = False
            verification_errors.append(f"R2 verification error: {str(e)}")
        
        # Verify Pinecone deletion
        try:
            remaining_chunks = self.pinecone_connector.find_chunks_by_video(hashed_identifier, namespace)
            pinecone_verified = len(remaining_chunks) == 0
            if not pinecone_verified:
                verification_errors.append(f"Pinecone verification failed: {len(remaining_chunks)} chunks still exist")
        except Exception as e:
            pinecone_verified = False
            verification_errors.append(f"Pinecone verification error: {str(e)}")
        
        result = VerificationResult(
            r2_verified=r2_verified,
            pinecone_verified=pinecone_verified,
            verification_errors=verification_errors,
            timestamp=datetime.utcnow()
        )
        
        if r2_verified and pinecone_verified:
            logger.info("Deletion verification successful: both systems confirmed")
        else:
            logger.warning(f"Deletion verification issues: {verification_errors}")
            
        return result

    async def delete_video(self, hashed_identifier: str, namespace: str = "web-demo") -> DeletionResult:
        """
        Delete a video and all associated data from both storage systems.
        
        Args:
            hashed_identifier: Base64-encoded identifier of the video
            namespace: Namespace for data partitioning (default: "web-demo")
            
        Returns:
            DeletionResult: Complete result of the deletion operation
        """
        logger.info(f"Video deletion request: identifier={hashed_identifier}, namespace={namespace}")
        
        # Environment validation
        if not self._validate_environment():
            error_msg = f"Deletion not allowed in {self.environment} environment"
            logger.warning(f"Unauthorized deletion attempt in {self.environment} environment: identifier={hashed_identifier}, namespace={namespace}")
            
            # Return failed result with empty sub-results
            return DeletionResult(
                success=False,
                hashed_identifier=hashed_identifier,
                namespace=namespace,
                r2_result=R2DeletionResult(
                    success=False,
                    bucket="",
                    key="",
                    file_existed=False,
                    error_message="Not attempted due to environment restriction"
                ),
                pinecone_result=PineconeDeletionResult(
                    success=False,
                    chunks_found=0,
                    chunks_deleted=0,
                    chunk_ids=[],
                    namespace=namespace,
                    error_message="Not attempted due to environment restriction"
                ),
                error_message=error_msg
            )

        # Identifier validation
        try:
            bucket_name, decoded_namespace, filename = self._decode_identifier(hashed_identifier)
            logger.info(f"Decoded identifier: bucket={bucket_name}, namespace={decoded_namespace}, filename={filename}")
        except ValueError as e:
            error_msg = f"Invalid identifier: {str(e)}"
            logger.error(error_msg)
            
            return DeletionResult(
                success=False,
                hashed_identifier=hashed_identifier,
                namespace=namespace,
                r2_result=R2DeletionResult(
                    success=False,
                    bucket="",
                    key="",
                    file_existed=False,
                    error_message="Invalid identifier format"
                ),
                pinecone_result=PineconeDeletionResult(
                    success=False,
                    chunks_found=0,
                    chunks_deleted=0,
                    chunk_ids=[],
                    namespace=namespace,
                    error_message="Invalid identifier format"
                ),
                error_message=error_msg
            )

        # Perform parallel deletion operations
        logger.info("Starting parallel deletion operations")
        try:
            r2_task = self._delete_from_r2(hashed_identifier)
            pinecone_task = self._delete_from_pinecone(hashed_identifier, namespace)
            
            r2_result, pinecone_result = await asyncio.gather(r2_task, pinecone_task)
            
        except Exception as e:
            error_msg = f"Error during parallel deletion: {str(e)}"
            logger.error(error_msg)
            
            return DeletionResult(
                success=False,
                hashed_identifier=hashed_identifier,
                namespace=namespace,
                r2_result=R2DeletionResult(
                    success=False,
                    bucket="",
                    key="",
                    file_existed=False,
                    error_message="Parallel deletion failed"
                ),
                pinecone_result=PineconeDeletionResult(
                    success=False,
                    chunks_found=0,
                    chunks_deleted=0,
                    chunk_ids=[],
                    namespace=namespace,
                    error_message="Parallel deletion failed"
                ),
                error_message=error_msg
            )

        # Check if R2 deletion failed - if so, don't proceed with verification
        if not r2_result.success and r2_result.error_message and "not found" not in r2_result.error_message.lower():
            logger.error("R2 deletion failed with error, skipping verification")
            return DeletionResult(
                success=False,
                hashed_identifier=hashed_identifier,
                namespace=namespace,
                r2_result=r2_result,
                pinecone_result=pinecone_result,
                error_message="R2 deletion failed, operation incomplete"
            )

        # Determine overall success
        overall_success = (
            (r2_result.success or (not r2_result.file_existed)) and
            (pinecone_result.success or (pinecone_result.chunks_found == 0))
        )

        # Perform verification if deletion was successful
        verification_result = None
        if overall_success:
            try:
                verification_result = await self._verify_deletion(hashed_identifier, namespace)
                if verification_result.verification_errors:
                    overall_success = False
            except Exception as e:
                logger.error(f"Verification failed: {e}")
                overall_success = False

        # Log final result with detailed summary
        if overall_success:
            logger.info(f"Video deletion completed successfully: identifier={hashed_identifier}, namespace={namespace}, "
                       f"r2_success={r2_result.success}, pinecone_success={pinecone_result.success}")
        else:
            logger.error(f"Video deletion failed or incomplete: identifier={hashed_identifier}, namespace={namespace}, "
                        f"r2_success={r2_result.success}, pinecone_success={pinecone_result.success}, "
                        f"error={error_msg if 'error_msg' in locals() else 'verification_failed'}")

        return DeletionResult(
            success=overall_success,
            hashed_identifier=hashed_identifier,
            namespace=namespace,
            r2_result=r2_result,
            pinecone_result=pinecone_result,
            verification_result=verification_result,
            error_message=None if overall_success else "Deletion incomplete or failed verification"
        )
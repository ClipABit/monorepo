import os
import logging
import boto3
from botocore.exceptions import ClientError
from typing import Optional, Tuple
import base64
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class R2DeletionResult:
    """Result of R2 deletion operation."""
    success: bool
    bucket: str
    key: str
    file_existed: bool
    error_message: Optional[str] = None
    bytes_deleted: Optional[int] = None

class R2Connector:
    """
    R2 Connector Class for managing Cloudflare R2 storage interactions.
    Supports uploading videos to folders and retrieving them by hashed identifiers.

    Web client videos are stored under folders named after the web client name.
    Plugin client videos are stored under folders named after the user/project.
    """
    
    def __init__(
        self,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        environment: str = "dev"  # dev or prod
    ):
        """
        Initialize R2 connector with bucket credentials.

        Args:
            account_id: Cloudflare account ID
            access_key_id: R2 access key ID
            secret_access_key: R2 secret access key
            environment: Environment name (dev/prod) which maps directly to bucket name
        """
        self.bucket_name = environment
        self.endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        
        # Initialize S3 client (R2 is S3-compatible)
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            endpoint_url=self.endpoint_url,
            region_name='auto'  # R2 uses 'auto' for region
        )
        
        logger.info(f"Initialized R2Connector for bucket: {self.bucket_name}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent directory traversal attacks.
        
        Args:
            filename: Original filename
        
        Returns:
            str: Sanitized filename
        """
        filename = os.path.basename(filename)  # Remove any path traversal components
        filename = filename.replace(" ", "_")  # Replace spaces with underscores
        if not filename:
            raise ValueError("Sanitized filename is empty. Please provide a valid filename.")
        return filename

    def _encode_path(self, bucket_name: str, namespace: str, filename: str) -> str:
        """
        Encode bucket name, user ID, and filename into a URL-safe identifier.
        
        Args:
            bucket_name: Name of the R2 bucket
            namespace: Namespace (web client name or user/project)
            filename: Name of the video file
        
        Returns:
            str: URL-safe base64-encoded identifier
        """
        # Create the full path string
        path = f"{bucket_name}/{namespace}/{filename}"
        
        # Convert to base64 for URL-safe representation
        encoded = base64.urlsafe_b64encode(path.encode('utf-8')).decode('utf-8').rstrip('=')
        return encoded
    
    def _decode_path(self, identifier: str) -> Tuple[str, str, str]:
        """
        Decode the URL-safe identifier back into bucket name, user ID, and filename.
        
        Args:
            identifier: The base64-encoded identifier
        Returns:
            Tuple[str, str, str]: (bucket_name, namespace, filename)
        """
        # Add padding if necessary
        padding = '=' * (-len(identifier) % 4)
        decoded_bytes = base64.urlsafe_b64decode(identifier + padding)
        decoded_str = decoded_bytes.decode('utf-8')
        
        # Split into components
        parts = decoded_str.split('/', 2)
        if len(parts) != 3:
            raise ValueError("Invalid identifier format")
        
        bucket_name, namespace, filename = parts
        return bucket_name, namespace, filename
    
    def _get_object_key_from_identifier(self, identifier: str) -> Optional[str]:
        """
        Decode identifier and validate bucket name matches, then return object key.
        
        Args:
            identifier: The base64-encoded identifier
        
        Returns:
            str: The object key if valid, None otherwise
        """
        try:
            # Decode the identifier
            bucket_name, namespace, filename = self._decode_path(identifier)
            
            # Validate bucket name matches
            if bucket_name != self.bucket_name:
                logger.warning(f"Bucket mismatch: expected {self.bucket_name}, got {bucket_name}")
                return None
            
            # Construct object key
            object_key = f"{namespace}/{filename}"
            logger.info(f"Decoded identifier to object key: {object_key}")
            return object_key
            
        except (ValueError, Exception) as e:
            logger.error(f"Error decoding identifier {identifier}: {e}")
            return None
        
    def _determine_content_type(self, filename: str) -> str:
        """
        Determine the MIME type of a file based on its extension.
        
        Args:
            filename: Name of the file
        
        Returns:
            str: MIME type string
        """
        ext = os.path.splitext(filename)[1].lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm'
        }
        return mime_types.get(ext, 'application/octet-stream')
    
    def upload_video(
        self,
        video_data: bytes,
        filename: str,
        namespace: str = "__default__" # corresponds to pinecone namespace
    ) -> Tuple[bool, str]:
        """
        Upload a video to R2 storage and return a hashed identifier.
        
        Args:
            video_data: The video file as bytes
            filename: Name of the video file
            namespace: Namespace to organize videos (default: "__default__")
        
        Returns:
            Tuple[bool, str]: (Success flag, hashed identifier or error message)
        """
        try:
            filename = self._sanitize_filename(filename)
            
            # Append timestamp to filename to ensure uniqueness
            import time
            filename = f"{int(time.time())}_{filename}"
            
            # Create encoded identifier
            identifier = self._encode_path(self.bucket_name, namespace, filename)
            
            # Determine file MIME type
            content_type = self._determine_content_type(filename)

            # Construct the object key (namespace/filename)
            object_key = f"{namespace}/{filename}"

            # Upload the video
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=video_data,
                ContentType=content_type
            )
            
            logger.info(f"Uploaded {filename} to R2 with identifier: {identifier}")
            return True, identifier
            
        except Exception as e:
            logger.error(f"Error uploading video to R2: {e}")
            return False, str(e)
    
    def fetch_video(self, identifier: str) -> Optional[bytes]:
        """
        Fetch a video from R2 storage using its identifier.
        
        Args:
            identifier: The base64-encoded identifier of the video
        
        Returns:
            bytes: The video file as bytes, or None if fetch failed
        """
        try:
            # Get object key from identifier
            object_key = self._get_object_key_from_identifier(identifier)
            if not object_key:
                return None
            
            # Fetch the object
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            
            video_data = response['Body'].read()
            logger.info(f"Fetched video with identifier {identifier}: ({len(video_data)} bytes)")
            return video_data
            
        except Exception as e:
            logger.error(f"Error fetching video from R2: {e}")
            return None
    
    def generate_presigned_url(self, identifier: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for temporary access to a video using its identifier.
        Validates that the bucket name in the identifier matches the connector's bucket.
        
        Args:
            identifier: The base64-encoded identifier of the video
            expiration: URL expiration time in seconds (default: 1 hour)
        
        Returns:
            str: Presigned URL that can be used to access the video directly, or None if failed
        """
        try:
            # Get object key from identifier
            object_key = self._get_object_key_from_identifier(identifier)
            if not object_key:
                logger.warning(f"Cannot generate presigned URL: invalid identifier {identifier}")
                return None
            
            # Generate presigned URL
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            
            logger.info(f"Generated presigned URL for identifier: {identifier}")
            return presigned_url
            
        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None

    def delete_video(self, identifier: str) -> bool:
        """
        Delete a video from R2 storage using its identifier.
        
        Args:
            identifier: The base64-encoded identifier of the video
        
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Get object key from identifier
            object_key = self._get_object_key_from_identifier(identifier)
            if not object_key:
                logger.warning(f"Cannot delete video: invalid identifier {identifier}")
                return False
            
            # Delete the object
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            
            logger.info(f"Deleted video with identifier: {identifier}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting video from R2: {e}")
            return False

    def fetch_all_video_data(self, namespace: str = "__default__", expiration: int = 3600) -> list[dict]:
        """
        Fetch all video data for a namespace, including filenames, identifiers, and presigned URLs.
        For WEB this will be the web client name, and for PLUGIN this will be user/project
        
        Args:
            namespace: The namespace name
            expiration: URL expiration time in seconds (default: 1 hour)
        
        Returns:
            list[dict]: List of dictionaries containing file_name, hashed_identifier, and presigned_url
        """
        video_data_list = []
        try:
            # Ensure namespace ends with /
            prefix = f"{namespace}/"
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    object_key = obj['Key']
                    
                    # Skip if it's just the folder placeholder itself
                    if object_key == prefix:
                        continue
                        
                    try:
                        # Extract filename from object key
                        # object_key is namespace/filename
                        filename = object_key.split('/', 1)[1]
                        
                        # Generate hashed identifier
                        identifier = self._encode_path(self.bucket_name, namespace, filename)
                        
                        # Generate presigned URL directly from object key
                        url = self.s3_client.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': self.bucket_name, 'Key': object_key},
                            ExpiresIn=expiration
                        )
                        
                        if url:
                            video_data_list.append({
                                "file_name": filename,
                                "hashed_identifier": identifier,
                                "presigned_url": url
                            })
                    except ClientError as e:
                        logger.error(f"Error processing video {object_key}: {e}")
            
            logger.info(f"Fetched data for {len(video_data_list)} videos for user {namespace}")
            return video_data_list
            
        except Exception as e:
            logger.error(f"Error listing objects for user {namespace}: {e}")
            return []

    def delete_video_file(self, hashed_identifier: str) -> R2DeletionResult:
        """
        Delete video file from R2 storage using its hashed identifier.
        
        Args:
            hashed_identifier: The base64-encoded identifier of the video
        
        Returns:
            R2DeletionResult: Result of the deletion operation
        """
        try:
            # Get object key from identifier
            object_key = self._get_object_key_from_identifier(hashed_identifier)
            if not object_key:
                return R2DeletionResult(
                    success=False,
                    bucket=self.bucket_name,
                    key="",
                    file_existed=False,
                    error_message=f"Invalid hashed identifier: {hashed_identifier}"
                )
            
            # Check if file exists and get size before deletion
            file_existed = False
            bytes_deleted = None
            try:
                response = self.s3_client.head_object(
                    Bucket=self.bucket_name,
                    Key=object_key
                )
                file_existed = True
                bytes_deleted = response.get('ContentLength', 0)
                logger.info(f"Found video file {object_key} ({bytes_deleted} bytes) for deletion")
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.info(f"Video file {object_key} does not exist in R2")
                    return R2DeletionResult(
                        success=True,  # Not an error if file doesn't exist
                        bucket=self.bucket_name,
                        key=object_key,
                        file_existed=False,
                        error_message="File not found in R2 storage"
                    )
                else:
                    raise e
            
            # Delete the object
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            
            logger.info(f"Successfully deleted video file {object_key} from R2")
            return R2DeletionResult(
                success=True,
                bucket=self.bucket_name,
                key=object_key,
                file_existed=file_existed,
                bytes_deleted=bytes_deleted
            )
            
        except ClientError as e:
            error_msg = f"Error deleting video from R2: {e}"
            logger.error(error_msg)
            return R2DeletionResult(
                success=False,
                bucket=self.bucket_name,
                key=object_key if 'object_key' in locals() else "",
                file_existed=False,
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error during R2 deletion: {e}"
            logger.error(error_msg)
            return R2DeletionResult(
                success=False,
                bucket=self.bucket_name,
                key="",
                file_existed=False,
                error_message=error_msg
            )
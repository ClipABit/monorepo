import os
import math
import logging
import time
import boto3
from botocore.exceptions import ClientError
from typing import Optional, Tuple, List
import base64

from database.url_cache_connector import UrlCacheConnector

logger = logging.getLogger(__name__)

DEFAULT_PRESIGNED_URL_TTL = 60 * 60  # 1 hour

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

        self._video_cache = UrlCacheConnector(environment=environment)
    
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
    
    def generate_presigned_url(
        self,
        identifier: str,
        expiration: int = 3600,
        validate_exists: bool = False,
    ) -> Optional[str]:
        """
        Generate a presigned URL for temporary access to a video using its identifier.
        Validates that the bucket name in the identifier matches the connector's bucket.
        
        Args:
            identifier: The base64-encoded identifier of the video
            expiration: URL expiration time in seconds (default: 1 hour)
            validate_exists: When True, perform a HEAD request to ensure the object exists before signing
        
        Returns:
            str: Presigned URL that can be used to access the video directly, or None if failed
        """
        try:
            # Get object key from identifier
            object_key = self._get_object_key_from_identifier(identifier)
            if not object_key:
                logger.warning(f"Cannot generate presigned URL: invalid identifier {identifier}")
                return None
            
            # Optionally validate the object exists before signing
            if validate_exists:
                try:
                    self.s3_client.head_object(
                        Bucket=self.bucket_name,
                        Key=object_key,
                    )
                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code") if getattr(e, "response", None) else None
                    if error_code in {"404", "NoSuchKey", "NotFound"}:
                        logger.warning(
                            "Cannot generate presigned URL: object missing for identifier %s (key=%s)",
                            identifier,
                            object_key,
                        )
                        return None
                    logger.error(
                        "Error validating object existence for identifier %s: %s",
                        identifier,
                        e,
                    )
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

    CURSOR_PREFIX = "cursor:"

    @staticmethod
    def _encode_cursor_token(object_key: str) -> str:
        return f"{R2Connector.CURSOR_PREFIX}{base64.urlsafe_b64encode(object_key.encode('utf-8')).decode('utf-8')}"

    @staticmethod
    def _decode_cursor_token(token: str) -> Optional[str]:
        if not token.startswith(R2Connector.CURSOR_PREFIX):
            return None
        raw = token[len(R2Connector.CURSOR_PREFIX):]
        try:
            padding = '=' * (-len(raw) % 4)
            return base64.urlsafe_b64decode(raw + padding).decode('utf-8')
        except Exception:
            return None

    def fetch_video_page(
        self,
        namespace: str = "__default__",
        page_size: int = 20,
        continuation_token: Optional[str] = None,
        expiration: int = DEFAULT_PRESIGNED_URL_TTL,
    ) -> Tuple[List[dict], Optional[str]]:
        """Fetch a single page of video metadata from R2.

        Returns a tuple of (videos, next_continuation_token).
        """
        try:
            if page_size <= 0:
                raise ValueError("page_size must be a positive integer")

            prefix = f"{namespace}/"
            params = {
                "Bucket": self.bucket_name,
                "Prefix": prefix,
                "MaxKeys": min(page_size + 1, 1000),
            }

            if continuation_token:
                cursor_key = self._decode_cursor_token(continuation_token)
                if cursor_key:
                    params["StartAfter"] = cursor_key
                else:
                    params["ContinuationToken"] = continuation_token

            response = self.s3_client.list_objects_v2(**params)

            contents = response.get("Contents", [])
            filtered: List[dict] = []

            for obj in contents:
                object_key = obj.get("Key")
                if not object_key or object_key == prefix:
                    continue
                filtered.append(obj)

            has_more_flag = response.get("IsTruncated", False)
            items = filtered[:page_size]
            has_more = has_more_flag or len(filtered) > page_size

            videos: List[dict] = []
            for obj in items:
                object_key = obj.get("Key")
                try:
                    filename = object_key.split('/', 1)[1]
                    identifier = self._encode_path(self.bucket_name, namespace, filename)
                    url = self.s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': self.bucket_name, 'Key': object_key},
                        ExpiresIn=expiration,
                    )
                    if url:
                        videos.append({
                            "file_name": filename,
                            "hashed_identifier": identifier,
                            "presigned_url": url,
                        })
                except ClientError as e:
                    logger.error(f"Error processing video {object_key}: {e}")

            next_token = response.get("NextContinuationToken") if has_more_flag else None
            if not next_token and has_more and videos:
                next_token = self._encode_cursor_token(items[-1].get("Key"))

            logger.info(
                "Fetched %s video objects for namespace %s (has_more=%s)",
                len(videos),
                namespace,
                bool(next_token),
            )
            return videos, next_token

        except Exception as e:
            logger.error(
                "Error fetching video page for namespace %s (token=%s): %s",
                namespace,
                continuation_token,
                e,
            )
            return [], None

    def list_videos_page(
        self,
        namespace: str = "__default__",
        page_size: int = 20,
        continuation_token: Optional[str] = None,
    ) -> Tuple[List[dict], Optional[str], int, int]:
        normalized_token = continuation_token or None
        videos: List[dict] = []
        next_token: Optional[str] = None
        total_videos: Optional[int] = None
        cache_hit = False

        if self._video_cache:
            cached = self._video_cache.get_page(namespace, normalized_token, page_size)
            if cached:
                videos = cached.get("videos", [])
                next_token = cached.get("next_token")
                cache_hit = True

            metadata = self._video_cache.get_namespace_metadata(namespace)
            if metadata is not None:
                total_videos = metadata.get("total_videos")

        if not cache_hit:
            videos, next_token = self.fetch_video_page(
                namespace=namespace,
                page_size=page_size,
                continuation_token=normalized_token,
            )
            if self._video_cache:
                self._video_cache.set_page(namespace, normalized_token, page_size, videos, next_token)

        if total_videos is None:
            total_videos = self.count_videos(namespace=namespace)
            if self._video_cache:
                self._video_cache.set_namespace_metadata(
                    namespace,
                    {
                        "total_videos": int(total_videos),
                    },
                )

        total_pages = math.ceil(total_videos / page_size) if page_size and total_videos else 0
        return videos, next_token, total_videos, total_pages

    def count_videos(self, namespace: str = "__default__") -> int:
        """Return total number of stored video objects for a namespace."""
        try:
            prefix = f"{namespace}/"
            continuation_token: Optional[str] = None
            total = 0

            while True:
                params = {
                    "Bucket": self.bucket_name,
                    "Prefix": prefix,
                    "MaxKeys": 1000,
                }
                if continuation_token:
                    params["ContinuationToken"] = continuation_token

                response = self.s3_client.list_objects_v2(**params)

                contents = response.get("Contents", [])
                for obj in contents:
                    key = obj.get("Key")
                    if key and key != prefix:
                        total += 1

                if response.get("IsTruncated"):
                    continuation_token = response.get("NextContinuationToken")
                else:
                    break

            logger.info(
                "Counted %s video objects for namespace %s",
                total,
                namespace,
            )
            return total
        except Exception as exc:
            logger.error(
                "Error counting videos for namespace %s: %s",
                namespace,
                exc,
            )
            return 0

    def fetch_all_video_data(
        self,
        namespace: str = "__default__",
        expiration: int = DEFAULT_PRESIGNED_URL_TTL,
    ) -> list[dict]:
        """
        Fetch all video data for a namespace, including filenames, identifiers, and presigned URLs.
        For WEB this will be the web client name, and for PLUGIN this will be user/project
        
        Args:
            namespace: The namespace name
            expiration: URL expiration time in seconds (default: 1 hour)
        
        Returns:
            list[dict]: List of dictionaries containing file_name, hashed_identifier, and presigned_url
        """
        video_data_list: List[dict] = []
        next_token: Optional[str] = None

        while True:
            page, next_token = self.fetch_video_page(
                namespace=namespace,
                page_size=1000,
                continuation_token=next_token,
                expiration=expiration,
            )
            video_data_list.extend(page)

            if not next_token:
                break

        logger.info(
            "Fetched complete dataset (%s videos) for namespace %s",
            len(video_data_list),
            namespace,
        )
        return video_data_list

    def clear_cache(self, namespace: str) -> int:
        return self._video_cache.clear_namespace(namespace or "__default__")
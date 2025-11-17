import logging
import boto3
from botocore.exceptions import ClientError
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class R2Connector:
    """
    R2 Connector Class for managing Cloudflare R2 storage interactions.
    Supports uploading videos to folders and retrieving them by URL.
    """
    
    def __init__(
        self,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        environment: str = "dev"  # dev, test, or prod
    ):
        """
        Initialize R2 connector with bucket credentials.
        
        Args:
            environment: Environment name (dev/test/prod)
            account_id: Cloudflare account ID
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
    
    def upload_video(
        self,
        video_data: bytes,
        filename: str,
        user_id: str,
        content_type: str = "video/mp4"
    ) -> Tuple[bool, str]:
        """
        Upload a video to R2 storage in a specified folder.
        Creates the folder if it doesn't exist (folders are virtual in S3/R2).
        
        Args:
            video_data: The video file as bytes
            filename: Name of the video file
            user_id: User ID to organize videos
            content_type: MIME type of the video
        
        Returns:
            Tuple[bool, str]: (Success flag, R2 URL of the uploaded video or error message)
        """
        try:
            # Construct the object key (folder/filename)
            object_key = f"{user_id}/{filename}"
            
            # Upload the video
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=video_data,
                ContentType=content_type
            )
            
            # Construct the R2 URL
            r2_url = f"{self.endpoint_url}/{self.bucket_name}/{object_key}"
            
            logger.info(f"Uploaded video to R2: {object_key}")
            return True, r2_url
            
        except ClientError as e:
            logger.error(f"Error uploading video to R2: {e}")
            return False, ""
    
    def fetch_video(self, r2_url: str) -> Optional[bytes]:
        """
        Fetch a video from R2 storage using its URL.
        
        Args:
            r2_url: The full R2 URL of the video
        
        Returns:
            bytes: The video file as bytes, or None if fetch failed
        """
        try:
            # Extract object key from URL
            # URL format: https://{account_id}.r2.cloudflarestorage.com/{bucket_name}/{object_key}
            url_parts = r2_url.replace(f"{self.endpoint_url}/{self.bucket_name}/", "")
            object_key = url_parts
            
            # Fetch the object
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            
            video_data = response['Body'].read()
            logger.info(f"Fetched video from R2: {object_key} ({len(video_data)} bytes)")
            return video_data
            
        except ClientError as e:
            logger.error(f"Error fetching video from R2: {e}")
            return None
    
    def _generate_presigned_url(self, r2_url: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for temporary access to a video.
        
        Args:
            r2_url: The R2 URL of the video
            expiration: URL expiration time in seconds (default: 1 hour)
        
        Returns:
            str: Presigned URL that can be used to access the video directly
        """
        try:
            # Extract object key from URL
            url_parts = r2_url.replace(f"{self.endpoint_url}/{self.bucket_name}/", "")
            object_key = url_parts
            
            # Generate presigned URL
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            
            logger.info(f"Generated presigned URL for: {object_key}")
            return presigned_url
            
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None

    def generate_presigned_urls_batch(self, r2_urls: list[str], expiration: int = 3600) -> dict[str, Optional[str]]:
        """
        Generate presigned URLs for multiple videos. These can then used upstream to access videos directly without authentication.
        
        Args:
            r2_urls: List of R2 URLs
            expiration: URL expiration time in seconds
        
        Returns:
            dict: Mapping of original URL to presigned URL
        """
        results = {}
        for url in r2_urls:
            presigned = self._generate_presigned_url(url, expiration)
            results[url] = presigned
        
        logger.info(f"Generated {len(results)} presigned URLs")
        return results

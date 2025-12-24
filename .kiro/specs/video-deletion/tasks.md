# Implementation Plan: Video Deletion System

## Overview

This implementation plan integrates the video deletion feature seamlessly into the existing ClipABit architecture. All components follow established patterns and are placed within the current directory structure. The implementation extends existing connectors, adds a new Modal endpoint, and enhances the Streamlit frontend.

## File Structure Integration

```
backend/
├── database/
│   ├── r2_connector.py          # EXTEND: Add delete_video_file() method
│   ├── pinecone_connector.py    # EXTEND: Add delete_by_metadata() method
│   └── deletion_service.py      # NEW: Core deletion orchestrator
├── main.py                      # EXTEND: Add delete_video endpoint
└── tests/
    ├── unit/
    │   ├── test_r2_connector.py         # EXTEND: Add deletion tests
    │   ├── test_pinecone_connector.py   # EXTEND: Add deletion tests
    │   └── test_deletion_service.py     # NEW: Core service tests
    ├── integration/
    │   └── test_deletion_endpoint.py    # NEW: API integration tests
    └── property/
        └── test_deletion_properties.py  # NEW: Property-based tests

frontend/streamlit/
├── config.py                    # EXTEND: Add DELETE_API_URL
├── pages/search_demo.py         # EXTEND: Add delete buttons and handlers
└── tests/                       # NEW: Frontend test directory
    └── test_deletion_ui.py      # NEW: UI component tests
```

## Tasks

- [x] 1. Extend R2Connector with deletion capabilities
  - Add `delete_video_file()` method to existing `R2Connector` class
  - Add `verify_deletion()` method for confirmation
  - Follow existing error handling patterns from `upload_video()` method
  - _Requirements: 1.1, 2.1, 2.3_

- [x] 1.1 Write property tests for R2 deletion
  - **Property 1: Complete R2 Deletion**
  - **Validates: Requirements 1.1**

- [x] 1.2 Write unit tests for R2 deletion edge cases
  - Test missing files, network failures, invalid identifiers
  - _Requirements: 2.1, 2.3, 2.5_

- [x] 2. Extend PineconeConnector with deletion capabilities
  - Add `delete_by_metadata()` method to existing `PineconeConnector` class
  - Add `batch_delete_chunks()` method for efficient multi-chunk removal
  - Add `find_chunks_by_video()` method for chunk discovery
  - Follow existing patterns from `upsert_chunk()` and `query_chunks()` methods
  - _Requirements: 1.2, 1.5, 6.2_

- [x] 2.1 Write property tests for Pinecone deletion
  - **Property 2: Complete Pinecone Deletion**
  - **Validates: Requirements 1.2, 1.5, 6.3**

- [x] 2.2 Write unit tests for Pinecone deletion scenarios
  - Test multi-chunk videos, missing chunks, batch operations
  - _Requirements: 2.2, 2.4, 6.2_

- [x] 3. Create VideoDeletionService orchestrator
  - Create new file `backend/database/deletion_service.py`
  - Implement core deletion logic with parallel R2/Pinecone operations
  - Add environment validation using existing environment patterns
  - Add comprehensive logging using existing logger configuration
  - _Requirements: 1.3, 1.4, 3.1, 3.2, 7.1, 7.2_

- [x] 3.1 Write property tests for deletion service
  - **Property 3: Dual System Confirmation**
  - **Property 8: Dev Environment Access**
  - **Property 9: Prod Environment Restriction**
  - **Validates: Requirements 1.4, 3.1, 3.2**

- [x] 3.2 Write property tests for error handling
  - **Property 4: Graceful R2 Missing Data Handling**
  - **Property 5: Graceful Pinecone Missing Data Handling**
  - **Property 6: R2 Failure Isolation**
  - **Property 7: Pinecone Failure Reporting**
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [x] 4. Add Modal API endpoint
  - Add `delete_video()` method to existing `Server` class in `backend/main.py`
  - Use `@modal.method()` decorator following existing endpoint patterns
  - Integrate with existing startup logic and connector initialization
  - Follow existing error handling and response patterns from `search()` endpoint
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 4.1 Write property tests for API endpoint
  - **Property 11: Identifier Validation**
  - **Property 12: Namespace Isolation**
  - **Property 13: HTTP Status Code Correctness**
  - **Property 14: Structured Error Responses**
  - **Validates: Requirements 4.2, 4.3, 4.4, 4.5**

- [x] 4.2 Write integration tests for deletion endpoint
  - Test complete end-to-end deletion flow
  - Test error scenarios and status codes
  - _Requirements: 4.1, 4.4, 4.5_

- [ ] 5. Checkpoint - Backend implementation complete
  - Ensure all backend tests pass, ask the user if questions arise.

- [x] 6. Extend frontend configuration
  - Add `DELETE_API_URL` to existing `Config` class in `frontend/streamlit/config.py`
  - Follow existing URL pattern: `f"https://clipabit01--{ENVIRONMENT}-server-delete{url_portion}.modal.run"`
  - Add environment-based delete button visibility flag
  - _Requirements: 5.6_

- [x] 7. Add delete functionality to search demo
  - Modify `frontend/streamlit/pages/search_demo.py` to add delete buttons
  - Add delete confirmation dialog using Streamlit's `@st.dialog` decorator
  - Add delete API call function following existing `search_videos()` pattern
  - Update video display grid to include delete buttons (dev environment only)
  - Add success/error message handling using existing `st.toast()` patterns
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [x] 7.1 Write frontend component tests
  - Test delete button rendering based on environment
  - Test confirmation dialog behavior
  - Test API integration and error handling
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [x] 8. Add comprehensive logging
  - Extend existing logging in `VideoDeletionService` using current logger patterns
  - Add security logging for unauthorized attempts
  - Add verification result logging
  - Follow existing log format from `preprocessor.py` and `main.py`
  - _Requirements: 3.4, 7.1, 7.2, 7.3, 7.5_

- [x] 8.1 Write property tests for logging
  - **Property 10: Unauthorized Attempt Logging**
  - **Property 16: Request Logging**
  - **Property 17: Result Logging**
  - **Property 18: Error Logging**
  - **Property 19: Verification Logging**
  - **Validates: Requirements 3.4, 7.1, 7.2, 7.3, 7.5**

- [x] 9. Add deletion verification
  - Implement verification logic in `VideoDeletionService`
  - Add verification methods to both R2 and Pinecone connectors
  - Follow existing verification patterns from upload pipeline
  - _Requirements: 6.4_

- [x] 9.1 Write property tests for verification
  - **Property 15: Deletion Verification**
  - **Validates: Requirements 6.4**

- [x] 10. Final integration and testing
  - Run complete test suite including property-based tests
  - Test in dev environment with real data
  - Verify prod environment properly blocks deletion
  - Test frontend integration with backend
  - _Requirements: All_

- [x] 11. Final checkpoint - Complete system verification
  - Ensure all tests pass, ask the user if questions arise.

## Implementation Details

### Backend Integration Points

**R2Connector Extension** (`backend/database/r2_connector.py`):
```python
def delete_video_file(self, hashed_identifier: str) -> R2DeletionResult:
    """Delete video file from R2 storage."""
    # Follows same pattern as upload_video() method
    
def verify_deletion(self, hashed_identifier: str) -> bool:
    """Verify video file no longer exists in R2."""
    # Uses existing _decode_path() method
```

**PineconeConnector Extension** (`backend/database/pinecone_connector.py`):
```python
def delete_by_metadata(self, video_metadata: dict, namespace: str) -> PineconeDeletionResult:
    """Delete all chunks matching video metadata."""
    # Uses existing index connection patterns
    
def find_chunks_by_video(self, hashed_identifier: str, namespace: str) -> List[str]:
    """Find all chunk IDs for a video."""
    # Uses existing query_chunks() patterns
```

**Modal Endpoint** (`backend/main.py`):
```python
@modal.method()
async def delete_video(self, hashed_identifier: str, namespace: str = "web-demo") -> dict:
    """Delete video and all associated data."""
    # Uses existing self.r2_connector and self.pinecone_connector
    # Follows same error handling as process_video() method
```

**Frontend Configuration** (`frontend/streamlit/config.py`):
```python
DELETE_API_URL = f"https://clipabit01--{ENVIRONMENT}-server-delete{url_portion}.modal.run"
SHOW_DELETE_BUTTONS = ENVIRONMENT == "dev"
```

**Frontend Integration** (`frontend/streamlit/pages/search_demo.py`):
```python
def delete_video(hashed_identifier: str, filename: str):
    """Delete video via API call."""
    # Follows same pattern as search_videos() function
    
@st.dialog("Delete Video")
def delete_confirmation_dialog(hashed_identifier: str, filename: str):
    """Show delete confirmation dialog."""
    # Uses existing st.dialog pattern
```

### Testing Integration

**Property-Based Tests** (`backend/tests/property/test_deletion_properties.py`):
- Uses Hypothesis library (already in dependencies)
- Follows existing test patterns from `test_preprocessor.py`
- Minimum 100 iterations per property
- Tagged with feature and property references

**Unit Tests**: Extend existing test files following current pytest patterns

**Integration Tests**: New files following existing integration test structure

## Notes

- All tasks build incrementally on existing architecture
- No new dependencies required (Hypothesis already available)
- Environment validation uses existing ENVIRONMENT variable
- Logging follows existing patterns and configuration
- API endpoints follow existing Modal patterns
- Frontend follows existing Streamlit patterns
- All tasks are required for comprehensive testing and robust implementation
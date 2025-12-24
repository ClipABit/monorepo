# Video Deletion System Architecture

```mermaid
graph TB
    %% User Interface Layer
    UI[Streamlit UI<br/>search_demo.py] --> |Delete Button Click| DEL_HANDLER[delete_video_handler]
    
    %% Frontend Functions
    DEL_HANDLER --> |HTTP POST| API_CALL[requests.post<br/>/delete-video]
    DEL_HANDLER --> |On Success| CACHE_CLEAR[fetch_all_videos.clear]
    DEL_HANDLER --> |On Success| UI_REFRESH[st.rerun]
    
    %% Modal API Layer
    API_CALL --> |Modal Endpoint| DELETE_EP[delete_video_endpoint<br/>main.py]
    DELETE_EP --> |Initialize Services| INIT_SERVICES[get_r2_connector<br/>get_pinecone_connector<br/>VideoDeletionService]
    
    %% Core Deletion Service
    INIT_SERVICES --> |Call| VDS_DELETE[VideoDeletionService.delete_video]
    
    %% Environment & Validation
    VDS_DELETE --> |Check| ENV_VAL[_validate_environment]
    VDS_DELETE --> |Decode| ID_DECODE[_decode_identifier]
    
    %% Parallel Deletion Operations
    VDS_DELETE --> |asyncio.gather| PARALLEL{Parallel Execution}
    PARALLEL --> |Task 1| R2_DELETE[_delete_from_r2]
    PARALLEL --> |Task 2| PC_DELETE[_delete_from_pinecone]
    
    %% R2 Storage Operations
    R2_DELETE --> |Call| R2_CONN[R2Connector.delete_video_file]
    R2_CONN --> |Decode Path| R2_DECODE[_decode_path]
    R2_CONN --> |Delete Object| R2_DEL_OBJ[delete_object]
    R2_CONN --> |Check Existence| R2_EXISTS[head_object]
    
    %% Pinecone Database Operations
    PC_DELETE --> |Call| PC_CONN[PineconeConnector.delete_by_metadata]
    PC_CONN --> |Find Chunks| PC_FIND[find_chunks_by_video]
    PC_FIND --> |Try Variations| PC_VARIATIONS[Base64 Padding Variants]
    PC_VARIATIONS --> |Query| PC_QUERY[index.query with metadata filter]
    PC_CONN --> |Delete IDs| PC_DEL_IDS[index.delete with ids]
    
    %% Verification Phase
    PARALLEL --> |After Completion| VERIFY[_verify_deletion]
    VERIFY --> |Check R2| R2_VERIFY[R2Connector.verify_deletion]
    VERIFY --> |Check Pinecone| PC_VERIFY[find_chunks_by_video]
    
    %% Results Flow
    VERIFY --> |Compile Results| RESULT[DeletionResult]
    RESULT --> |Return| DELETE_EP
    DELETE_EP --> |JSON Response| API_CALL
    API_CALL --> |Success/Error| DEL_HANDLER
    
    %% Data Structures
    subgraph "Key Data Structures"
        DR[DeletionResult<br/>- success: bool<br/>- r2_result: R2DeletionResult<br/>- pinecone_result: PineconeDeletionResult<br/>- verification_result: VerificationResult]
        R2R[R2DeletionResult<br/>- success: bool<br/>- bucket: str<br/>- key: str<br/>- file_existed: bool<br/>- bytes_deleted: int]
        PCR[PineconeDeletionResult<br/>- success: bool<br/>- chunks_found: int<br/>- chunks_deleted: int<br/>- chunk_ids: list]
    end
    
    %% Environment & Security
    subgraph "Security & Environment"
        ENV[Environment Check<br/>Only allows 'dev' environment]
        ID_VAL[Identifier Validation<br/>Base64 decode validation]
        LOG[Comprehensive Logging<br/>All operations logged]
    end
    
    %% Storage Systems
    subgraph "Storage Systems"
        R2_STORAGE[(Cloudflare R2<br/>Video Files)]
        PC_DB[(Pinecone Database<br/>Vector Embeddings)]
    end
    
    R2_DEL_OBJ --> R2_STORAGE
    PC_DEL_IDS --> PC_DB
    
    %% Styling
    classDef uiClass fill:#e1f5fe
    classDef serviceClass fill:#f3e5f5
    classDef storageClass fill:#e8f5e8
    classDef dataClass fill:#fff3e0
    
    class UI,DEL_HANDLER,CACHE_CLEAR,UI_REFRESH uiClass
    class VDS_DELETE,R2_DELETE,PC_DELETE,VERIFY serviceClass
    class R2_STORAGE,PC_DB storageClass
    class DR,R2R,PCR dataClass
```

## Key Functions Breakdown

### Frontend (Streamlit)
- **`delete_video_handler`**: Main UI deletion handler with confirmation dialog
- **`fetch_all_videos.clear()`**: Cache invalidation for immediate UI updates
- **`st.rerun()`**: Force UI refresh after deletion

### Backend API (Modal)
- **`delete_video_endpoint`**: HTTP endpoint that orchestrates the deletion
- **Service initialization functions**: Setup R2 and Pinecone connectors

### Core Deletion Service
- **`VideoDeletionService.delete_video`**: Main orchestrator function
- **`_validate_environment`**: Security check (dev-only deletion)
- **`_decode_identifier`**: Base64 identifier parsing
- **`_delete_from_r2`** & **`_delete_from_pinecone`**: Parallel deletion operations
- **`_verify_deletion`**: Post-deletion verification

### R2 Connector
- **`delete_video_file`**: Main R2 deletion method
- **`_decode_path`**: Base64 to bucket/key conversion
- **`delete_object`**: Actual S3-compatible deletion
- **`verify_deletion`**: Confirm file no longer exists

### Pinecone Connector
- **`delete_by_metadata`**: Main Pinecone deletion method
- **`find_chunks_by_video`**: Find chunks with base64 padding handling
- **`index.query`**: Vector database query with metadata filters
- **`index.delete`**: Bulk deletion of vector chunks

## Async Flow
The system uses `asyncio.gather()` to run R2 and Pinecone deletions in parallel, significantly improving performance. The async pattern allows both storage systems to be updated simultaneously rather than sequentially.

## Error Handling & Security
- Environment-based access control (dev-only)
- Comprehensive logging at each step
- Graceful error handling with detailed error messages
- Verification phase to ensure deletion success
- Base64 padding variation handling for identifier matching
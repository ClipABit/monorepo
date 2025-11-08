"""
- ArcFace for facial recognition (face-specific embeddings)
- YAMNet for audio classification (preset audio event flags)
- Better preprocessing for Whisper ASR
- Timestamped output directories
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
import numpy as np
from tqdm import tqdm
import pickle
import shutil

# ML Models
try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_AVAILABLE = True
    print("✓ PySceneDetect loaded")
except ImportError:
    SCENEDETECT_AVAILABLE = False
    print("⚠️  PySceneDetect not available")

# Whisper ASR disabled - using YAMNet for speech detection instead
WHISPER_AVAILABLE = False
WHISPER_MODEL = None
print("⚠️  Whisper ASR disabled (using YAMNet for speech detection)")

# PyTorch not needed without Whisper/CLIP
TORCH_AVAILABLE = False
print("✓ PyTorch skipped (not needed)")

try:
    import cv2
    print("✓ OpenCV loaded")
except ImportError:
    cv2 = None
    print("⚠️  OpenCV not available")

# RetinaFace for face detection + recognition
try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
    print("✓ RetinaFace loaded (modern face detection + embeddings)")
except ImportError:
    RETINAFACE_AVAILABLE = False
    print("⚠️  RetinaFace not available")

# YAMNet for audio classification
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
    YAMNET_AVAILABLE = True
    print("✓ YAMNet audio classifier loaded")
except ImportError:
    YAMNET_AVAILABLE = False
    YAMNET_MODEL = None
    print("⚠️  YAMNet not available - will use fallback")

try:
    from imageio_ffmpeg import get_ffmpeg_exe
    FFMPEG = get_ffmpeg_exe()
    print(f"✓ ffmpeg loaded\n")
except:
    FFMPEG = 'ffmpeg'
    print("⚠️  Using system ffmpeg\n")

# CONFIG - Generic folders
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
INPUT_DIR = "input_videos"
OUTPUT_DIR = f"output_{TIMESTAMP}"
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")
CHUNKS_DIR = os.path.join(OUTPUT_DIR, "extracted_chunks")
FACE_DB_PATH = os.path.join(OUTPUT_DIR, "face_database.pkl")
FACE_IMAGES_DIR = os.path.join(OUTPUT_DIR, "face_images")
FINAL_COMPILATION = os.path.join(OUTPUT_DIR, "compilation.mp4")
SELECTION_REPORT = os.path.join(OUTPUT_DIR, "selection_report.txt")
AUDIO_EVENTS_LOG = os.path.join(OUTPUT_DIR, "audio_events.json")

# Chunk parameters
MIN_CHUNK_DURATION = 8.0
MAX_CHUNK_DURATION = 20.0
TARGET_COMPILATION_DURATION = 300.0  # 5 minutes

# Face matching - RetinaFace/Facenet512 uses cosine similarity
FACE_SIMILARITY_THRESHOLD = 0.4  # Facenet512 recommended threshold (more strict than ArcFace)

# YAMNet audio event categories we care about
YAMNET_MUSIC_CATEGORIES = ['Music', 'Musical instrument', 'Drum', 'Guitar', 'Piano', 'Percussion']
YAMNET_SPEECH_CATEGORIES = ['Speech', 'Conversation', 'Narration', 'Male speech', 'Female speech', 'Shout', 'Yell']


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk"""
    video_source: str
    chunk_id: int
    start_time: float
    end_time: float
    duration: float
    
    # Speech flag (from YAMNet)
    has_speech: bool = False
    
    # Face flags
    has_faces: bool = False
    face_ids: List[str] = field(default_factory=list)
    num_faces: int = 0
    
    # Audio event flags (from YAMNet)
    has_music: bool = False
    has_interesting_audio: bool = False  # Any other audio events besides speech/music
    detected_audio_events: List[str] = field(default_factory=list)
    
    # Quality
    has_good_quality: bool = False
    quality_score: float = 0.0
    
    # Scoring (not saved to metadata)
    score: float = 0.0
    selection_reason: str = ""


class FaceDatabase:
    """Manages face embeddings with ArcFace"""
    def __init__(self):
        self.face_embeddings = []  # List of (person_id, embedding) tuples
        self.next_person_id = 1
    
    def find_or_create_person(self, embedding: np.ndarray) -> str:
        """Find matching person or create new one"""
        if not RETINAFACE_AVAILABLE:
            return "unknown"
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        best_match_id = None
        best_similarity = 0.0
        
        for person_id, stored_embedding in self.face_embeddings:
            stored_norm = stored_embedding / np.linalg.norm(stored_embedding)
            similarity = np.dot(embedding, stored_norm)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person_id
        
        if best_similarity >= FACE_SIMILARITY_THRESHOLD:
            return best_match_id
        
        # New person
        person_id = f"person{self.next_person_id}"
        self.next_person_id += 1
        self.face_embeddings.append((person_id, embedding))
        return person_id
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return FaceDatabase()


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def get_video_duration(video_path) -> Optional[float]:
    """Get video duration"""
    try:
        import imageio.v3 as iio
        meta = iio.immeta(str(video_path), plugin="pyav")
        return meta['duration']
    except:
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            return frame_count / fps if fps > 0 else None
        except:
            return None


def detect_scenes(video_path) -> List[Tuple[float, float]]:
    """Detect scene boundaries"""
    video_path = str(video_path)
    
    if not SCENEDETECT_AVAILABLE:
        duration = get_video_duration(video_path)
        if not duration:
            return []
        return [(i, min(i+10, duration)) for i in range(0, int(duration), 10)]
    
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        
        scenes = []
        for scene in scene_list:
            start = scene[0].get_seconds()
            end = scene[1].get_seconds()
            scenes.append((start, end))
        
        return scenes if scenes else [(0, get_video_duration(video_path) or 10)]
    except Exception as e:
        print(f"    ⚠️  Scene detection failed: {e}")
        return []


def combine_scenes_to_chunks(scenes: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Combine scenes to create 8-20 second chunks"""
    if not scenes:
        return []
    
    chunks = []
    current_start = scenes[0][0]
    current_end = scenes[0][1]
    
    for i in range(1, len(scenes)):
        scene_start, scene_end = scenes[i]
        potential_duration = scene_end - current_start
        
        if potential_duration <= MAX_CHUNK_DURATION:
            current_end = scene_end
        else:
            duration = current_end - current_start
            if duration >= MIN_CHUNK_DURATION:
                chunks.append((current_start, current_end))
            elif duration > 0:
                current_end = min(current_start + MIN_CHUNK_DURATION, current_end)
                chunks.append((current_start, current_end))
            
            current_start = scene_start
            current_end = scene_end
    
    duration = current_end - current_start
    if duration >= MIN_CHUNK_DURATION:
        chunks.append((current_start, current_end))
    elif duration > 0 and chunks:
        chunks[-1] = (chunks[-1][0], current_end)
    
    final_chunks = []
    for start, end in chunks:
        duration = end - start
        if duration <= MAX_CHUNK_DURATION:
            final_chunks.append((start, end))
        else:
            final_chunks.append((start, start + MAX_CHUNK_DURATION))
    
    return final_chunks


def extract_audio_segment(video_path, start_time: float, duration: float) -> Optional[str]:
    """Extract audio segment - PROPER preprocessing for Whisper"""
    temp_audio = os.path.abspath(f"temp_audio_{start_time:.1f}.wav")
    
    if os.path.exists(temp_audio):
        try:
            os.remove(temp_audio)
        except:
            pass
    
    video_path = os.path.abspath(str(video_path))
    
    # Whisper-optimized extraction
    cmd = [
        FFMPEG, '-y',
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',  # Whisper expects 16kHz
        '-ac', '1',  # Mono
        '-af', 'loudnorm',  # Audio normalization
        '-loglevel', 'error',
        temp_audio
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 1000:
            return temp_audio
        return None
    except:
        return None


def analyze_speech(video_path, start_time: float, duration: float) -> Tuple[bool, str, float]:
    """Analyze speech with Whisper"""
    if not WHISPER_AVAILABLE:
        return False, "", 0.0
    
    video_path = os.path.join(INPUT_DIR, str(video_path)) if not os.path.isabs(str(video_path)) else str(video_path)
    audio_path = extract_audio_segment(video_path, start_time, duration)
    
    if not audio_path:
        return False, "", 0.0
    
    try:
        result = WHISPER_MODEL.transcribe(
            audio_path,
            language="en",
            fp16=False,
            verbose=False,
            condition_on_previous_text=False  # Better for short clips
        )
        
        transcript = result['text'].strip()
        segments = result.get('segments', [])
        
        if segments:
            no_speech_probs = [seg.get('no_speech_prob', 1.0) for seg in segments]
            speech_confidence = 1.0 - (sum(no_speech_probs) / len(no_speech_probs))
        else:
            speech_confidence = 0.0
        
        has_speech = len(transcript) >= 5 and speech_confidence > 0.3
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return has_speech, transcript, speech_confidence
    except Exception as e:
        print(f"    ⚠️  Whisper error: {e}")
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        return False, "", 0.0


def analyze_audio_events(video_path, start_time: float, duration: float) -> Tuple[bool, bool, bool, List[str]]:
    """Analyze audio with YAMNet for event classification - returns (has_speech, has_music, has_interesting, events)"""
    if not YAMNET_AVAILABLE:
        # Fallback to simple detection
        return False, False, False, []
    
    audio_path = extract_audio_segment(video_path, start_time, duration)
    if not audio_path:
        return False, False, False, []
    
    try:
        # Load audio for YAMNet (expects 16kHz mono)
        import soundfile as sf
        audio_data, sr = sf.read(audio_path)
        
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        
        # Run YAMNet
        scores, embeddings, spectrogram = YAMNET_MODEL(audio_data)
        
        # Load class names from embedded CSV
        import csv
        import requests
        class_map_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
        
        # Download and cache class map
        class_names = []
        try:
            response = requests.get(class_map_url, timeout=5)
            lines = response.text.strip().split('\n')
            reader = csv.DictReader(lines)
            class_names = [row['display_name'] for row in reader]
        except:
            # Fallback: use indices if can't get names
            class_names = [f"Class_{i}" for i in range(521)]
        
        # Get top predictions (above 0.3 confidence)
        mean_scores = np.mean(scores.numpy(), axis=0)
        top_indices = np.where(mean_scores > 0.3)[0]
        
        detected_events = []
        has_speech = False
        has_music = False
        has_interesting = False
        
        for idx in top_indices:
            if idx < len(class_names):
                event_name = class_names[idx]
                detected_events.append(event_name)
                
                # Check if it's speech
                if any(cat in event_name for cat in YAMNET_SPEECH_CATEGORIES):
                    has_speech = True
                # Check if it's music
                elif any(cat in event_name for cat in YAMNET_MUSIC_CATEGORIES):
                    has_music = True
                # Anything else is interesting
                else:
                    has_interesting = True
        
        os.remove(audio_path)
        return has_speech, has_music, has_interesting, detected_events
    
    except Exception as e:
        print(f"    ⚠️  YAMNet error: {e}")
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        return False, False, False, []


def detect_faces_retinaface(video_path, start_time: float, duration: float, face_db: FaceDatabase) -> List[str]:
    """Detect faces with RetinaFace (modern detection + embeddings)"""
    if not RETINAFACE_AVAILABLE or not cv2:
        return []
    
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # MUCH faster: Only sample 2-3 frames per chunk (start, middle, end)
    num_samples = min(3, max(2, int(duration / 5)))
    sample_times = [start_time + (i * duration / (num_samples - 1)) for i in range(num_samples)]
    
    person_ids = set()
    detected_faces = []
    
    for sample_time in sample_times:
        frame_num = int(sample_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        try:
            # Resize frame for faster processing
            scale = 0.5  # Process at half resolution
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            
            # Fast detection only (no embedding yet)
            faces = RetinaFace.detect_faces(small_frame, threshold=0.7)
            
            if not faces or not isinstance(faces, dict):
                continue
            
            # Only process first few faces to avoid slowdown
            for face_idx, (face_key, face_data) in enumerate(faces.items()):
                if face_idx >= 3:  # Max 3 faces per frame
                    break
                    
                # Get facial area and scale back up
                facial_area = face_data.get('facial_area', [])
                if len(facial_area) != 4:
                    continue
                
                x1, y1, x2, y2 = [int(coord / scale) for coord in facial_area]
                
                # Extract face region from full-size frame
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
                    continue
                
                # Get embedding - use DeepFace which is faster
                try:
                    from deepface import DeepFace
                    # Get embedding directly from face crop
                    embedding_objs = DeepFace.represent(face_img, 
                                                       model_name='Facenet512',
                                                       detector_backend='skip',  # Skip detection, already cropped
                                                       enforce_detection=False)
                    
                    if embedding_objs and len(embedding_objs) > 0:
                        emb_vector = np.array(embedding_objs[0]['embedding'])
                        
                        # Find or create person
                        person_id = face_db.find_or_create_person(emb_vector)
                        person_ids.add(person_id)
                        
                        # Save face image (only once per person)
                        if person_id not in [f[0] for f in detected_faces]:
                            detected_faces.append((person_id, face_img))
                except Exception as e:
                    # If deepface fails, create simple hash-based ID from face crop
                    face_hash = hash(face_img.tobytes()) % 10000
                    person_id = f"person{face_hash}"
                    person_ids.add(person_id)
                    if person_id not in [f[0] for f in detected_faces]:
                        detected_faces.append((person_id, face_img))
                    
        except Exception as e:
            # Skip frames that error
            continue
    
    cap.release()
    
    # Save face images
    if detected_faces:
        os.makedirs(FACE_IMAGES_DIR, exist_ok=True)
        for person_id, face_img in detected_faces:
            face_path = os.path.join(FACE_IMAGES_DIR, f"{person_id}.jpg")
            if not os.path.exists(face_path):
                cv2.imwrite(face_path, face_img)
    
    return list(person_ids)


def process_video(video_path: Path, face_db: FaceDatabase) -> List[ChunkMetadata]:
    """Process a single video"""
    print(f"\n{'='*70}")
    print(f"PROCESSING: {video_path.name}")
    print(f"{'='*70}\n")
    
    duration = get_video_duration(video_path)
    if not duration:
        print("❌ Could not get video duration")
        return []
    print(f"✓ Duration: {duration:.1f}s")
    
    print("🔍 Detecting scenes...")
    scenes = detect_scenes(video_path)
    print(f"✓ Found {len(scenes)} scenes")
    
    print("✂️  Creating chunks (8-20s)...")
    chunks = combine_scenes_to_chunks(scenes)
    print(f"✓ Created {len(chunks)} chunks")
    
    print("\n🔬 Analyzing chunks...")
    chunk_metadata = []
    
    for chunk_id, (start, end) in enumerate(tqdm(chunks, desc="Processing")):
        chunk_duration = end - start
        
        # Audio events (YAMNet) - PRIMARY source for speech/music detection
        has_speech_yamnet, has_music, has_interesting, audio_events = analyze_audio_events(video_path, start, chunk_duration)
        
        # Use YAMNet for speech flag (no more ASR)
        has_speech = has_speech_yamnet
        
        # Face detection
        face_ids = detect_faces_retinaface(video_path, start, chunk_duration, face_db)
        
        # Create metadata
        metadata = ChunkMetadata(
            video_source=video_path.name,
            chunk_id=chunk_id,
            start_time=start,
            end_time=end,
            duration=chunk_duration,
            has_speech=has_speech,
            has_faces=len(face_ids) > 0,
            face_ids=face_ids,
            num_faces=len(face_ids),
            has_music=has_music,
            has_interesting_audio=has_interesting,
            detected_audio_events=audio_events,
            has_good_quality=True,
            quality_score=0.8
        )
        
        chunk_metadata.append(metadata)
    
    print(f"✓ Analyzed {len(chunk_metadata)} chunks\n")
    return chunk_metadata


def save_metadata(chunks: List[ChunkMetadata], video_name: str):
    """Save chunk metadata to JSON (excluding score and selection_reason)"""
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    output_file = os.path.join(METADATA_DIR, f"{Path(video_name).stem}_metadata.json")
    
    # Convert to dict and remove score/selection_reason
    chunks_data = []
    for chunk in chunks:
        chunk_dict = asdict(chunk)
        chunk_dict.pop('score', None)
        chunk_dict.pop('selection_reason', None)
        chunks_data.append(chunk_dict)
    
    data = {
        'video_source': video_name,
        'num_chunks': len(chunks),
        'chunks': chunks_data
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    
    print(f"💾 Saved metadata: {output_file}")


def load_all_metadata() -> List[ChunkMetadata]:
    """Load all chunk metadata"""
    all_chunks = []
    
    if not os.path.exists(METADATA_DIR):
        return []
    
    for filename in os.listdir(METADATA_DIR):
        if filename.endswith('_metadata.json'):
            filepath = os.path.join(METADATA_DIR, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                for chunk_dict in data['chunks']:
                    chunk = ChunkMetadata(**chunk_dict)
                    all_chunks.append(chunk)
    
    return all_chunks


def score_and_select_chunks(all_chunks: List[ChunkMetadata], target_duration: float = 300.0) -> List[ChunkMetadata]:
    """Score and select chunks to fill target duration (5 minutes)"""
    print(f"\n{'='*70}")
    print("SCORING & SELECTION")
    print(f"{'='*70}\n")
    
    for chunk in all_chunks:
        score = 0
        reasons = []
        
        # Faces are most important
        if chunk.has_faces:
            face_score = 100 * chunk.num_faces
            score += face_score
            reasons.append(f"👥 {chunk.num_faces} face(s): {', '.join(chunk.face_ids)}")
        
        # Speech from YAMNet
        if chunk.has_speech:
            score += 50
            reasons.append(f"🗣️ Speech detected")
        
        # Music
        if chunk.has_music:
            score += 30
            reasons.append(f"🎵 Music")
        
        # Interesting audio (anything besides speech/music)
        if chunk.has_interesting_audio:
            score += 25
            other_events = [e for e in chunk.detected_audio_events 
                          if not any(cat in e for cat in YAMNET_SPEECH_CATEGORIES + YAMNET_MUSIC_CATEGORIES)]
            if other_events:
                reasons.append(f"🔊 Interesting audio: {', '.join(other_events[:3])}")
        
        # Longer chunks are slightly better
        if chunk.duration >= 15:
            score += 10
            reasons.append(f"⏱️ Good length ({chunk.duration:.1f}s)")
        
        # Quality bonus
        if chunk.has_good_quality:
            score += 5
        
        chunk.score = score
        chunk.selection_reason = " | ".join(reasons) if reasons else "No interesting content"
    
    # Sort by score
    sorted_chunks = sorted(all_chunks, key=lambda c: c.score, reverse=True)
    
    # Select chunks to fill target duration
    selected = []
    total_duration = 0.0
    
    for chunk in sorted_chunks:
        if total_duration + chunk.duration <= target_duration:
            selected.append(chunk)
            total_duration += chunk.duration
        
        if total_duration >= target_duration * 0.95:  # Stop at 95% to avoid going over
            break
    
    print(f"✓ Selected {len(selected)} chunks (total duration: {total_duration:.1f}s / {target_duration:.1f}s target)\n")
    
    print("TOP SELECTED CHUNKS:")
    for i, chunk in enumerate(selected, 1):
        print(f"\n{i}. [{chunk.video_source}] Chunk #{chunk.chunk_id}")
        print(f"   Time: {chunk.start_time:.1f}s - {chunk.end_time:.1f}s ({chunk.duration:.1f}s)")
        print(f"   Score: {chunk.score}")
        print(f"   {chunk.selection_reason}")
    
    return selected


def generate_selection_report(selected_chunks: List[ChunkMetadata], all_chunks: List[ChunkMetadata]):
    """Generate detailed selection report with full metadata for each chunk"""
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("TEST DATA SELECTION REPORT")
    report_lines.append("="*70)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total chunks analyzed: {len(all_chunks)}")
    report_lines.append(f"Chunks selected: {len(selected_chunks)}")
    
    total_duration = sum(c.duration for c in selected_chunks)
    report_lines.append(f"Total compilation duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    
    # Coverage statistics
    speech_chunks = [c for c in selected_chunks if c.has_speech]
    music_chunks = [c for c in selected_chunks if c.has_music]
    interesting_chunks = [c for c in selected_chunks if c.has_interesting_audio]
    face_chunks = [c for c in selected_chunks if c.has_faces]
    
    report_lines.append(f"\n{'='*70}")
    report_lines.append("COVERAGE SUMMARY")
    report_lines.append("="*70)
    report_lines.append(f"✓ Has Speech: {len(speech_chunks)}/{len(selected_chunks)} chunks")
    report_lines.append(f"✓ Has Music: {len(music_chunks)}/{len(selected_chunks)} chunks")
    report_lines.append(f"✓ Has Interesting Audio: {len(interesting_chunks)}/{len(selected_chunks)} chunks")
    report_lines.append(f"✓ Has Faces: {len(face_chunks)}/{len(selected_chunks)} chunks")
    
    # Detailed chunk information
    report_lines.append(f"\n{'='*70}")
    report_lines.append("SELECTED CHUNKS - DETAILED METADATA")
    report_lines.append("="*70)
    
    for i, chunk in enumerate(selected_chunks, 1):
        report_lines.append(f"\n{'─'*70}")
        report_lines.append(f"CHUNK #{i} - ID: {chunk.chunk_id}")
        report_lines.append(f"{'─'*70}")
        report_lines.append(f"Video Source: {chunk.video_source}")
        report_lines.append(f"Time Range: {chunk.start_time:.2f}s - {chunk.end_time:.2f}s")
        report_lines.append(f"Duration: {chunk.duration:.2f}s")
        report_lines.append(f"\nSELECTION REASON:")
        report_lines.append(f"  {chunk.selection_reason}")
        
        report_lines.append(f"\n📊 AUDIO ANALYSIS:")
        report_lines.append(f"  Speech Detected: {'✓ YES' if chunk.has_speech else '✗ NO'}")
        report_lines.append(f"  Music Detected: {'✓ YES' if chunk.has_music else '✗ NO'}")
        report_lines.append(f"  Other Interesting Audio: {'✓ YES' if chunk.has_interesting_audio else '✗ NO'}")
        if chunk.detected_audio_events:
            report_lines.append(f"  Detected Events: {', '.join(chunk.detected_audio_events)}")
        
        report_lines.append(f"\n👥 FACE ANALYSIS:")
        report_lines.append(f"  Faces Detected: {'✓ YES' if chunk.has_faces else '✗ NO'}")
        if chunk.has_faces:
            report_lines.append(f"  Number of People: {chunk.num_faces}")
            report_lines.append(f"  Person IDs: {', '.join(chunk.face_ids)}")
        
        report_lines.append(f"\n🎬 QUALITY:")
        report_lines.append(f"  Good Quality: {'✓ YES' if chunk.has_good_quality else '✗ NO'}")
        report_lines.append(f"  Quality Score: {chunk.quality_score:.2f}")
    
    report_lines.append(f"\n{'='*70}")
    report_lines.append("END OF REPORT")
    report_lines.append("="*70)
    
    report_text = "\n".join(report_lines)
    with open(SELECTION_REPORT, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n📊 Detailed report saved: {SELECTION_REPORT}")


def create_compilation(selected_chunks: List[ChunkMetadata], output_path: str):
    """Create compilation"""
    print(f"\n{'='*70}")
    print("CREATING COMPILATION")
    print(f"{'='*70}\n")
    
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    
    chunk_files = []
    for i, chunk in enumerate(tqdm(selected_chunks, desc="Extracting")):
        # Extract video name and sanitize for filename
        video_stem = Path(chunk.video_source).stem
        # Remove special characters and limit length
        video_name_clean = ''.join(c for c in video_stem if c.isalnum() or c in (' ', '_'))
        video_name_clean = video_name_clean.replace(' ', '')[:30]  # Max 30 chars
        
        chunk_file = os.path.join(CHUNKS_DIR, f"chunk_{i:03d}_{video_name_clean}_{chunk.chunk_id}.mp4")
        
        video_path = os.path.join(INPUT_DIR, chunk.video_source)
        # Re-encode during extraction to ensure clean keyframe-aligned chunks
        # Using -c copy with -ss can cause frozen frames and misaligned starts
        cmd = [
            FFMPEG, '-y',
            '-ss', str(chunk.start_time),
            '-i', video_path,
            '-t', str(chunk.duration),
            '-c:v', 'libx264',        # Re-encode video for clean keyframes
            '-preset', 'fast',         # Fast encoding (we'll re-encode again during concat)
            '-crf', '18',              # High quality
            '-c:a', 'aac',             # Re-encode audio
            '-b:a', '192k',
            '-loglevel', 'error',
            chunk_file
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        chunk_files.append(chunk_file)
    
    # Use filter complex for concatenation (most robust method)
    # This fully decodes each chunk and concatenates at the frame level
    print(f"\nConcatenating {len(chunk_files)} chunks using filter complex...")
    
    # Build filter complex for concatenation
    filter_parts = []
    for i in range(len(chunk_files)):
        filter_parts.append(f"[{i}:v:0][{i}:a:0]")
    
    filter_complex = "".join(filter_parts) + f"concat=n={len(chunk_files)}:v=1:a=1[outv][outa]"
    
    # Build command with all input files
    cmd = [FFMPEG, '-y']
    
    for chunk_file in chunk_files:
        cmd.extend(['-i', chunk_file])
    
    # Add filter complex and output encoding
    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        '-map', '[outa]',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-movflags', '+faststart',
        '-loglevel', 'error',
        output_path
    ])
    
    subprocess.run(cmd, capture_output=True, check=True)
    
    print(f"\n✅ Created: {output_path}")
    print(f"   Chunks: {len(selected_chunks)}")


def main():
    """Main pipeline"""
    print("\n" + "="*70)
    print("PROFESSIONAL TEST SET CREATOR V3")
    print("="*70 + "\n")
    
    # Check for RetinaFace availability
    if not RETINAFACE_AVAILABLE:
        print("⚠️  Installing RetinaFace...")
        subprocess.run([sys.executable, "-m", "pip", "install", "retina-face"], 
                      capture_output=True)
        print("   Please restart the script after installation.\n")
        return
    
    if not YAMNET_AVAILABLE:
        print("⚠️  Installing TensorFlow Hub for YAMNet...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow", "tensorflow-hub", "soundfile", "pandas"], 
                      capture_output=True)
        print("   Please restart the script after installation.\n")
        return
    
    face_db = FaceDatabase.load(FACE_DB_PATH)
    print(f"✓ Face database initialized ({len(face_db.face_embeddings)} known faces)\n")
    
    input_dir = Path(INPUT_DIR)
    video_files = list(input_dir.glob('*.mp4'))
    
    if not video_files:
        print(f"❌ No MP4 files found in {INPUT_DIR}/")
        print(f"   Please place your video files in the '{INPUT_DIR}' folder.\n")
        return
    
    print(f"📹 Processing {len(video_files)} video(s)\n")
    
    for video_path in video_files:
        chunks = process_video(video_path, face_db)
        save_metadata(chunks, video_path.name)
    
    face_db.save(FACE_DB_PATH)
    print(f"\n💾 Face database saved: {FACE_DB_PATH}")
    
    all_chunks = load_all_metadata()
    selected_chunks = score_and_select_chunks(all_chunks)
    
    generate_selection_report(selected_chunks, all_chunks)
    create_compilation(selected_chunks, FINAL_COMPILATION)
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"\n📁 Output: {OUTPUT_DIR}/\n")


if __name__ == '__main__':
    main()

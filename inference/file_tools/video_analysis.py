import base64
import io
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Literal, Optional, TypedDict, Union
from urllib.parse import urlparse

import requests
from PIL import Image
from openai import OpenAI
from qwen_agent.log import logger
from qwen_agent.tools.base import BaseTool, register_tool

# Configuration constants
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
SUPPORTED_VIDEO_TYPES = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
SUPPORTED_AUDIO_TYPES = {'.mp3', '.wav', '.aac', '.ogg', '.flac'}
DEFAULT_FRAMES = 8
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1


class AnalysisResult(TypedDict):
    """Type definition for analysis results"""
    status: Literal['success', 'error']
    data: Optional[Dict]
    error: Optional[Dict]


@contextmanager
def temp_directory():
    """Context manager for temporary directory handling"""
    temp_dir = tempfile.TemporaryDirectory()
    try:
        logger.debug(f"Created temp directory: {temp_dir.name}")
        yield Path(temp_dir.name)
    finally:
        try:
            temp_dir.cleanup()
            logger.debug("Cleaned up temp directory")
        except Exception as e:
            logger.warning(f"Temp directory cleanup failed: {str(e)}")


@register_tool('video_analysis')
class VideoAnalysis(BaseTool):
    """Improved tool for analyzing video and audio content"""
    parameters = [
        {
            'name': 'prompt',
            'type': 'string',
            'description': 'Detailed question or instruction for video/audio analysis',
            'required': True
        },
        {
            'name': 'url',
            'type': 'string',
            'description': 'Media file URL/path (supports video/audio)',
            'required': True
        },
        {
            'name': 'num_frames',
            'type': 'number',
            'description': 'Number of key frames to extract (default: 8)',
            'required': False
        }
    ]

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg or {})
        self.config = self._init_config(cfg or {})
        self.client = OpenAI(
            api_key=self.config['api_key'],
            base_url=self.config['api_base'],
            timeout=self.config['timeout']
        )
        self.http_session = self._init_http_client()
        self._check_dependencies()
        logger.info("Video analysis tool initialized")

    def _init_config(self, cfg: Dict) -> Dict:
        """Initialize configuration with sensible defaults"""
        return {
            'api_key': os.getenv('DASHSCOPE_API_KEY', ''),
            'api_base': cfg.get('api_base') or os.getenv('DASHSCOPE_API_BASE', ''),
            'video_model': cfg.get('video_model') or os.getenv('VIDEO_MODEL_NAME', 'qwen-omni-turbo'),
            'analysis_model': cfg.get('analysis_model') or os.getenv('VIDEO_ANALYSIS_MODEL_NAME', 'qwen-plus-latest'),
            'timeout': min(cfg.get('timeout', 30), 300),  # Cap at 300 seconds
            'max_frames': min(cfg.get('max_frames', 20), 50),  # Cap at 50 frames
            'max_file_size': MAX_FILE_SIZE
        }

    def _init_http_client(self) -> requests.Session:
        """Initialize HTTP client with retry logic"""
        session = requests.Session()
        retry_adapter = requests.adapters.HTTPAdapter(
            max_retries=requests.packages.urllib3.util.Retry(
                total=RETRY_ATTEMPTS,
                backoff_factor=RETRY_DELAY,
                status_forcelist=[500, 502, 503, 504]
            )
        )
        session.mount('http://', retry_adapter)
        session.mount('https://', retry_adapter)
        return session

    def _check_dependencies(self):
        """Check for required dependencies"""
        # Check for FFmpeg wrapper library
        try:
            # Try to import as ffmpeg_python to avoid name collisions
            import ffmpeg as ffmpeg_lib
            # Verify it's the correct library by checking for the input method
            if hasattr(ffmpeg_lib, 'input'):
                self.ffmpeg = ffmpeg_lib
                logger.debug("Successfully loaded ffmpeg-python library")
            else:
                logger.warning(
                    "Found 'ffmpeg' module but it's not the ffmpeg-python package. Will use subprocess fallback.")
                self.ffmpeg = None
        except ImportError:
            logger.warning("ffmpeg-python not installed. Will use subprocess fallback for media operations.")
            self.ffmpeg = None

        # Check for scene detection capability
        try:
            from scenedetect import SceneManager, VideoManager
            from scenedetect.detectors import ContentDetector
            self._scene_detect_available = True
        except ImportError:
            logger.warning("SceneDetect not available. Using basic frame extraction.")
            self._scene_detect_available = False

    def call(self, params: Union[str, Dict], **kwargs) -> AnalysisResult:
        """Execute media analysis"""
        result: AnalysisResult = {
            'status': 'success',
            'data': None,
            'error': None
        }

        try:
            # Parse and validate parameters
            params = self._parse_params(params)
            logger.info(f"Starting analysis task: {params['url']}")

            with temp_directory() as temp_dir:
                # Process input file
                media_path = self._process_input(params['url'], temp_dir)
                self._validate_media_file(media_path)

                # Determine media type
                is_audio = self._is_audio_only(media_path)

                # Audio transcription
                audio_path = media_path if is_audio else self._extract_audio(media_path, temp_dir)
                transcript = self._transcribe_audio(audio_path)

                # Key frame extraction (for videos only)
                frames = []
                if not is_audio:
                    frames = self._extract_keyframes(
                        media_path,
                        min(params['num_frames'], self.config['max_frames'])
                    )

                # AI analysis
                analysis_result = self._analyze_media(
                    prompt=params['prompt'],
                    transcript=transcript,
                    frames=frames,
                    is_audio=is_audio
                )

                result['data'] = {
                    'transcript': transcript,
                    'frame_count': len(frames),
                    'analysis': analysis_result
                }

        except Exception as e:
            result.update({
                'status': 'error',
                'error': {
                    'type': type(e).__name__,
                    'message': str(e),
                    'details': getattr(e, 'details', '')
                }
            })
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)

        return result

    def _parse_params(self, params: Union[str, Dict]) -> Dict:
        """Parse and validate parameters"""
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON parameters: {str(e)}")

        required = ['url', 'prompt']
        missing = [f for f in required if f not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")

        return {
            'url': params['url'],
            'prompt': params['prompt'],
            'num_frames': min(
                int(params.get('num_frames', DEFAULT_FRAMES)),
                self.config['max_frames']
            )
        }

    def _process_input(self, url: str, temp_dir: Path) -> Path:
        """Process input URL/path and get local file path"""
        parsed = urlparse(url)
        if parsed.scheme in ('http', 'https'):
            return self._download_media(url, temp_dir)
        return self._resolve_local_path(url)

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration in seconds"""
        # Try ffmpeg-python first
        if self.ffmpeg:
            try:
                probe = self.ffmpeg.probe(str(video_path))
                return float(probe['format']['duration'])
            except Exception as e:
                logger.warning(f"ffmpeg-python probe failed: {str(e)}")

        # Fallback to subprocess
        try:
            import subprocess
            import json
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'json', str(video_path)
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            data = json.loads(result.stdout)
            return float(data['format']['duration'])
        except Exception as e:
            logger.warning(f"Subprocess duration check failed: {str(e)}")
            # Default to a reasonable duration if all else fails
            return 60.0  # Assume 1 minuteimport base64

    def _download_media(self, url: str, save_dir: Path) -> Path:
        """Download remote media file with validation"""
        logger.info(f"Starting download: {url}")

        try:
            # Pre-validate request
            head_res = self.http_session.head(url, timeout=10)
            head_res.raise_for_status()

            # Validate content type
            content_type = head_res.headers.get('Content-Type', '')
            file_ext = self._get_file_extension(content_type, url)
            if not self._is_supported_type(file_ext):
                raise ValueError(f"Unsupported file type: {file_ext}")

            # Validate file size
            content_length = int(head_res.headers.get('Content-Length', 0))
            if content_length > self.config['max_file_size']:
                raise ValueError(
                    f"File size ({content_length / 1e6:.2f}MB) exceeds limit ({self.config['max_file_size'] / 1e6}MB)"
                )

            # Download file in chunks
            save_path = save_dir / f"media_{int(time.time())}{file_ext}"
            with self.http_session.get(url, stream=True, timeout=self.config['timeout']) as res:
                res.raise_for_status()
                self._stream_write_file(res, save_path)

            logger.info(f"Download completed: {save_path}")
            return save_path

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Download failed: {str(e)}") from e

    def _stream_write_file(self, response: requests.Response, save_path: Path) -> None:
        """Stream file content to disk with progress tracking"""
        total_size = 0
        start_time = time.time()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)

                    # Log progress periodically
                    if time.time() - start_time > 1:
                        logger.debug(f"Downloaded: {total_size / 1e6:.2f}MB")
                        start_time = time.time()

                    if total_size > self.config['max_file_size']:
                        raise ValueError("File size exceeds limit")

    def _resolve_local_path(self, path: str) -> Path:
        """Resolve local file path, handling relative paths"""
        media_path = Path(path)
        if not media_path.is_absolute():
            base_path = Path(os.getenv('PROJECT_ROOT', os.getcwd()))
            media_path = base_path / media_path

        if not media_path.exists():
            raise FileNotFoundError(f"File not found: {media_path}")
        return media_path

    def _validate_media_file(self, path: Path) -> None:
        """Validate media file existence and size"""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_size = path.stat().st_size
        if file_size > self.config['max_file_size']:
            raise ValueError(
                f"File size ({file_size / 1e6:.2f}MB) exceeds limit ({self.config['max_file_size'] / 1e6}MB)"
            )

        if not self._is_supported_type(path.suffix):
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def _is_supported_type(self, extension: str) -> bool:
        """Check if file type is supported"""
        ext = extension.lower().lstrip('.')
        return ext in {ext.lstrip('.') for ext in SUPPORTED_VIDEO_TYPES | SUPPORTED_AUDIO_TYPES}

    def _get_file_extension(self, content_type: str, url: str) -> str:
        """Get file extension from content type or URL"""
        # Try from Content-Type
        if content_type:
            type_map = {
                'video/mp4': '.mp4',
                'video/quicktime': '.mov',
                'audio/mpeg': '.mp3',
                'audio/wav': '.wav',
                'audio/aac': '.aac'
            }
            if ext := type_map.get(content_type.split(';')[0]):
                return ext

        # Try from URL path
        if path_ext := Path(urlparse(url).path).suffix:
            return path_ext

        return '.mp4'  # Default extension

    def _is_audio_only(self, path: Path) -> bool:
        """Detect if file is audio-only"""
        # Check by extension first
        if path.suffix.lower() in SUPPORTED_AUDIO_TYPES:
            return True

        # Then try to use ffmpeg probe
        if self.ffmpeg:
            try:
                probe = self.ffmpeg.probe(str(path))
                return not any(s['codec_type'] == 'video' for s in probe['streams'])
            except Exception as e:
                logger.warning(f"Media probe failed: {str(e)}")

        # If ffmpeg-python not available, use subprocess
        try:
            import subprocess
            cmd = ['ffprobe', '-v', 'error', '-show_entries',
                   'stream=codec_type', '-of', 'json', str(path)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            import json
            probe_data = json.loads(result.stdout)
            return not any(s.get('codec_type') == 'video'
                           for s in probe_data.get('streams', []))
        except Exception as e:
            logger.warning(f"Subprocess probe failed: {str(e)}")
            # If all else fails, use file extension
            return path.suffix.lower() in SUPPORTED_AUDIO_TYPES

    def _extract_audio(self, video_path: Path, temp_dir: Path) -> Path:
        """Extract audio from video"""
        logger.info(f"Extracting audio: {video_path}")
        output_path = temp_dir / f"audio_{video_path.stem}.mp3"

        # First try using ffmpeg-python if available
        if self.ffmpeg:
            try:
                (
                    self.ffmpeg.input(str(video_path))
                    .output(str(output_path), vn=None, acodec='libmp3lame', loglevel='error')
                    .run(overwrite_output=True)
                )
                return output_path
            except Exception as e:
                logger.warning(f"ffmpeg-python extraction failed: {str(e)}. Trying subprocess fallback.")
                # Fall through to subprocess method

        # Fallback to direct subprocess call
        try:
            import subprocess
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vn', '-acodec', 'libmp3lame',
                '-y', str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Audio extraction failed: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Audio extraction failed: {str(e)}") from e

    def _transcribe_audio(self, audio_path: Path) -> str:
        """Transcribe audio to text"""
        logger.info(f"Starting transcription: {audio_path}")
        start_time = time.time()

        try:
            with open(audio_path, 'rb') as f:
                base64_audio = base64.b64encode(f.read()).decode()

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Completely transcribe this audio content with all details"},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:audio/mp3;base64,{base64_audio}",
                            "format": "mp3"
                        }
                    }
                ]
            }]
            response = self.client.chat.completions.create(
                model=self.config['video_model'],
                messages=messages,
                stream=True
            )

            transcript = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    transcript.append(chunk.choices[0].delta.content)

            final_text = ''.join(transcript).strip()
            logger.info(f"Transcription completed (time: {time.time() - start_time:.1f}s, chars: {len(final_text)})")
            return final_text

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return ""

    def _extract_keyframes(self, video_path: Path, num_frames: int) -> List[str]:
        """Extract key frames intelligently"""
        logger.info(f"Extracting key frames: {video_path}")
        frames = []

        try:
            import ffmpeg

            # Use scene detection if available
            if self._scene_detect_available:
                frames = self._extract_frames_with_scene_detection(video_path, num_frames)
            else:
                frames = self._extract_frames_uniform(video_path, num_frames)

            return frames

        except ImportError as e:
            logger.error(f"Missing dependency: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Frame extraction failed: {str(e)}")
            return []

    def _extract_frames_with_scene_detection(self, video_path: Path, num_frames: int) -> List[str]:
        """Extract frames based on scene changes"""
        try:
            from scenedetect import detect, ContentDetector

            # Detect scene changes
            scene_list = detect(str(video_path), ContentDetector(threshold=30))
            timestamps = [scene[0].get_seconds() for scene in scene_list]

            # Get video duration
            duration = self._get_video_duration(video_path)

            # If no scenes detected or too few, use uniform sampling
            if not timestamps or len(timestamps) < num_frames:
                # Calculate how many additional frames we need
                additional_needed = num_frames - len(timestamps)
                if additional_needed > 0:
                    # Create evenly spaced timestamps for remaining frames
                    interval = duration / (additional_needed + 1)
                    extra_timestamps = [interval * (i + 1) for i in range(additional_needed)]
                    timestamps.extend(extra_timestamps)
                    timestamps.sort()

            # If too many scenes detected, select a representative subset
            if len(timestamps) > num_frames:
                step = len(timestamps) // num_frames
                timestamps = [timestamps[i] for i in range(0, len(timestamps), step)][:num_frames]

            # Capture frames at timestamps
            return [
                self._frame_to_base64(self._capture_frame(video_path, ts))
                for ts in timestamps[:num_frames]
                if self._capture_frame(video_path, ts) is not None
            ]

        except Exception as e:
            logger.warning(f"Scene detection failed, falling back to uniform sampling: {str(e)}")
            return self._extract_frames_uniform(video_path, num_frames)

    def _extract_frames_uniform(self, video_path: Path, num_frames: int) -> List[str]:
        """Extract frames at uniform intervals"""
        try:
            # Get video duration
            duration = self._get_video_duration(video_path)

            # Calculate evenly spaced timestamps
            interval = duration / (num_frames + 1)
            timestamps = [interval * (i + 1) for i in range(num_frames)]

            # Capture frames
            return [
                self._frame_to_base64(self._capture_frame(video_path, ts))
                for ts in timestamps
                if self._capture_frame(video_path, ts) is not None
            ]

        except Exception as e:
            logger.error(f"Uniform frame extraction failed: {str(e)}")
            return []

    def _capture_frame(self, video_path: Path, timestamp: float) -> Optional[Image.Image]:
        """Capture a video frame at specified timestamp"""
        output_file = video_path.parent / f"frame_{timestamp}.jpg"

        # Try ffmpeg-python if available
        if self.ffmpeg:
            try:
                (
                    self.ffmpeg.input(str(video_path), ss=timestamp)
                    .output(str(output_file), vframes=1, q=2, loglevel='error')
                    .run(overwrite_output=True)
                )
                return Image.open(output_file)
            except Exception as e:
                logger.warning(f"ffmpeg-python frame capture failed: {str(e)}")
                # Fall through to subprocess method

        # Fallback to subprocess
        try:
            import subprocess
            cmd = [
                'ffmpeg', '-ss', str(timestamp),
                '-i', str(video_path),
                '-vframes', '1', '-q:v', '2',
                '-y', str(output_file)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return Image.open(output_file)
        except Exception as e:
            logger.warning(f"Frame capture failed at {timestamp}s: {str(e)}")
            return None

    def _frame_to_base64(self, image: Image.Image) -> str:
        """Convert image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffered.getvalue()).decode()

    def _analyze_media(self, prompt: str, transcript: str, frames: List[str], is_audio: bool) -> str:
        """Analyze media using AI model"""
        logger.info(f"Starting AI analysis ({'audio' if is_audio else 'video'})")
        messages = self._build_analysis_messages(prompt, transcript, frames, is_audio)
        try:
            response = self.client.chat.completions.create(
                model=self.config['analysis_model'],
                messages=messages,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"AI analysis failed: {str(e)}")
            return "Analysis generation failed"

    def _build_analysis_messages(self, prompt: str, transcript: str, frames: List[str], is_audio: bool) -> List[Dict]:
        """Build prompt messages for analysis"""
        system_prompt = (
            f"You are a professional {'audio' if is_audio else 'video'} analysis expert. "
            "Your task is to analyze the provided content by:\n"
            "1. Identifying key information and contextual relationships\n"
            "2. Noting time-sequence information\n"
            "3. Providing expert answers to the user's question"
        )

        content = [
            {"type": "text", "text": f"User question: {prompt}\n\nAudio transcription:\n{transcript}"}
        ]

        if not is_audio:
            content.extend([
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                for img in frames
            ])

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content}
        ]
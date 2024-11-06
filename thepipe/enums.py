from typing import Dict, Any, Optional, List, Union
from enum import Enum

class YouTubeEnum(Enum):
    TITLE = 'title'
    DESCRIPTION = 'description'
    UPLOAD_DATE = 'upload_date'
    UPLOADER = 'uploader'
    VIEW_COUNT = 'view_count'
    LIKE_COUNT = 'like_count'
    DURATION = 'duration'
    TAGS = 'tags'
    CATEGORY = 'categories'
    THUMBNAIL_URL = 'thumbnail'
    
    @classmethod
    def extract(cls, info: Dict[str, Any]) -> Dict[str, Any]:
        return {field.name.lower(): info.get(field.value, 'N/A') for field in cls}

    @staticmethod
    def process_options(options: Dict[str, Any], text_only: Optional[Union[bool, str]], verbose: bool) -> Dict[str, Any]:
        ydl_opts = options.copy()  # Start with user-provided options

        ydl_opts['quiet'] = not verbose
        ydl_opts['ignoreerrors'] = True
        ydl_opts['extract_flat'] = 'in_playlist'

        if text_only == 'transcribe':
            ydl_opts['format'] = 'bestaudio/best'
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
            ydl_opts['skip_download'] = False
        elif text_only in [True, 'ai', 'uploaded']:
            ydl_opts['writesubtitles'] = True
            ydl_opts['writeautomaticsub'] = True
            if text_only == 'ai':
                ydl_opts['subtitleslangs'] = ['a.en,a.*', 'en,*']
            elif text_only == 'uploaded':
                ydl_opts['subtitleslangs'] = ['en,*', 'a.en,a.*']
            else:  # text_only is True
                ydl_opts['subtitleslangs'] = ['en,*', 'a.en,a.*']
            ydl_opts['skip_download'] = True
        else:  # text_only is False or None
            ydl_opts['format'] = 'bestvideo+bestaudio/best'
            ydl_opts['writesubtitles'] = True
            ydl_opts['writeautomaticsub'] = True
            ydl_opts['subtitleslangs'] = ['en,*', 'a.en,a.*']
            ydl_opts['skip_download'] = False

        return ydl_opts

    @staticmethod
    def extract_metadata(video_info: Dict[str, Any], metadata_fields: Optional[List[Enum]] = None) -> Dict[str, Any]:
        if metadata_fields is None:
            metadata_fields = DEFAULT_METADATA_FIELDS
        return {field.name.lower(): YouTubeEnum.extract(video_info)[field.name.lower()] for field in metadata_fields}

    @staticmethod
    def format_metadata(metadata: Dict[str, Any]) -> str:
        formatted_metadata = []
        for key, value in metadata.items():
            if isinstance(value, list):
                value = ', '.join(map(str, value))
            formatted_metadata.append(f"{key.replace('_', ' ').title()}: {value}")
        return '\n'.join(formatted_metadata)

DEFAULT_METADATA_FIELDS = list(YouTubeEnum)
from typing import Dict, Any

class YouTubeEnum:
    @staticmethod
    def process_options(options: Dict[str, Any], text_only: bool, verbose: bool) -> Dict[str, Any]:
        ydl_opts = options.copy()  # Start with user-provided options

        # Override or set options based on thepipe arguments
        if text_only:
            ydl_opts['format'] = 'bestaudio/best'
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
        else:
            ydl_opts.setdefault('format', 'bestvideo+bestaudio/best')

        ydl_opts['quiet'] = not verbose

        return ydl_opts
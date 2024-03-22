import setup
import argparse
from extractor import get_all_contents
from representor import create_compressed_project_context

def create_prompt_from_context(context):
    messages = [
        {
          "role": "user",
          "content": [
                {
                  "type": "text",
                  "text": context['text']
                }] + [
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                  }
                } for base64_image in context['images']
            ]
        }
    ]
    return messages

def extract(source, ignore=[], limit=64000, verbose=False, use_mathpix=False, use_text=False):
    files = get_all_contents(source, substrings_to_ignore=ignore, verbose=verbose, use_mathpix=use_mathpix, use_text=use_text)
    context = create_compressed_project_context(files, token_limit=limit, verbose=verbose)
    messages = create_prompt_from_context(context)
    return messages
    
def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Compress project files into a context for NanoGPT.')
    parser.add_argument('--source', type=str, required=True,
                        help='The input source directory containing the files to be processed.')
    parser.add_argument('--output', type=str, required=True,
                        help='The output file to save the compressed context to.')
    parser.add_argument('--ignore', type=str, nargs='*',
                        help='Substrings to ignore in the files. Provide each substring as a separate argument.',
                        default=[])
    parser.add_argument('--limit', type=float, default=64000,
                        help='The token limit for the compressed project context.')
    parser.add_argument('--mathpix', action='store_true', help='Use Mathpix to extract text from images.')
    parser.add_argument('--silent', action='store_true', help='Use Mathpix to extract text from images.')
    parser.add_argument('--text', action='store_true', help='Scrapes all images for text.')
    
    args = parser.parse_args()
    # Make context from source
    messages = extract(args.source, args.ignore, args.limit, verbose=not args.silent, use_mathpix=args.mathpix, use_text=args.text)
    # Save output to file
    with open(args.output, 'w', encoding='utf-8') as output_file:
        output_file.write(messages[0]['content'][0]['text'])

if __name__ == '__main__':
    main()
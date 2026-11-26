"""Download or verify pinned public Qwen3.8/Qwen3.6 checkpoints and file hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

MODELS = {
    'qwen38': ('Qwen/Qwen3.8-27B', '37561816e66b5b9153b1b64c10968c18c1e6f4a0'),
    'qwen36': ('Qwen/Qwen3.6-35B-A3B', '345d467b117d1fd5eda9b879a5988de9ae5f47a7'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, default='qwen38')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    model_id, revision = MODELS[args.model]
    url = f'https://huggingface.co/api/models/{model_id}/revision/{revision}?blobs=true'
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    if metadata['sha'] != revision:
        raise RuntimeError('Hub returned a different model revision')
    entries = [entry for entry in metadata['siblings']
               if Path(entry['rfilename']).name == entry['rfilename']
               and entry['rfilename'] not in ('.gitattributes', 'README.md', 'crc32.txt')]
    if not args.verify_only:
        from huggingface_hub import snapshot_download
        snapshot_download(model_id, revision=revision, local_dir=args.output_dir,
                          cache_dir=args.output_dir/'.cache', max_workers=4,
                          allow_patterns=[entry['rfilename'] for entry in entries])
    manifest = dict(repo=model_id, revision=revision, files={})
    for entry in entries:
        name = entry['rfilename']
        path = args.output_dir/name
        if path.stat().st_size != entry['size']:
            raise RuntimeError('File size mismatch: '+name)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        expected = entry.get('lfs', {}).get('sha256')
        if expected and digest != expected:
            raise RuntimeError('SHA-256 mismatch: '+name)
        manifest['files'][name] = dict(size=path.stat().st_size, sha256=digest)
        print('Verified', name, flush=True)
    target = args.output_dir/'download_manifest.json'
    target.write_text(json.dumps(manifest, indent=2)+'\n')
    print('Saved', target, flush=True)


if __name__ == '__main__':
    main()

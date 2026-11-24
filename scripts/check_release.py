"""Check tracked files/history for credentials and private infrastructure references.

Only locations and categories are reported; matching values are never printed.
"""
from pathlib import Path
import argparse
import ipaddress
import json
import re
import subprocess
from urllib.parse import urlsplit

RULES = {
    'private_key': r'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----',
    'provider_token': r'\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{40,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|xox[baprs]-[A-Za-z0-9-]{10,})\b',
    'secret_assignment': r'''(?:api[_-]?key|secret|token|password|passwd|pwd)\s*[:=]\s*["'][^"'\s]{16,}["']''',
    'internal_absolute_path': r'/(?:dev_share|ai_share|prod_share|data/ai_share|root|home)/',
    'private_hostname': r'\b[\w.-]+\.(?:internal|corp|local|lan)\b',
}
PUBLIC_HOSTS = {'github.com', 'huggingface.co', 'users.noreply.github.com'}
PUBLIC_NAMESPACES = {'http://www.w3.org/2000/svg'}


def scan(text, origin):
    found=[]
    for i,line in enumerate(text.splitlines(),1):
        for label,pattern in RULES.items():
            if re.search(pattern,line,re.I):
                found.append({'file':origin,'line':i,'type':label})
        for token in re.findall(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])',line):
            try:
                ipaddress.ip_address(token)
                found.append({'file':origin,'line':i,'type':'ip_address'})
            except ValueError:
                pass
        for url in re.findall(r'https?://[^\s)\]>"\']+',line):
            if url in PUBLIC_NAMESPACES:
                continue
            parsed=urlsplit(url)
            if parsed.username or parsed.password:
                found.append({'file':origin,'line':i,'type':'url_credentials'})
            if parsed.hostname and parsed.hostname not in PUBLIC_HOSTS:
                found.append({'file':origin,'line':i,'type':'unreviewed_endpoint'})
    return found


def scan_blob(data, origin):
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        # Image pixels are binary. Inspect chunks rather than decode pixels as
        # text; reject text metadata and EXIF that could contain private paths.
        offset = 8
        while offset + 12 <= len(data):
            size = int.from_bytes(data[offset:offset+4], 'big')
            kind = data[offset+4:offset+8]
            if offset + size + 12 > len(data):
                return [{'file':origin, 'type':'invalid_png'}]
            if kind in {b'tEXt', b'zTXt', b'iTXt', b'eXIf'}:
                return [{'file':origin, 'type':'image_metadata_requires_review'}]
            offset += size + 12
            if kind == b'IEND':
                return [] if offset == len(data) else [{'file':origin, 'type':'png_trailing_data'}]
        return [{'file':origin, 'type':'invalid_png'}]
    try:
        return scan(data.decode('utf-8'), origin)
    except UnicodeDecodeError:
        return [{'file':origin, 'type':'unreviewed_binary_file'}]


def git(*args):
    return subprocess.check_output(['git',*args])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history',action='store_true')
    args=p.parse_args()
    findings=[]
    files=git('ls-files','-z').decode().split('\0')
    for name in filter(None,files):
        f=Path(name)
        if f.is_symlink():
            findings.append({'file':name,'type':'symlink'})
        elif f.is_file():
            findings.extend(scan_blob(f.read_bytes(),name))
    blobs=set()
    if args.history:
        for row in git('rev-list','--objects','--all').decode().splitlines():
            oid,_,name=row.partition(' ')
            if git('cat-file','-t',oid).strip()==b'blob' and oid not in blobs:
                blobs.add(oid)
                findings.extend(scan_blob(git('cat-file','blob',oid),f'history:{name}'))
        findings.extend(scan(git('log','--all','--format=fuller').decode(),'commit_metadata'))
    print(json.dumps({'files':sum(bool(x) for x in files),'history_blobs':len(blobs),'findings':findings},indent=2))
    raise SystemExit(2 if findings else 0)


if __name__=='__main__':
    main()

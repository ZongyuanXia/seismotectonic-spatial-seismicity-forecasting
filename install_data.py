"""Install the companion data tree into workflow without overwriting source code."""
from pathlib import Path
import argparse,hashlib,shutil

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data_directory',type=Path);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();source=a.data_directory.resolve();dest=Path(__file__).resolve().parent/'workflow'
    assert (source/'manuscript/tables/Table1.csv').is_file(),'Not the companion data directory'
    files=[f for f in source.rglob('*') if f.is_file() and f.name not in ('README.md','LICENSE','FILE_MANIFEST.csv','SHA256SUMS.txt')]
    for f in files:
        q=dest/f.relative_to(source)
        if q.exists():assert sha(q)==sha(f),f'Would overwrite a different file: {q}'
    if not a.dry_run:
        for f in files:
            q=dest/f.relative_to(source)
            if not q.exists():q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,q)
    print(('Validated' if a.dry_run else 'Installed'),len(files),'data files; no model execution.')
if __name__=='__main__':main()

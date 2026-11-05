"""Run an unchanged source snapshot with explicit runtime path overrides."""
import argparse
from pathlib import Path
import importlib
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--script',type=Path,required=True)
    p.add_argument('--path',action='append',default=[])
    a,extra=p.parse_known_args()
    path=a.script.resolve();sys.path.insert(0,str(path.parent))
    if extra[:1]==['--']:extra=extra[1:]
    sys.argv=[str(path),*extra]
    # A real importable module name is required by multiprocessing spawn.
    namespace=importlib.import_module(path.stem)
    function=namespace.main
    for override in a.path:
        name,value=override.split('=',1);function.__globals__[name]=Path(value)
    function()


if __name__=='__main__':main()

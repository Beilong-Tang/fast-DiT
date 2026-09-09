import shutil
import os
import time
import datetime
import argparse
import glob
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type = str)
    parser.add_argument("tgt_dir", type = str)
    parser.add_argument("--sleep_time", type = int, default = 30, help='check frequency in seconds')
    parser.add_argument("--num_files_keep", type = int, default = 1, help='num files to keep in the src_dir')
    return args

def main(args):
    assert os.path.isdir(args.src_dir), f"src_dir {args.src_dir} does not exist"
    os.makedirs(args.tgt_dir, exist_ok=True)
    while True:
        files = glob.glob(os.path.join(args.src_dir, "*.pt"))
        files = sorted(files, key=lambda x: int(Path(x).stem))
        print(f"{datetime.datetime.now()} - found files {files}")
        for index, f in enumerate(files):
            if index < args.num_files_keep:
                pass
            else:
                shutil.move(f, args.tgt_dir)
        time.sleep(args.sleep_time)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
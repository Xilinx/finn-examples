import argparse
import glob
import json
import os
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", action="store_true", help="download data files")
parser.add_argument("-b", "--board", help="select which board's bitfiles to download")
parser.add_argument("-r", "--remove", action="store_true", help="delete data and bitfiles")

data_links_dir = "/../finn_examples/data"
bit_links_dir = "/../finn_examples/bitfiles"
bitfile = "bitfiles.zip.link"
bitfile_zip_dir = "bitfiles.zip.d"


def runcmd(cmd, verbose=False, *args, **kwargs):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


def load_data_links(file):
    with open(file, "r") as f:
        data = json.load(f)
    url = data["url"]
    print("found url: {}".format(url))
    return url


def load_bitfile_links(file, board):
    with open(file, "r") as f:
        data = json.load(f)
    url = data[board]["url"]
    print("found bitfile url: {}".format(url))
    return url


if __name__ == "__main__":
    args = parser.parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))

    if args.remove:
        os.chdir(root_dir + data_links_dir)
        runcmd("rm -rf audio_samples")
        runcmd("rm -rf *.npy")
        runcmd("rm -rf *.zip")
        os.chdir(root_dir + bit_links_dir)
        runcmd("rm -rf {}".format(bitfile_zip_dir))
        print("Files deleted")
        sys.exit()

    # Download the data files needed for certain notebooks
    if args.data:
        os.chdir(root_dir + data_links_dir)

        for link_file in glob.glob("*.link"):
            url = load_data_links(link_file)
            runcmd("wget {}".format(url))
            zip_file = os.path.basename(url)
            runcmd("unzip {}".format(zip_file))

    # Download bitstreams for the requested board
    if args.board:
        os.chdir(root_dir + bit_links_dir)

        bitfile = "bitfiles.zip.link"
        bitfile_zip_dir = "bitfiles.zip.d"
        bitfile_url = load_bitfile_links(bitfile, args.board)

        # make bitstream dir and
        runcmd("mkdir {}".format(bitfile_zip_dir))
        os.chdir(bitfile_zip_dir)
        runcmd("wget {}".format(bitfile_url))
        zip_file = os.path.basename(bitfile_url)
        runcmd("unzip {}".format(zip_file))

print("Done")

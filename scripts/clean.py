import glob
import os
import shutil


def clean_dir(directory: str):

    file_list = [file_ for file_ in glob.glob(directory + "/*")]

    for file_ in file_list:
        os.remove(file_)


if __name__ == "__main__":

    tmpdir = "./tmpdir"
    outdir_detection = "data/output/detection"
    outdir_reassembly = "data/output/reassembly"

    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)

    clean_dir(outdir_detection)
    clean_dir(outdir_reassembly)

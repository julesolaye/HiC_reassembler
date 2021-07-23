# Script used to delete the previous output files before make a new detection/reassembly.
import glob
import os
import shutil


def clean_dir(directory: str):

    file_list = [file_ for file_ in glob.glob(directory + "/*")] # Delete every file inside the directory

    for file_ in file_list:
        os.remove(file_)


if __name__ == "__main__":

    tmpdir = "./tmpdir"
    outdir_detection = "data/output/detection"
    outdir_reassembly = "data/output/reassembly"

    if os.path.exists(tmpdir): # If an issue has occured during the detection 
                                # and the tmpdir has not been deleted, it will delete it.
        shutil.rmtree(tmpdir)

    clean_dir(outdir_detection)
    clean_dir(outdir_reassembly)

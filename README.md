# Hi-C reassembler

This repo contains a program to detect SVs on scrambled Hi-C maps with machine learning methods and reassemble the maps with a bruteforce method. It handles the following structural variations:

* Deletion: Chunk of sequence removed
* Insertion: New chunk of sequence introduced
* Inversion(short): Chunk of sequence flipped
* Translocation: Chunk of sequence moved from one place to another

These alterations can be happen sequentially and be superimposed on each other, which result in "complex events". These programs allow to reassemble this complex events after they have been detected.

The Hi-C map must be generated with the aligner "bwa".

## Setup

To install python dependencies, you can use the requirements.txt file as follows:

```bash
pip install --user -r requirements.txt
```

To setup install the project as a local package, run:

```bash
pip install --user -e .
```

## Training

This program works with machine learning methods, it needs to have in input training set for each detector:

* Matrixdetector, which will detect SVs on Hi-C maps,
* BAMdetector, which detect the exact position on bam files of the structural variations,
* RepeatsFinder and BadMappedFinder, which respectively find repeats or positions where there are a lot of reads which are not correctly mapped. These two detector are used by BAMdetector.

When all training sets are in the folder "data/training", you can train the models:

```train
make train
```

## Detection & reassembly

After the training has been done, the program can detect SVs on Hi-C maps and reassemble the matrix. The program needs to have a Hi-C matrix (format npy), the sequence associated (format fasta), the name of the chromosome linked to the matrix and the binsize which has been used to generate the matrix. To reassemble, you must run the script with the good arguments:

```reassembler
binsize = 10000
hic_file = "data/testing/scrambled.npy"
seq_file = "data/testing/seq.fa"
bam_file = "data/testing/scrambled.for.bam"
chrom_name = "Sc_chr04"

python ./scripts/reassemble.py -b binsize hic_file seq_file bam_file chrom_name
```

It is also possible to just detect the structural variations, without reassembly:

```detection
binsize = 10000
hic_file = "data/testing/scrambled.npy"
seq_file = "data/testing/seq.fa"
bam_file = "data/testing/scrambled.for.bam"
chrom_name = "Sc_chr04"

python ./scripts/detect.py -b binsize hic_file seq_file bam_file chrom_name
```

Before the reassembly, it can be important to clean the ouput directory with the script associated:

```clean
python ./scripts/clean.py
```

## Output

The programs will generate an output directory containing different files. Firstly, it will generate in a folder four files linked to the detection:

* "DEL_detected.npy", a npy file with the coordinates on the BAM files of the deletions detected, 
* "INS_detected.npy", a npy file with the coordinates on the BAM files of the insertions detected, 
* "INV_detected.npy", a npy file with the coordinates on the BAM files of the inversions detected,
* "TRA_detected.npy", a npy file with the coordinates on the BAM files of the translocations detected.  

If you proceed to the reassembly, the program will also generate 3 files in a different folder linked to the reassembly of the Hi-C map:

* "mat_reassembled.npy", a file with the Hi-C matrix which has been reassembled,
* "seq_reassembled.fa", a fasta file with the sequence which has been reassembled,
* "difference.png", a png file with a picture where we can see the Hi-C map before and after the reassembly (in order to compare).
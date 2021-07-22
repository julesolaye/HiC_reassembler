.PHONY: clean

clean: 
	@python ./scripts/clean.py
	
detect: clean
	@python ./scripts/detect.py data/testing/scrambled.npy data/testing/mod_genome.fa data/testing/scrambled.for.bam Sc_chr04
	
reassemble: clean
	@python ./scripts/reassemble.py data/testing/scrambled.npy data/testing/mod_genome.fa data/testing/scrambled.for.bam Sc_chr04
	
train: 
	@python ./scripts/train.py

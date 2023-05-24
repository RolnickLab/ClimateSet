#!/bin/sh

rsync -rv --ignore-existing $SLURM_TMPDIR/causalpaca /home/venka97/scratch
rm -f `find $SLURM_TMPDIR/causalpaca -type f` 

#!/bin/sh

rsync -r --ignore-existing $SLURM_TMPDIR/causalpaca_co2 /home/venka97/scratch
find $SLURM_TMPDIR/causalpaca -type f -delete
find $SLURM_TMPDIR/causalpaca -type d -delete

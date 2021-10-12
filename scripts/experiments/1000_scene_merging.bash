#!/bin/bash

EXPERIMENT_NAME=`basename "$0"`

# merging [x]
sbatch \
    ./scripts/lopri_run.bash \
    data/collation_functions=voxelize_collate_merge

# merging [ ]
sbatch \
    ./scripts/lopri_run.bash

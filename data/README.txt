This readme is for internal use, explaining how to setup the code on the cluster.

Mila Cluster:

1. Sharing data directories

If you want to access the same data dirs with your co-workers:

1. 1. Make subdirs and dirs searchable -> important: your $USER dir must become searchable [x]

setfacl -m   user:$USER2:x   /network/scratch/$USER/$DIR/$DIR2/
setfacl -m   user:$USER2:x   /network/scratch/$USER/$DIR/
setfacl -m   user:$USER2:x   /network/scratch/$USER/

[x]: exectuable
[w]: writeable
[r]: readable

See also: https://github.com/mila-iqia/mila-docs/pull/121

1. 2. use chmod to give user access to dirs and files within your user dir


Explanation of structure:

TODO

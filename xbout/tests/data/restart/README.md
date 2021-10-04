These restart files were created by running the 'conduction' example from the
BOUT-dev repo (at commit 696735c37ce31491c730aaf547795ff7cd593ae5).
```
$ git clone git@github.com:boutproject/BOUT-dev.git
$ cd BOUT-dev
$ ./configure; make
$ cd examples/conduction
$ mpirun -np 2 ./conduction
```
The files are created in the `data` subdirectory: `data/BOUT.restart.{0,1}.nc`.

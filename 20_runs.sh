#!/bin/bash

cd ./build/

# If make works it returns 0 and hence exit wont run
# due to short circuting
make || exit 1
cd ./Linux-x86_64/

# 10 runs
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 1 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 2 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 3 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 4 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 5 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 6 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 7 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 8 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 9 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 10 Done\n\n\n"


./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 11 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 12 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 13 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 14 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 15 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 16 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 17 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 18 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 19 Done\n\n\n"
./popsift-demo --input-file ../../sample_640×426.pgm || exit 1
echo -e "\n\n\n Nr. 20 Done\n\n\n"


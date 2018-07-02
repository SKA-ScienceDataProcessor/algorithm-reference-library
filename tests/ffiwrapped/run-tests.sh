#!/bin/bash

# ######################################################## #
#             Script to run serial tests                   #
# ######################################################## #

#make clean;

TESTPATH=$ARLROOT/workflows/ffiwrapped/serial
TESTLIST='timg_serial ical_demo'

if [ $1 ] 
then
    filelist=$1
else
  #  filelist=`ls *.c`;
   filelist=$TESTLIST;
fi

# clean before we start
cd $TESTPATH
make clean;

export LD_LIBRARY_PATH=$ARLROOT:$LD_LIBRARY_PATH

for testfile in $filelist; do
  
  # file=`basename $testfile .c`;
  file=$testfile
  echo "Running $file";
  make $file >& make.out
  if [ $? -ne 0 ]
  then 
      echo "Compilation failed"
      # store the output for reference
      cp make.out $file.make.out;
      continue
  fi
  # Run the serial test
  ./$file >& run.out
  if [ $? -eq 0 ]
  then
      echo "$testfile passed"
  else
      # store the output and diff file for reference
      echo "$testfile failed"
      cp run.out $file.run.out;
  fi
  rm -f make.out run.out 
done

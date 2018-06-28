#!/bin/bash

# ######################################################## #
#             Script to run serial tests                   #
# ######################################################## #

#make clean;

TESTPATH=$ARLROOT/workflows/ffiwrapped/serial
TESTLIST='timg_serial ical_serial'

if [ $1 ] 
then
    filelist=$1
else
  #  filelist=`ls *.c`;
   filelist=$TESTLIST;
fi

for testfile in $filelist; do
  
  # file=`basename $testfile .c`;
  # Assumes the folder has the same name as the executable file
  file=$testfile
  cd $file
  make clean;
  echo "Running $file";
  make $file >& make.out
  if [ $? -ne 0 ]
  then 
      echo "Compilation failed"
      continue
  fi
  # Run the serial test
  #./$file >& run.out
  make run >& run.out
  if [ $? -eq 0 ]
  then
      echo "$testfile passed"
  else
      # store the output and diff file for reference
      echo "$testfile failed"
      cp make.out $file.make.out;
      cp run.out $file.run.out;
  fi
  rm -f make.out run.out 
  cd ..
done

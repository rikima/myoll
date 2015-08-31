#!/bin/sh
cur=$(dirname $0)
jars=$cur/target/myoll-0.1-SNAPSHOT.jar
for jar in $(ls $cur/lib/*jar) ; do
    jars=$jars:$jar
done
program=com.rikima.ml.oll.ClassifyDriver

java -cp $jars $program $*

#!/usr/bin/bash

# Generate site-specific hazard curves by calling HazardCurveCalc.java

longitude=-122.259
latitude=37.871
vs30=733.4

cd src

# make a directory to store the output
# (if it does not exist)
site_hazard_path="../analysis/site_hazard/"
mkdir -p $site_hazard_path

# download the .jar file (if it is not there)
jar_file_path="../lib/opensha-all.jar"
if [ -f "$jar_file_path" ]; then
    echo "The file exists."
else
    echo "The file does not exist. Downloading file."
    wget -P ../lib/ "http://opensha.usc.edu/apps/opensha/nightlies/latest/opensha-all.jar"
fi

# compile java code if it has not been compiled already
javafile_path="HazardCurveCalc.class"
if [ -f "$javafile_path" ]; then
    echo "Already compiled"
else
    echo "Compiling java file."
    javac -classpath $jar_file_path:. HazardCurveCalc.java
fi

# obtain hazard curve data
periods=(0.010 0.020 0.030 0.050 0.075 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.75 1.0 1.5 2.0 3.0 4.0 5.0 7.5 10.0)
curve_names=(0p01 0p02 0p03 0p05 0p075 0p1 0p15 0p2 0p25 0p3 0p4 0p5 0p75 1p0 1p5 2p0 3p0 4p0 5p0 7p5 10p0)

# run calculations in batches of three
for idx in {0..7}
do
    for idxadd in {0..2}
    do
	subidx=$(expr $idx + $idxadd)
	java -classpath $jar_file_path:. HazardCurveCalc ${periods[$subidx]} 37.871 -122.259 733.4 "$site_hazard_path""${curve_names[$subidx]}.txt" &
    done
    wait
done



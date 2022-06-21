#!/usr/bin/bash

# Perform seismic hazard deaggregation using DisaggregationCalc.java
# and get GMM mean and stdev results using GMMCalc.java

longitude=$(cat data/study_vars/longitude)
latitude=$(cat data/study_vars/latitude)
vs30=$(cat data/study_vars/vs30)

cd src

site_hazard_path="../analysis/site_hazard/"

# get the codes of the archetypes of this study
arch_codes=$(cat ../data/archetype_codes_response | uniq)


# download the .jar file (if it is not there)
jar_file_path="../lib/opensha-all.jar"
if [ -f "$jar_file_path" ]; then
    echo "The file exists."
else
    echo "The file does not exist. Downloading file."
    wget -P ../lib/ "http://opensha.usc.edu/apps/opensha/nightlies/latest/opensha-all.jar"
fi

# compile java code if it has not been compiled already
javafile_path="DisaggregationCalc.class"
if [ -f "$javafile_path" ]; then
    echo "Already compiled DisaggregationCalc"
else
    echo "Compiling DisaggregationCalc.java"
    javac -classpath $jar_file_path:. DisaggregationCalc.java
fi

javafile_path="GMMCalc.class"
if [ -f "$javafile_path" ]; then
    echo "Already compiled GMMCalc"
else
    echo "Compiling GMMCalc.java"
    javac -classpath $jar_file_path:. GMMCalc.java
fi


for code in $arch_codes
do
    
    # Get the period of that archetype
    period=$(cat ../data/$code/period_closest)

    # Get the hazard level midpoint Sa's
    sas=$(awk -F, '{if (NR!=1) {print $2}}' ../analysis/$code/site_hazard/Hazard_Curve_Interval_Data.csv)

    i=1
    for sa in $sas
    do

	# perform seismic hazard deaggregation
	java -classpath ../lib/opensha-all.jar:. DisaggregationCalc $period $latitude $longitude $vs30 $sa ../analysis/$code/site_hazard/deaggregation_$i.txt

	# generate GMM results
	mbar=$(cat ../analysis/$code/site_hazard/deaggregation_$i.txt | grep Mbar | awk '{print $3}')
	dbar=$(cat ../analysis/$code/site_hazard/deaggregation_$i.txt | grep Dbar | awk '{print $3}')
	java -classpath ../lib/opensha-all.jar:. GMMCalc $mbar $dbar $vs30 ../analysis/$code/site_hazard/gmm_$i.txt

	i=$(($i+1))	

    done
        
done





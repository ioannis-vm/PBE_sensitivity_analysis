// Compile:
//   javac -classpath <path_to_jar> GMMCalc.java

// Execute:
//   java -classpath <path_to_jar>:. GMMCalc <Mw> <rRup> <vs30>
//   e.g.: clear && javac -classpath ../lib/opensha-all.jar GMMCalc.java && java -classpath ../lib/opensha-all.jar:. GMMCalc 7.5 25.00 730.0

// e.g.
// clear && javac -classpath ../lib/opensha-all.jar GMMCalc.java && java -classpath ../lib/opensha-all.jar:. GMMCalc 7.5 25.00 730.0 out.txt

import java.awt.geom.Point2D;

import org.opensha.commons.data.Site;
import org.opensha.commons.data.TimeSpan;
import org.opensha.commons.data.function.ArbitrarilyDiscretizedFunc;
import org.opensha.commons.data.function.DiscretizedFunc;
import org.opensha.commons.geo.Location;
import org.opensha.commons.param.Parameter;
import org.opensha.sha.calc.HazardCurveCalculator;
import org.opensha.sha.earthquake.ERF;
import org.opensha.sha.earthquake.ERF_Ref;
import org.opensha.sha.gui.infoTools.IMT_Info;
import org.opensha.sha.imr.AttenRelRef;
import org.opensha.sha.imr.ScalarIMR;
import org.opensha.sha.imr.param.IntensityMeasureParams.SA_Param;
import static java.lang.Math.exp;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;


class GMMCalc {

    public static void main(String[] args) {

	double Mw = Double.parseDouble(args[0]);
	double rRup = Double.parseDouble(args[1]);
	double vs30 = Double.parseDouble(args[2]);
	String output_file_path = args[3];

	ScalarIMR gmm = AttenRelRef.NGAWest_2014_AVG.instance(null);
	double periods[] = { 0.010, 0.020, 0.030, 0.050, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0 };
	double[] means = new double[21];
	double[] stdvs = new double[21];

		
	for (int i = 0; i < periods.length; i++) {
	    double period = periods[i];
	    gmm.setParamDefaults();
	    gmm.setIntensityMeasure(SA_Param.NAME);
	    gmm.getSiteParams().setValue("vs30", vs30);
	    SA_Param.setPeriodInSA_Param(gmm.getIntensityMeasure(), period);
	    means[i] = exp(gmm.getMean());
	    stdvs[i] = gmm.getStdDev();
	}

	// write results to a text file
	try {
	    File myObj = new File(output_file_path);
	    if (myObj.createNewFile()) {
		System.out.println("File created: " + myObj.getName());
	    } else {
		// file already exists
		myObj.delete();
		myObj.createNewFile();
	    }
	} catch (IOException e) {
	    System.out.println("An error occurred.");
	    e.printStackTrace();
	}
	try {
	    FileWriter myWriter = new FileWriter(output_file_path);

	    myWriter.write("GMM: "+gmm.getName());
	    myWriter.write(""+gmm.getSiteParams());
	    for (int i = 0; i < periods.length; i++) {
		myWriter.write(periods[i]+" "+means[i]+" "+stdvs[i]+"\n");
	    }
	    myWriter.close();
	} catch (IOException e) {
	    System.out.println("An error occurred.");
	    e.printStackTrace();
	}

    }

}

package org.jeevkulk.dataanalytics.ml.datapreparation;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.Bucketizer;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.col;

public class DataCorrelator {

    public static void main(String[] args) {

        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession sparkSession = SparkSession.builder()
                .appName("SparkML")
                .master("local[*]")
                .getOrCreate();

        //******************************************Reading the data file***********************************************
        Dataset<Row> rawDataset = sparkSession.read().option("header", true)
                .option("inferschema", true)
                .csv(DataCorrelator.class.getClassLoader().getResource("data/datapreparation/medical_charges.csv").toString());
        rawDataset = rawDataset.select(col("age").cast("Double"), col("sex"), col("children"),
                col("bmi").cast("Double"), col("smoker"),
                col("region"), col("charges").cast("Double"));
        rawDataset.show();

        //******************************************Removing Records with missing values***********************************
        rawDataset = rawDataset.na().drop();
        double[] splits = {Double.NEGATIVE_INFINITY, 25, 35, 45, 55, Double.POSITIVE_INFINITY};
        Bucketizer bucketizer = new Bucketizer()
                .setInputCol("age")
                .setOutputCol("Bucketed-Age")
                .setSplits(splits);

        //Dividing a continuous variable age into buckets or sub groups using the transformer bucketizer
        //In bucketizer you need to mention the minimum value possible and maximum value possible
        Dataset<Row> aggregatedDataset = bucketizer.transform(rawDataset).select("age", "Bucketed-Age", "sex", "bmi", "smoker", "children", "region", "charges");
        //Grouping the data based on Age-Buckets
        aggregatedDataset.groupBy("Bucketed-Age").agg(functions.avg("charges"), functions.max("bmi")).orderBy("Bucketed-Age").show();
        //Grouping data based on region and smoking
        aggregatedDataset.groupBy("region", "smoker").agg(functions.min("charges")).orderBy("region").filter(col("smoker").equalTo("yes")).show();

        //handling Categorical inputs such as Male/Female or Yes/No
        StringIndexer sexIndex = new StringIndexer()    //String Indexer transforms categorical attributes into numerical attributes
                .setInputCol("sex")
                .setOutputCol("NumSex");
        StringIndexer smokerIndex = new StringIndexer()    //String Indexer transforms categorical attributes into numerical attributes
                .setInputCol("smoker")
                .setOutputCol("NumSmoker");
        Dataset<Row> sexIndexedDataset = sexIndex.fit(aggregatedDataset).transform(aggregatedDataset);
        Dataset<Row> sexSmokerIndexedDataset = smokerIndex.fit(aggregatedDataset).transform(sexIndexedDataset);
        sexSmokerIndexedDataset.show();

        StructType Schema = sexSmokerIndexedDataset.schema(); //Inferring Schema
        for (StructField field : Schema.fields()) {    //Running through each column and performing Correlation Analysis
            if (!field.dataType().equals(DataTypes.StringType)) {
                System.out.println("Correlation between Charges and " + field.name()
                        + " = " + sexSmokerIndexedDataset.stat().corr("charges", field.name()));
            }
        }
        sexSmokerIndexedDataset = sexSmokerIndexedDataset.na().drop();

        //Selecting the attributes only with positive correlation
        Dataset<Row> correlatedDataset = sexSmokerIndexedDataset.select(col("charges").as("label"), col("age"), col("bmi"), col("children"));
        //Assembling the vector
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"age", "bmi", "children"})
                .setOutputCol("features");

        Dataset<Row> assembledDataset = assembler.transform(correlatedDataset).select("label", "features");
        assembledDataset.show();
    }
}
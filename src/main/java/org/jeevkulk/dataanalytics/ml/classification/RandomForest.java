package org.jeevkulk.dataanalytics.ml.classification;

import static org.apache.spark.sql.functions.col;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.jeevkulk.dataanalytics.ml.datapreparation.TfIdfCalculator;

public class RandomForest {

	public static void main(String[] args) {

		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession sparkSession = SparkSession.builder()
                .appName("SparkML") 
                .master("local[*]") 
                .getOrCreate();

        Dataset<Row> userKnowDf = sparkSession.read()
		        .option("header", true)
		        .option("inferSchema",true)
		        .csv(TfIdfCalculator.class.getClassLoader().getResource("data/classification/user_know_modeling_dataset_train.csv").toString());
        userKnowDf.show(); //Displaying Samples
        userKnowDf.printSchema(); //Printing Schema
        userKnowDf.describe().show(); // Statistically Summarizing about the data

        //**************************************String Indexer***************************************************//
		StringIndexer indexer = new StringIndexer().setInputCol("SKL").setOutputCol("IND_SKL");
		StringIndexerModel indModel = indexer.fit(userKnowDf);
		Dataset<Row> indexedUserKnow = indModel.transform(userKnowDf);
		indexedUserKnow.groupBy(col("SKL"), col("IND_SKL")).count().show();
		indexedUserKnow.show();
		
		//**********************************Assembling the vector and label************************//
		Dataset<Row> df = indexedUserKnow.select(
				col("IND_SKL").as("label"),
				col("SST"),
				col("SRT"),
				col("SAT"),
				col("SAP"),
				col("SEP")
		);

        //Assembling the features in the dataFrame as Dense Vector
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"SST","SRT","SAT","SAP","SEP"})
                .setOutputCol("features");

        Dataset<Row> LRdf = assembler.transform(df).select("label","features");    
        LRdf.show();

        //*****************************Model Building *****************************************//
		Dataset<Row>[] splits = LRdf.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];		//Training Data
        Dataset<Row> testData = splits[1];			//Testing Data
        
        RandomForestClassifier rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features");

		RandomForestClassificationModel Model = rf.fit(trainingData);
		System.out.println("Learned Random Forest" + Model.toDebugString());

		// Convert indexed labels back to original labels.
		IndexToString labelConverter = new IndexToString().setInputCol("label").setOutputCol("labelStr")
				.setLabels(indModel.labels());
		IndexToString predConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictionStr")
				.setLabels(indModel.labels());

		Dataset<Row> rawPredictions = Model.transform(testData);
		Dataset<Row> predictions = predConverter.transform(labelConverter.transform(rawPredictions));
		predictions.select("predictionStr", "labelStr", "features").show();

		/*************************Model Evaluation*********************/
		// View confusion matrix
		System.out.println("Confusion Matrix :");
		predictions.groupBy(col("labelStr"), col("predictionStr")).count().show();

		// Accuracy computation
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
				.setPredictionCol("prediction");
		double fscore = evaluator.evaluate(predictions);
		System.out.println("fscore = " + fscore );
	}
}

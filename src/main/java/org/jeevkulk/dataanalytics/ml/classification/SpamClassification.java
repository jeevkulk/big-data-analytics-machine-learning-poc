package org.jeevkulk.dataanalytics.ml.classification;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;


public class SpamClassification {
public static void main(String args[]) {
		
		
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);

        SparkSession sparkSession = SparkSession.builder()  //SparkSession  
                .appName("SparkML") 
                .master("local[*]") 
                .getOrCreate(); //
  
	//read the file as data
        String pathTrain = "data/spam.csv";	
        Dataset<Row> data = sparkSession.read().format("csv").option("header","true").load(pathTrain);
        
        //performing splits on the 
		Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainset = splits[0];		//Training Set
        Dataset<Row> testset = splits[1];			//Testing Set
        
        //Isolate the relevant columns
	Dataset<Row> traindata = trainset.select(trainset.col("v2"), 			trainset.col("v1")); 
	traindata.show();
	Dataset<Row> traindataClean = traindata.na().drop();

		//Isolate the relevant columns
	Dataset<Row> testdata = testset.select(testset.col("v2"), 			testset.col("v1")); 
	testdata.show();
	Dataset<Row> testdataClean = testdata.na().drop();
	

		// Configure an ML pipeline, which consists of multiple stages: indexer, tokenizer, hashingTF, idf, lr/rf etc 
		// and labelindexer.		
		//Relabel the target variable
		StringIndexerModel indexer = new StringIndexer()
				.setInputCol("v1")
				.setOutputCol("label").fit(traindata);
		
		// Tokenize the input text
		Tokenizer tokenizer = new Tokenizer()
		  .setInputCol("v2")
		  .setOutputCol("words");
		
		// Remove the stop words
		StopWordsRemover remover = new StopWordsRemover()
				  .setInputCol(tokenizer.getOutputCol())
				  .setOutputCol("filtered");		

		// Create the Term Frequency Matrix
		HashingTF hashingTF = new HashingTF()
		  .setNumFeatures(1000)
		  .setInputCol(remover.getOutputCol())
		  .setOutputCol("numFeatures");
	
		// Calculate the Inverse Document Frequency 
		IDF idf = new IDF()
				.setInputCol(hashingTF.getOutputCol())
				.setOutputCol("features");
		
		// Set up the Random Forest Model
		RandomForestClassifier rf = new RandomForestClassifier();
		
		//Set up Decision Tree
		DecisionTreeClassifier dt = new DecisionTreeClassifier();
		
		IndexToString labelConverter = new IndexToString()
		  .setInputCol("prediction")
		  .setOutputCol("predictedLabel").setLabels(indexer.labels());

		// Create and Run Random Forest Pipeline
		Pipeline pipelineRF = new Pipeline()
		  .setStages(new PipelineStage[] {indexer, tokenizer, remover, hashingTF, idf, rf, labelConverter});	
		// Fit the pipeline to training documents.
		PipelineModel modelRF = pipelineRF.fit(traindataClean);	
		System.out.println("c6");
		// Make predictions on test documents.
		Dataset<Row> predictionsRF = modelRF.transform(testdataClean);
		System.out.println("Predictions from Random Forest Model are:");
		predictionsRF.show(10);

		// Create and Run Decision Tree Pipeline
		Pipeline pipelineDT = new Pipeline()
		.setStages(new PipelineStage[] {indexer, tokenizer, remover, hashingTF, idf, dt, labelConverter});	
		// Fit the pipeline to training documents.
		PipelineModel modelDT = pipelineDT.fit(traindataClean);		
		// Make predictions on test documents.
		Dataset<Row> predictionsDT = modelDT.transform(testdataClean);
		System.out.println("Predictions from Random Forest Model are:");
		predictionsDT.show(10);		

		// Select (prediction, true label) and compute test error.
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
		  .setLabelCol("label")
		  .setPredictionCol("prediction")
		  .setMetricName("accuracy");		


		
		//Evaluate Random Forest
		double accuracyRF = evaluator.evaluate(predictionsRF);
		System.out.println("Test Error = " + (1.0 - accuracyRF));
		
		//Evaluate Random Forest
		double accuracyDT = evaluator.evaluate(predictionsDT);
		System.out.println("Test Error = " + (1.0 - accuracyDT));
	
		
	}

}

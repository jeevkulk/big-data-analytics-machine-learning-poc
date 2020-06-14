package org.jeevkulk.dataanalytics.ml.datapreparation;

import static org.apache.spark.sql.functions.col;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class TfIdfCalculator {

	public static void main(String[] args) {
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		
		// create a spark session
		SparkSession sparkSession = SparkSession.builder().appName("TFIDF Example").master("local")
				.getOrCreate();

		// import data with the schema
		Dataset<Row> inputData = sparkSession.read()
				.option("header", "true")
				.option("inferSchema", true)
				.csv(TfIdfCalculator.class.getClassLoader().getResource("data/datapreparation/hotel_review.csv").toString());

		Dataset<Row> sentenceData = inputData.select(
				"Positive Review",
				"Hotel Name",
				"Negative Review",
				"Nationality",
				"Reviewer Score",
				"Average Score"
		);
		sentenceData.printSchema();

		// split the sentence into words
		Tokenizer tokenizerPositive = new Tokenizer().setInputCol("Positive Review").setOutputCol("PositiveWords");
		Tokenizer tokenizerNegative = new Tokenizer().setInputCol("Negative Review").setOutputCol("NegativeWords");
		Dataset<Row> tokenData = tokenizerPositive.transform(sentenceData);
		tokenData = tokenizerNegative.transform(tokenData);
		tokenData.show();

		//Removing unnecessary words and Stop Words
		StopWordsRemover removerPositive = new StopWordsRemover().setInputCol("PositiveWords").setOutputCol("NewPositiveWords");
		StopWordsRemover removerNegative = new StopWordsRemover().setInputCol("NegativeWords").setOutputCol("NewNegativeWords");
		Dataset<Row> stopData = removerPositive.transform(tokenData);
		stopData = removerNegative.transform(stopData)
					.select(
							"Hotel Name",
							"Reviewer Score",
							"Nationality",
							"NewPositiveWords",
							"NewNegativeWords",
							"Average Score"
					);
		stopData.show();

		//Define Transformer, HashingTF
		//HashingTF will map each word to its frequency in the text.
		//This mapping is done by the hash function of the HashingTF.
		//You are required to give a SetNumFeatures. It defines the size of the hash function.
		//A greater size gives a better unique word mapping and no collision or duplication.
		HashingTF hashingTF1 = new HashingTF()
				.setInputCol("NewPositiveWords")
				.setOutputCol("PW").setNumFeatures(10000);
		HashingTF hashingTF2 = new HashingTF()
				.setInputCol("NewNegativeWords")
				.setOutputCol("NW").setNumFeatures(10000);
		Dataset<Row> featurizedData = hashingTF1.transform(stopData);
		featurizedData = hashingTF2.transform(featurizedData);
		featurizedData.show();

		// IDF is an Estimator which is fit on a dataset and produces an IDFModel
		// TF-IDF is a numerical statistic that is intended to reflect how important a word is to a document in a collection
		IDF idf1 = new IDF().setInputCol("PW").setOutputCol("Positive");
		IDF idf2 = new IDF().setInputCol("NW").setOutputCol("Negative");

		// The IDFModel takes feature vectors (generally created from HashingTF or CountVectorizer) and scales each column
		IDFModel idfModel1 = idf1.fit(featurizedData);
		IDFModel idfModel2 = idf2.fit(featurizedData);
		Dataset<Row> idfData1 = idfModel1.transform(featurizedData);
		Dataset<Row> idfData2 = idfModel2.transform(idfData1);
		//Selecting the required columns and attributes
		Dataset<Row> dataset = idfData2.select(
				col("Hotel Name"),
				col("Reviewer Score").as("label"),
				col("Nationality"),
				col("Positive"),
				col("Negative"),
				col("Average Score")
		);

		//Converts categorical attribute into a number.
		StringIndexer indexer = new StringIndexer().setInputCol("Nationality").setOutputCol("IndexNationality");
		Dataset<Row> indexData = indexer.fit(dataset).transform(dataset);
		indexData.show();

		System.out.println( "Correlation between Reviewer Score and " + "IndexNationality"
		+ " = " + indexData.stat().corr("label", "IndexNationality"));
		
		//Assembles the attributes for the model
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[]{"Positive", "Negative","Average Score"})
				.setOutputCol("features");
		Dataset<Row> output = assembler.transform(indexData).select("label","features");
		output.show();
		
		Normalizer normalizer = new Normalizer()
				.setInputCol("features")
				.setOutputCol("normalizedFeatures")
				.setP(1.0);
		Dataset<Row> normOutput = normalizer.transform(output);

		Dataset<Row>[] dataSplit = normOutput.randomSplit(new double[]{0.7, 0.3});
		Dataset<Row> trainingData = dataSplit[0];
		Dataset<Row> testingData = dataSplit[1];

//		Performing Linear regression
		LinearRegression lr = new LinearRegression()
				.setLabelCol("label")
				.setFeaturesCol("normalizedFeatures");
//				.setRegParam(0.3)
//				.setElasticNetParam(0.8);
		LinearRegressionModel lrm = lr.fit(trainingData);

//		Testing data by using linearRegressionModel
		Dataset<Row> predictionValues = lrm.transform(testingData);
		predictionValues.printSchema();
		predictionValues.show();

		RegressionEvaluator evaluator2 = new RegressionEvaluator()
				.setMetricName("rmse")
				.setLabelCol("label")
				.setPredictionCol("prediction");

		Double rmse2 = evaluator2.evaluate(predictionValues);
		System.out.println("Root-mean-square error for the Model with default values " + rmse2);
	}
}
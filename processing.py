from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import StandardScaler, MinMaxScaler, VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.impute import KNNImputer
import pandas as pd
import logging
import sys

# Setting up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

# Spark session + loading data
spark = SparkSession.builder.appName('PreProcessing').getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
df = spark.read.csv('hdfs://hadoop1:9000/user/sp3/data/input/cars_cleaned.csv', header=True, inferSchema=True)
logger.info("Data loaded successfully.")

# KNN Imputation for EngineSize(cc) and EnginePower(HP)
impute_cols = ['EngineSize(cc)', 'EnginePower(HP)']
imputed_cols = pd.DataFrame(
    KNNImputer(n_neighbors=5).fit_transform(df.select(*impute_cols).toPandas()), 
    columns=impute_cols
)
imputed_cols['id'] = range(len(imputed_cols))
df_imputed = spark.createDataFrame(imputed_cols)
df = df.withColumn('id', monotonically_increasing_id())
df = df.drop(*impute_cols)
df = df.join(df_imputed, 'id').drop('id')
logger.info("KNN Imputation completed.")

# Feature scaling
scale_cols = ['Price(TRY)', 'Year', 'Mileage(km)', 'EngineSize(cc)', 'EnginePower(HP)', 'ListingYear', 'ListingMonth', 'ListingDay']
pipeline = Pipeline(stages=[
    VectorAssembler(inputCols=scale_cols, outputCol='features'),
    StandardScaler(inputCol='features', outputCol='scaled_features', withMean=True, withStd=True),
    MinMaxScaler(inputCol='scaled_features', outputCol='normalized_features')
])
df_normalized = pipeline.fit(df).transform(df)
final_df = df_normalized.drop('features', 'scaled_features')
final_df = final_df.withColumn("normalized_features_array", vector_to_array("normalized_features"))
for i, col_name in enumerate(scale_cols):
    final_df = final_df.withColumn(col_name, final_df["normalized_features_array"][i])
final_df = final_df.drop("normalized_features", "normalized_features_array")
logger.info("Feature scaling completed.")

# Stratifed train-test split
fractions = final_df.select('Brand').distinct().withColumn('fraction', F.lit(0.8)).rdd.collectAsMap()
train_df = final_df.stat.sampleBy('Brand', fractions, seed=42)
test_df = final_df.subtract(train_df)
logger.info("Data splitting done")

# Getting labelled class for Brand column
indexer_model = StringIndexer(inputCol='Brand', outputCol='label').fit(train_df)
train_df = indexer_model.transform(train_df)
test_df = indexer_model.transform(test_df)

# Assemble predictors 
assembler = VectorAssembler(inputCols=[c for c in train_df.columns if c not in ('Brand', 'label')], outputCol='features')
models = {
    'Logistic Regression': Pipeline(stages=[
        assembler,
        LogisticRegression(featuresCol='features', labelCol='label')
    ]),
    'Random Forest': Pipeline(stages=[
        assembler,
        RandomForestClassifier(featuresCol='features', labelCol='label')
    ]),
    'Decision Tree': Pipeline(stages=[
        assembler,
        DecisionTreeClassifier(featuresCol='features', labelCol='label')
    ]),
}
results = []
for name, classy_model in models.items():
    # Set up a param grid for cross-validation based on model type
    # if name == 'Logistic Regression':
    #     paramGrid = ParamGridBuilder() \
    #         .addGrid(classy_model.getStages()[1].regParam, [0.01, 0.1, 1.0, 2.5, 5.0])\
            # .addGrid(classy_model.getStages()[1].maxIter, [10, 20,])
            # .addGrid(classy_model.getStages()[1].elasticNetParam, [0.0, 0.5, 1.0]) \
            # .build()
    # elif name == 'Random Forest':
    #     paramGrid = ParamGridBuilder() \
    #         .addGrid(classy_model.getStages()[1].numTrees, [10, 30, 50, 100]) \
    #         .addGrid(classy_model.getStages()[1].maxDepth, [5, 10, 15, 20]) \
    #         .addGrid(classy_model.getStages()[1].minInstancesPerNode, [1, 2, 4]) \
    #         .addGrid(classy_model.getStages()[1].featureSubsetStrategy, ['auto', 'sqrt', 'log2']) \
    #         .build()
    # elif name == 'Decision Tree':
    #     paramGrid = ParamGridBuilder() \
    #         .addGrid(classy_model.getStages()[1].maxDepth, [3, 5, 7]) \
    #         .addGrid(classy_model.getStages()[1].minInstancesPerNode, [1, 2, 4]) \
    #         .addGrid(classy_model.getStages()[1].impurity, ['gini', 'entropy']) \
    #         .build()
    # else:
    #     paramGrid = ParamGridBuilder().build()

    paramGrid = ParamGridBuilder().build()
    
    cv = CrossValidator(
        estimator=classy_model,
        evaluator=MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy'),
        estimatorParamMaps=paramGrid,
        numFolds=2)
    cvModel = cv.fit(train_df)

    best_model = cvModel.bestModel
    predictions = best_model.transform(test_df)

    best_stage = best_model.stages[1] if hasattr(best_model, 'stages') else best_model
    best_params = {param.name: best_stage.getOrDefault(param) for param in best_stage.extractParamMap()}

    metric = {}
    metric['accuracy'] = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy').evaluate(predictions)
    metric['wPrecision'] = MulticlassClassificationEvaluator(labelCol='label', metricName='weightedPrecision').evaluate(predictions)
    metric['wRecall'] = MulticlassClassificationEvaluator(labelCol='label', metricName='weightedRecall').evaluate(predictions)
    metric['F1'] = MulticlassClassificationEvaluator(labelCol='label', metricName='f1').evaluate(predictions)
    results.append((name, metric['accuracy'], metric['wPrecision'], metric['wRecall'], metric['F1']))

results_df = pd.DataFrame(results, columns=["Model", "Accuracy","Precision", "Recall", "F1"])
print(results_df)
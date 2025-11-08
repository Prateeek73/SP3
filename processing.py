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
LOG = logging.getLOG()
LOG.setLevel(logging.INFO)

# Spark session + loading data
spark_sess = SparkSession.builder.appName('PreProcessing').getOrCreate()
df = spark_sess.read.csv('hdfs://hadoop1:9000/user/sp3/data/input/cars_cleaned.csv', header=True, inferSchema=True)
LOG.info("Data loaded successfully.")

# KNN Imputation for EngineSize(cc) and EnginePower(HP)
miss_cols = ['EngineSize(cc)', 'EnginePower(HP)']
imput_cols = pd.DataFrame(
    KNNImputer(n_neighbors=5).fit_transform(df.select(*miss_cols).toPandas()), 
    columns=miss_cols
)
imput_cols['id'] = range(len(imput_cols))
df_impu = spark_sess.createDataFrame(imput_cols)
df = df.withColumn('id', monotonically_increasing_id())
df = df.drop(*miss_cols)
df = df.join(df_impu, 'id').drop('id')
LOG.info("KNN Imputation completed.")

# Feature scaling
scale_cols = ['Price(TRY)', 'Year', 'Mileage(km)', 'EngineSize(cc)', 'EnginePower(HP)', 'ListingYear', 'ListingMonth', 'ListingDay']
pipeline = Pipeline(stages=[
    VectorAssembler(inputCols=scale_cols, outputCol='feat'),
    StandardScaler(inputCol='feat', outputCol='scal', withMean=True, withStd=True),
    MinMaxScaler(inputCol='scal', outputCol='norm')
])
df_norm = pipeline.fit(df).transform(df)
f_df = df_norm.drop('feat', 'scal')
f_df = f_df.withColumn("norm_arry", vector_to_array("norm"))
for i, col_name in enumerate(scale_cols):
    f_df = f_df.withColumn(col_name, f_df["norm_arry"][i])
f_df = f_df.drop("norm", "norm_arry")
LOG.info("Feature scaling completed.")

# Stratifed train-test split
fractions = f_df.select('Brand').distinct().withColumn('fraction', F.lit(0.8)).rdd.collectAsMap()
train = f_df.stat.sampleBy('Brand', fractions, seed=42)
test = f_df.subtract(train)
LOG.info("Data splitting done")

# Getting labelled class for Brand column
indexer_model = StringIndexer(inputCol='Brand', outputCol='label').fit(train)
train = indexer_model.transform(train)
test = indexer_model.transform(test)

# Assemble predictors 
assembler = VectorAssembler(inputCols=[c for c in train.columns if c not in ('Brand', 'label')], outputCol='feat')
mandals = {
    'Logistic Regression': Pipeline(stages=[
        assembler,
        LogisticRegression(featCol='feat', labelCol='label')
    ]),
    'Random Forest': Pipeline(stages=[
        assembler,
        RandomForestClassifier(featCol='feat', labelCol='label')
    ]),
    'Decision Tree': Pipeline(stages=[
        assembler,
        DecisionTreeClassifier(featCol='feat', labelCol='label')
    ]),
}
results = []
for name, classy_model in mandals.items():

    paramGrid = ParamGridBuilder().build()
    
    cv = CrossValidator(
        estimator=classy_model,
        evaluator=MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy'),
        estimatorParamMaps=paramGrid,
        numFolds=3)
    cvModel = cv.fit(train)

    best_model = cvModel.bestModel
    predictions = best_model.transform(test)

    best_stage = best_model.stages[1] if hasattr(best_model, 'stages') else best_model
    best_params = {param.name: best_stage.getOrDefault(param) for param in best_stage.extractParamMap()}

    kps = {}
    kps['accuracy'] = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy').evaluate(predictions)
    kps['wPrecision'] = MulticlassClassificationEvaluator(labelCol='label', metricName='weightedPrecision').evaluate(predictions)
    kps['wRecall'] = MulticlassClassificationEvaluator(labelCol='label', metricName='weightedRecall').evaluate(predictions)
    kps['F1'] = MulticlassClassificationEvaluator(labelCol='label', metricName='f1').evaluate(predictions)
    results.append((name, kps['accuracy'], kps['wPrecision'], kps['wRecall'], kps['F1']))

results_df = pd.DataFrame(results, columns=["Model", "Accuracy","Precision", "Recall", "F1"])
print(results_df)
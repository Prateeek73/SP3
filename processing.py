from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import StandardScaler, MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
import pandas as pd
from sklearn.impute import KNNImputer

# Spark session + loading data
spark = SparkSession.builder.appName('PreProcessing')
df = spark.read.csv('hdfs://hadoop1:9000/user/sp3/data/input/cars_cleaned.csv', header=True, inferSchema=True)

# KNN Imputation for EngineSize(cc) and EnginePower(HP)
impute_cols = ['EngineSize(cc)', 'EnginePower(HP)']
imputed_cols = pd.DataFrame(
    KNNImputer(n_neighbors=5).fit_transform(df.select(*impute_cols).toPandas()), 
    columns=impute_cols
)
imputed_cols['id'] = range(len(imputed_cols))
df_imputed = spark.createDataFrame(imputed_cols)        
df = df.withColumn('id', monotonically_increasing_id()).join(df_imputed, 'id').drop('id')

# Feature scaling
scale_cols = {'Price(TRY)', 'Year', 'Mileage(km)', 'EngineSize(cc)', 'EnginePower(HP)', 'ListingYear', 'ListingMonth', 'ListingDay'}
pipeline = Pipeline(stages=[
    VectorAssembler(inputCols=scale_cols, outputCol='features'),
    StandardScaler(inputCol='features', outputCol='scaled_features', withMean=True, withStd=True),
    MinMaxScaler(inputCol='scaled_features', outputCol='normalized_features')
])
df_normalized = pipeline.fit(df).transform(df)

# Preprocessed dataset
final_df = df_normalized.select("*")
final_df = df_normalized.drop("features", "scaled_features")

# Stratifed train-test split
fractions = final_df.select('Brand').distinct().withColumn('fraction', F.lit(0.8)).rdd.collectAsMap()
train_df = final_df.stat.sampleBy('Brand', fractions, seed=42)
test_df = final_df.subtract(train_df)

# Save processed train and test data
train_df.write.csv('hdfs://hadoop1:9000/user/sp3/data/output/data_preprocessed_train.csv', header=True)
test_df.write.csv('hdfs://hadoop1:9000/user/sp3/data/output/data_preprocessed_test.csv', header=True)
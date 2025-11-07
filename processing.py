from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, monotonically_increasing_id
from pyspark.ml.feature import StandardScaler, MinMaxScaler, VectorAssembler
from pyspark.sql.functions import rand
from pyspark.ml.functions import vector_to_array

import logging
import time
import pandas as pd
from sklearn.impute import KNNImputer

# Spark session
spark = (
    SparkSession.builder
    .appName('DataPreprocessing')
    .config('spark.sql.adaptive.enabled', 'true')
    .config('spark.executor.memory', '4g')
    .config('spark.driver.memory', '2g')
    .config('spark.eventLog.enabled', 'false')
    .config("spark.executor.cores", "2")
    .getOrCreate()
)

logging.info("Loading cleaned data")
# Loading clearned data
input_path = 'hdfs://hadoop1:9000/user/sp3/data/input/cars_clean.csv'
df = spark.read.csv(input_path, header=True, inferSchema=True)

logging.info("Starting KNN imputation")
# KNN Imputation for EngineSize(cc) and EnginePower(HP)
impute_cols = ['EngineSize(cc)', 'EnginePower(HP)']
df_pd = df.select(*impute_cols).toPandas()
knn_imputer = KNNImputer(n_neighbors=5)
df_pd_imputed = pd.DataFrame(knn_imputer.fit_transform(df_pd), columns=impute_cols)
for col_name in impute_cols:
    df = df.withColumn(col_name, col(col_name).cast('double'))
    df_pd_imputed[col_name] = df_pd_imputed[col_name].astype(float)
df_pd_imputed['id'] = range(len(df_pd_imputed))
df = df.withColumn('id', monotonically_increasing_id())

# Drop original impute columns before join to avoid ambiguity
for col_name in impute_cols:
    df = df.drop(col_name)
df_imputed = df.join(spark.createDataFrame(df_pd_imputed), on='id').drop('id')
logging.info('KNN imputation completed')


logging.info("Starting feature scaling and normalization")
scale_cols = {
    'Price(TRY)', 'Year', 'Mileage(km)', 'EngineSize(cc)', 'EnginePower(HP)',
    'ListingYear', 'ListingMonth', 'ListingDay'
}

assembler = VectorAssembler(inputCols=list(scale_cols), outputCol='features')
df_vec = assembler.transform(df_imputed)

scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withMean=True, withStd=True)
df_scaled = scaler.fit(df_vec).transform(df_vec)

minmax = MinMaxScaler(inputCol='scaled_features', outputCol='normalized_features')
df_normalized = minmax.fit(df_scaled).transform(df_scaled)

df_final = df_normalized.withColumn('normalized_array', vector_to_array(col('normalized_features')))
for i, col_name in enumerate(scale_cols):
    df_final = df_final.withColumn(col_name, col('normalized_array')[i])
df_final = df_final.drop('features', 'scaled_features', 'normalized_features', 'normalized_array')
logging.info('Data transformation (scaling + normalization) completed')

# Stratified train-test split 
logging.info("Starting train-test split")
target_col = 'Brand'

fractions = df_final.select(target_col).distinct().withColumn('fraction', F.lit(0.8)).rdd.collectAsMap()
train_df = df_final.stat.sampleBy(target_col, fractions, seed=42)
test_df = df_final.subtract(train_df)

logging.info("Train-test split completed")

logging.info(f"Train set count: {train_df.count()}, Test set count: {test_df.count()}")
logging.info("Saving processed data to HDFS")

# Save processed train and test data
train_output_path = 'hdfs://hadoop1:9000/user/sp3/data/output/data_preprocessed_train.csv'
test_output_path = 'hdfs://hadoop1:9000/user/sp3/data/output/data_preprocessed_test.csv'
train_df.write.csv(train_output_path, header=True, mode='overwrite')
test_df.write.csv(test_output_path, header=True, mode='overwrite')
logging.info("Processed data saved successfully")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id
from pyspark.ml.feature import StandardScaler, MinMaxScaler, VectorAssembler
from pyspark.sql.functions import rand
from pyspark.ml.functions import vector_to_array

import logging
import time
import pandas as pd
from sklearn.impute import KNNImputer


# Start Spark session
spark = SparkSession.builder.appName('DataPreprocessing').getOrCreate()
logging.basicConfig(level=logging.INFO)

start_time = time.time()


# Load cleaned data
input_path = 'hdfs://hadoop1:9000/user/sp3/data/input/cars_clean.csv'

df = spark.read.csv(input_path, header=True, inferSchema=True)

# KNN Imputation for EngineSize(cc) and EnginePower(HP)
impute_cols = ['EngineSize(cc)', 'EnginePower(HP)']
knn_start_time = time.time()
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
logging.info(f'KNN imputation completed in {time.time() - knn_start_time:.2f} seconds')

# Center, scale, and normalize all relevant columns
transform_start_time = time.time()
scale_cols = [
    'Price(TRY)', 'Year', 'Mileage(km)', 'EngineSize(cc)', 'EnginePower(HP)',
    'ListingYear', 'ListingMonth', 'ListingDay'
]
assembler = VectorAssembler(inputCols=scale_cols, outputCol='features')
df_vec = assembler.transform(df_imputed)

scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withMean=True, withStd=True)
df_scaled = scaler.fit(df_vec).transform(df_vec)

minmax = MinMaxScaler(inputCol='scaled_features', outputCol='normalized_features')
df_normalized = minmax.fit(df_scaled).transform(df_scaled)

df_final = df_normalized.withColumn('normalized_array', vector_to_array(col('normalized_features')))
for i, col_name in enumerate(scale_cols):
    df_final = df_final.withColumn(col_name, col('normalized_array')[i])
df_final = df_final.drop('features', 'scaled_features', 'normalized_features', 'normalized_array')
logging.info(f'Data transformation (scaling + normalization) completed in {time.time() - transform_start_time:.2f} seconds')

end_time = time.time()
logging.info(f'Total preprocessing time: {end_time - start_time:.2f} seconds')

# Data splitting: 80% train, 20% test, stratified by Brand
target_col = 'Brand'
if target_col in df_final.columns:
    df_final = df_final.withColumn('rand', rand())
    brands = [row[target_col] for row in df_final.select(target_col).distinct().collect()]
    train_dfs = []
    test_dfs = []
    for brand in brands:
        brand_df = df_final.filter(col(target_col) == brand)
        train, test = brand_df.randomSplit([0.8, 0.2], seed=42)
        train_dfs.append(train)
        test_dfs.append(test)
    train_df = train_dfs[0]
    test_df = test_dfs[0]
    for df_ in train_dfs[1:]:
        train_df = train_df.union(df_)
    for df_ in test_dfs[1:]:
        test_df = test_df.union(df_)
    train_df = train_df.drop('rand')
    test_df = test_df.drop('rand')
    train_path = 'hdfs://hadoop1:9000/user/sp3/data/output/data_preprocessed_train.csv'
    test_path = 'hdfs://hadoop1:9000/user/sp3/data/output/data_preprocessed_test.csv'
    train_df.coalesce(1).write.csv(train_path, header=True, mode='overwrite')
    test_df.coalesce(1).write.csv(test_path, header=True, mode='overwrite')
    print(f'Train set saved to {train_path}')
    print(f'Test set saved to {test_path}')
    logging.info('Note: For classification models, Brand (target) should be encoded in training.')
else:
    print('Brand column not found for stratified split.')
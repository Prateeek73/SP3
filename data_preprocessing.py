
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, monotonically_increasing_id
from pyspark.ml.feature import Imputer, StandardScaler, MinMaxScaler, VectorAssembler
import time

# Start Spark session
spark = SparkSession.builder.appName('DataPreprocessing').getOrCreate()

start_time = time.time()

# Load cleaned data
input_path = 'data/data_clean.csv'
df = spark.read.csv(input_path, header=True, inferSchema=True)

# KNN Imputation (approximate using Spark's Imputer for mean imputation, as Spark does not have KNNImputer)
numeric_cols = [field.name for field in df.schema.fields if str(field.dataType) in ['IntegerType', 'DoubleType', 'FloatType', 'LongType']]
imputer = Imputer(inputCols=numeric_cols, outputCols=numeric_cols).setStrategy('mean')
df_imputed = imputer.fit(df).transform(df)

# Near Zero Variance (NZV) analysis
def nzv_analysis(df, cols):
    nzv_cols = []
    for col_name in cols:
        unique_count = df.select(countDistinct(col(col_name))).collect()[0][0]
        if unique_count <= 1:
            nzv_cols.append(col_name)
        else:
            freq = df.groupBy(col_name).count().orderBy('count', ascending=False).collect()
            if len(freq) > 1:
                freq_ratio = freq[0]['count'] / freq[1]['count']
            else:
                freq_ratio = float('inf')
            percent_unique = 100 * unique_count / df.count()
            if freq_ratio > 20 and percent_unique < 10:
                nzv_cols.append(col_name)
    return nzv_cols

nzv_cols = nzv_analysis(df_imputed, numeric_cols)
print(f'Near Zero Variance columns: {nzv_cols}')
df_nzv = df_imputed.drop(*nzv_cols)


# Center, scale, and normalize using StandardScaler and MinMaxScaler
assembler = VectorAssembler(inputCols=numeric_cols, outputCol='features')
df_vec = assembler.transform(df_nzv)

scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withMean=True, withStd=True)
df_scaled = scaler.fit(df_vec).transform(df_vec)

minmax = MinMaxScaler(inputCol='scaled_features', outputCol='normalized_features')
df_normalized = minmax.fit(df_scaled).transform(df_scaled)

# Convert normalized features back to columns
from pyspark.ml.functions import vector_to_array
df_final = df_normalized.withColumn('normalized_array', vector_to_array(col('normalized_features')))
for i, col_name in enumerate(numeric_cols):
    df_final = df_final.withColumn(col_name, col('normalized_array')[i])
df_final = df_final.drop('features', 'scaled_features', 'normalized_features', 'normalized_array')

end_time = time.time()
print(f'Total preprocessing time: {end_time - start_time:.2f} seconds')


# Data splitting: 80% train, 20% test, stratified by Brand
target_col = 'Brand'
if target_col in df_final.columns:
    # Add a random column for splitting
    from pyspark.sql.functions import rand
    df_final = df_final.withColumn('rand', rand())
    # Get unique Brand values
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
    # Save train and test sets
    train_path = 'data/data_preprocessed_train.csv'
    test_path = 'data/data_preprocessed_test.csv'
    train_df.coalesce(1).write.csv(train_path, header=True, mode='overwrite')
    test_df.coalesce(1).write.csv(test_path, header=True, mode='overwrite')
    print(f'Train set saved to {train_path}')
    print(f'Test set saved to {test_path}')
else:
    print('Brand column not found for stratified split.')
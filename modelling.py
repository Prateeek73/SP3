from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, DecisionTreeClassifier
from pyspark.ml.classification import OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator

spark = SparkSession.builder.appName('SnapModelling')

# Loading train nd test data
train_df = spark.read.csv('hdfs://hadoop1:9000/user/sp3/data/output/data_preprocessed_train.csv', header=True)
test_df = spark.read.csv('hdfs://hadoop1:9000/user/sp3/data/output/data_preprocessed_test.csv', header=True)

# Getting labelled class for Brand column
indexer_model = StringIndexer(inputCol='Brand', outputCol='label').fit(train_df)
train_df = indexer_model.transform(train_df)
test_df = indexer_model.transform(test_df)

# Assemble predictors (exclude target and label if present)
assembler = VectorAssembler(inputCols=[c for c in train_df.columns if c not in ('Brand', 'label')], outputCol='features')

models = []
models.append(('LogisticRegression', LogisticRegression(featuresCol='features', labelCol='label', maxIter = 20)))
models.append(('RandomForest', RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=20)))
models.append(('GBT_OneVsRest', OneVsRest(classifier=GBTClassifier(featuresCol='features', labelCol='label', maxIter=20))))
models.append(('DecisionTree', DecisionTreeClassifier(featuresCol='features', labelCol='label',max_depth=10)))

results = []
for name, classy_model in models:
    cv = CrossValidator(
        estimator=Pipeline(stages=[assembler, classy_model]),
        evaluator=MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy'),
        umFolds=10)
    
    cvModel = cv.fit(train_df)
    predictions = cvModel.transform(test_df)

    metric = {}
    metric['accuracy'] = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy').evaluate(predictions)
    metric['wPrecision'] = MulticlassClassificationEvaluator(labelCol='label', metricName='weightedPrecision').evaluate(predictions)
    metric['wRecall'] = MulticlassClassificationEvaluator(labelCol='label', metricName='weightedRecall').evaluate(predictions)
    metric['F1'] = MulticlassClassificationEvaluator(labelCol='label', metricName='f1').evaluate(predictions)
    
    results.append((name, metric['accuracy'], metric['wPrecision'], metric['wRecall'], metric['f1']))
    
results_df = pd.DataFrame(results, columns=["Model", "Accuracy","Precision", "Recall", "F1"])
print(results_df)
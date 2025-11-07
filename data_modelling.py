from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, DecisionTreeClassifier, NaiveBayes, MultilayerPerceptronClassifier
from pyspark.ml.classification import OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Start Spark session with reduced logging and console output
spark = (
    SparkSession.builder
    .appName('DataModelling')
    .config('spark.sql.adaptive.enabled', 'true')
    .config('spark.executor.memory', '4g')
    .config('spark.driver.memory', '2g')
    .config('spark.eventLog.enabled', 'false')
    .config("spark.executor.cores", "2")
    .config('spark.ui.showConsoleProgress', 'false')
    .getOrCreate()
)
spark.sparkContext.setLogLevel('WARN')

# Load processed train and test data
train_path = 'hdfs://hadoop1:9000/user/sp3/data/output/data_preprocessed_train.csv'
test_path = 'hdfs://hadoop1:9000/user/sp3/data/output/data_preprocessed_test.csv'
train_df = spark.read.csv(train_path, header=True, inferSchema=True)
test_df = spark.read.csv(test_path, header=True, inferSchema=True)

# Index target column labels and get number of classes
target_col = 'Brand'
num_classes = train_df.select(target_col).distinct().count()
indexer = StringIndexer(inputCol=target_col, outputCol='label')
indexer_model = indexer.fit(train_df)
train_df = indexer_model.transform(train_df)
test_df = indexer_model.transform(test_df)

# Assemble predictors (exclude target and label if present)
feature_cols = [c for c in train_df.columns if c not in (target_col, 'label')]
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

# Define classifiers
classifiers = []
classifiers.append(('LogisticRegression', LogisticRegression(featuresCol='features', labelCol='label', maxIter=20)))
classifiers.append(('RandomForest', RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=50)))
classifiers.append(('GBT_OneVsRest', OneVsRest(classifier=GBTClassifier(featuresCol='features', labelCol='label', maxIter=20))))
classifiers.append(('DecisionTree', DecisionTreeClassifier(featuresCol='features', labelCol='label')))

results = []
for name, clf in classifiers:
    pipeline = Pipeline(stages=[assembler, clf])
    # Reduced param grid for each classifier to avoid large task binaries
    if name == 'LogisticRegression':
        paramGrid = (ParamGridBuilder()
            .addGrid(clf.maxIter, [20])
            .addGrid(clf.regParam, [0.0])
            .addGrid(clf.elasticNetParam, [0.0])
            .build())
    elif name == 'RandomForest':
        paramGrid = (ParamGridBuilder()
            .addGrid(clf.numTrees, [50])
            .addGrid(clf.maxDepth, [5])
            .build())
    elif name == 'GBT_OneVsRest':
        # For OneVsRest wrapper, access the underlying classifier via getClassifier()
        paramGrid = (ParamGridBuilder()
            .addGrid(clf.getClassifier().maxIter, [20])
            .addGrid(clf.getClassifier().maxDepth, [5])
            .build())
    elif name == 'DecisionTree':
        paramGrid = (ParamGridBuilder()
            .addGrid(clf.maxDepth, [5])
            .addGrid(clf.maxBins, [32])
            .build())
    else:
        paramGrid = ParamGridBuilder().build()

    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=paramGrid,
                        evaluator=MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy'),
                        numFolds=3)
    cvModel = cv.fit(train_df)
    predictions = cvModel.transform(test_df)

    # Metrics (use multiclass metrics; compute AUC only for binary)
    evaluator_acc = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy')
    evaluator_precision = MulticlassClassificationEvaluator(labelCol='label', metricName='weightedPrecision')
    evaluator_recall = MulticlassClassificationEvaluator(labelCol='label', metricName='weightedRecall')
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol='label', metricName='f1')
    acc = evaluator_acc.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    auc = None
    if num_classes == 2:
        evaluator_auc = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='rawPrediction', metricName='areaUnderROC')
        try:
            auc = evaluator_auc.evaluate(predictions)
        except Exception:
            auc = None

    results.append((name, acc, auc, recall, precision, f1))
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"AUC: {auc:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"F1 (weighted): {f1:.4f}")

# Save results to CSV
import pandas as pd
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "AUC", "Recall", "Precision", "F1"])
results_df.to_csv("model_results.csv", index=False)
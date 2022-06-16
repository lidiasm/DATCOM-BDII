import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.ml.feature.{IndexToString, MinMaxScaler, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.instance.ASMOTE
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.FMClassifier
import org.apache.spark.ml.classification.kNN_IS.kNN_ISClassifier

object Practica {

  // Rutas hacia los ficheros en el cluster
  val trainClusterPath = "hdfs://192.168.10.1/user/datasets/master/susy/susyMaster-Train.data"
  val testClusterPath = "hdfs://192.168.10.1/user/datasets/master/susy/susyMaster-Test.data"

  // Rutas hacia los ficheros locales
  val trainLocalPath = "/home/usuario/datasets/susy-10k-tra.data"
  val testLocalPath = "/home/usuario/datasets/susy-10k-tst.data"

  /**
   * Algoritmo de balanceo de clases ROS (random oversampling)
   * @param train conjunto de entrenamiento a balancear
   * @param overRate porcentaje de equilibrio entre clases
   * @return conjunto de entrenamiento balanceado
   */
  def ROS(train: DataFrame, overRate: Double): DataFrame = {
    var oversample: DataFrame = train.limit(0) //empty DF

    val train_positive = train.where("label == 1")
    val train_negative = train.where("label == 0")
    val num_neg = train_negative.count().toDouble
    val num_pos = train_positive.count().toDouble

    if (num_pos > num_neg) {
      val fraction = (num_pos * overRate) / num_neg
      oversample = train_positive.union(train_negative.sample(withReplacement = true, fraction, seed = 1000))
    } else {
      val fraction = (num_neg * overRate) / num_pos
      oversample = train_negative.union(train_positive.sample(withReplacement = true, fraction, seed = 1000))
    }
    oversample.repartition(train.rdd.getNumPartitions)
  }

  /**
   * Algoritmo de balanceo de clases RUS (random undersampling)
   * @param train conjunto de entrenamiento a balancear
   * @return conjunto de entrenamiento balanceado
   */
  def RUS(train: DataFrame): DataFrame = {
    var undersample: DataFrame = train.limit(0) //empty DF

    val train_positive = train.where("label == 1")
    val train_negative = train.where("label == 0")
    val num_neg = train_negative.count().toDouble
    val num_pos = train_positive.count().toDouble

    if (num_pos > num_neg) {
      val fraction = num_neg / num_pos
      undersample = train_negative.union(train_positive.sample(withReplacement = false, fraction, seed = 1000))
    } else {
      val fraction = num_pos / num_neg
      undersample = train_positive.union(train_negative.sample(withReplacement = false, fraction, seed = 1000))
    }
    undersample.repartition(train.rdd.getNumPartitions)
  }

  /**
   * Calcula la tasa de muestras positivas y negativas bien clasificadas
   * a partir de un conjunto de predicciones realizadas por cualquier
   * modelo sobre el conjunto de entrenamiento o de test
   * @param predictions conjunto de predicciones
   * @param predColumn nombre de la columna que contiene las predicciones
   */
  def calculateQualityMetrics(predictions: DataFrame, predColumn: String): Unit = {
    // Muestras positivas bien clasificadas
    println("TP = " + predictions.filter(predictions(predColumn) === 1
      && predictions("label") === predictions(predColumn)).count())

    // Muestras negativas bien clasificadas
    println("TN = " + predictions.filter(predictions(predColumn) === 0 &&
      predictions("label") === predictions(predColumn)).count())
  }

  def main(arg: Array[String]): Unit = {
    // Iniciamos una nueva sesión de Spark
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("SparkApp")
      .getOrCreate()

    // Lectura del conjunto de entrenamiento
    val dfTrain = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(trainLocalPath) //trainClusterPath
    // Modificamos el nombre de la variable clase
    var train = dfTrain.withColumnRenamed("_c18", "label")

    // Lectura del conjunto de test
    val dfTest = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(testLocalPath) //testClusterPath
    // Modificamos el nombre de la variable clase
    var test = dfTest.withColumnRenamed("_c18", "label")

    // Une las variables de entrada en una columna llamada 'features'
    val assembler = new VectorAssembler()
      .setInputCols(dfTrain.columns.init)
      .setOutputCol("features")
    // Conjunto de entrenamiento y test en caché para mayor velocidad
    train = assembler.transform(train).repartition(100).cache()
    test = assembler.transform(test).repartition(100).cache()

    /*---------------------------------------------------------------*/
    // ALGORITMO DE BALANCEO DE CLASES: ROS
    val balancedTrain = ROS(train.select("label", "features"),1.0)
    println("DISTRIBUCIÓN DE CLASES TRAS ROS")
    balancedTrain.groupBy("label").count().show
    /*---------------------------------------------------------------*/
    // ALGORITMO DE BALANCEO DE CLASES: RUS
    val balancedTrain = RUS(train.select("label", "features"))
    println("DISTRIBUCIÓN DE CLASES TRAS RUS")
    balancedTrain.groupBy("label").count().show
    /*---------------------------------------------------------------*/
    // ALGORITMO DE BALANCEO DE CLASES: ASMOTE
    // Convertimos la variable clase en entera 
    train = train.withColumn("label", col("label").cast(IntegerType))
    test = test.withColumn("label", col("label").cast(IntegerType))
    // Balanceo de clases con semilla para resultados reproducibles
    val asmote = new ASMOTE().setK(5).setPercOver(5).setSeed(1000)
    val balancedTrain = asmote.transform(train.select("label", "features"))
    println("DISTRIBUCIÓN DE CLASES TRAS ASMOTE")
    balancedTrain.groupBy("label").count().show
    /*---------------------------------------------------------------*/

    /*---------------------------------------------------------------*/
    // ALGORITMO DE CLASIFICACIÓN: ÁRBOLES DE DECISIÓN
    // Indexa las variables del dataset
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(balancedTrain)
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(balancedTrain)

    // Construye el clasificador
    val classifierSettings = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
    // Traduce las etiquetas predichas al rango de valores del dataset
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Pipeline con el procedimiento
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, classifierSettings, labelConverter))

    // Entrenamiento del modelo
    val model = pipeline.fit(balancedTrain)
    // Predicciones y evaluación sobre train
    val trainPredictions = model.transform(balancedTrain).select("label", "predictedLabel")
    println("EVALUACIÓN SOBRE TRAIN:")
    calculateQualityMetrics(trainPredictions, "predictedLabel")
    // Predicciones y evaluación sobre test
    val testPredictions = model.transform(test).select("label", "predictedLabel")
    println("EVALUACIÓN SOBRE TEST:")
    calculateQualityMetrics(testPredictions, "predictedLabel")
    /*---------------------------------------------------------------*/
  }
}
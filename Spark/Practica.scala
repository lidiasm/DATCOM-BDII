import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.instance.ASMOTE
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, LogisticRegression}
import org.apache.spark.ml.classification.kNN_IS.kNN_ISClassifier
import com.microsoft.azure.synapse.ml.lightgbm.LightGBMClassifier

import org.apache.spark.mllib.feature._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.{DecisionTree, GradientBoostedTrees, RandomForest}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.classification.kNN_IS.kNN_IS


object Practica {

  // Rutas hacia los ficheros en el cluster
  val trainClusterPath = "hdfs://192.168.10.1/user/datasets/master/susy/susyMaster-Train.data"
  val testClusterPath = "hdfs://192.168.10.1/user/datasets/master/susy/susyMaster-Test.data"

  // Rutas hacia los ficheros ampliados en local
  val trainLocalPath = "/home/usuario/datasets/susyMaster-Train.data"
  val testLocalPath = "/home/usuario/datasets/susyMaster-Test.data"

  // Rutas hacia los ficheros reducidos en local
  val train10kLocalPath = "/home/usuario/datasets/susy-10k-tra.data"
  val test10kLocalPath = "/home/usuario/datasets/susy-10k-tst.data"

  /**
   * Algoritmo de balanceado de clases ROS (random oversampling)
   * utilizando la librería ML (DataFrames).
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
   * utilizando la librería ML (DataFrames).
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
   * a partir de un conjunto de predicciones y etiquetas reales generadas
   * por un algoritmo utilizando la librería ML (DataFrames).
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

  /**
   * Función que aplica un algoritmo de preprocesamiento (ROS, RUS o ASMOTE)
   * para balancear el conjunto de entrenamiento y construir un clasificador
   * utilizando el algoritmo Árboles de Decisión utilizando la librería ML (DataFrames).
   * Finalmente se evalúa su calidad sobre los conjuntos de entrenamiento y test
   * calculando el número de muestras positivas y negativas bien clasificadas.
   * @param trainPath ruta hacia el fichero de entrenamiento
   * @param testPath ruta hacia el fichero de test
   * @param balAlg algoritmo de balanceo de clases a aplicar sobre el conjunto
   *               de entrenamiento desbalanceado. Opciones: ROS, RUS o ASMOTE.
   */
  def applyDecisionTreesDF(trainPath: String, testPath: String, balAlg: String): Unit = {
    // Inicia una nueva sesión de Spark
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("SparkApp")
      .getOrCreate()

    // Lectura del conjunto de entrenamiento
    val dfTrain = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(trainPath)
    // Modificamos el nombre de la variable clase
    var train = dfTrain.withColumnRenamed("_c18", "label")

    // Lectura del conjunto de test
    val dfTest = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(testPath)
    // Modifica el nombre de la variable de clase
    var test = dfTest.withColumnRenamed("_c18", "label")

    // Une las variables de entrada en una columna llamada 'features'
    val assembler = new VectorAssembler()
      .setInputCols(dfTrain.columns.init)
      .setOutputCol("features")

    // Conjunto de entrenamiento y test en caché para mayor velocidad
    train = assembler.transform(train).repartition(100).cache()
    test = assembler.transform(test).repartition(100).cache()

    /*---------------------------------------------------------------*/
    // ALGORITMO DE BALANCEADO DE CLASES
    // Almacena el dataframe balanceado resultante
    var balancedTrain: DataFrame = null
    // Aplica el algoritmo de balanceo de clases seleccionado
    balAlg match {
      case "ROS" =>
        balancedTrain = ROS(train.select("label", "features"),1.0)
        println("\nDISTRIBUCIÓN DE CLASES TRAS ROS")
        balancedTrain.groupBy("label").count().show

      case "RUS" =>
        balancedTrain = RUS(train.select("label", "features"))
        println("\nDISTRIBUCIÓN DE CLASES TRAS RUS")
        balancedTrain.groupBy("label").count().show

      case "ASMOTE" =>
        // Balanceo de clases con semilla para resultados reproducibles
        val asmote = new ASMOTE().setK(5).setPercOver(800).setSeed(1000)
        balancedTrain = asmote.transform(train.select("label", "features"))
        println("\nDISTRIBUCIÓN DE CLASES TRAS ASMOTE")
        balancedTrain.groupBy("label").count().show
    }
    /*---------------------------------------------------------------*/

    /*---------------------------------------------------------------*/
    // ALGORITMO DE CLASIFICACIÓN
    // Indexa las variables del dataset
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(balancedTrain)
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(balancedTrain)

    // Construye el clasificador con Árboles de Decisión
    val classifierSettings = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    // Traduce las etiquetas predichas al rango de valores del dataset
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Pipeline con todas las actividades
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

  /**
   * Función que aplica un algoritmo de preprocesamiento (ROS, RUS o ASMOTE)
   * para balancear el conjunto de entrenamiento y construir un clasificador
   * utilizando el algoritmo Gradient-Boosted Trees utilizando la librería ML (DataFrames).
   * Finalmente se evalúa su calidad sobre los conjuntos de entrenamiento y test
   * calculando el número de muestras positivas y negativas bien clasificadas.
   * @param trainPath ruta hacia el fichero de entrenamiento
   * @param testPath ruta hacia el fichero de test
   * @param balAlg algoritmo de balanceo de clases a aplicar sobre el conjunto
   *               de entrenamiento desbalanceado. Opciones: ROS, RUS o ASMOTE.
   */
  def applyGradientBoostedTreesDF(trainPath: String, testPath: String, balAlg: String): Unit = {
    // Inicia una nueva sesión de Spark
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("SparkApp")
      .getOrCreate()

    // Lectura del conjunto de entrenamiento
    val dfTrain = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(trainPath)
    // Modificamos el nombre de la variable clase
    var train = dfTrain.withColumnRenamed("_c18", "label")

    // Lectura del conjunto de test
    val dfTest = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(testPath)
    // Modifica el nombre de la variable de clase
    var test = dfTest.withColumnRenamed("_c18", "label")

    // Une las variables de entrada en una columna llamada 'features'
    val assembler = new VectorAssembler()
      .setInputCols(dfTrain.columns.init)
      .setOutputCol("features")

    // Conjunto de entrenamiento y test en caché para mayor velocidad
    train = assembler.transform(train).repartition(100).cache()
    test = assembler.transform(test).repartition(100).cache()

    /*---------------------------------------------------------------*/
    // ALGORITMO DE BALANCEADO DE CLASES
    // Almacena el dataframe balanceado resultante
    var balancedTrain: DataFrame = null
    // Aplica el algoritmo de balanceo de clases seleccionado
    balAlg match {
      case "ROS" =>
        balancedTrain = ROS(train.select("label", "features"),1.0)
        println("\nDISTRIBUCIÓN DE CLASES TRAS ROS")
        balancedTrain.groupBy("label").count().show

      case "RUS" =>
        balancedTrain = RUS(train.select("label", "features"))
        println("\nDISTRIBUCIÓN DE CLASES TRAS RUS")
        balancedTrain.groupBy("label").count().show

      case "ASMOTE" =>
        // Balanceo de clases con semilla para resultados reproducibles
        val asmote = new ASMOTE().setK(5).setPercOver(800).setSeed(1000)
        balancedTrain = asmote.transform(train.select("label", "features"))
        println("\nDISTRIBUCIÓN DE CLASES TRAS ASMOTE")
        balancedTrain.groupBy("label").count().show
    }
    /*---------------------------------------------------------------*/

    /*---------------------------------------------------------------*/
    // ALGORITMO DE CLASIFICACIÓN
    // Indexa las variables del dataset
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(balancedTrain)
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(balancedTrain)

    // Construye el clasificador con Gradient-Boosted Trees
    val classifierSettings = new GBTClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    // Traduce las etiquetas predichas al rango de valores del dataset
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Pipeline con todas las actividades
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

  /**
   * Función que aplica un algoritmo de preprocesamiento (ROS, RUS o ASMOTE)
   * para balancear el conjunto de entrenamiento y construir un clasificador
   * utilizando el algoritmo Random Forest utilizando la librería ML (DataFrames).
   * Finalmente se evalúa su calidad sobre los conjuntos de entrenamiento y test
   * calculando el número de muestras positivas y negativas bien clasificadas.
   * @param trainPath ruta hacia el fichero de entrenamiento
   * @param testPath ruta hacia el fichero de test
   * @param balAlg algoritmo de balanceo de clases a aplicar sobre el conjunto
   *               de entrenamiento desbalanceado. Opciones: ROS, RUS o ASMOTE.
   */
  def applyRandomForestDF(trainPath: String, testPath: String, balAlg: String): Unit = {
    // Inicia una nueva sesión de Spark
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("SparkApp")
      .getOrCreate()

    // Lectura del conjunto de entrenamiento
    val dfTrain = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(trainPath)
    // Modificamos el nombre de la variable clase
    var train = dfTrain.withColumnRenamed("_c18", "label")

    // Lectura del conjunto de test
    val dfTest = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(testPath)
    // Modifica el nombre de la variable de clase
    var test = dfTest.withColumnRenamed("_c18", "label")

    // Une las variables de entrada en una columna llamada 'features'
    val assembler = new VectorAssembler()
      .setInputCols(dfTrain.columns.init)
      .setOutputCol("features")

    // Conjunto de entrenamiento y test en caché para mayor velocidad
    train = assembler.transform(train).repartition(100).cache()
    test = assembler.transform(test).repartition(100).cache()

    /*---------------------------------------------------------------*/
    // ALGORITMO DE BALANCEADO DE CLASES
    // Almacena el dataframe balanceado resultante
    var balancedTrain: DataFrame = null
    // Aplica el algoritmo de balanceo de clases seleccionado
    balAlg match {
      case "ROS" =>
        balancedTrain = ROS(train.select("label", "features"),1.0)
        println("\nDISTRIBUCIÓN DE CLASES TRAS ROS")
        balancedTrain.groupBy("label").count().show

      case "RUS" =>
        balancedTrain = RUS(train.select("label", "features"))
        println("\nDISTRIBUCIÓN DE CLASES TRAS RUS")
        balancedTrain.groupBy("label").count().show

      case "ASMOTE" =>
        // Balanceo de clases con semilla para resultados reproducibles
        val asmote = new ASMOTE().setK(5).setPercOver(800).setSeed(1000)
        balancedTrain = asmote.transform(train.select("label", "features"))
        println("\nDISTRIBUCIÓN DE CLASES TRAS ASMOTE")
        balancedTrain.groupBy("label").count().show
    }
    /*---------------------------------------------------------------*/

    /*---------------------------------------------------------------*/
    // ALGORITMO DE CLASIFICACIÓN
    // Indexa las variables del dataset
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(balancedTrain)
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(balancedTrain)

    // Construye el clasificador con Random Forest
    val classifierSettings = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    // Traduce las etiquetas predichas al rango de valores del dataset
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Pipeline con todas las actividades
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

  /**
   * Función que aplica un algoritmo de preprocesamiento (ROS, RUS o ASMOTE)
   * para balancear el conjunto de entrenamiento y construir un clasificador
   * utilizando el algoritmo Regresión Logística utilizando la librería ML (DataFrames).
   * Finalmente se evalúa su calidad sobre los conjuntos de entrenamiento y test
   * calculando el número de muestras positivas y negativas bien clasificadas.
   * @param trainPath ruta hacia el fichero de entrenamiento
   * @param testPath ruta hacia el fichero de test
   * @param balAlg algoritmo de balanceo de clases a aplicar sobre el conjunto
   *               de entrenamiento desbalanceado. Opciones: ROS, RUS o ASMOTE.
   */
  def applyLogisticRegressionDF(trainPath: String, testPath: String, balAlg: String): Unit = {
    // Inicia una nueva sesión de Spark
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("SparkApp")
      .getOrCreate()

    // Lectura del conjunto de entrenamiento
    val dfTrain = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(trainPath)
    // Modificamos el nombre de la variable clase
    var train = dfTrain.withColumnRenamed("_c18", "label")

    // Lectura del conjunto de test
    val dfTest = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(testPath)
    // Modifica el nombre de la variable de clase
    var test = dfTest.withColumnRenamed("_c18", "label")

    // Une las variables de entrada en una columna llamada 'features'
    val assembler = new VectorAssembler()
      .setInputCols(dfTrain.columns.init)
      .setOutputCol("features")

    // Conjunto de entrenamiento y test en caché para mayor velocidad
    train = assembler.transform(train).repartition(100).cache()
    test = assembler.transform(test).repartition(100).cache()

    /*---------------------------------------------------------------*/
    // ALGORITMO DE BALANCEADO DE CLASES
    // Almacena el dataframe balanceado resultante
    var balancedTrain: DataFrame = null
    // Aplica el algoritmo de balanceo de clases seleccionado
    balAlg match {
      case "ROS" =>
        balancedTrain = ROS(train.select("label", "features"),1.0)
        println("\nDISTRIBUCIÓN DE CLASES TRAS ROS")
        balancedTrain.groupBy("label").count().show

      case "RUS" =>
        balancedTrain = RUS(train.select("label", "features"))
        println("\nDISTRIBUCIÓN DE CLASES TRAS RUS")
        balancedTrain.groupBy("label").count().show

      case "ASMOTE" =>
        // Balanceo de clases con semilla para resultados reproducibles
        val asmote = new ASMOTE().setK(5).setPercOver(800).setSeed(1000)
        balancedTrain = asmote.transform(train.select("label", "features"))
        println("\nDISTRIBUCIÓN DE CLASES TRAS ASMOTE")
        balancedTrain.groupBy("label").count().show
    }
    /*---------------------------------------------------------------*/

    /*---------------------------------------------------------------*/
    // ALGORITMO DE CLASIFICACIÓN
    // Indexa las variables del dataset
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(balancedTrain)
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(balancedTrain)

    // Construye el clasificador con Regresión Logística Binomial
    val classifierSettings = new LogisticRegression()
      .setFamily("binomial")
      .setMaxIter(10)
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    // Traduce las etiquetas predichas al rango de valores del dataset
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Pipeline con todas las actividades
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

  /**
   * Función que aplica un algoritmo de preprocesamiento (ROS, RUS o ASMOTE)
   * para balancear el conjunto de entrenamiento y construir un clasificador
   * utilizando el algoritmo k-Nearest Neighbors Iterative Spark-based
   * haciendo referencia a la librería ML (DataFrames). Finalmente se evalúa su
   * calidad sobre los conjuntos de entrenamiento y test calculando el número
   * de muestras positivas y negativas bien clasificadas.
   * @param trainPath ruta hacia el fichero de entrenamiento
   * @param testPath ruta hacia el fichero de test
   * @param balAlg algoritmo de balanceo de clases a aplicar sobre el conjunto
   *               de entrenamiento desbalanceado. Opciones: ROS, RUS o ASMOTE.
   */
  def applykNNISDF(trainPath: String, testPath: String, balAlg: String): Unit = {
    // Inicia una nueva sesión de Spark
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("SparkApp")
      .getOrCreate()

    // Lectura del conjunto de entrenamiento
    val dfTrain = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(trainPath)
    // Modificamos el nombre de la variable clase
    var train = dfTrain.withColumn("label", dfTrain("_c18").cast("Integer"))

    // Lectura del conjunto de test
    val dfTest = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(testPath)
    // Modifica el nombre de la variable de clase
    var test = dfTest.withColumn("label", dfTest("_c18").cast("Integer"))

    // Une las variables de entrada en una columna llamada 'features'
    val assembler = new VectorAssembler()
      .setInputCols(dfTrain.columns.init)
      .setOutputCol("features")

    // Conjunto de entrenamiento y test en caché para mayor velocidad
    train = assembler.transform(train).repartition(100).cache()
    test = assembler.transform(test).repartition(100).cache()

    /*---------------------------------------------------------------*/
    // ALGORITMO DE BALANCEADO DE CLASES
    // Almacena el dataframe balanceado resultante
    var balancedTrain: DataFrame = null
    // Aplica el algoritmo de balanceo de clases seleccionado
    balAlg match {
      case "ROS" =>
        balancedTrain = ROS(train.select("label", "features"),1.0)
        println("\nDISTRIBUCIÓN DE CLASES TRAS ROS")
        balancedTrain.groupBy("label").count().show

      case "RUS" =>
        balancedTrain = RUS(train.select("label", "features"))
        println("\nDISTRIBUCIÓN DE CLASES TRAS RUS")
        balancedTrain.groupBy("label").count().show

      case "ASMOTE" =>
        // Balanceo de clases con semilla para resultados reproducibles
        val asmote = new ASMOTE().setK(5).setPercOver(800).setSeed(1000)
        balancedTrain = asmote.transform(train.select("label", "features"))
        println("\nDISTRIBUCIÓN DE CLASES TRAS ASMOTE")
        balancedTrain.groupBy("label").count().show
    }
    /*---------------------------------------------------------------*/

    /*---------------------------------------------------------------*/
    // ALGORITMO DE CLASIFICACIÓN
    // Construye un clasificador con kNN-IS
    val outPathArray: Array[String] = new Array[String](1)
    outPathArray(0) = "."
    val classifierSettings = new kNN_ISClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setK(3)
      .setDistanceType(2)
      .setNumClass(2)
      .setNumFeatures(balancedTrain.columns.size-1)
      .setNumPartitionMap(15)
      .setNumReduces(15)
      .setNumIter(10)
      .setMaxWeight(1)
      .setNumSamplesTest(test.count.toInt)
      .setOutPath(outPathArray)

    // Pipeline con todas las actividades
    val pipeline = new Pipeline().setStages(Array(classifierSettings))

    // Entrenamiento del modelo
    val model = pipeline.fit(balancedTrain)
    // Predicciones y evaluación sobre train
    val trainPredictions = model.transform(balancedTrain).select("label", "prediction")
    println("EVALUACIÓN SOBRE TRAIN:")
    calculateQualityMetrics(trainPredictions, "prediction")
    // Predicciones y evaluación sobre test
    val testPredictions = model.transform(test).select("label", "prediction")
    println("EVALUACIÓN SOBRE TEST:")
    calculateQualityMetrics(testPredictions, "prediction")
    /*---------------------------------------------------------------*/
  }

  /**
   * Función que aplica un algoritmo de preprocesamiento (ROS, RUS o ASMOTE)
   * para balancear el conjunto de entrenamiento y construir un clasificador
   * utilizando el algoritmo Light Gradient Boosted Machines haciendo
   * referencia a la librería ML (DataFrames). Finalmente se evalúa su
   * calidad sobre los conjuntos de entrenamiento y test calculando el número
   * de muestras positivas y negativas bien clasificadas.
   * @param trainPath ruta hacia el fichero de entrenamiento
   * @param testPath ruta hacia el fichero de test
   * @param balAlg algoritmo de balanceo de clases a aplicar sobre el conjunto
   *               de entrenamiento desbalanceado. Opciones: ROS, RUS o ASMOTE.
   */
  def applyLightGBMDF(trainPath: String, testPath: String, balAlg: String): Unit = {
    // Inicia una nueva sesión de Spark
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("SparkApp")
      .getOrCreate()

    // Lectura del conjunto de entrenamiento
    val dfTrain = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(trainPath)
    // Modificamos el nombre de la variable clase
    var train = dfTrain.withColumnRenamed("_c18", "label")

    // Lectura del conjunto de test
    val dfTest = spark.read
      .format("csv")
      .option("inferSchema", true)
      .option("header", false)
      .load(testPath)
    // Modifica el nombre de la variable de clase
    var test = dfTest.withColumnRenamed("_c18", "label")

    // Une las variables de entrada en una columna llamada 'features'
    val assembler = new VectorAssembler()
      .setInputCols(dfTrain.columns.init)
      .setOutputCol("features")

    // Conjunto de entrenamiento y test en caché para mayor velocidad
    train = assembler.transform(train).repartition(100).cache()
    test = assembler.transform(test).repartition(100).cache()

    /*---------------------------------------------------------------*/
    // ALGORITMO DE BALANCEADO DE CLASES
    // Almacena el dataframe balanceado resultante
    var balancedTrain: DataFrame = null
    // Aplica el algoritmo de balanceo de clases seleccionado
    balAlg match {
      case "ROS" =>
        balancedTrain = ROS(train.select("label", "features"),1.0)
        println("\nDISTRIBUCIÓN DE CLASES TRAS ROS")
        balancedTrain.groupBy("label").count().show

      case "RUS" =>
        balancedTrain = RUS(train.select("label", "features"))
        println("\nDISTRIBUCIÓN DE CLASES TRAS RUS")
        balancedTrain.groupBy("label").count().show

      case "ASMOTE" =>
        // Balanceo de clases con semilla para resultados reproducibles
        val asmote = new ASMOTE().setK(5).setPercOver(800).setSeed(1000)
        balancedTrain = asmote.transform(train.select("label", "features"))
        println("\nDISTRIBUCIÓN DE CLASES TRAS ASMOTE")
        balancedTrain.groupBy("label").count().show
    }
    /*---------------------------------------------------------------*/

    /*---------------------------------------------------------------*/
    // ALGORITMO DE CLASIFICACIÓN
    // Indexa las variables del dataset
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(balancedTrain)
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .fit(balancedTrain)

    // Construye el clasificador con Light Gradient Boosted Machines
    val classifierSettings = new LightGBMClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumIterations(100)

    // Traduce las etiquetas predichas al rango de valores del dataset
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Pipeline con todas las actividades
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

  /**
   * Función que aplica un algoritmo de preprocesamiento (ROS, RUS o SMOTE)
   * para balancear el conjunto de entrenamiento y construir un clasificador
   * utilizando el algoritmo Árboles de Decisión utilizando la libreria MLLib (RDD).
   * Finalmente se evalúa su calidad sobre los conjuntos de entrenamiento y
   * test calculando el número de muestras positivas y negativas bien clasificadas.
   * @param trainPath ruta hacia el fichero de entrenamiento
   * @param testPath ruta hacia el fichero de test
   * @param balAlg algoritmo de preprocesamiento de ruido para eliminar instancias
   *               ruidosas del conjunto de entrenamiento. Opciones: HME, HTE o ENN.
   */
  def applyDecisionTreesRDD(trainPath: String, testPath: String, balAlg: String): Unit = {
    // Inicia una nueva sesión de Spark
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("SparkApp")
      .getOrCreate()

    // Lectura del conjunto de entrenamiento como RDD
    val rddTrain = spark.sparkContext.textFile(trainPath).map { line =>
      val featureVector = Vectors.dense(line.split(",").map(f => f.toDouble).init)
      val label = line.split(",").map(f => f.toDouble).last
      LabeledPoint(label, featureVector)
    }.persist

    // Lectura del conjunto de test como RDD
    val rddTest = spark.sparkContext.textFile(testPath).map { line =>
      val featureVector = Vectors.dense(line.split(",").map(f => f.toDouble).init)
      val label = line.split(",").map(f => f.toDouble).last
      LabeledPoint(label, featureVector)
    }.persist

    /*---------------------------------------------------------------*/
    // ALGORITMO DE BALANCEADO DE CLASES
    // Almacena el dataframe balanceado resultante
    var balancedTrain: RDD[LabeledPoint] = null
    // Aplica el algoritmo de balanceo de clases seleccionado
    balAlg match {
      case "HME" =>
        balancedTrain = new HME_BD(rddTrain, 100, 4, 10, 1000).runFilter()
        println("\nDISTRIBUCIÓN DE CLASES TRAS HME")

      case "HTE" =>
        balancedTrain = new HTE_BD(rddTrain, 100, 4, 0, 3, 10, 1000).runFilter()
        println("\nDISTRIBUCIÓN DE CLASES TRAS HTE")

      case "ENN" =>
        balancedTrain = new ENN_BD(rddTrain, 3).runFilter()
        println("\nDISTRIBUCIÓN DE CLASES TRAS ENN")
    }
    // Convierte el RDD a Dataframe para observar el balanceo de clases
    import spark.implicits._
    val balancedTrainDF = balancedTrain.map(e => (e.label, e.features)).toDF("label", "features")
    balancedTrainDF.groupBy("label").count().show
    /*---------------------------------------------------------------*/

    /*---------------------------------------------------------------*/
    // ALGORITMO DE CLASIFICACIÓN
    // Configuración y entrenamiento de un modelo con Árboles de Decisión
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]() // Características continuas
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32
    val model = DecisionTree.trainClassifier(balancedTrain,
      numClasses,
      categoricalFeaturesInfo,
      impurity,
      maxDepth,
      maxBins)

    // Predicciones y evaluación sobre entrenamiento
    val trainPredictions = balancedTrain.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    println("TP = " + trainPredictions.filter(r => r._1 == r._2 && r._1 == 1).count())
    println("TN = " + trainPredictions.filter(r => r._1 == r._2 && r._1 == 0).count())
    // Predicciones y evaluación sobre test
    val testPredictions = rddTest.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    println("TP = " + testPredictions.filter(r => r._1 == r._2 && r._1 == 1).count())
    println("TN = " + testPredictions.filter(r => r._1 == r._2 && r._1 == 0).count())
    /*---------------------------------------------------------------*/
  }

  /**
   * Función que aplica un algoritmo de preprocesamiento (ROS, RUS o SMOTE)
   * para balancear el conjunto de entrenamiento y construir un clasificador
   * utilizando el algoritmo Gradient-Boosted Trees utilizando la libreria MLLib (RDD).
   * Finalmente se evalúa su calidad sobre los conjuntos de entrenamiento y
   * test calculando el número de muestras positivas y negativas bien clasificadas.
   * @param trainPath ruta hacia el fichero de entrenamiento
   * @param testPath ruta hacia el fichero de test
   * @param balAlg algoritmo de preprocesamiento de ruido para eliminar instancias
   *               ruidosas del conjunto de entrenamiento. Opciones: HME, HTE o ENN.
   */
  def applyGradientBoostedTreesRDD(trainPath: String, testPath: String, balAlg: String): Unit = {
    // Inicia una nueva sesión de Spark
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("SparkApp")
      .getOrCreate()

    // Lectura del conjunto de entrenamiento como RDD
    val rddTrain = spark.sparkContext.textFile(trainPath).map { line =>
      val featureVector = Vectors.dense(line.split(",").map(f => f.toDouble).init)
      val label = line.split(",").map(f => f.toDouble).last
      LabeledPoint(label, featureVector)
    }.persist

    // Lectura del conjunto de test como RDD
    val rddTest = spark.sparkContext.textFile(testPath).map { line =>
      val featureVector = Vectors.dense(line.split(",").map(f => f.toDouble).init)
      val label = line.split(",").map(f => f.toDouble).last
      LabeledPoint(label, featureVector)
    }.persist

    /*---------------------------------------------------------------*/
    // ALGORITMO DE BALANCEADO DE CLASES
    // Almacena el dataframe balanceado resultante
    var balancedTrain: RDD[LabeledPoint] = null
    // Aplica el algoritmo de balanceo de clases seleccionado
    balAlg match {
      case "HME" =>
        balancedTrain = new HME_BD(rddTrain, 100, 4, 10, 1000).runFilter()
        println("\nDISTRIBUCIÓN DE CLASES TRAS HME")

      case "HTE" =>
        balancedTrain = new HTE_BD(rddTrain, 100, 4, 0, 3, 10, 1000).runFilter()
        println("\nDISTRIBUCIÓN DE CLASES TRAS HTE")

      case "ENN" =>
        balancedTrain = new ENN_BD(rddTrain, 3).runFilter()
        println("\nDISTRIBUCIÓN DE CLASES TRAS ENN")
    }
    // Convierte el RDD a Dataframe para observar el balanceo de clases
    import spark.implicits._
    val balancedTrainDF = balancedTrain.map(e => (e.label, e.features)).toDF("label", "features")
    balancedTrainDF.groupBy("label").count().show
    /*---------------------------------------------------------------*/

    /*---------------------------------------------------------------*/
    // ALGORITMO DE CLASIFICACIÓN
    // Configuración de un modelo con Gradient-Boosted Trees
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
      boostingStrategy.treeStrategy.numClasses = 2
      boostingStrategy.treeStrategy.maxDepth = 5
      boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    // Entrenamiento de un clasificador
    val model = GradientBoostedTrees.train(balancedTrain, boostingStrategy)

    // Predicciones y evaluación sobre entrenamiento
    val trainPredictions = balancedTrain.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    println("TP = " + trainPredictions.filter(r => r._1 == r._2 && r._1 == 1).count())
    println("TN = " + trainPredictions.filter(r => r._1 == r._2 && r._1 == 0).count())
    // Predicciones y evaluación sobre test
    val testPredictions = rddTest.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    println("TP = " + testPredictions.filter(r => r._1 == r._2 && r._1 == 1).count())
    println("TN = " + testPredictions.filter(r => r._1 == r._2 && r._1 == 0).count())
    /*---------------------------------------------------------------*/
  }

  /**
   * Función que aplica un algoritmo de preprocesamiento (ROS, RUS o SMOTE)
   * para balancear el conjunto de entrenamiento y construir un clasificador
   * utilizando el algoritmo Random Forest utilizando la libreria MLLib (RDD).
   * Finalmente se evalúa su calidad sobre los conjuntos de entrenamiento y
   * test calculando el número de muestras positivas y negativas bien clasificadas.
   * @param trainPath ruta hacia el fichero de entrenamiento
   * @param testPath ruta hacia el fichero de test
   * @param balAlg algoritmo de preprocesamiento de ruido para eliminar instancias
   *               ruidosas del conjunto de entrenamiento. Opciones: HME, HTE o ENN.
   */
  def applyRandomForestRDD(trainPath: String, testPath: String, balAlg: String): Unit = {
    // Inicia una nueva sesión de Spark
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("SparkApp")
      .getOrCreate()

    // Lectura del conjunto de entrenamiento como RDD
    val rddTrain = spark.sparkContext.textFile(trainPath).map { line =>
      val featureVector = Vectors.dense(line.split(",").map(f => f.toDouble).init)
      val label = line.split(",").map(f => f.toDouble).last
      LabeledPoint(label, featureVector)
    }.persist

    // Lectura del conjunto de test como RDD
    val rddTest = spark.sparkContext.textFile(testPath).map { line =>
      val featureVector = Vectors.dense(line.split(",").map(f => f.toDouble).init)
      val label = line.split(",").map(f => f.toDouble).last
      LabeledPoint(label, featureVector)
    }.persist

    /*---------------------------------------------------------------*/
    // ALGORITMO DE BALANCEADO DE CLASES
    // Almacena el dataframe balanceado resultante
    var balancedTrain: RDD[LabeledPoint] = null
    // Aplica el algoritmo de balanceo de clases seleccionado
    balAlg match {
      case "HME" =>
        balancedTrain = new HME_BD(rddTrain, 100, 4, 10, 1000).runFilter()
        println("\nDISTRIBUCIÓN DE CLASES TRAS HME")

      case "HTE" =>
        balancedTrain = new HTE_BD(rddTrain, 100, 4, 0, 3, 10, 1000).runFilter()
        println("\nDISTRIBUCIÓN DE CLASES TRAS HTE")

      case "ENN" =>
        balancedTrain = new ENN_BD(rddTrain, 3).runFilter()
        println("\nDISTRIBUCIÓN DE CLASES TRAS ENN")
    }
    // Convierte el RDD a Dataframe para observar el balanceo de clases
    import spark.implicits._
    val balancedTrainDF = balancedTrain.map(e => (e.label, e.features)).toDF("label", "features")
    balancedTrainDF.groupBy("label").count().show
    /*---------------------------------------------------------------*/

    /*---------------------------------------------------------------*/
    // ALGORITMO DE CLASIFICACIÓN
    // Configuración y entrenamiento de un modelo con Random Forest
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]() // Variables continuas
    val numTrees = 100
    val featureSubsetStrategy = "auto" // El algoritmo decide la estrategia automáticamente
    val impurity = "gini"
    val maxDepth = 10
    val maxBins = 32
    val model = RandomForest.trainClassifier(balancedTrain,
      numClasses,
      categoricalFeaturesInfo,
      numTrees,
      featureSubsetStrategy,
      impurity,
      maxDepth,
      maxBins)

    // Predicciones y evaluación sobre entrenamiento
    val trainPredictions = balancedTrain.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    println("TP = " + trainPredictions.filter(r => r._1 == r._2 && r._1 == 1).count())
    println("TN = " + trainPredictions.filter(r => r._1 == r._2 && r._1 == 0).count())
    // Predicciones y evaluación sobre test
    val testPredictions = rddTest.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    println("TP = " + testPredictions.filter(r => r._1 == r._2 && r._1 == 1).count())
    println("TN = " + testPredictions.filter(r => r._1 == r._2 && r._1 == 0).count())
    /*---------------------------------------------------------------*/
  }

  /**
   * Función que aplica un algoritmo de preprocesamiento (ROS, RUS o SMOTE)
   * para balancear el conjunto de entrenamiento y construir un clasificador
   * utilizando el algoritmo Regresión Logística utilizando la libreria MLLib (RDD).
   * Finalmente se evalúa su calidad sobre los conjuntos de entrenamiento y
   * test calculando el número de muestras positivas y negativas bien clasificadas.
   * @param trainPath ruta hacia el fichero de entrenamiento
   * @param testPath ruta hacia el fichero de test
   * @param balAlg algoritmo de preprocesamiento de ruido para eliminar instancias
   *               ruidosas del conjunto de entrenamiento. Opciones: HME, HTE o ENN.
   */
  def applyLogisticRegressionRDD(trainPath: String, testPath: String, balAlg: String): Unit = {
    // Inicia una nueva sesión de Spark
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("SparkApp")
      .getOrCreate()

    // Lectura del conjunto de entrenamiento como RDD
    val rddTrain = spark.sparkContext.textFile(trainPath).map { line =>
      val featureVector = Vectors.dense(line.split(",").map(f => f.toDouble).init)
      val label = line.split(",").map(f => f.toDouble).last
      LabeledPoint(label, featureVector)
    }.persist

    // Lectura del conjunto de test como RDD
    val rddTest = spark.sparkContext.textFile(testPath).map { line =>
      val featureVector = Vectors.dense(line.split(",").map(f => f.toDouble).init)
      val label = line.split(",").map(f => f.toDouble).last
      LabeledPoint(label, featureVector)
    }.persist

    /*---------------------------------------------------------------*/
    // ALGORITMO DE BALANCEADO DE CLASES
    // Almacena el dataframe balanceado resultante
    var balancedTrain: RDD[LabeledPoint] = null
    // Aplica el algoritmo de balanceo de clases seleccionado
    balAlg match {
      case "HME" =>
        balancedTrain = new HME_BD(rddTrain, 100, 4, 10, 1000).runFilter()
        println("\nDISTRIBUCIÓN DE CLASES TRAS HME")

      case "HTE" =>
        balancedTrain = new HTE_BD(rddTrain, 100, 4, 0, 3, 10, 1000).runFilter()
        println("\nDISTRIBUCIÓN DE CLASES TRAS HTE")

      case "ENN" =>
        balancedTrain = new ENN_BD(rddTrain, 3).runFilter()
        println("\nDISTRIBUCIÓN DE CLASES TRAS ENN")
    }
    // Convierte el RDD a Dataframe para observar el balanceo de clases
    import spark.implicits._
    val balancedTrainDF = balancedTrain.map(e => (e.label, e.features)).toDF("label", "features")
    balancedTrainDF.groupBy("label").count().show
    /*---------------------------------------------------------------*/

    /*---------------------------------------------------------------*/
    // ALGORITMO DE CLASIFICACIÓN
    // Configuración y entrenamiento de un modelo con Regresión Logística
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(balancedTrain)

    // Predicciones y evaluación sobre entrenamiento
    val trainPredictions = balancedTrain.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    println("TP = " + trainPredictions.filter(r => r._1 == r._2 && r._1 == 1).count())
    println("TN = " + trainPredictions.filter(r => r._1 == r._2 && r._1 == 0).count())
    // Predicciones y evaluación sobre test
    val testPredictions = rddTest.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    println("TP = " + testPredictions.filter(r => r._1 == r._2 && r._1 == 1).count())
    println("TN = " + testPredictions.filter(r => r._1 == r._2 && r._1 == 0).count())
    /*---------------------------------------------------------------*/
  }

  /**
   * Función que aplica un algoritmo de preprocesamiento (HME, HTE o ENN)
   * para balancear el conjunto de entrenamiento y construir un clasificador
   * utilizando el algoritmo k-Nearest Neighbors Iterative Spark-based
   * haciendo referencia a la librería ML (DataFrames). Finalmente se evalúa su
   * calidad sobre los conjuntos de entrenamiento y test calculando el número
   * de muestras positivas y negativas bien clasificadas.
   * @param trainPath ruta hacia el fichero de entrenamiento
   * @param testPath ruta hacia el fichero de test
   * @param balAlg algoritmo de balanceo de clases a aplicar sobre el conjunto
   *               de entrenamiento desbalanceado. Opciones: HME, HTE o ENN.
   */
  def applykNNISRDD(trainPath: String, testPath: String, balAlg: String): Unit = {
    // Inicia una nueva sesión de Spark
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("SparkApp")
      .getOrCreate()

    // Lectura del conjunto de entrenamiento como RDD
    val rddTrain = spark.sparkContext.textFile(trainPath).map { line =>
      val featureVector = Vectors.dense(line.split(",").map(f => f.toDouble).init)
      val label = line.split(",").map(f => f.toDouble).last
      LabeledPoint(label, featureVector)
    }.persist

    // Lectura del conjunto de test como RDD
    val rddTest = spark.sparkContext.textFile(testPath).map { line =>
      val featureVector = Vectors.dense(line.split(",").map(f => f.toDouble).init)
      val label = line.split(",").map(f => f.toDouble).last
      LabeledPoint(label, featureVector)
    }.persist

    /*---------------------------------------------------------------*/
    // ALGORITMO DE BALANCEADO DE CLASES
    // Almacena el dataframe balanceado resultante
    var balancedTrain: RDD[LabeledPoint] = null
    // Aplica el algoritmo de balanceo de clases seleccionado
    balAlg match {
      case "HME" =>
        balancedTrain = new HME_BD(rddTrain, 100, 4, 10, 1000).runFilter()
        println("\nDISTRIBUCIÓN DE CLASES TRAS HME")

      case "HTE" =>
        balancedTrain = new HTE_BD(rddTrain, 100, 4, 0, 3, 10, 1000).runFilter()
        println("\nDISTRIBUCIÓN DE CLASES TRAS HTE")

      case "ENN" =>
        balancedTrain = new ENN_BD(rddTrain, 3).runFilter()
        println("\nDISTRIBUCIÓN DE CLASES TRAS ENN")
    }
    // Convierte el RDD a Dataframe para observar el balanceo de clases
    import spark.implicits._
    val balancedTrainDF = balancedTrain.map(e => (e.label, e.features)).toDF("label", "features")
    balancedTrainDF.groupBy("label").count().show
    /*---------------------------------------------------------------*/

    /*---------------------------------------------------------------*/
    // ALGORITMO DE CLASIFICACIÓN
    // Configuración y entrenamiento de un modelo con kNN-IS
    val k = 3
    val distanceType = 2
    val numClass = 2
    val numFeatures = balancedTrainDF.columns.size-1
    val numPartitionMap = 15
    val numReduces = 15
    val numIterations = 10
    val maxWeight = 1
    var model = kNN_IS.setup(balancedTrain,
      balancedTrain,
      k,
      distanceType,
      numClass,
      numFeatures,
      numPartitionMap,
      numReduces,
      numIterations,
      maxWeight)
    // Predicciones y evaluación sobre entrenamiento
    val trainPredictions = model.predict(spark.sparkContext)
    println("TP = " + trainPredictions.filter(r => r._1 == r._2 && r._1 == 1).count())
    println("TN = " + trainPredictions.filter(r => r._1 == r._2 && r._1 == 0).count())

    // Entrenamiento de un segundo modelo para evaluar sobre test
    model = kNN_IS.setup(balancedTrain,
      rddTest,
      k,
      distanceType,
      numClass,
      numFeatures,
      numPartitionMap,
      numReduces,
      numIterations,
      maxWeight)
    // Predicciones y evaluación sobre test
    val testPredictions = model.predict(spark.sparkContext)
    println("TP = " + testPredictions.filter(r => r._1 == r._2 && r._1 == 1).count())
    println("TN = " + testPredictions.filter(r => r._1 == r._2 && r._1 == 0).count())
    /*---------------------------------------------------------------*/
  }

  /**
   * Función principal que referencia a los procedimientos definidos anteriormente
   * para balancear/preprocesar el conjunto de datos de entrenamiento, construir
   * un clasificador y evaluar su bondad en entrenamiento y test.
   * @param arg conjunto de parámetros de entrada.
   */
  def main(arg: Array[String]): Unit = {
    // Árboles de Decisión + ROS
    applyDecisionTreesDF(trainLocalPath, testLocalPath, "ROS")
    // Árboles de Decisión + RUS
    applyDecisionTreesDF(trainLocalPath, testLocalPath, "RUS")
    // Árboles de Decisión + ASMOTE
    applyDecisionTreesDF(trainLocalPath, testLocalPath, "ASMOTE")
    // Árboles de Decisión + HME
    applyDecisionTreesRDD(trainLocalPath, testLocalPath, "HME")
    // Árboles de Decisión + HTE
    applyDecisionTreesRDD(trainLocalPath, testLocalPath, "HTE")
    // Árboles de Decisión + HTE
    applyDecisionTreesRDD(trainLocalPath, testLocalPath, "ENN")

    // Gradient-Boosted Trees + ROS
    applyGradientBoostedTreesDF(trainLocalPath, testLocalPath, "ROS")
    // Gradient-Boosted Trees + RUS
    applyGradientBoostedTreesDF(trainLocalPath, testLocalPath, "RUS")
    // Gradient-Boosted Trees + ASMOTE
    applyGradientBoostedTreesDF(trainLocalPath, testLocalPath, "ASMOTE")
    // Gradient-Boosted Trees + HME
    applyGradientBoostedTreesRDD(trainLocalPath, testLocalPath, "HME")
    // Gradient-Boosted Trees + HTE
    applyGradientBoostedTreesRDD(trainLocalPath, testLocalPath, "HTE")
    // Gradient-Boosted Trees + HTE
    applyGradientBoostedTreesRDD(trainLocalPath, testLocalPath, "ENN")

    // Random Forest + ROS
    applyRandomForestDF(trainLocalPath, testLocalPath, "ROS")
    // Random Forest + RUS
    applyRandomForestDF(trainLocalPath, testLocalPath, "RUS")
    // Random Forest + ASMOTE
    applyRandomForestDF(trainLocalPath, testLocalPath, "ASMOTE")
    // Random Forest + HME
    applyRandomForestRDD(trainLocalPath, testLocalPath, "HME")
    // Random Forest + HTE
    applyRandomForestRDD(trainLocalPath, testLocalPath, "HTE")
    // Random Forest + HTE
    applyRandomForestRDD(trainLocalPath, testLocalPath, "ENN")

    // Regresión Logística Binomial + ROS
    applyLogisticRegressionDF(trainLocalPath, testLocalPath, "ROS")
    // Regresión Logística Binomial + RUS
    applyLogisticRegressionDF(trainLocalPath, testLocalPath, "RUS")
    // Regresión Logística Binomial + ASMOTE
    applyLogisticRegressionDF(trainLocalPath, testLocalPath, "ASMOTE")
    // Regresión Logística Binomial + HME
    applyLogisticRegressionRDD(trainLocalPath, testLocalPath, "HME")
    // Regresión Logística Binomial + HTE
    applyLogisticRegressionRDD(trainLocalPath, testLocalPath, "HTE")
    // Regresión Logística Binomial + HTE
    applyLogisticRegressionRDD(trainLocalPath, testLocalPath, "ENN")

    // kNN-IS + ROS
    applykNNISDF(train10kLocalPath, test10kLocalPath, "ROS")
    // kNN-IS + RUS
    applykNNISDF(trainLocalPath, testLocalPath, "RUS")
    // kNN-IS + ASMOTE
    applykNNISDF(trainLocalPath, testLocalPath, "ASMOTE")
    // kNN-IS + HME
    applykNNISRDD(trainLocalPath, testLocalPath, "HME")
    // kNN-IS + HTE
    applykNNISDF(trainLocalPath, testLocalPath, "HTE")
    // kNN-IS + ENN
    applykNNISDF(trainLocalPath, testLocalPath, "ENN")

    // LightGBM + ROS
    applyLightGBMDF(trainLocalPath, testLocalPath, "ROS")
    // LightGBM + RUS
    applyLightGBMDF(trainLocalPath, testLocalPath, "RUS")
    // LightGBM + ASMOTE
    applyLightGBMDF(trainLocalPath, testLocalPath, "ASMOTE")
  }
}
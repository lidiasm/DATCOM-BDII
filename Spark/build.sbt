ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.15"

lazy val root = (project in file("."))
  .settings(
    name := "BDII",
    resolvers += "Spark Packages Repo" at "https://repos.spark-packages.org",
    libraryDependencies += "org.scala-lang" % "scala-library" % scalaVersion.value,
    libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.2.1",
    libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.2.1",
    libraryDependencies += "org.apache.spark" %% "spark-core" % "3.2.1",
    libraryDependencies += "mjuez" % "approx-smote" % "1.1.2",
    libraryDependencies += "com.microsoft.azure" % "synapseml_2.12" % "0.9.5",
    libraryDependencies += "com.nvidia" % "xgboost4j-spark_3.0" % "1.4.2-0.2.0"
  )

package nbexample

import javax.servlet.http.HttpServlet
import javax.servlet.http.HttpServletRequest
import javax.servlet.http.HttpServletResponse

import org.apache.mahout.classifier.naivebayes._
import org.apache.mahout.nlp.tfidf._
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import org.apache.mahout.sparkbindings._

import org.apache.spark.SparkConf

import org.apache.hadoop.io.Text
import org.apache.hadoop.io.IntWritable
import org.apache.hadoop.io.LongWritable






class NaiveBayesServlet extends HttpServlet {

  val master = System.getenv("MASTER")

  val conf = new SparkConf()
    .set("spark.executor.extraClassPath",
      "/home/andy/mbk/mahout-matrices/naiveBayesExample" +
      "/target/naiveBayes-mahout-1.0-SNAPSHOT.jar")

  implicit val sdc = mahoutSparkContext(
                       masterUrl = master,
                       appName = "NaiveBayesExample",
                       sparkConf = conf )

  val pathToData = "hdfs://localhost:9000/tmp/mahout-work-wiki/"
 //val pathToData = "/tmp/mahout-work-wiki/"

  val model = NBModel.dfsRead(pathToData+"model")

  val standardClassifier = new StandardNBClassifier(model)
  val dictionary = sdc.sequenceFile(pathToData +
                                    "wikipediaVecs/dictionary.file-0",
                                    classOf[Text],
                                    classOf[IntWritable])

  val documentFrequencyCount = sdc.sequenceFile(pathToData +
                                                "wikipediaVecs/df-count",
                                                classOf[IntWritable],
                                                classOf[LongWritable])


   // setup the dictionary and document frequency count as maps
  val dictionaryRDD = dictionary.map { case (wKey, wVal) =>
                                   wKey.asInstanceOf[Text].toString() -> wVal.get() }

  val documentFrequencyCountRDD = documentFrequencyCount.map { case (wKey, wVal) =>
                                       wKey.asInstanceOf[IntWritable].get() -> wVal.get() }

  val dictionaryMap = dictionaryRDD.collect.map(x => x._1.toString -> x._2.toInt).toMap
  val dfCountMap = documentFrequencyCountRDD.collect.map(x => x._1.toInt -> x._2.toLong).toMap



  override def doPost(request: HttpServletRequest, response: HttpServletResponse) {
    val txt = request.getReader().readLine()
    response.getWriter().append("\n"+classifyText(txt)+"\n\n")
  }

  // for this simple example, tokenize our document into unigrams using native string
  // methods and vectorize using our dictionary and document frequencies.
  // You could also use a lucene analyzer for bigrams, trigrams, etc.
  def vectorizeDocument(document: String,
                        dictionaryMap: Map[String,Int],
                        dfMap: Map[Int,Long]): Vector = {

    val wordCounts = document.replaceAll("[^\\p{L}\\p{Nd}]+", " ")
                             .toLowerCase.split(" ")
                             .groupBy(identity)
                             .mapValues(_.length)

    val vec = new RandomAccessSparseVector(dictionaryMap.size)

    val totalDFSize = dfMap(-1)
    val docSize = wordCounts.size

    val tfidf: TFIDF = new TFIDF()

    for (word <- wordCounts) {
      val term = word._1
      if (dictionaryMap.contains(term)) {
        val termFreq = word._2
        val dictIndex = dictionaryMap(term)
        val docFreq = dfMap(dictIndex)
        val currentTfIdf = tfidf.calculate(termFreq,
                                           docFreq.toInt,
                                           docSize,
                                           totalDFSize.toInt)
        vec(dictIndex) = currentTfIdf
      }
    }
    vec
  }


  val labelMap = model.labelIndex
  val numLabels = model.numLabels
  val reverseLabelMap = labelMap.map(x => x._2 -> x._1)

  // instantiate the correct type of classifier
  val classifier = model.isComplementary match {
    case true => new ComplementaryNBClassifier(model)
    case _ => new StandardNBClassifier(model)
  }

  // the label with the higest score wins the classification for a given document
  def argmax(v: Vector): (Int, Double) = {
    var bestIdx: Int = Int.MinValue
    var bestScore: Double = Double.MinValue
    for(i <- 0 until v.size) {
      if(v(i) > bestScore){
        bestScore = v(i)
        bestIdx = i
      }
    }
    (bestIdx, bestScore)
  }

  // our final vector classifier
  def classifyDocument(clvec: Vector) : String = {
    val cvec = classifier.classifyFull(clvec)
    val (bestIdx, bestScore) = argmax(cvec)
    reverseLabelMap(bestIdx)
  }

  // lump everything together for a text classier
  def classifyText(txt: String): String = {
    val v = vectorizeDocument(txt, dictionaryMap, dfCountMap)
    classifyDocument(v)
  }

}


object NaiveBayesServlet {
    val serialVersionUID: Long = 1L;
}


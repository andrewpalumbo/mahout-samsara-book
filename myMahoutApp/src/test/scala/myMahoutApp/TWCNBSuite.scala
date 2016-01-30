package myMahoutApp

import java.io.{BufferedWriter, FileWriter}

import myMahoutApp.BFGS.{MVFunc, MVFuncGrad}
import org.apache.log4j.Level
import org.apache.mahout.logging._
import org.apache.mahout.math._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite
import org.apache.mahout.classifier.naivebayes._
import org.scalatest.{FunSuite, Matchers}
import scala.collection.JavaConversions._
import scala.util.Random


class TWCNBSuite extends FunSuite with DistributedSparkSuite with Matchers {

  private final implicit val log = getLog(classOf[TWCNBSuite])


  //def simTfIdf(n: Int, m: Int) : Matrix

  test("TWCNB"){

    val nDocs = 50
    val nTerms = 100
    val nClasses = 4

    val rand: Random = new Random(1235L)

    // simulated training TF-IDF dataset
    val mxTfIdfTrain = Matrices.uniformView(nDocs, nTerms, 1234).cloned
    mxTfIdfTrain ::= {(r,c,v) => v + math.abs(rand.nextInt(50))}

    val drmTfIdfTrain = drmParallelize(mxTfIdfTrain, numPartitions = 2)

    // simulated TF-IDF test dataset
    val mxTfIdfTest = Matrices.uniformView(nDocs, nTerms, 2345).cloned
    mxTfIdfTest ::= {(r,c,v) => v + math.abs(rand.nextInt(50))}

    val drmTfIdfTest = drmParallelize(mxTfIdfTest, numPartitions = 2)

    // set the label indices for both training sets.
    // class labels by row
    val labelVec: Vector = new RandomAccessSparseVector(nDocs)
    for(i <- 0 until nDocs){
      labelVec(i) = Math.abs(rand.nextInt(nClasses))
    }

    // String class labels
    val labelIndexMap = new scala.collection.mutable.HashMap[String, Integer]

    // set a string label for each ordinal class
    for(i <- 0 until nClasses){
      labelIndexMap.put(String.valueOf(i), i.toInt)
    }

    // send to the mapBlock closure
    val bcastLabelVec = drmBroadcast(labelVec)

    // set the row keys
    val drmTrain = drmTfIdfTrain.mapBlock(){
      case (keys, block) =>
        val rowLabelVec = bcastLabelVec.value
        val newKeys: Array[Int] = new Array[Int](keys.size)
        for(i <- 0 until keys.size) {
          newKeys(i) = rowLabelVec(i).toInt
        }
        (newKeys -> block)
    }

    /*Not currently working because the transpose trick
     * retains its geometry leaving empty rows
     // train a model (W.t) using TWCNB -- Algorithm 8.2
     val drmTwcnbWtModel = TWCNB.twcnbTrain(drmTrain)
    */

    // cheat and aggregate our simulated data using the transpose trick.
    // note that though the transpose and aggregate trick does aggregate,
    // it leaves a matrix of original dimension with empty rows.
    val drmTfIdfAggregate = drmTrain.t.checkpoint()
    val mxTfIdfAggregate = drmTfIdfAggregate.collect.t
    printf("Aggregated transposed matrix: %s\n", mxTfIdfAggregate)

    // our CBayes model expects a matrix with non empty rows.
    val mxTfIdfAggregateNonEmpty = new SparseRowMatrix(nClasses, nTerms)
    for(i <- 0 until nClasses) {
      mxTfIdfAggregateNonEmpty(i, ::) = mxTfIdfAggregate(i, ::)
    }

    printf("Aggregated non-empty transposed matrix: %s\n", mxTfIdfAggregateNonEmpty)

    // parralelize a drm with non empty rows for CBayes training
    val drmCbayesTrain = drmParallelize(mxTfIdfAggregateNonEmpty, numPartitions = 2)

    // train a model (W.t) using TWCNB -- Algorithm 8.2
    // we'll use the aggregated CBayes data for now.
    val drmTwcnbWtModel = TWCNB.twcnbTrain(drmTrain)

    // Algorithrm 8.3, Training (1),(2),(3)
    // train a model (NBModel) using SparkNaiveBayes.train(...)
    val cBayesModel = NaiveBayes.train(drmCbayesTrain,
                                       labelIndexMap,
                                       true, 1.0f)


    // now classify our test set using both classifiers

    // TWCNB  (Alg 8.2, label assignment (2))
    // drmTfIdfTest %*% W.t
    val drmScoredTWCNBdocs = drmTfIdfTest %*% drmTwcnbWtModel
    val mxScoredTWCNBdocs = drmScoredTWCNBdocs.collect
    printf("drmTwcnbWtModel: %s \n", drmTwcnbWtModel.collect)

    //  classify using an canned complementary NB Classifier
    //  work on in-core matrix since batch distributed classification requires
    //  dropping down into native Spark code
    val classifier = new ComplementaryNBClassifier(cBayesModel)
    val mxScoredCBayesdocs = new SparseMatrix(nDocs, nClasses)

    // algorithm 8.3, Label Assignment (4)
    for (i <- 0 until nDocs){
      mxScoredCBayesdocs(i,::) := classifier.classifyFull(mxTfIdfTest(i,::))
    }


    printf("mxScoredTWCNBdocs rows: %s columns: %s\n", mxScoredTWCNBdocs.numRows(), mxScoredTWCNBdocs.numCols())
    printf("mxScoredCBayesdocs rows: %s columns: %s\n", mxScoredCBayesdocs.numRows(), mxScoredCBayesdocs.numCols())

    printf("mxScoredTWCNBdocs: %s\n", mxScoredTWCNBdocs)
    printf("mxScoredCBayesdocs: %s \n", mxScoredCBayesdocs)


    // compare both scored document matrices:
    (mxScoredTWCNBdocs + mxScoredCBayesdocs).norm should be < 1e-6



  }

}

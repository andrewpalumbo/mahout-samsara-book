package myMahoutApp

import java.io.{FileWriter, BufferedWriter}

import BFGS.{MVFuncGrad, MVFunc}
import org.apache.log4j.Level
import org.apache.mahout.logging._
import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import drm._
import RLikeDrmOps._
import RLikeOps._
import scala.util.Random
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite
import org.scalatest.{Matchers, FunSuite}
import scala.collection.JavaConversions._

/**
 * @author dmitriy
 */
class MyAppSuite extends FunSuite with DistributedSparkSuite with Matchers {

  private final implicit val log = getLog(classOf[MyAppSuite])


  /** Simulate regression data. Note, data dim = beta.length - 1 due to bias. */

  def simData(beta: Vector, m: Int, noiseSigma: Double = 0.04) = {
    val n = beta.length
    val rnd = new Random(1245)
    val mxData =
      Matrices.symmetricUniformView(m, n, 1234) cloned

    // Bias always 1
    mxData(::, 0) = 1
    val y = mxData %*% beta

    // Perturb y with a little noise for
    // things to be not so perfect.
    y := { v ⇒ v + noiseSigma * rnd.nextGaussian() }

    // Return simulated X and y.
    mxData(::, 1 until n) → y
  }

  def dumpToCSV(mxX: Matrix, y: Vector, fileName: String): Unit = {
    val w = new BufferedWriter(new FileWriter(fileName, false))
    try {

      var line = new StringBuffer()
      line.append("y,")
      line.append((0 until mxX.ncol).map(i ⇒ "X" + i).mkString(","))
      w.write(line.toString)
      w.newLine()

      for ((xrow, i) ← mxX.zipWithIndex) {
        line = new StringBuffer()
        line.append(y(i))
        line.append(",")
        line.append((0 until xrow.length).map(i ⇒ xrow(i)).mkString(","))
        w.write(line.toString)
        w.newLine()
      }


    } finally {
      w.close()
    }

  }

  test("ols") {

    import LinearRegression._

    setLogLevel(Level.TRACE)
    // Simulated beta.
    val betaSim = dvec(3, 25, 10, -4)
    // Simulated data with little noise added.
    val (mxX, y) = simData(betaSim, 250)

    // Run distributed ridge
    val drmX = drmParallelize(mxX, numPartitions = 2)
    val fittedBeta = dridge(drmX, y, 0)
    trace(s"beta = $fittedBeta.")
    (betaSim - fittedBeta).norm(1) should be < 1e-1
  }

  test("ols-coeff-tests") {
    import LinearRegression._

    // Simulated data
    val betaSim = dvec(-4, 3, 25, 12)
    val (mxX, y) = simData(betaSim, 250, noiseSigma = 10.0)

    // Distributed X
    val drmX = drmParallelize(mxX, numPartitions = 2)
    val fittedBeta = dridge(drmX, y, 0)

    trace(s"beta = $fittedBeta.")

    // Coefficient t-tests
    val (betaSE, betaT, betaPVal) = testBeta(drmX, y, fittedBeta)

    println("Beta tests:\n  #        beta          SE      t-stat     p-value")
    for (i ← 0 until fittedBeta.length) {
      println(f"${i}%3d${fittedBeta(i)}%12.4f${betaSE(i)}%12.4f${betaT(i)}%12.4f${betaPVal(i)}%12.4f")
    }

    // dump output for verification in R
    dumpToCSV(mxX, y, "ols-coeff-tests.csv")
  }

  test("bfgs") {
    setLogLevel(Level.TRACE)
    getLog(BFGS.getClass).setLevel(Level.TRACE)
    // Simple parabaloid with minimum at = (3, 5)
    val xminControl = dvec(3, 5)
    val mxQ = diagv((2, .5))
    val f: MVFunc = x ⇒ {
      val xp = x - xminControl
      xp dot (mxQ %*% xp) - 3.5
    }
    val gradF: MVFuncGrad = x ⇒ {
      2 *=: (mxQ.diagv *=: (x - xminControl))
    }

    // Where to start the search.
    val x0 = dvec(45, -32)
    val xargmin = BFGS.bfgs(f, gradF, x0, 40, BFGS.newtonStep)
    trace(s"xMin found:$xargmin")
    (xargmin - xminControl).norm(1) should be < 1e-7
  }

  test("Bahmani") {

    setLogLevel(Level.TRACE)
    val n = 5
    val k = 4

    // Number of points per center
    val km = 50

    // Simulate data.
    val mxCenters = Matrices.symmetricUniformView(k, n, 12345) * 30
    trace(s"centers=$mxCenters")

    val mxCenterPoints = Matrices.gaussianView(km, n, 12345)
    val mxData = dense(mxCenters.flatMap(c ⇒ mxCenterPoints.map(x ⇒ c + x)))
    val drmData = drmParallelize(mxData, numPartitions = 2)

    val (mxSketch, drmY) = BahmaniSketch.dSample(drmData, sketchSize = 30, iterations = 5)
    val weights = BahmaniSketch.computePointWeights(drmY, nC = mxSketch.nrow)

    debug(s"Obtained ${mxSketch.nrow} samples. Weights:\n$weights")
    debug(s"sum of weights:${weights.sum}.")
  }

  test("Untangling a1.t") {

    setLogLevel(Level.TRACE)

    val mxA = dense((1, 2, 3), (3, 2, -1), (5, 6, 7), (8, 6, 8))

    val drmA = drmParallelize(mxA, 2)
    val a: Vector = (3, 6, 2, 7)

    // Broadcast the vector \bm{a}
    val aBcast = drmBroadcast(a) // compute \mathbf{C}\leftarrow\mathbf{A}-\bm{a}\mathbf{1}^{\top}.

    val drmC = drmA.mapBlock() { case (keys, block) ⇒

      val a: Vector = aBcast
      block ::= { (r, _, v) ⇒ v - a(keys(r)) }
      keys → block
    }

    trace(s"C=${drmC.collect}")
  }

  test("Untangling A11.t") {

    setLogLevel(Level.TRACE)

    val mxA = dense((1, 2, 3), (3, 2, -1), (5, 6, 7), (8, 6, 8))

    val drmA = drmParallelize(mxA, 2)

    val drmC = drmA.mapBlock() { case (keys, block) ⇒

      val partialRS = block.rowSums()
      block ::= { (r, _, v) ⇒ v - partialRS(r) }
      keys → block
    }

    trace(s"C=${drmC.collect}")
  }
}

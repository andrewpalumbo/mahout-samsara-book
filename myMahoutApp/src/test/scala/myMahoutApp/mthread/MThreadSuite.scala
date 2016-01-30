package myMahoutApp.mthread

import org.apache.log4j.{BasicConfigurator, Level}
import org.scalatest.{Matchers, FunSuite}
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._

import org.apache.mahout.logging._

/**
 * @author dmitriy
 */
class MThreadSuite extends FunSuite with Matchers {

  BasicConfigurator.configure()
  private[mthread] final implicit val log = getLog(classOf[MThreadSuite])
  setLogLevel(Level.DEBUG)

  test("mthread-mmul") {

    val m = 5000
    val n = 300
    val s = 350

    val mxA = Matrices.symmetricUniformView(m, s, 1234).cloned
    val mxB = Matrices.symmetricUniformView(s, n, 1323).cloned

    // Just to warm up
    mxA %*% mxB
    MMul.mmulParA(mxA, mxB)

    val ntimes = 30

    val controlMsStart = System.currentTimeMillis()
    val mxControlC = mxA %*% mxB
    for (i ← 1 until ntimes) mxA %*% mxB
    val controlMs = System.currentTimeMillis() - controlMsStart

    val cMsStart = System.currentTimeMillis()
    val mxC = MMul.mmulParA(mxA, mxB)
    for (i ← 1 until ntimes) MMul.mmulParA(mxA, mxB)
    val cMs = System.currentTimeMillis() - cMsStart

    debug(f"control: ${controlMs/ntimes.toDouble}%.2f ms.")
    debug(f"mthread: ${cMs/ntimes.toDouble}%.2f ms.")

    trace(s"mxControlC:$mxControlC")
    trace(s"mxC:$mxC")

    (mxControlC - mxC).norm should be < 1e-5
  }

}

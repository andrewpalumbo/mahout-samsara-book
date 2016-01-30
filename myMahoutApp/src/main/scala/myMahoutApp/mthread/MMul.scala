package myMahoutApp.mthread

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._

import org.apache.mahout.logging._


/**
 * @author dmitriy
 */
object MMul {

  private[mthread] final implicit val log = getLog(MMul.getClass)

  import scala.concurrent.ExecutionContext.Implicits.global

  def createSplits(nrow: Int, nsplits: Int): TraversableOnce[Range] = {

    val step = nrow / nsplits
    val slack = nrow % nsplits

    // Ranges.
    // `slack` ranges `step+1 wide` each
    ((0 until slack * (step + 1) by (step + 1)) ++

        // And the remainder is `step` wide
        (slack * (step + 1) to nrow by step))
        .sliding(2).map(s ⇒ s(0) until s(1))
  }

  /** Parallelize over vertical blocks of A operand */
  def mmulParA(mxA: Matrix, mxB: Matrix): Matrix = {
    val result = if (mxA.getFlavor.isDense) mxA.like(mxA.nrow, mxB.ncol)
    else if (mxB.getFlavor.isDense) mxB.like(mxA.nrow, mxB.ncol)
    else mxA.like(mxA.nrow, mxB.ncol)

    val nsplits = Runtime.getRuntime.availableProcessors() min mxA.nrow
    val ranges = createSplits(mxA.nrow, nsplits)

    val blocks = ranges.map { r ⇒
      Future {
        r → (mxA(r, ::) %*% mxB)
      }
    }

    Await.result(Future.fold(blocks)(result) { case (result, (r, block)) ⇒
      result(r, ::) := block
      result
    }, Duration.Inf)

  }

}

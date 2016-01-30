package myMahoutApp

import org.apache.mahout.logging._
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import drm._
import RLikeDrmOps._

import collection._
import JavaConversions._
import scala.reflect.ClassTag
import scala.util.Random

/**
 * @author dmitriy
 */
object BahmaniSketch {

  private final implicit val log = getLog(BahmaniSketch.getClass)

  private def drmAtoY[K: ClassTag](drmA: DrmLike[K]): DrmLike[K] =
    drmA.mapBlock(drmA.ncol + 2) { case (keys, block) ⇒
      val yBlock = new DenseMatrix(block.nrow, 2) cbind block
      keys → yBlock
    }

  /**
   * Perform Bahmani sketch sampling without re-weighing
   * @param drmA input
   * @param sketchSize sketch size required
   * @param iterations number of iterations
   * @param seed seed
   * @tparam K row key type
   * @return Sketch as in-core matrix → Y-matrix (cluster idcs+sq distances added)
   */
  def dSample[K: ClassTag](drmA: DrmLike[K],
                           sketchSize: Int,
                           iterations: Int,
                           seed: Int = Random.nextInt()): (Matrix, DrmLike[K]) = {

    implicit val ctx = drmA.context

    val l = sketchSize * 2 / iterations + 1 max 1
    val n = drmA.ncol
    var drmY = drmAtoY(drmA).checkpoint()

    var mxC = drmSampleKRows(drmY, 1)
    mxC = mxC(::, 2 until n + 2)

    drmY = updateY(drmY, mxC, cStart = 0).checkpoint()

    for (iter ← 0 until (iterations - 1 max 0)) {

      val innerN = n
      val innerL = l

      // Compute phi
      val phi = drmY(::, 1 to 1).colSums()(0)
      val subseed = seed + iter

      // Resample C'
      val mxCPrime = drmY.allreduceBlock(

      { case (keys, yblock) ⇒

        if (yblock.nrow > 0) {
          val vDSq = yblock(::, 1)
          val ablock = yblock(::, 2 until innerN + 2)
          val rnd = new Random(subseed * keys(0).hashCode())

          // Perform draws
          val selected = new mutable.ArrayBuffer[Int](200)
          for (r ← 0 until ablock.nrow) {
            val p = innerL * vDSq(r) / phi
            if (rnd.nextDouble <= p) selected += r
          }
          val cPrimeBlock: Matrix = yblock.like(selected.size, innerN)
          selected.zipWithIndex.foreach { case (idx, i) ⇒
            cPrimeBlock(i, ::) = ablock(idx, ::)
          }
          cPrimeBlock
        } else {

          // Empty
          new DenseMatrix(0, innerN): Matrix
        }
      },

      // Reduce
      _ rbind _
      )

      drmY = updateY(drmY, mxCPrime, mxC.nrow).checkpoint()
      mxC = mxC rbind mxCPrime
    }

    mxC → drmY
  }

  private def updateY[K: ClassTag](drmY: DrmLike[K], mxC: Matrix,
                                   cStart: Int): DrmLike[K] = {
    implicit val ctx = drmY.context
    val mxCBcast = drmBroadcast(mxC)

    drmY.mapBlock() { case (keys, yblock) ⇒
      updateYBlock(yblock, mxCBcast, cStart)
      keys → yblock
    }
  }

  private def updateYBlock(mxY: Matrix,
                           mxC: Matrix,
                           cStart: Int): Unit = {
    val n = mxY.ncol - 2
    val k = mxC.nrow
    val mxA = mxY(::, 2 until n + 2)
    val vLabels = mxY(::, 0)
    val vSqD = mxY(::, 1)

    // We assume that mxC.nrow is small, << mxA.nrow.
    val mxCDsq = dist(mxC) /= 4

    for (row ← mxA) {
      var minC = 0
      var minCDsq = (row - mxC(minC, ::)) ^= 2 sum

      for (c ← 1 until k) {

        // Check lemma 1 per C. Elkan
        if (mxCDsq(minC, c) <= minCDsq &&

            // Triang. inequality
            sqr(mxC(c, ::).norm(2) - row.norm(2)) <= minCDsq) {

          val dSq = (row - mxC(c, ::)) ^= 2 sum

          if (dSq < minCDsq) {
            minC = c
            minCDsq = dSq
          }
        }
      }

      if (cStart == 0 || minCDsq < vSqD(row.index)) {
        vLabels(row.index) = minC + cStart
        vSqD(row.index) = minCDsq
      }
    }
  }

  /**
   *
   * @param drmY  the Y matrix as produced by [[dSample()]]
   * @param nC number of samples produces by [[dSample()]]
   * @tparam K
   * @return sampled point weights w.r.t. the rest of the data.
   */
  def computePointWeights[K: ClassTag](drmY: DrmLike[K], nC: Int): Vector = {

    // Single-row weight matrix with counts
    val vWeights = drmY.allreduceBlock({ case (keys, yblock) ⇒

        val wvec = new DenseVector(nC)
        val labels = yblock(::, 0).all
            .foreach(sgm ⇒ wvec(sgm.toInt) += 1)

        dense(wvec)

    })(0, ::)

    val s = vWeights.sum
    vWeights /= s
  }

  def sqr(x: Double) = x * x
}

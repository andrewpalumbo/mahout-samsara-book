package myMahoutApp

import org.apache.mahout.math._
import scalabindings._
import drm._

import RLikeDrmOps._
import RLikeOps._

import collection.JavaConversions._
import math._
import scala.collection.mutable.ArrayBuffer

/**
 * @author dmitriy
 */
object TWCNB {

  /**
   * TWCNB training algorithm.
   *
   * @param drmTfIdf: TF-IDF matrix rows correspond to a document; keys being class label ordinals
   *                c: c=0,1,...l-1.
   *
   * @return matrix \Theta' (transposed, term x class) of the trained parameters. This is essentially
   *         TWNCB's model.
   */
  def twcnbTrain(drmTfIdf: DrmLike[Int]): DrmLike[Int] = {

    implicit val dc = drmTfIdf.context

    val nTerm = drmTfIdf.ncol

    // alpha_i -- parameter of a Dirichlet prior.
    val alpha_i = 1.0
    val alpha = alpha_i * drmTfIdf.ncol

    /*
     * This is done in seq2sparse currently so we can remove for now.
     */
    // Normalize to matrix D (step 3)
    // val drmD: DrmLike[Int] = drmTfIdf.mapBlock() { case (keys, block) ⇒
    //  for (row ← block) row /= row.norm(2)
    //  keys → block
    // }


     // Aggregate N. Actually, we force transposition that possesses the aggregate properties.
     // val drmNt = drmD.t.checkpoint()
    val drmNt = drmTfIdf.t.checkpoint()

    // Remove empty rows from drmNt
    // the result will be a (class x term) in-core matrix
    val mxNNonEmpty = drmNt.t.allreduceBlock( {
      case(keys, block) =>
        val nonEmptyRows = ArrayBuffer[(Vector, Int)]()

        // Keep only the aggregated rows
        for(i <- 0 until block.nrow) {
          if (block(i, ::).getNumNondefaultElements() > 0){
            // get the class label (the key) for the row
            nonEmptyRows += ((block(i, ::), keys(i)))
          }
        }

        // Now that we know the number of aggregated rows in this partition
        // create a new matrix for those rows. and their class labels.
        val blockB = new SparseRowMatrix(nonEmptyRows.size, block.ncol + 1)
        for(i <- 0 until nonEmptyRows.size){
          blockB(i, ::) := nonEmptyRows(i)._1

          //append the class label to the last column
          blockB(i, block.ncol) = nonEmptyRows(i)._2
        }

        blockB: Matrix

    },

      // Reduce by stacking aggregated rows
      _ rbind _

    )


    // Parallelize the aggregated mxNNonEmpty and set the keys
    // to the correct class label. the result will be
    // N-transpose, a (class x term) drm matrix
    val drmNtNonEmpty = drmParallelize(mxNNonEmpty)
      .mapBlock(ncol = mxNNonEmpty.ncol - 1){
        case (keys, block) =>

          // strip the classes off the matrix
          val classes = block(::, block.ncol - 1)

          for(i <- 0 until keys.size){
            // set the keys as the class for that row
            keys.update(i, classes(i).toInt)
         }
        (keys , block(::, 0 until block.ncol - 1))
      }
        .t

        .checkpoint()



    // Complement N
    val drmNc = drmNtNonEmpty.mapBlock() { case (keys, block) ⇒
      val termTotals = block.rowSums()
      block ::= { (r, _, v) ⇒ termTotals(r) - v }
      keys → block
    }
      .checkpoint()

    // Nc * 1 + alpha*1: this is denomitator for compuation of Theta.
    val ncSum = drmNc.colSums() += alpha
    val ncSumBcast = drmBroadcast(ncSum)

    // theta/W:
    val drmWtUnweighted = drmNc.mapBlock() { case (keys, block) ⇒
      // Pin to the task memory
      val ncSum: Vector = ncSumBcast

      // Compute (finish) the step 4 and 5 (theta and its log).
      (block += alpha_i) := { (_, c, v) ⇒ log(v / ncSum(c)) }

      keys → block
    }


    // Sum up the absolute value of each Term Weight
    // per term per class
    val cTotals = dabs(drmWtUnweighted).colSums()
    val cTotalsBcast = drmBroadcast(cTotals)

    // Make all weights per class to sum to 1:
    drmWtUnweighted.mapBlock() { case (keys, block) ⇒

            // Pin to the task memory
            val cTotals: Vector = cTotalsBcast

            // normalize
            block ::= { (_, c, v) ⇒ v / cTotals(c) }

            keys → block
        }
    }

}

package myMahoutApp.mthread

import org.apache.mahout.math.Matrix

/**
 * @author dmitriy
 */
trait BlockMatrix {

  def apply(i:Int,j:Int):Matrix
  val nrowBlocks:Int
  val ncolBlocks:Int
  val nrows:Int
  val ncols:Int

}

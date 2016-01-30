package myMahoutApp

import org.apache.mahout.logging._
import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._

/**
 * @author dmitriy
 */
object BFGS {

  private final implicit val log = getLog(BFGS.getClass)

  type MVFunc = Vector ⇒ Double
  type MVFuncGrad = Vector ⇒ Vector
  type LineSearchFunc = (MVFunc, MVFuncGrad, Vector, Vector) ⇒ Double

  def bfgs(f: MVFunc,
           grad: MVFuncGrad,
           x0: Vector,
           maxIterations: Int = 0,
           lineSearch: LineSearchFunc,
           epsilon: Double = 1e-7): Vector = {
    val d = x0.length
    var mxBInv = eye(d): Matrix
    var stop = false
    var k = 0

    var x = x0

    var gradk = grad(x)
    while (k < maxIterations && !stop) {
      val p = (mxBInv %*% gradk) := (-_)

      // Step length.

      val alpha = lineSearch(f, grad, x, p)

      // Step
      val s = alpha * p
      val x1 = x + s
      // Compute and cache \nabla f_{k+1}

      val gradk1 = grad(x1)

      // Check convergence.
      stop = gradk1.norm(2) < epsilon

      if (!stop) {
        // Update BInv.
        val y = gradk1 - gradk
        val rho = 1.0 / (y dot s)
        val mxT = -rho * s cross y
        mxT.diagv += 1.0
        mxBInv = mxT %*% mxBInv %*% mxT.t + (rho * s cross s)
      }
      // Next iteration:
      x = x1
      gradk = gradk1
      k += 1
    }
    require(stop,
      s"Convergence not reached in $k iterations.")
    trace(s"BFGS convergence reached after $k iterations.")
    x
  }

  def newtonStep(f: MVFunc, fgrad: MVFuncGrad, x: Vector, p: Vector) = 1.0
}

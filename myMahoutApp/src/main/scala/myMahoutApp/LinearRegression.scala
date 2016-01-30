package myMahoutApp

import org.apache.commons.math3.distribution.TDistribution
import org.apache.mahout.math._
import scalabindings._
import drm._
import RLikeDrmOps._
import RLikeOps._
import math._

/**
 * Linear regression and its regularized variations.
 */
object LinearRegression {

  /** Fittig the distributed ridge. */
  def dridge(drmX: DrmLike[Int], y: Vector, lambda: Double): Vector = {

    require(drmX.nrow == y.length,
      "Target and data set have different point count.")

    // Add bias term.
    val drmXB = (1 cbind drmX).checkpoint()

    // A = X'X + lambda*I
    val mxA: Matrix = drmXB.t %*% drmXB
    mxA.diagv += lambda

    // b = X'y
    val b = (drmXB.t %*% y).collect(::, 0)

    // Solve A*beta = b for beta.
    solve(mxA, b)
  }

  /** Sum of square resiguals */
  def ssr(drmX: DrmLike[Int], y: Vector, beta: Vector): Double = {

    val m = drmX.nrow
    val n = drmX.ncol + 1

    require(beta.size == n, "beta.size must be X.ncol + 1.")
    require(y.size == m, "y.size must be X.nrow.")

    // X*beta produces OLS estimators
    (((1 cbind drmX) %*% beta).collect(::, 0) - y) ^= 2 sum
  }

  /** Regression variance (sigma^2) estimator */
  def regVarEstimate(drmX: DrmLike[Int], y: Vector, beta: Vector): Double = {
    val m = drmX.nrow
    val n = drmX.ncol + 1

    // We shy away from an underdefined problem
    require(m > n, "Underdefined problem")
    ssr(drmX, y, beta) / (m - n)
  }

  def testBeta(drmX: DrmLike[Int], y: Vector, beta: Vector): (Vector, Vector, Vector) = {

    val m = drmX.nrow
    val n = drmX.ncol + 1

    require(beta.length == n, "beta.length must be X.ncol + 1.")
    require(y.length == m, "y.length must be X.nrow.")

    // We shy away from an underdefined problem
    require(m > n, "Underdefined problem")

    // Estimate regression variance.
    val drmBX = (1 cbind drmX).checkpoint()

    // Compute sum of square residuals.
    val ssr = ((drmBX %*% beta).collect(::, 0) - y) ^= 2 sum

    // Regression variance (sigma^2) estimator. DF = m - n
    val regVar = ssr / (m - n)

    // C := inv(X'X); compute main diagonal of C as c-vector
    val c = solve(drmBX.t %*% drmBX) diagv

    // Standard errors of all betas except intercept
    val seBeta = (c *= regVar) := sqrt _

    // t-statistics
    val tBeta = beta / seBeta

    // Standard t-distr for (n-p) degrees of freedom
    val tDistribution = new TDistribution(m - n)

    // p-values, 2-side test
    val pValBeta = tBeta.cloned := { t ⇒
      2 * (1 - tDistribution.cumulativeProbability(abs(t)))
    }

    (seBeta, tBeta, pValBeta)
  }

}

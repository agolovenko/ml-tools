package regression

import breeze.linalg.DenseVector
import breeze.optimize.{DiffFunction, _}

import scala.collection.mutable

/**
  * Created by ashot.golovenko on 14/11/2016.
  */
abstract class LBFGSRegressor(val data: Iterable[(DenseVector[Double], Double)],
                              val featureCount: Int,
                              val l2Reg: Double,
                              val maxIterations: Int) extends Regressor {

  protected val regVector: DenseVector[Double] = DenseVector.fill[Double](featureCount, l2Reg)
  //no regularization for bias
  regVector(0) = 0.0d

  private def costFunctionAndGradient(weights: DenseVector[Double]): (Double, DenseVector[Double]) = {

    var totalCost = 0.0d
    val grad = DenseVector.zeros[Double](featureCount)

    val totalExamples = data.iterator.map { case (x, y) =>
      val h: Double = predict(x, weights)

      totalCost += costOfPrediction(h, y)

      //grad :+= x.*(h - y) - in place impl
      val delta = h - y
      for (i <- 0 until grad.length) {
        grad(i) += x(i) * delta
      }

      1.0d
    }.sum

    val regCost = (regVector dot (weights :^ 2.0d)) / 2.0d
    val regGrad = regVector :* weights

    ((totalCost + regCost) / totalExamples, (grad :+ regGrad) :/ totalExamples)
  }

  override protected def learn: (DenseVector[Double], Seq[Double]) = {
    val f = new DiffFunction[DenseVector[Double]] {
      val iterations = mutable.Buffer[Double]()

      def calculate(weights: DenseVector[Double]) = {
        val costAndGrad = costFunctionAndGradient(weights)

        iterations += costAndGrad._1
        costAndGrad
      }
    }

    (minimize(f, DenseVector.zeros[Double](featureCount), MaxIterations(maxIterations)), f.iterations)
  }
}

class LinearLBFGSRegressor(data: Iterable[(DenseVector[Double], Double)],
                           featureCount: Int,
                           l2Reg: Double,
                           maxIterations: Int)
  extends LBFGSRegressor(data, featureCount, l2Reg, maxIterations) with LinearRegressor {

}

class LogisticLBFGSRegressor(data: Iterable[(DenseVector[Double], Double)],
                             featureCount: Int,
                             l2Reg: Double,
                             maxIterations: Int)
  extends LBFGSRegressor(data, featureCount, l2Reg, maxIterations) with LogisticRegressor {

}

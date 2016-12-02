package regression

import breeze.linalg.DenseVector
import breeze.optimize.{DiffFunction, _}

import scala.collection.mutable

/**
  * Created by ashot.golovenko on 14/11/2016.
  */
abstract class BatchRegressor(val data: Iterable[(DenseVector[Double], Double)],
                              val featureCount: Int,
                              val l2Reg: Double,
                              val maxIterations: Int) extends Regressor {

  protected val regVector: DenseVector[Double] = DenseVector.fill[Double](featureCount, l2Reg)
  //no regularization for bias
  regVector(0) = 0.0d

  protected def costFunctionAndGradient(weights: DenseVector[Double]): (Double, DenseVector[Double]) = {
    var totalCost = 0.0d
    val grad = DenseVector.zeros[Double](featureCount)

    val totalExamples = data.map { case (x, y) =>
      val h = predict(x, weights)

      totalCost += costOfPrediction(h, y)

      //      grad :+= (x :* (h - y)) // - in place impl
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
}

abstract class LBFGSRegressor(data: Iterable[(DenseVector[Double], Double)],
                              featureCount: Int,
                              l2Reg: Double,
                              maxIterations: Int) extends BatchRegressor(data, featureCount, l2Reg, maxIterations) {
  override protected def learn: (DenseVector[Double], Seq[Double]) = {
    val f = new DiffFunction[DenseVector[Double]] {
      val iterations = mutable.Buffer[Double]()

      def calculate(weights: DenseVector[Double]) = {
        val costAndGrad = costFunctionAndGradient(weights)

        iterations += costAndGrad._1
        costAndGrad
      }
    }

    (minimize(f, DenseVector.zeros[Double](featureCount), MaxIterations(maxIterations)), f.iterations.toList)
  }
}

abstract class GDRegressor(data: Iterable[(DenseVector[Double], Double)],
                           featureCount: Int,
                           l2Reg: Double,
                           maxIterations: Int,
                           val learningRate: Double) extends BatchRegressor(data, featureCount, l2Reg, maxIterations) {
  override protected def learn: (DenseVector[Double], Seq[Double]) = {
    val weights = DenseVector.zeros[Double](featureCount)

    val iterations = (1 to maxIterations).map { i =>
      val costAndGrad = costFunctionAndGradient(weights)

      weights :-= (costAndGrad._2 :* learningRate)

      costAndGrad._1
    }

    (weights, iterations)
  }
}

class LinearGDRegressor(data: Iterable[(DenseVector[Double], Double)],
                        featureCount: Int,
                        l2Reg: Double,
                        maxIterations: Int,
                        learningRate: Double)
  extends GDRegressor(data, featureCount, l2Reg, maxIterations, learningRate) with LinearLike {

}

class LinearLBFGSRegressor(data: Iterable[(DenseVector[Double], Double)],
                           featureCount: Int,
                           l2Reg: Double,
                           maxIterations: Int)
  extends LBFGSRegressor(data, featureCount, l2Reg, maxIterations) with LinearLike {

}

class LogisticLBFGSRegressor(data: Iterable[(DenseVector[Double], Double)],
                             featureCount: Int,
                             l2Reg: Double,
                             maxIterations: Int)
  extends LBFGSRegressor(data, featureCount, l2Reg, maxIterations) with LogisticLike {

}

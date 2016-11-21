package util

import breeze.generic.UFunc
import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics._
import breeze.optimize.{DiffFunction, _}

import scala.collection.mutable

/**
  * Created by ashot.golovenko on 14/11/2016.
  */
abstract class LBFGSRegressor[V](val data: V,
                                 val featureCount: Int,
                                 val l2Reg: Double,
                                 val maxIterations: Int) extends LogisticRegressor {


  protected val regVect = DenseVector.fill[Double](featureCount, l2Reg)
  //no regularization for bias
  regVect(0) = 0.0d

  def costFunctionAndGradient(weights: DenseVector[Double]): (Double, DenseVector[Double])

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

class IterativeLBFGSRegressor(data: () => Iterator[(DenseVector[Double], Double)],
                              featureCount: Int,
                              l2Reg: Double,
                              maxIterations: Int)
  extends LBFGSRegressor[() => Iterator[(DenseVector[Double], Double)]](data, featureCount, l2Reg, maxIterations) {

  def cost(h: Double, y: Double): Double = {
    -y * log(h) - (1.0 - y) * log(1.0 - h)
  }

  // in place implementation of grad :+= x .* (h - y)
  object gradUpdate extends UFunc {
    implicit object implV extends InPlaceImpl3[DenseVector[Double], DenseVector[Double], Double] {
      override def apply(grad: DenseVector[Double], x: DenseVector[Double], delta: Double): Unit =
        for (i <- 0 until grad.length)
          grad(i) += +x(i) * delta
    }
  }

  override def costFunctionAndGradient(weights: DenseVector[Double]): (Double, DenseVector[Double]) = {
    var totalCost = 0.0d
    val grad = DenseVector.zeros[Double](featureCount)

    val totalExamples = data().map { case (x, y) =>
      val h: Double = sigmoid(x dot weights)

      totalCost += cost(h, y)

      //grad :+= x.*(h - y)
      gradUpdate.inPlace(grad, x, h - y)

      1.0d
    }.sum

    val regCost = (regVect dot (weights :^ 2.0d)) / 2.0d
    val regGrad = regVect :* weights

    ((totalCost + regCost) / totalExamples, (grad :+ regGrad) / totalExamples)
  }
}

class MatrixLBFGSRegressor(data: (DenseMatrix[Double], DenseVector[Double]),
                           featureCount: Int,
                           l2Reg: Double,
                           maxIterations: Int)
  extends LBFGSRegressor[(DenseMatrix[Double], DenseVector[Double])](data, featureCount, l2Reg, maxIterations) {

  def cost(h: DenseVector[Double], y: DenseVector[Double]): Double = {
    sum(-y :* log(h) :+ (y :- 1.0) :* log(1.0 - h))
  }

  override def costFunctionAndGradient(weights: DenseVector[Double]): (Double, DenseVector[Double]) = {
    val totalExamples = data._1.cols.toDouble

    val h: DenseVector[Double] = sigmoid(data._1 * weights)
    val totalCost = cost(h, data._2)
    val grad = data._1.t * (h :- data._2)

    val regCost = (regVect dot (weights :^ 2.0d)) / 2.0d
    val regGrad = regVect :* weights

    ((totalCost + regCost) / totalExamples, (grad :+ regGrad) / totalExamples)
  }
}

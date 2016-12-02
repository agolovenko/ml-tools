package regression

import breeze.linalg.DenseVector
import breeze.numerics.{log, sigmoid}

/**
  * Created by ashot.golovenko on 18/11/2016.
  */
trait Regressor {
  private lazy val weightsWithIterations = learn

  def weights: DenseVector[Double] = weightsWithIterations._1

  def iterations: Seq[Double] = weightsWithIterations._2

  protected def predict(x: DenseVector[Double], weights: DenseVector[Double]): Double

  protected def costOfPrediction(h: Double, y: Double): Double

  protected def learn: (DenseVector[Double], Seq[Double])

  def predict(x: DenseVector[Double]): Double = {
    predict(x, weights)
  }

  def cost(x: DenseVector[Double], y: Double): Double = {
    costOfPrediction(predict(x), y)
  }

  def meanCost(data: Iterable[(DenseVector[Double], Double)]): Double = {
    var cost = 0.0d
    var total = 0L
    data.foreach { case (x, y) =>
      total += 1
      cost += costOfPrediction(predict(x), y)
    }
    cost / total
  }
}

trait LinearLike {
  protected def predict(x: DenseVector[Double], weights: DenseVector[Double]): Double = {
    x dot weights
  }

  protected def costOfPrediction(h: Double, y: Double): Double = {
    val error = h - y
    error * error / 2
  }
}

trait LogisticLike {
  protected def predict(x: DenseVector[Double], weights: DenseVector[Double]): Double = {
    sigmoid(x dot weights)
  }

  protected def costOfPrediction(h: Double, y: Double): Double = {
    -y * log(h) - (1.0 - y) * log(1.0 - h)
  }
}


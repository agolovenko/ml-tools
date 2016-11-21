package util

import breeze.linalg.DenseVector
import breeze.numerics.sigmoid
import breeze.stats.distributions.Gaussian

/**
  * Created by ashot.golovenko on 15/11/2016.
  */
class SGDRegressor(val data: () => Iterator[(DenseVector[Double], Double)],
                   val featureCount: Int,
                   val learningRate: Double,
                   val maxIterations: Int) extends LogisticRegressor {

  override protected def learn: (DenseVector[Double], Seq[Double]) = {
    val normal = Gaussian(0, .01)
    val weights = normal.samplesVector(featureCount)
    weights(0) = 0.0

    val iterations = (1 to maxIterations).map { i =>
      val cost = data().map {case (x, y) =>
        val p = sigmoid(x dot weights)
        val error = y - p

        val gradient = x.map(error * _)
        weights += gradient :* learningRate

        math.abs(error)
      }.sum

      println(s"SGDRegressor: Iteration #$i, Cost: $cost")

      cost
    }

    (weights, iterations)
  }
}

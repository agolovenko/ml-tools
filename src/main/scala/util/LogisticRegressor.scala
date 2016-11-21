package util

import breeze.linalg.DenseVector
import breeze.numerics._

/**
  * Created by ashot.golovenko on 18/11/2016.
  */
trait LogisticRegressor {
  private lazy val weightsWithIterations = learn

  def weights = weightsWithIterations._1

  def iterations = weightsWithIterations._2

  def predict(features: DenseVector[Double]): Double = {
    val z: Double = features dot weightsWithIterations._1
    sigmoid(z)
  }

  protected def learn: (DenseVector[Double], Seq[Double])
}


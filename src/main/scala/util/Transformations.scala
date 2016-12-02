package util

import breeze.linalg.DenseVector

/**
  * Created by ashot.golovenko on 15/11/2016.
  */
object Transformations {
  def minMax(med: DenseVector[Double], halfRange: DenseVector[Double])(features: DenseVector[Double]): DenseVector[Double] = {
    (features :- med) / halfRange
  }

  def zScore(means: DenseVector[Double], stddevs: DenseVector[Double])(features: DenseVector[Double]): DenseVector[Double] = {
    (features :- means) :/ stddevs
  }

  def filter(indices: Set[Int])(features: DenseVector[Double]): DenseVector[Double] = {
    val result = features.keysIterator.collect {
      case i if indices.contains(i) => features(i)
    }.toArray

    DenseVector(result)
  }

  def addPolynomialFeatures(mask: DenseVector[Boolean], maxPower: Int)(features: DenseVector[Double]): DenseVector[Double] = {
    val f = features(mask).toArray
    val len = (maxPower - 1) * f.length

    val polyFeatures = new Array[Double](len)

    var j = 0
    for (p <- 2 to maxPower) {
      f.indices.foreach { i =>
        polyFeatures(j) = Math.pow(f(i), p)
        j += 1
      }
    }

    DenseVector.vertcat(features, DenseVector(polyFeatures))
  }

}

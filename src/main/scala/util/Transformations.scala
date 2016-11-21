package util

import breeze.linalg.DenseVector

/**
  * Created by ashot.golovenko on 15/11/2016.
  */
object Transformations {
  def minMax(mins: DenseVector[Double], maxs: DenseVector[Double])(features: DenseVector[Double]): DenseVector[Double] = {
    (features :- ((maxs :+ mins) :/ 2.0)) / (maxs :- mins)
  }

  def zScore(means: DenseVector[Double], stddevs: DenseVector[Double])(features: DenseVector[Double]): DenseVector[Double] = {
    (features :- means) :/ stddevs
  }
}

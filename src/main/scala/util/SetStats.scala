package util

import breeze.linalg.DenseVector

class SetStats private (val data: Iterable[DenseVector[Double]], val featureCount: Int) {
  private val stats = learnStats()

  val count: Long = stats._1

  val mean: DenseVector[Double] = stats._2

  val stddev: DenseVector[Double] = stats._3

  val min: DenseVector[Double] = stats._4

  val max: DenseVector[Double] = stats._5

  val med: DenseVector[Double] = {
    ((max :+ min) :/ 2.0).mapPairs{(i, v) => if (max(i) == min(i)) 0.0d else v}
  }

  val halfRange: DenseVector[Double] = {
    ((max :- min) :/ 2.0).map(v => if (v == 0) 1.0 else v)
  }

  val meanForZScore: DenseVector[Double] = {
    mean.mapPairs { (i, v) => if (stddev(i) == 0.0d) 0.0d else v }
  }

  val stddevForZScore: DenseVector[Double] = {
    stddev.map(v => if (v == 0) 1.0 else v)
  }

  private def learnStats(): (Long, DenseVector[Double], DenseVector[Double], DenseVector[Double], DenseVector[Double]) = {
    println("Learning stats...")
    val min = DenseVector.fill[Double](featureCount)(Double.MaxValue)
    val max = DenseVector.fill[Double](featureCount)(Double.MinValue)
    val mean = DenseVector.fill[Double](featureCount)(0.0)
    val stddev = DenseVector.fill[Double](featureCount)(0.0)

    var count = 0L

    data.foreach { v =>
      count += 1
      if (count % 1000000 == 0) println(s"${count/1000000}M rows processed...")
      for (i <- 0 until featureCount) {

        //Welford algorythm
        val d = v(i) - mean(i)
        mean(i) += d / count
        stddev(i) += d * (v(i) - mean(i))

        min(i) = Math.min(min(i), v(i))
        max(i) = Math.max(max(i), v(i))
      }
    }

    println("Finished...")

    (
      count,
      mean,
      if (count > 1) (stddev :/ (count - 1).toDouble) :^ 0.5d else DenseVector.fill[Double](featureCount)(0.0),
      min,
      max
    )
  }
}

object SetStats {
  def apply(data: Iterable[DenseVector[Double]], featureCount: Int): SetStats = new SetStats(data, featureCount)
}

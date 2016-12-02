package examples.mlstanford.week2

import breeze.linalg.DenseVector
import regression.{LinearGDRegressor, LinearLBFGSRegressor}
import util.SetStats
import util.Transformations.zScore

import scala.io.Source

object Task1 {
  def parse(line: String): (DenseVector[Double], Double) = {
    val values = line.trim.split(",")

    val features = DenseVector(1.0d +: values.slice(0, values.length - 1).map(_.toDouble))

    (features, values.last.toDouble)
  }

  def main(args: Array[String]): Unit = {
    val f = getClass.getResource("ex1data1.txt").getPath

    val data = Source.fromFile(f).getLines().map(parse).toSeq

    val fCount = data.head._1.length
    val stats = SetStats(data.map(_._1), fCount)

    //Try different normalisations to see how many iterations occured in LBFGSRegressor
//    def norm: (DenseVector[Double]) => DenseVector[Double] = zScore(stats.meanForZScore, stats.stddevForZScore)
    def norm: (DenseVector[Double]) => DenseVector[Double] = identity[DenseVector[Double]]

    val normalized = data.map { case (x, y) => (norm(x), y)}

    //Get the default cost
    val dummy = new LinearGDRegressor(data, fCount, 0.0d, 1, 0.01)
    println("Default avg cost: " + dummy.iterations.head)

    val lr = new LinearGDRegressor(normalized, fCount, 0.0d, 1500, 0.01)

    //actually LBFGSRegressor seems to find another local minimum.
//    val lr = new LinearLBFGSRegressor(normalized, fCount, 0.0d, 50)

    println("Finished in: " + lr.iterations.size + " iterations. History: " + lr.iterations)
    println("Final weights: " + lr.weights)

    println("Profit for foodtruck in 35K town: " + lr.predict(norm(DenseVector(1.0d, 3.5d))))
    println("Profit for foodtruck in 70K town: " + lr.predict(norm(DenseVector(1.0d, 7.0d))))
  }
}

package examples.mlstanford.week2

import breeze.linalg.DenseVector
import regression.{LinearGDRegressor, LinearLBFGSRegressor}
import util.SetStats
import util.Transformations.{minMax, zScore}

import scala.io.Source

/**
  * Created by ashot.golovenko on 15/11/2016.
  */
object Task2 {
  def parse(line: String): (DenseVector[Double], Double) = {
    val values = line.trim.split(",")

    val features = DenseVector(1.0d +: values.slice(0, values.length - 1).map(_.toDouble))

    (features, values.last.toDouble)
  }

  def main(args: Array[String]): Unit = {
    val f = getClass.getResource("ex1data2.txt").getPath

    val data = Source.fromFile(f).getLines().map(parse).toSeq

    val fCount = data.head._1.length
    val stats = SetStats(data.map(_._1), fCount)

    def norm: (DenseVector[Double]) => DenseVector[Double] = zScore(stats.meanForZScore, stats.stddevForZScore)
//    def norm: (DenseVector[Double]) => DenseVector[Double] = identity[DenseVector[Double]]

    val normalized = data.map { case (x, y) => (norm(x), y)}

    //Get the default cost
//    val dummy = new LinearGDRegressor(data, fCount, 0.0d, 1, 0.01)
//    println("Default avg cost: " + dummy.iterations.head)

    val lr = new LinearGDRegressor(normalized, fCount, 0.0d, 1500, 0.01)
//    val lr = new LinearLBFGSRegressor(normalized, fCount, 0.0d, 50)

    println("Finished in: " + lr.iterations.size + " iterations. History: " + lr.iterations)
    println("Final weights: " + lr.weights)

    println("Price for 1650sqm 3br house: " + lr.predict(norm(DenseVector(1.0d, 1650.0d, 3d))))
  }
}

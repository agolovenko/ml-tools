package examples.mlstanford.week2

import breeze.linalg.DenseVector
import regression.LinearLBFGSRegressor
import util.{SetStats, Transformations}

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

    val data = Source.fromFile(f).getLines()
      .map(parse).toList

    val fCount = data.head._1.length
    val stats = new SetStats(data.map(_._1), fCount)

    val normalized = data.map { case (x, y) => (Transformations.minMax(stats.med, stats.halfRange)(x), y) }

    val lr = new LinearLBFGSRegressor(normalized, fCount, 0.0d, 50)

    println("Final weights: " + lr.weights)
    println("Iterations: " + lr.iterations)
  }
}

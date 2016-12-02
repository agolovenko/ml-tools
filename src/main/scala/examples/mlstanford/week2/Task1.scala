package examples.mlstanford.week2

import breeze.linalg.DenseVector
import regression.LinearLBFGSRegressor

import scala.io.Source

object Task1 {
  def parse(line: String): (DenseVector[Double], Double) = {
    val values = line.trim.split(",")

    val features = DenseVector(1.0d +: values.slice(0, values.length - 1).map(_.toDouble))

    (features, values.last.toDouble)
  }

  def main(args: Array[String]): Unit = {
    val f = getClass.getResource("ex1data1.txt").getPath

    val data = Source.fromFile(f).getLines()
      .map(parse).toSeq

    val fCount = data.head._1.length

    val lr = new LinearLBFGSRegressor(data, fCount, 0.0d, 50)

    println("Final weights: " + lr.weights)
    println("Profit for foodtruck in 35000 town: " + lr.predict(DenseVector(1.0d, 3.5d)))
    println("Profit for foodtruck in 70000 town: " + lr.predict(DenseVector(1.0d, 7.0d)))
  }
}

package examples.mlstanford.week3

import java.io.File

import breeze.linalg.DenseVector
import util.IterativeLBFGSRegressor

import scala.io.Source

/**
  * Created by ashot.golovenko on 15/11/2016.
  */
object Task2 {
  def parse(line: String): (DenseVector[Double], Double) = {
    val values = line.trim.split(",")
    val features = createPolynomialFeatures(values(0).toDouble, values(1).toDouble, 6)

    (features, values.last.toDouble)
  }

  def createPolynomialFeatures(f1: Double, f2: Double, maxPower: Int): DenseVector[Double] = {
    val len = (maxPower + 1) * (maxPower + 2) / 2

    val polyFeatures = new Array[Double](len)
    var i = 0

    for (p1 <- 0 to maxPower) {
      for (p2 <- 0 to p1) {
        polyFeatures(i) = Math.pow(f1, p1 - p2) * Math.pow(f2, p2)
        i += 1
      }
    }

    DenseVector(polyFeatures)
  }

  def main(args: Array[String]): Unit = {
    val f = new File(getClass.getResource("ex2data2.txt").getPath)

    val data = Source.fromFile(f).getLines()
      .map(parse).toSeq

    val fCount = data.head._1.length

    val lr = new IterativeLBFGSRegressor(() => data.iterator, fCount, 0.1d, 50)

    println("Final weights: " + lr.weights)
  }
}

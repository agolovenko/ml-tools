package examples.mlstanford.week3

import java.io.File

import breeze.linalg.DenseVector
import util.IterativeLBFGSRegressor

import scala.io.Source

/**
  * Created by ashot.golovenko on 14/11/2016.
  */
object Task1 {
  def parse(line: String): (DenseVector[Double], Double) = {
    val values = line.trim.split(",")

    val features = DenseVector(1.0d +: values.slice(0, values.length - 1).map(_.toDouble))

    (features, values.last.toDouble)
  }

  def main(args: Array[String]): Unit = {
    val f = new File(getClass.getResource("ex2data1.txt").getPath)

    val data = Source.fromFile(f).getLines()
      .map(parse).toSeq

    val fCount = data.head._1.length

    val lr = new IterativeLBFGSRegressor(() => data.iterator, fCount, 0.0d, 50)

    println("Final weights: " + lr.weights)
    println("Student with grades 45 and 85 has chances: " + lr.predict(DenseVector(1.0d, 45.0d, 85.0d)))
  }
}


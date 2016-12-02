package util

import scala.collection.IterableView
import scala.io.Source

class FileLineView private(val filePath: String, val encoding: String) extends IterableView[String, Iterable[String]] {
  override protected lazy val underlying: Iterable[String] = this.asInstanceOf[Iterable[String]]

  override def iterator: Iterator[String] = Source.fromFile(filePath, encoding).getLines()
}

object FileLineView {
  def apply(filePath: String): FileLineView = apply(filePath, "UTF-8")

  def apply(filePath: String, encoding: String): FileLineView = new FileLineView(filePath, encoding)
}



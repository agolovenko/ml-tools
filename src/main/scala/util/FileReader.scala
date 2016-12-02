package util

import scala.io.Source

class FileReader(val filePath: String, val encoding: String = "UTF-8") extends Iterable[String] {
  private val source = Source.fromFile(filePath, encoding)

  override def iterator: Iterator[String] = source.getLines
}

package util

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import regression.Regressor

/**
  * Created by ashot.golovenko on 18/11/2016.
  */
object Evaluation {
  def confusion(lr: Regressor, data: () => Iterator[(DenseVector[Double], Double)]): DenseMatrix[Double] = {
    val confusion = DenseMatrix.zeros[Double](2, 2)

    data().map { case (x, y) =>
      (y.toInt, if (lr.predict(x) > 0.5) 1 else 0)
    } foreach { case (truth, predicted) =>
      confusion(truth, predicted) += 1.0
    }

    confusion
  }

  def printConfusionMtx(confusion: DenseMatrix[Double]): Unit = {
    val negatives = confusion(0, 0) + confusion(0, 1)
    val positives = confusion(1, 0) + confusion(1, 1)
    val total = sum(confusion)

    val falseNegatives = confusion(1, 0)
    val falsePositives = confusion(0, 1)
    val accuracy = (confusion(0, 0) + confusion(1, 1)) / total

    println("============= Stats =============\n")

    println(f"Positive examples: $positives%1.0f")
    println(f"Negative examples: $negatives%1.0f")
    println(f"Total: $total%1.0f")
    println(f"Pos/Neg ratio: ${positives/negatives}%1.2f")

    println("\n============= Results =============\n")

    println("Confusion Matrix:")
    println(confusion)
    println(f"Accuracy: ${accuracy * 100}%2.2f%%")
    println(f"False positives: ${falsePositives * 100 / negatives}%2.2f%%")
    println(f"False negatives: ${falseNegatives * 100 / positives}%2.2f%%")
  }
}


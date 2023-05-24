package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.InputStream
import org.jetbrains.kotlinx.multik.ndarray.operations.*

internal data class LicensePlateResult(
    var outputBitmap: Bitmap, var outputBox: Array<FloatArray>
)



internal class LicensePlateDetector {
    fun detect(inputStream: InputStream, ortEnv: OrtEnvironment, ortSession: OrtSession) {
        val bitmap = BitmapFactory.decodeStream(inputStream)
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        val resizedImage = Mat()
        val newmat = Mat(512, 512, CvType.CV_32FC3)

        Imgproc.resize(mat, resizedImage, Size(512.0, 512.0))
        Imgproc.cvtColor(resizedImage, resizedImage, Imgproc.COLOR_BGRA2RGB)
        resizedImage.convertTo(newmat, CvType.CV_32FC3, 2.0 / 255, -1.0)

        val matrix = Array(newmat.rows()) { row ->
            FloatArray(newmat.cols()) { col ->
                newmat.get(row, col)[0].toFloat()
            }
        }
        val transposedMatrix = transposeMatrix(matrix)
        val a = mk.ndarray(mk[1, 2, 3])


        val copiedImage = FloatArray((newmat.total() * newmat.channels()).toInt()) { 0f }
        newmat.get(0, 0, copiedImage)
        val flattenImage = prepareImage(copiedImage)
        val stop = 1
    }

    private fun transposeMatrix(matrix: Array<FloatArray>): Array<FloatArray> {
        val rows = matrix.size
        val cols = matrix[0].size

        val transposedMatrix = Array(cols) { FloatArray(rows) }

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                transposedMatrix[j][i] = matrix[i][j]
            }
        }

        return transposedMatrix
    }

    private fun prepareImage(data: FloatArray): FloatArray {
        return FloatArray(1){0f}
    }

    private fun byteArrayToBitmap(data: ByteArray): Bitmap {
        return BitmapFactory.decodeByteArray(data, 0, data.size)
    }
}
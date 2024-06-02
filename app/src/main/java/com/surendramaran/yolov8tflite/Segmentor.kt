package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

class Segmentor(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val segmentorListener: SegmentorListener
) {

    private var interpreter: Interpreter? = null
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private var xPoints = 0
    private var yPoints = 0
    private var numMasks = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    fun setup() {
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options()
        options.numThreads = 4
        interpreter = Interpreter(model, options)

        val inputShape = interpreter?.getInputTensor(0)?.shape() ?: return
        val coordsShape = interpreter?.getOutputTensor(0)?.shape() ?: return // Coordinates
        val masksShape = interpreter?.getOutputTensor(1)?.shape() ?: return // Masks

        tensorWidth = inputShape[1]
        tensorHeight = inputShape[2]

        numChannel = coordsShape[1]
        numElements = coordsShape[2]

        xPoints = masksShape[1]
        yPoints = masksShape[2]
        numMasks = masksShape[3]

        try {
            val inputStream: InputStream = context.assets.open(labelPath)
            val reader = BufferedReader(InputStreamReader(inputStream))

            var line: String? = reader.readLine()
            while (line != null && line != "") {
                labels.add(line)
                line = reader.readLine()
            }

            reader.close()
            inputStream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    fun clear() {
        interpreter?.close()
        interpreter = null
    }

    fun segment(frame: Bitmap) {
        interpreter ?: return
        if (tensorWidth == 0) return
        if (tensorHeight == 0) return
        if (numChannel == 0) return
        if (numElements == 0) return
        if (xPoints == 0) return
        if (yPoints == 0) return
        if (numMasks == 0) return

        var inferenceTime = SystemClock.uptimeMillis()

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)

        val tensor = TensorImage(DataType.FLOAT32)
        tensor.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensor)
        val imageBuffer = processedImage.buffer

        val coordinatesBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1 , numChannel, numElements),
            OUTPUT_IMAGE_TYPE
        )

        val maskProtoBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, xPoints, yPoints, numMasks),
            OUTPUT_IMAGE_TYPE
        )

        val output = mapOf<Int, Any>(
            0 to coordinatesBuffer.buffer.rewind(),
            1 to maskProtoBuffer.buffer.rewind()
        )

        interpreter?.run(imageBuffer, output)

        val coordinates = coordinatesBuffer.floatArray
        val masks = maskProtoBuffer.floatArray

        val bestBoxes = bestMask(coordinates)?.sortedByDescending { it.cnf }?.get(0)
        val output1 = reshape(masks)

        val multiply = mutableListOf<Mat>()
        for (index in 0 until numMasks) {
            multiply.add(output1[index].multiplyDouble(bestBoxes?.maskWeight?.get(index)?.toDouble()))
        }

        val final = multiply[0].clone()
        for (i in 1 until multiply.size) {
            Core.add(final, multiply[i], final)
        }

        val mask = Mat()
        Core.compare(final, Scalar(0.0), mask, Core.CMP_GT)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime


        if (bestBoxes == null) {
            segmentorListener.onEmptySegment()
            return
        }

        segmentorListener.onSegment(bestBoxes, inferenceTime)

    }

    private fun bestMask(array: FloatArray) : List<MaskBox>? {

        val maskBoxes = mutableListOf<MaskBox>()

        for (c in 0 until numElements) {
            var maxConf = -1.0f
            var maxIdx = -1
            var j = 4
            var arrayIdx = c + numElements * j
            while (j < numChannel){
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += numElements
            }

            if (maxConf > Segmentor.CONFIDENCE_THRESHOLD) {
                val clsName = labels[maxIdx]
                val cx = array[c] // 0
                val cy = array[c + numElements] // 1
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]

                val x1 = cx - (w/2F)
                val y1 = cy - (h/2F)
                val x2 = cx + (w/2F)
                val y2 = cy + (h/2F)
                if (x1 < 0F || x1 > 1F) continue
                if (y1 < 0F || y1 > 1F) continue
                if (x2 < 0F || x2 > 1F) continue
                if (y2 < 0F || y2 > 1F) continue

                val maskWeight = mutableListOf<Float>()
                for (index in 0 until numMasks) {
                    maskWeight.add(array[c + numElements * (index + 5)])
                }
                maskBoxes.add(
                    MaskBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = maxConf, cls = maxIdx, clsName = clsName,
                        maskWeight = maskWeight,
                    )
                )
            }
        }

        if (maskBoxes.isEmpty()) return null

        return applyNMS(maskBoxes)
    }
    private fun reshape(masks: FloatArray) : List<Mat> {
        val all = mutableListOf<Mat>()
        for (mask in 0 until numMasks) {
            val mat = Mat(xPoints, yPoints, CvType.CV_32F)
            for (x in 0 until xPoints) {
                for (y in 0 until yPoints) {
                    mat.put(y, x, masks[ numMasks * yPoints *y + numMasks *x + mask].toDouble())
                }
            }
            all.add(mat)
        }
        return all
    }


    private fun applyNMS(boxes: List<MaskBox>) : MutableList<MaskBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

        while(sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= Segmentor.IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return selectedBoxes
    }
    private fun calculateIoU(b1: MaskBox, b2: MaskBox): Float {
        val x1 = maxOf(b1.cx - (b1.w/2F), b2.cx - (b2.w/2F))
        val y1 = maxOf(b1.cy - (b1.h/2F), b2.cy - (b2.h/2F))
        val x2 = minOf(b1.cx + (b1.w/2F), b2.cx + (b2.w/2F))
        val y2 = minOf(b1.cy + (b1.h/2F), b2.cy + (b2.h/2F))

        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = b1.w * b1.h
        val box2Area = b2.w * b2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }
    interface SegmentorListener {
        fun onEmptySegment()
        fun onSegment(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.3F
        private const val IOU_THRESHOLD = 0.5F
    }
}
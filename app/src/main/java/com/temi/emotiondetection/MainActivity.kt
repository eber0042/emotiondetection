package com.temi.emotiondetection

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.Toast
import androidx.annotation.RequiresApi
import com.google.android.gms.tasks.Task
import com.google.android.gms.tflite.java.TfLite
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import org.opencv.android.CameraActivity
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2
import org.opencv.android.JavaCameraView
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfRect
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import org.tensorflow.lite.InterpreterApi
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption
import kotlin.math.abs
import com.google.gson.Gson

class MainActivity : CameraActivity(), CvCameraViewListener2 {
    private val coroutineScope = CoroutineScope(Dispatchers.Main)

    // OpenCV params
    private lateinit var mOpenCvCameraView: CameraBridgeViewBase
    private lateinit var faceCascade: CascadeClassifier

    // LiteRT params
    private val initializeTask: Task<Void> by lazy { TfLite.initialize(this) }
    private lateinit var interpreter: InterpreterApi

    /**
     * Emotion dictionary to map indices to emotion labels
     */
    val emotionDict = mapOf(
        0 to "Angry",
        1 to "Disgusted",
        2 to "Fearful",
        3 to "Happy",
        4 to "Neutral",
        5 to "Sad",
        6 to "Surprised"
    )

    /**
     * List that stores previous generation of faces and corresponding classifier results.
     */
    var facesArrayMemory: MutableList< Array< Pair<Rect, FloatArray> > > = mutableListOf() // Change size to change how far system remembers

    // In Emotion Detection app, in the activity where you want to send the message
    // val message = "Hello from Emotion Detection!"

    /**
     * Number of previous iterations of faces stored in memory.
     */
    var memorySize = 1

    var currentframe = 5

    init {
        Log.i(TAG, "Instantiated new " + this.javaClass)
        for (i in 0 until memorySize) {
            facesArrayMemory.add( arrayOf< Pair<Rect, FloatArray> >() ) // Add an empty array of Rect
        }
        Log.i("SIZE", "Boo  ${facesArrayMemory.size}")
    }

    @RequiresApi(Build.VERSION_CODES.O)
    public override fun onCreate(savedInstanceState: Bundle?) {
        Log.i(TAG, "called onCreate")
        super.onCreate(savedInstanceState)

        val installedApps = getInstalledTemiApps(packageManager)

        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully")
        } else {
            Log.e(TAG, "OpenCV initialization failed!")
            Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG)
                .show()
            return
        }

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContentView(R.layout.activity_main)

        mOpenCvCameraView =
            findViewById<JavaCameraView>(R.id.tutorial1_activity_java_surface_view) as CameraBridgeViewBase

        mOpenCvCameraView.visibility = SurfaceView.VISIBLE

        mOpenCvCameraView.setCvCameraViewListener(this)

        getFrontCameraID(this)?.let { id: String ->
            mOpenCvCameraView.setCameraIndex(id.toInt())
        }

        faceCascade = loadCascade()

        initializeTask.addOnSuccessListener {
            val interpreterOption =
                InterpreterApi.Options()
                    .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
            interpreter = InterpreterApi.create(
                loadModelFile(this),
                interpreterOption
            )
        }
            .addOnFailureListener { e ->
                Log.e("Interpreter", "Cannot initialize interpreter", e)
            }
    }

    public override fun onPause() {
        super.onPause()
        mOpenCvCameraView.disableView()
    }

    public override fun onResume() {
        super.onResume()
        mOpenCvCameraView.enableView()
    }

    override fun getCameraViewList(): List<CameraBridgeViewBase> {
        return listOf<CameraBridgeViewBase>(mOpenCvCameraView)
    }

    public override fun onDestroy() {
        super.onDestroy()
        mOpenCvCameraView.disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        // Set desired resolution (e.g., 640x480)
//        mOpenCvCameraView.setCameraResolution(640, 480)
//        mOpenCvCameraView.setMaxFrameSize(640, 480) // Example resolution
    }

    override fun onCameraViewStopped() {
    }

    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat {
        // Get the RGBA frame from the camera input
        val output = inputFrame.rgba()
        val grayFrame = inputFrame.gray()

        // Define the region of interest (x, y, width, height)
        // 1920 by 1080
        val roiX = 560  // Change these values as needed
        val roiY = 0
        val roiWidth = 800
        val roiHeight = 1080

        Imgproc.rectangle(
            output, // Change this to roi for checking range
            Point(roiX.toDouble(), roiY.toDouble()),
            Point((roiX + roiWidth).toDouble(), (roiY + roiHeight).toDouble()),
            Scalar(0.0, 0.0, 0.0), // Green color for the rectangle
            3
        )

        // Extract the region of interest from the gray frame
        val roi = grayFrame.submat(roiY, roiY + roiHeight, roiX, roiX + roiWidth)

        /**
         * Check if two points are within `threshold` distance of each other on the x and y plane.
         *
         * @param corner1 first point
         * @param corner2 second point
         * @param threshold threshold distance
         *
         * @return if both points are within `threshold` distance of each other.
         */
        fun isWithinThreshold(corner1: Point, corner2: Point, threshold: Int): Boolean {
            val dx = abs(corner1.x - corner2.x)
            val dy = abs(corner1.y - corner2.y)
            return dx <= threshold && dy <= threshold
        }

        /**
         * Check if two faces have corners that are within `threshold` distance of each other.
         *
         * @param rect1 first face to compare
         * @param rect2 second face to compare
         * @param threshold threshold distance
         *
         * @return if both faces are within `threshold` distance of each other.
         */
        fun compareRects(rect1: Rect, rect2: Rect, threshold: Int): Boolean {
            // Extract corners for rect1
            val rect1TopLeft = rect1.tl()  // Top-left corner
            val rect1BottomRight = rect1.br()  // Bottom-right corner
            val rect1TopRight = rect1.tl().apply { x += rect1.width }  // Top-right corner
            val rect1BottomLeft = rect1.tl().apply { y += rect1.height }  // Bottom-left corner

            // Extract corners for rect2
            val rect2TopLeft = rect2.tl()
            val rect2BottomRight = rect2.br()
            val rect2TopRight = rect2.tl().apply { x += rect2.width }
            val rect2BottomLeft = rect2.tl().apply { y += rect2.height }

            // Compare the corresponding corners
            return isWithinThreshold(rect1TopLeft, rect2TopLeft, threshold) &&
                    isWithinThreshold(rect1TopRight, rect2TopRight, threshold) &&
                    isWithinThreshold(rect1BottomLeft, rect2BottomLeft, threshold) &&
                    isWithinThreshold(rect1BottomRight, rect2BottomRight, threshold)
        }

        /**
         * Get classifier result for detected face.
         *
         * @param face face detected
         *
         * @return float array containing results of classifier model.
         */
        fun getClassifierResult(face: Rect): FloatArray {
            // Process the face for emotion recognition
            val faceRegion = roi.submat(face)
            val resizedFace = Mat()
            Imgproc.resize(faceRegion, resizedFace, Size(48.0, 48.0))

            val faceFloat32 = Mat()
            resizedFace.convertTo(faceFloat32, CvType.CV_32F, 1.0 / 255.0)

            val faceArray = FloatArray(48 * 48)
            faceFloat32.get(0, 0, faceArray)

            // Prepare the input tensor for TFLite
            val inputArray = Array(1) { Array(48) { Array(48) { FloatArray(1) } } }
            for (i in 0 until 48) {
                for (j in 0 until 48) {
                    inputArray[0][i][j][0] = faceArray[i * 48 + j]
                }
            }

            // Prepare output for the interpreter: [1, 7]
            val outputArray = Array(1) { FloatArray(7) }

            // Run TFLite inference
            interpreter.run(inputArray, outputArray)

            // Process the output
            val predictions = outputArray[0]

            return predictions
        }

        /**
         * Predict emotion for face with past `memorySize` generations of faces.
         *
         * @param face face to be classified
         * @param threshold threshold distance to determine if faces of different generations are the same face
         *
         * @return Prediction result. If face is not valid, returns `null`.
         */
        fun predictEmotion(face: Rect, threshold: Int = 15): String? {
            // Perform a BFS to find if there are predecessors in the past (memorySize) generations
            val queue = mutableListOf( Pair(face, getClassifierResult(face)) )

            for (generation: Int in 1 until memorySize) {
                val queueLen: Int = queue.size
                for (i in 1..queueLen) {
                    val (popFace: Rect, popResult: FloatArray) = queue.removeAt(index = 0)

                    for ((predecessorFace: Rect, predecessorResult: FloatArray) in facesArrayMemory[generation]) {
                        if (compareRects(popFace, predecessorFace, threshold)) {
                            queue.add( Pair(predecessorFace, popResult.mapIndexed { index, fl -> fl + predecessorResult[index] }.toFloatArray() ) )
                        }
                    }
                }

                if (queue.size == 0) return null
            }

            val finalResult: FloatArray = queue[0].second
            val predictedClassIndex = finalResult.indices.maxByOrNull { finalResult[it] } ?: -1
            return if (predictedClassIndex != -1) emotionDict[predictedClassIndex] else "Unknown"
        }

        // Main face detection logic
        val faces = MatOfRect()
        faceCascade.detectMultiScale(roi, faces)

        val facesArray = faces.toArray()
        val facesArrayResult: List< Pair<Rect, FloatArray> > = facesArray.map{ face: Rect ->
            Pair(face, getClassifierResult(face))
        }

        // Add the new array at index 0
        facesArrayMemory.add(0, facesArrayResult.toTypedArray())

        // Remove the last element
        facesArrayMemory.removeAt(facesArrayMemory.size - 1)


        Log.i("Face", "Number of faces detected: ${facesArray.size}")

        Log.i("Size", "Size of memory: ${facesArrayMemory.size}")

        // Create a list to store faces and their associated areas
        val detectedFaces = mutableListOf<Pair<Rect, String>>() // Pair of Rect and emotion label

        facesArray.forEach { face: Rect ->
            val adjustedTopLeft = Point(face.x + roiX.toDouble(), face.y + roiY.toDouble())
            val adjustedBottomRight = Point(
                face.x + roiX.toDouble() + face.width.toDouble(),
                face.y + roiY.toDouble() + face.height.toDouble()
            )
             var i = 0
            predictEmotion(face)?.let { emotionLabel: String ->
                // Store the face and its emotion label in the list
                detectedFaces.add(Pair(face, emotionLabel))

                // Draw rectangle
                Imgproc.rectangle(
                    output,
                    adjustedTopLeft,
                    adjustedBottomRight,
                    Scalar(0.0, 255.0, 0.0),
                    3
                )

                val labelPosition = Point(adjustedTopLeft.x, adjustedTopLeft.y - 10) // Position above the rectangle
                Imgproc.putText(
                    output,
                    emotionLabel,
                    labelPosition,
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    Scalar(0.0, 255.0, 0.0),
                    10
                )
            } ?: run {
                // Draw rectangle
                Imgproc.rectangle(
                    output,
                    adjustedTopLeft,
                    adjustedBottomRight,
                    Scalar(255.0, 0.0, 0.0),
                    3
                )
            }
        }

        // Sort the detected faces based on area (largest first)
        detectedFaces.sortByDescending { it.first.width * it.first.height }

        var i = 0
        // Now, send out the broadcasts in order of the area
        for ((face, emotionLabel) in detectedFaces) {
            val adjustedTopLeft = Point(face.x.toDouble(), face.y.toDouble())
            val adjustedBottomRight = Point((face.x + face.width).toDouble(), (face.y + face.height).toDouble())

            // Send the broadcast message

            // Log.i("FUCK!", "Sent ${convertToJson(emotionLabel)}" + " $i ${face.width * face.height}")
            val returnIntent = Intent().apply {
                action = "com.quizapp.RECEIVE_DATA"
                putExtra("message", convertToJson(emotionLabel)  + " Face: $i, Area: ${face.width * face.height} ")
                // Optional: Set the Quiz app’s package explicitly to ensure it’s handled by the right app
                `package` = "com.temi.temiSDK" // Replace with actual Quiz app package
            }

            sendBroadcast(returnIntent)
            i++
        }

        currentframe = (currentframe + 1) % 6
        return output
    }

    //**************************************************************************************
    companion object {
        private const val TAG = "OCVSample::Activity"
    }

    private fun loadCascade(): CascadeClassifier {
        val tempFile = File.createTempFile("haarcascade_frontalface_alt", ".xml", cacheDir)

        val inputStream: InputStream =
            resources.openRawResource(R.raw.haarcascade_frontalface_default)
        val outputStream: OutputStream = FileOutputStream(tempFile)
        inputStream.use { input ->
            outputStream.use { output ->
                input.copyTo(output)
            }
        }

        val classifier = CascadeClassifier(tempFile.absolutePath)

        tempFile.delete()

        return classifier
    }

    private fun getFrontCameraID(context: Context): String? {
        val cManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

        for (cameraId in cManager.getCameraIdList()) {
            val characteristics: CameraCharacteristics =
                cManager.getCameraCharacteristics(cameraId);
            val cOrientation = characteristics.get(CameraCharacteristics.LENS_FACING)

            if (cOrientation == CameraCharacteristics.LENS_FACING_FRONT) return cameraId
        }

        return null
    }

    // Example function to convert your data structure to JSON string
    fun convertToJson(facesArrayMemory: String): String {
        val gson = Gson()
        return gson.toJson(facesArrayMemory)
    }
}

fun getInstalledTemiApps(packageManager: PackageManager): List<String> {
    val apps = mutableListOf<String>()
    val packages = packageManager.getInstalledApplications(PackageManager.GET_META_DATA)

    for (packageInfo in packages) {
        // Check if the package name starts with "com.temi"
        if (packageInfo.packageName.startsWith("com.temi")) {
            // Log the found app
            // Log.d("FUCK!", "Found Temi app: ${packageInfo.packageName}")
            apps.add(packageInfo.packageName)
        }
    }

    return apps
}


@RequiresApi(Build.VERSION_CODES.O)
fun loadModelFile(context: Context): ByteBuffer {
    // Create a temporary file to store the raw resource
    val tempFile = File(context.cacheDir, "temp_file.dat")

    // Write the raw resource to the temporary file

    context.resources.openRawResource(R.raw.model).use { inputStream ->
        FileOutputStream(tempFile).use { outputStream ->
            inputStream.copyTo(outputStream)
        }
    }

    // Create a MappedByteBuffer from the temporary file
    return FileChannel.open(tempFile.toPath(), StandardOpenOption.READ).use { channel ->
        channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
    }
}

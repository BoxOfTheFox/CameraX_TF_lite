package com.example.cameraxapp

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.AttributeSet
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.doOnPreDraw
import com.example.cameraxapp.ml.MlModel
import com.example.cameraxapp.util.YuvToRgbConverter
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.model.Model
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

typealias BusDetectorListener = (detections: List<MlModel.DetectionResult>) -> Unit

class OverlayView(context: Context, attrs: AttributeSet) : View(context, attrs) {
    private val paint = Paint()
    private val pen = Paint()
    private val targets: MutableList<MlModel.DetectionResult> = ArrayList()

    init {
        pen.style = Paint.Style.FILL_AND_STROKE
        pen.color = Color.YELLOW
        pen.strokeWidth = 2F

        pen.textSize = 80F
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        synchronized(this) {
            for (entry in targets) {
                canvas.drawRect(entry.locationAsRectF, paint)
                canvas.drawText(
                    "${entry.categoryAsString} ${
                        entry.scoreAsFloat.times(100).toInt()
                    }%",
                    entry.locationAsRectF.centerX() - pen.measureText(entry.categoryAsString),
                    entry.locationAsRectF.top + 80,
                    pen
                )
            }
        }
    }

    fun setTargets(sources: List<MlModel.DetectionResult>) {
        synchronized(this) {
            targets.clear()
            targets.addAll(sources)
            this.postInvalidate()
        }
    }

    init {
        val density = context.resources.displayMetrics.density
        paint.strokeWidth = 2.0f * density
        paint.color = Color.BLUE
        paint.style = Paint.Style.STROKE
    }
}

class MainActivity : AppCompatActivity() {
    private var imageCapture: ImageCapture? = null

    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewFinder.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(1080, 1920))
                .build()
                .also {
                    viewFinder.doOnPreDraw { view ->
                        it.setAnalyzer(
                            cameraExecutor,
                            BusDetector(this, view.width, view.height) { detections ->
                                overlayView.setTargets(detections)
                            })
                    }
                }


            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer
                )


            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }


    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }


    companion object {
        private const val TAG = "CameraXBasic"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    private class BusDetector(
        ctx: Context,
        private val width: Int,
        private val height: Int,
        private val listener: BusDetectorListener
    ) :
        ImageAnalysis.Analyzer {

        val model: MlModel by lazy {
            val compatibilityList = CompatibilityList()

            val options = if (compatibilityList.isDelegateSupportedOnThisDevice) {
                Log.d(TAG, "This device is GPU Compatible")
                Model.Options.Builder().setDevice(Model.Device.GPU).build()
            } else {
                Log.d(TAG, "This device is GPU Incompatible")
                Model.Options.Builder().setNumThreads(4).build()
            }

            MlModel.newInstance(ctx, options)
        }

        override fun analyze(image: ImageProxy) {

            val resultToDisplay = model.process(
                TensorImage.fromBitmap(
                    toBitmap(image)?.let {
                        Bitmap.createScaledBitmap(it, width, height, false)
                    }
                )
            )
                .detectionResultList
                .filter { it.categoryAsString == "bus" && it.scoreAsFloat > 0.3 }

            listener(resultToDisplay)

            image.close()
        }

        /**
         * Convert Image Proxy to Bitmap
         */
        private val yuvToRgbConverter = YuvToRgbConverter(ctx)
        private lateinit var bitmapBuffer: Bitmap
        private lateinit var rotationMatrix: Matrix

        @SuppressLint("UnsafeExperimentalUsageError", "UnsafeOptInUsageError")
        private fun toBitmap(imageProxy: ImageProxy): Bitmap? {

            val image = imageProxy.image ?: return null

            // Initialise Buffer
            if (!::bitmapBuffer.isInitialized) {
                // The image rotation and RGB image buffer are initialized only once
                Log.d(TAG, "Initalise toBitmap()")
                rotationMatrix = Matrix()
                rotationMatrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                bitmapBuffer = Bitmap.createBitmap(
                    imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888
                )
            }

            // Pass image to an image analyser
            yuvToRgbConverter.yuvToRgb(image, bitmapBuffer)

            // Create the Bitmap in the correct orientation
            return Bitmap.createBitmap(
                bitmapBuffer,
                0,
                0,
                bitmapBuffer.width,
                bitmapBuffer.height,
                rotationMatrix,
                false
            )
        }
    }
}

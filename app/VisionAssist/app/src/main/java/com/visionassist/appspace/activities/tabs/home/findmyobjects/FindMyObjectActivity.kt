@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.tabs.home.findmyobjects

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.Size
import android.view.KeyEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.tween
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.tabs.LightManager
import com.visionassist.appspace.activities.tabs.MotionManager
import com.visionassist.appspace.activities.tabs.reports.EnvironmentReportsManagerKt
import com.visionassist.appspace.jetpack.design.BackArrowLargeFab
import com.visionassist.appspace.jetpack.design.CustomSlider
import com.visionassist.appspace.jetpack.design.NextArrowLargeFab
import com.visionassist.appspace.jetpack.design.ThumbStyle
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager
import com.visionassist.appspace.models.classifier.YOLOClassifier
import com.visionassist.appspace.models.detector.DetectionResult
import com.visionassist.appspace.models.detector.YOLODetector
import com.visionassist.appspace.models.detector.YOLODetectorPool
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.BackgroundTaskExecutor
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.ImageUtils
import com.visionassist.appspace.utils.PermissionChecker
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.startBatteryLevelCheck
import com.visionassist.appspace.utils.vibrate
import java.util.concurrent.CompletableFuture
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

class FindMyObjectActivity : ComponentActivity() {
    private val TAG = "FindMyObjectActivity"

    // Models
    private lateinit var motionMonitor: MotionManager
    private lateinit var lightMonitor: LightManager
    private lateinit var currentDetectorModel: YOLODetectorPool
    private var preferNanoModel = false
    private val classifier: YOLOClassifier =
        PhoneStatusMonitor.getInstance().modelManager.classifier
    private var canSwitchModels = false

    // Detection data
    private lateinit var objectsToFind: MutableMap<Int, String>
    private lateinit var remainingClassIndices: MutableList<Int>

    // Results
    private var originalBitmap: Bitmap? = null
    private val resultBitmap = mutableStateOf<Bitmap?>(null)
    private var detectionResult: DetectionResult? = null
    private val threadCount = AtomicInteger(0)
    private var avgDetectorLatency = 0L
    private var avgClassifierLatency = 0L

    // Camera
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private var currentCamera: Camera? = null
    private var isFlashlightOn = false

    // Threads Control States
    private val stopDetection = AtomicBoolean(false)
    private val resultsReady = AtomicBoolean(false)
    private val displayReady = AtomicBoolean(false)

    // Handlers
    private val mainHandler = Handler(Looper.getMainLooper())
    private val updateHandler = Handler(Looper.getMainLooper())
    private var updateRunnable: Runnable? = null

    // Compose states
    private val showResult = mutableStateOf(false)
    private val showBatteryWarning = mutableStateOf(false)

    // Monitor parameters
    private val avgBatteryMoreUsed = mutableStateOf(0f)
    private val batteryCheckRunning = mutableStateOf(false)

    // Resize bbox + text parameters
    private var currentBBoxOffset = 0f
    private var currentTextRatio = Constants.TEXT_SIZE_WIDTH_SCREEN

    data class ThreadResult(
        val detectionResult: DetectionResult?,
        val bitmap: Bitmap?,
        val sceneClassId: Int,
        val detectorLatency: Long,
        val classifierLatency: Long
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        extractIntentData()
        cameraExecutor = Executors.newSingleThreadExecutor()

        currentDetectorModel = PhoneStatusMonitor.getInstance().modelManager.detector

        motionMonitor = MotionManager(
            this,
            mainHandler
        ) {
            preferNanoModel =
                motionMonitor.linearSpeed > Constants.LINEAR_SPEED_THRESHOLD || motionMonitor.rotationSpeed > Constants.ROTATION_SPEED_THRESHOLD
        }

        lightMonitor= LightManager(
            this,
            mainHandler
        ) {
            if (lightMonitor.isDark) {
                turnFlashlightOn()
            } else {
                turnFlashlightOff()
            }
        }

        setContent {
            FindMyObjectScreen(
                showResult = showResult.value,
                resultBitmap = resultBitmap.value,
                hasMoreObjects = remainingClassIndices.isNotEmpty(),
                showBatteryWarning = showBatteryWarning.value,
                onBackClick = ::handleBackClick,
                onNextClick = if (remainingClassIndices.isNotEmpty()) ::handleNextClick else null,
                onCameraReady = { previewView ->
                    startCameraX(previewView)
                },
                onBBoxResize = ::handleBBoxResize,
                onTextResize = ::handleTextResize
            )
        }
    }

    override fun onResume() {
        super.onResume()
        if (PhoneStatusMonitor.getInstance().isReturningFromPermissions) {
            PermissionChecker.checkAndRequestPermissions(this, false) {
                reloadDetectionPhase()
                startDetectionProcess()
            }
        } else {
            if (stopDetection.get() && !showResult.value) {
                Log.d(TAG, "Activity resumed - restarting detection")
                reloadDetectionPhase()
                startDetectionProcess()
            }
        }
    }

    private fun extractIntentData() {
        val classIndices = intent.getIntArrayExtra(Constants.EXTRA_MATCHED_INDICES) ?: intArrayOf()
        val matchedWords = intent.getStringArrayExtra(Constants.EXTRA_SYNONYMS_WORDS) ?: arrayOf()

        objectsToFind = mutableMapOf()
        remainingClassIndices = mutableListOf()

        for (i in classIndices.indices) {
            objectsToFind[classIndices[i]] = matchedWords[i]
            remainingClassIndices.add(classIndices[i])
        }

        Log.d(TAG, "Objects to find (detector_class:synonym): $objectsToFind")
    }

    private fun turnFlashlightOn() {
        try {
            if (currentCamera?.cameraInfo?.hasFlashUnit() == true && !isFlashlightOn) {
                currentCamera?.cameraControl?.enableTorch(true)
                isFlashlightOn = true
                Log.d(TAG, "🔦 Flashlight turned ON via CameraX")
            } else {
                Log.w(TAG, "Cannot turn ON flashlight: hasFlash=${currentCamera?.cameraInfo?.hasFlashUnit()}, isOn=$isFlashlightOn")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error turning flashlight ON via CameraX", e)
        }
    }

    private fun turnFlashlightOff() {
        try {
            if (currentCamera?.cameraInfo?.hasFlashUnit() == true && isFlashlightOn) {
                currentCamera?.cameraControl?.enableTorch(false)
                isFlashlightOn = false
                Log.d(TAG, "🔦 Flashlight turned OFF via CameraX")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error turning flashlight OFF via CameraX", e)
        }
    }

    private fun handleBBoxResize(offsetDp: Float) {
        currentBBoxOffset = offsetDp

        // Cancel any pending update
        updateRunnable?.let { updateHandler.removeCallbacks(it) }

        // Schedule new update with delay
        updateRunnable = Runnable {
            redrawDetections()
        }
        updateHandler.postDelayed(updateRunnable!!, Constants.PREVIEW_UPDATE_DELAY.toLong())
    }

    private fun handleTextResize(textSizeRatio: Float) {
        currentTextRatio = textSizeRatio

        // Cancel any pending update
        updateRunnable?.let { updateHandler.removeCallbacks(it) }

        // Schedule new update with delay
        updateRunnable = Runnable {
            redrawDetections()
        }
        updateHandler.postDelayed(updateRunnable!!, Constants.PREVIEW_UPDATE_DELAY.toLong())
    }

    private fun redrawDetections() {
        BackgroundTaskExecutor.getInstance().executeAsync(
            {
                // Use the smart drawing method that handles everything
                YOLODetector.drawDetectionsWithSmartResize(
                    originalBitmap,
                    detectionResult,
                    currentBBoxOffset,  // Current bbox offset
                    currentTextRatio,   // Current text ratio (can be null for default)
                    resources.displayMetrics
                )
            },
            object : BackgroundTaskExecutor.TaskCallback<Bitmap?> {
                override fun onSuccess(result: Bitmap?) {
                    if (result != null) {
                        resultBitmap.value = result
                        // Trigger recomposition
                        showResult.value = true
                    }
                }

                override fun onError(e: Exception) {
                    Log.e(TAG, "Error redrawing detections", e)
                }
            }
        )
    }

    private fun startCameraX(previewView: PreviewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCamera(previewView)
                startDetectionProcess()
            } catch (e: Exception) {
                Log.e(TAG, "Error starting camera", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCamera(previewView: PreviewView) {
        val preview = Preview.Builder().build()
        preview.surfaceProvider = previewView.surfaceProvider

        val screenWidth = resources.displayMetrics.widthPixels
        val screenHeight = resources.displayMetrics.heightPixels

        imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .setTargetResolution(Size(screenWidth, screenHeight))
            .setFlashMode(ImageCapture.FLASH_MODE_OFF)
            .build()

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            cameraProvider?.unbindAll()
            currentCamera=cameraProvider?.bindToLifecycle(
                this, cameraSelector, preview, imageCapture
            )
            Log.d(TAG, "Camera bound")
        } catch (e: Exception) {
            Log.e(TAG, "Error binding camera", e)
        }
    }

    private fun startDetectionProcess() {
        // Start phone status monitoring
        PhoneStatusMonitor.getInstance().startMonitoring(mainHandler) {
            resultsReady.set(true);batteryCheckRunning.value = false;motionMonitor.stopMonitoring();lightMonitor.stopMonitoring();turnFlashlightOff()
        }
        if (canSwitchModels)
            motionMonitor.startMonitoring()
        lightMonitor.startMonitoring()
        // Start battery level check
        batteryCheckRunning.value = true
        startBatteryLevelCheck(batteryCheckRunning, showBatteryWarning, avgBatteryMoreUsed)

        // Start detection loop
        stopDetection.set(false)
        resultsReady.set(false)
        displayReady.set(false)
        startDetectionLoop()
    }

    private fun startDetectionLoop() {
        // Launch detection thread
        launchDetectionThread()

        mainHandler.post(object : Runnable {
            override fun run() {
                if (stopDetection.get()) {
                    // Stop loop and show results
                    showResults()
                    return
                }

                // Schedule next iteration
                mainHandler.postDelayed(this, 500)
            }
        })
    }

    private fun launchDetectionThread() {
        val toRun = {
            threadCount.incrementAndGet()
            if (!resultsReady.get()) {
                BackgroundTaskExecutor.getInstance().executeAsync(
                    {
                        // Capture image
                        val currentThreadNo = "Thread_" + threadCount.get()
                        val detectorWrapper = currentDetectorModel.acquireDetector(
                            preferNanoModel,  // Prefer nano if moving fast
                            5  // 5 second timeout
                        )
                        if (detectorWrapper == null)
                            return@executeAsync ThreadResult(null, null, -1, 0, 0)

                        val detector = detectorWrapper.detector

                        val bitmap = captureImageSync() ?: return@executeAsync null

                        var sceneClassId = -1
                        var classifierLatency = 0L

                        val classifierLatch = CountDownLatch(1)

                        if (AppConfig.env_reports) {
                            BackgroundTaskExecutor.getInstance().executeAsync(
                                {
                                    val classifierStart = System.currentTimeMillis()
                                    sceneClassId = classifier.detectScene(bitmap, currentThreadNo)
                                    classifierLatency = System.currentTimeMillis() - classifierStart
                                    null
                                },
                                object : BackgroundTaskExecutor.TaskCallback<Any?> {
                                    override fun onSuccess(result: Any?) {
                                        classifierLatch.countDown()
                                    }

                                    override fun onError(e: Exception) {
                                        Log.e(TAG, currentThreadNo + "Classifier error", e)
                                        classifierLatch.countDown()
                                    }
                                }
                            )
                        } else {
                            classifierLatch.countDown() // Skip waiting if disabled
                        }

                        // Run detector
                        val detectorStart = System.currentTimeMillis()
                        val detectionResult = detector.detectObjects(bitmap, currentThreadNo)
                        val detectorLatency = System.currentTimeMillis() - detectorStart

                        // Check if found target objects
                        val detectedClasses = detectionResult.classIndices
                        val foundClasses = remainingClassIndices.filter { it in detectedClasses }

                        if (foundClasses.isEmpty()) {
                            currentDetectorModel.releaseDetector(detectorWrapper)
                            bitmap.recycle()
                            return@executeAsync ThreadResult(null, null, -1, detectorLatency, 0)
                        }

                        // Filter detection result
                        val filteredResult =
                            filterDetectionResult(detectionResult, foundClasses, detector)
                        currentDetectorModel.releaseDetector(detectorWrapper)

                        try {
                            val finished = classifierLatch.await(1, TimeUnit.SECONDS)
                            if (!finished) {
                                Log.w(TAG, currentThreadNo + "Classifier timeout after 3 seconds")
                            }
                        } catch (e: InterruptedException) {
                            Log.e(TAG, currentThreadNo + "Wait interrupted", e)
                        }

                        // Now classifier is done, return result
                        ThreadResult(
                            filteredResult,
                            bitmap,
                            sceneClassId,
                            detectorLatency,
                            classifierLatency
                        )
                    },
                    object : BackgroundTaskExecutor.TaskCallback<ThreadResult?> {
                        override fun onSuccess(result: ThreadResult?) {
                            if (result == null || result.detectionResult == null) {
                                updateLatencyStats(
                                    result?.detectorLatency ?: 0,
                                    result?.classifierLatency ?: 0
                                )
                                //threadCount.decrementAndGet()
                                return
                            }

                            // Update latency statistics
                            updateLatencyStats(result.detectorLatency, result.classifierLatency)

                            // Check if another thread already finished
                            if (resultsReady.compareAndSet(false, true)) {
                                processFoundObjects(result)
                            }
                        }

                        override fun onError(e: Exception) {
                            Log.e(TAG, "Detection thread error", e)
                            //threadCount.decrementAndGet()
                        }
                    }
                )
            }
        }

        val runtime = Runtime.getRuntime()
        val maxMemory = runtime.maxMemory()
        val usedMemory = runtime.totalMemory() - runtime.freeMemory()
        var availableMemory: Long = maxMemory - usedMemory
        val checkRunnable = object : Runnable {
            override fun run() {
                if (availableMemory > currentDetectorModel.ACTUAL_THREAD_MEM_USED_MB) {
                    mainHandler.post(toRun)
                } else {
                    val runtime = Runtime.getRuntime()
                    val maxMemory = runtime.maxMemory()
                    val usedMemory = runtime.totalMemory() - runtime.freeMemory()
                    availableMemory = maxMemory - usedMemory
                    Log.w(
                        TAG,
                        "Low memory (${availableMemory / 1_048_576}MB), WAITING FOR RELEASE OR USER EXIT"
                    )
                    mainHandler.postDelayed(this, Constants.WAIT_CHECK)
                }
            }
        }
        mainHandler.post(checkRunnable)
    }

    private fun captureImageSync(): Bitmap? {
        return try {
            val captureResult = CompletableFuture<Bitmap?>()

            imageCapture?.takePicture(
                ContextCompat.getMainExecutor(this),
                object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(image: ImageProxy) {
                        try {
                            if (!resultsReady.get()) {
                                mainHandler.postDelayed(
                                    {
                                        launchDetectionThread()
                                    },
                                    Constants.CAMERA_RECOVERY_MS.toLong()
                                )  // 200ms camera recovery time
                            }

                            // Convert ImageProxy to Bitmap using ImageUtils
                            val bitmap = ImageUtils.imageProxyToBitmap(image)
                            captureResult.complete(bitmap)

                            if (bitmap != null) {
                                Log.d(TAG, "Image captured: ${bitmap.width}x${bitmap.height}")
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Error converting ImageProxy to Bitmap", e)
                            showCameraError(Constants.CAMERA_FAIL_CONVERT_IMGPROXY)
                            captureResult.complete(null)
                        } finally {
                            // CRITICAL: Always close ImageProxy to free memory
                            // If you don't close it, you'll run out of memory buffers
                            image.close()
                        }
                    }

                    override fun onError(exception: ImageCaptureException) {
                        Log.e(TAG, "Image capture failed: ${exception.message}", exception)
                        captureResult.complete(null)
                    }
                }
            )

            // Block and wait for result with timeout (5 seconds)
            // This makes the async callback synchronous
            captureResult.get(5, TimeUnit.SECONDS)

        } catch (e: TimeoutException) {
            Log.e(TAG, "Image capture timeout after 5 seconds", e)
            null
        } catch (e: Exception) {
            Log.e(TAG, "Error capturing image synchronously", e)
            showCameraError(Constants.STD_CAMERA_FAIL)
            null
        }
    }

    private fun filterDetectionResult(
        original: DetectionResult,
        foundClasses: List<Int>,
        detector: YOLODetector
    ): DetectionResult {
        // Get original detection data
        val originalBoxes = original.boundingBoxes
        val originalConfidences = original.confidences
        val originalClassIndices = original.classIndices

        // Create filtered lists
        val filteredBoxes = mutableListOf<RectF>()
        val filteredConfidences = mutableListOf<Float>()
        val filteredClassIndices = mutableListOf<Int>()
        val filteredLabels = mutableListOf<String>()

        // Iterate through all detections
        for (i in originalClassIndices.indices) {
            val classIdx = originalClassIndices[i]

            // Check if this detection matches one of our target objects
            if (classIdx in foundClasses) {
                // Add detection data to filtered lists
                filteredBoxes.add(originalBoxes[i])
                filteredConfidences.add(originalConfidences[i])
                filteredClassIndices.add(classIdx)

                // IMPORTANT: Use synonym from objectsToFind, not YOLO label
                // Example: User said "keys" → Use "keys" not "key"(detector label)
                val synonym = objectsToFind[classIdx] ?: detector.getClassName(classIdx)
                filteredLabels.add(synonym)

                Log.d(
                    TAG,
                    "Filtered detection: $synonym (class $classIdx) with confidence ${originalConfidences[i]}"
                )
            }
        }

        Log.d(
            TAG,
            "Filtered ${filteredLabels.size} detections from ${originalClassIndices.size} total"
        )

        // Return new DetectionResult with only matched objects
        return DetectionResult(
            filteredBoxes,
            filteredConfidences,
            filteredLabels,
            filteredClassIndices
        )
    }

    private fun updateLatencyStats(detectorLatency: Long, classifierLatency: Long) {
        synchronized(this) {
            if (detectorLatency > 0) {
                avgDetectorLatency = (avgDetectorLatency + detectorLatency) /
                        (2)
            }
            if (classifierLatency > 0) {
                avgClassifierLatency = (avgClassifierLatency + classifierLatency) /
                        (2)
            }
        }
    }

    private fun processFoundObjects(result: ThreadResult) {
        // Signal stop
        stopDetection.set(true)
        PhoneStatusMonitor.getInstance().stopMonitoring()
        motionMonitor.stopMonitoring();lightMonitor.stopMonitoring();turnFlashlightOff()
        batteryCheckRunning.value = false

        // Remove found classes
        val foundSynonyms = mutableListOf<String>()
        result.detectionResult?.classIndices?.forEach { classIdx ->
            if (classIdx in remainingClassIndices) {
                foundSynonyms.add(objectsToFind[classIdx] ?: "")
                remainingClassIndices.remove(classIdx)
            }
        }

        val listIndices = result.detectionResult?.classIndices as List<Int>
        // Write environment report
        if (AppConfig.env_reports) {
            val batteryUsageIncrease = if (avgBatteryMoreUsed.value > 0f) {
                1f - avgBatteryMoreUsed.value
            } else {
                0f
            }
            EnvironmentReportsManagerKt.writeDetectionReport(
                this,
                result.sceneClassId, listIndices,
                threadCount.get(),
                avgDetectorLatency,
                avgClassifierLatency,
                batteryUsageIncrease
            )
        }


        originalBitmap = result.bitmap
        detectionResult = result.detectionResult

        // Set result bitmap
        resultBitmap.value = YOLODetector.drawDetections(originalBitmap, detectionResult)

        currentBBoxOffset = 0f
        currentTextRatio = Constants.TEXT_SIZE_WIDTH_SCREEN

        // Signal display ready
        displayReady.set(true)
    }

    private fun showResults() {
        mainHandler.post(object : Runnable {
            override fun run() {
                if (displayReady.get()) {
                    // Stop loop and show results
                    if (AppConfig.haptics) {
                        vibrate(haptic_model0())
                    }
                    showResult.value = true
                    return
                }
                // Schedule next iteration
                mainHandler.postDelayed(this, 500)
            }
        })
    }

    private fun handleBackClick() {
        if (AppConfig.haptics) {
            vibrate(haptic_model0())
        }
        finish()
        //startActivity(Intent(this, HomeActivity::class.java))
    }

    private fun reloadDetectionPhase() {
        // Reset states
        showResult.value = false
        originalBitmap?.recycle()
        resultBitmap.value?.recycle()
        originalBitmap = null
        detectionResult = null
    }

    private fun handleNextClick() {
        if (AppConfig.haptics) {
            vibrate(haptic_model0())
        }
        PermissionChecker.checkAndRequestPermissions(this, false) {
            reloadDetectionPhase()
            startDetectionProcess()
        }
    }

    private fun showCameraError(reason: Int) {
        val monitor = PhoneStatusMonitor.getInstance()
        val errorDialog = ErrorDialogManager(monitor.currentActivity)
        errorDialog.setupDialog(reason)
        monitor.shutdownApp(errorDialog, monitor.currentContext)
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        return when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                handleBackClick()
                true
            }

            KeyEvent.KEYCODE_VOLUME_UP -> {
                if (remainingClassIndices.isNotEmpty() && showResult.value) {
                    handleNextClick()
                }
                true
            }

            else -> super.onKeyDown(keyCode, event)
        }
    }

    override fun onPause() {
        super.onPause()
        PhoneStatusMonitor.getInstance().stopMonitoring()
        motionMonitor.stopMonitoring();lightMonitor.stopMonitoring();turnFlashlightOff()
        batteryCheckRunning.value = false
        mainHandler.removeCallbacksAndMessages(null)
        updateHandler.removeCallbacksAndMessages(null)
        stopDetection.set(true)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraProvider?.unbindAll()
        cameraExecutor.shutdown()
    }
}

@Composable
fun FindMyObjectScreen(
    showResult: Boolean,
    resultBitmap: Bitmap?,
    hasMoreObjects: Boolean,
    showBatteryWarning: Boolean,
    onBackClick: () -> Unit,
    onNextClick: (() -> Unit)?,
    onCameraReady: (PreviewView) -> Unit,
    onBBoxResize: (Float) -> Unit,
    onTextResize: (Float) -> Unit
) {
    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val screenHeight = maxHeight
        val screenWidth = maxWidth

        if (!showResult) {
            // Detection phase - Camera + FPS Slider
            DetectionPhase(
                onCameraReady = onCameraReady
            )
        } else {
            // Result phase - Result image + Navigation
            ResultPhaseWithSettings(
                screenHeight = screenHeight, screenWidth = screenWidth,
                resultBitmap = resultBitmap,
                hasMoreObjects = hasMoreObjects,
                onBackClick = onBackClick,
                onNextClick = onNextClick,
                onBBoxResize = onBBoxResize,
                onTextResize = onTextResize
            )
        }

        // Battery warning notification
        AnimatedVisibility(
            visible = showBatteryWarning,
            enter = slideInVertically(initialOffsetY = { -it }),
            exit = slideOutVertically(targetOffsetY = { -it }),
            modifier = Modifier.align(Alignment.TopCenter)
        ) {
            BatteryWarningNotification()
        }
    }
}

@Composable
fun DetectionPhase(
    onCameraReady: (PreviewView) -> Unit
) {
    Box(modifier = Modifier.fillMaxSize()) {

        // Camera Preview
        AndroidView(
            factory = { context ->
                PreviewView(context).also { previewView ->
                    onCameraReady(previewView)
                }
            },
            modifier = Modifier.fillMaxSize()
        )

    }
}

@Composable
fun BatteryWarningNotification() {
    Column(
        modifier = Modifier
            .fillMaxWidth(0.8f)
            .background(
                color = colorResource(R.color.notification_white), // Semi-transparent black
                shape = RoundedCornerShape(28.dp)
            )
            .padding(horizontal = 24.dp, vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(
            imageVector = Icons.Filled.Warning,
            contentDescription = "Warning",
            modifier = Modifier.size(24.dp),
            tint = Color(0xFFFF9800)
        )

        Text(
            text = "The application detected an higher usage of battery than usual",
            color = colorResource(R.color.notification_text_gray),
            fontSize = Constants.STD_FONT_SIZE.sp,
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            textAlign = TextAlign.Center,
            lineHeight = 20.sp
        )
    }
}

@Composable
fun ResultPhaseWithSettings(
    screenHeight: Dp,
    screenWidth: Dp,
    resultBitmap: Bitmap?,
    hasMoreObjects: Boolean,
    onBackClick: () -> Unit,
    onNextClick: (() -> Unit)?,
    onBBoxResize: (Float) -> Unit,
    onTextResize: (Float) -> Unit
) {
    Box(modifier = Modifier.fillMaxSize()) {
        // Result image
        if (resultBitmap != null) {
            Image(
                bitmap = resultBitmap.asImageBitmap(),
                contentDescription = "Detection Result",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit
            )
        }

        // Settings Panel at bottom
        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
        ) {
            SettingsPanel(
                screenHeight = screenHeight,
                screenWidth = screenWidth,
                hasMoreObjects = hasMoreObjects,
                onBackClick = onBackClick,
                onNextClick = onNextClick,
                onBBoxResize = onBBoxResize,
                onTextResize = onTextResize
            )
        }
    }
}

@Composable
fun SettingsPanel(
    screenHeight: Dp,
    screenWidth: Dp,
    hasMoreObjects: Boolean,
    onBackClick: () -> Unit,
    onNextClick: (() -> Unit)?,
    onBBoxResize: (Float) -> Unit,
    onTextResize: (Float) -> Unit
) {
    var isExpanded by remember { mutableStateOf(false) }
    var currentSliderSection by remember { mutableIntStateOf(1) } // 1 = BBox, 2 = Text

    var bboxOffset by remember { mutableFloatStateOf(0f) }
    var textSizeRatio by remember { mutableFloatStateOf(Constants.TEXT_SIZE_WIDTH_SCREEN) }

    // Animated offsets for slide animations
    val navigationOffsetY by animateDpAsState(
        targetValue = if (isExpanded) screenHeight * 0.246f else 0.dp,
        animationSpec = tween(
            durationMillis = 250,
            delayMillis = if (isExpanded) 0 else 250,
            easing = FastOutSlowInEasing
        ),
        label = "navigation_offset"
    )

    val sliderOffsetY by animateDpAsState(
        targetValue = if (isExpanded) 0.dp else screenHeight * 0.246f,
        animationSpec = tween(
            durationMillis = 250,
            delayMillis = if (!isExpanded) 0 else 250,
            easing = FastOutSlowInEasing
        ),
        label = "slider_offset"
    )

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(screenHeight * 0.246f)
    ) {
        // ============================================
        // COLUMN 1: NAVIGATION + SETTINGS BUTTON
        // (No background, slides down when expanded)
        // ============================================
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
                .offset(y = navigationOffsetY)
                .padding(bottom = 43.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (hasMoreObjects) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(screenWidth * 0.08f)
                    ) {
                        BackArrowLargeFab(onClick = onBackClick)
                        NextArrowLargeFab(onClick = onNextClick as () -> Unit)
                    }
                }
            } else {
                Box(
                    modifier = Modifier.fillMaxWidth(),
                    contentAlignment = Alignment.Center
                ) {
                    BackArrowLargeFab(onClick = onBackClick)
                }
            }

            Spacer(modifier = Modifier.height(15.dp))

            // Settings button (triggers expansion)
            ExtendedFloatingActionButton(
                onClick = {
                    isExpanded = true
                    if (AppConfig.haptics) vibrate(haptic_model0())
                },
                containerColor = colorResource(R.color.std_purple),
                contentColor = Color.White,
                shape = RoundedCornerShape(
                    topStart = 0.dp,
                    topEnd = 0.dp,
                    bottomEnd = 16.dp,
                    bottomStart = 16.dp
                ),
                modifier = Modifier
                    .width(Constants.NAV_BUTTONS_WIDTH.dp * 2 + screenWidth * 0.08f)
                    .height(Constants.NAV_BUTTONS_HEIGHT.dp / 2)
            ) {
                Row(
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Filled.Settings,
                        contentDescription = "Settings",
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(5.dp))
                    Text(
                        text = "Adjust Detection",
                        fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                        fontFamily = robotoExtraBold
                    )
                }
            }
        }

        // ============================================
        // COLUMN 2: SLIDER PANEL
        // (Has background, slides up when expanded)
        // ============================================
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
                .offset(y = sliderOffsetY)
                .background(
                    color = colorResource(R.color.notification_button_white),
                    shape = RoundedCornerShape(topStart = 28.dp, topEnd = 28.dp)
                )
                .padding(bottom = 43.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Close button at top
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(screenHeight * 0.03f)
                    .clip(RoundedCornerShape(topStart = 28.dp, topEnd = 28.dp))
                    .background(colorResource(R.color.std_purple))
                    .clickable {
                        isExpanded = false
                        if (AppConfig.haptics) vibrate(haptic_model0())
                    },
                contentAlignment = Alignment.Center
            ) {
                Row(
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.KeyboardArrowDown,
                        contentDescription = "Collapse",
                        tint = Color.White,
                        modifier = Modifier.size(20.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Section selector tabs
            Row(
                modifier = Modifier
                    .fillMaxWidth(0.85f)
                    .height(screenHeight * 0.045f),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // BBox tab
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                        .clip(RoundedCornerShape(16.dp))
                        .background(
                            if (currentSliderSection == 1)
                                colorResource(R.color.std_cyan)
                            else
                                Color(0xFFDCD8E0)
                        )
                        .clickable {
                            currentSliderSection = 1
                            if (AppConfig.haptics) vibrate(haptic_model0())
                        },
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "Box Size",
                        color = if (currentSliderSection == 1)
                            Color.White
                        else
                            colorResource(R.color.std_purple),
                        fontSize = Constants.STD_FONT_SIZE.sp,
                        fontFamily = robotoExtraBold
                    )
                }

                // Text tab
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                        .clip(RoundedCornerShape(16.dp))
                        .background(
                            if (currentSliderSection == 2)
                                colorResource(R.color.std_cyan)
                            else
                                Color(0xFFDCD8E0)
                        )
                        .clickable {
                            currentSliderSection = 2
                            if (AppConfig.haptics) vibrate(haptic_model0())
                        },
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "Text Size",
                        color = if (currentSliderSection == 2)
                            Color.White
                        else
                            colorResource(R.color.std_purple),
                        fontSize = Constants.STD_FONT_SIZE.sp,
                        fontFamily = robotoExtraBold
                    )
                }
            }

            Spacer(modifier = Modifier.height(5.dp))

            // Animated slider content
            AnimatedContent(
                targetState = currentSliderSection,
                transitionSpec = {
                    slideInHorizontally(
                        initialOffsetX = { if (targetState > initialState) it else -it },
                        animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY)
                    ) togetherWith slideOutHorizontally(
                        targetOffsetX = { if (targetState > initialState) -it else it },
                        animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY)
                    )
                },
                label = "slider_section"
            ) { section ->
                when (section) {
                    1 -> {
                        // BBox size slider
                        BBoxSizeSlider(
                            bboxOffset = bboxOffset,
                            onBBoxChange = { newOffset ->
                                bboxOffset = newOffset
                                if (AppConfig.haptics) {
                                    vibrate(haptic_model0())
                                }
                                onBBoxResize(newOffset)
                            }
                        )
                    }

                    2 -> {
                        // Text size slider
                        TextSizeSlider(
                            textSizeRatio = textSizeRatio,
                            onTextChange = { newRatio ->
                                textSizeRatio = newRatio
                                if (AppConfig.haptics) {
                                    vibrate(haptic_model0())
                                }
                                onTextResize(newRatio)
                            }
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun BBoxSizeSlider(
    bboxOffset: Float,
    onBBoxChange: (Float) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth(0.85f)
            .padding(horizontal = 16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Current value display
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "+${(bboxOffset / Constants.BBOX_RESIZE_MAX * 100).toInt()} %",
                color = colorResource(R.color.std_cyan),
                fontSize = Constants.STD_SUBTITLE_SIZE.sp,
                fontFamily = robotoExtraBold
            )
        }

        CustomSlider(
            value = bboxOffset,
            onValueChange = onBBoxChange,
            valueRange = 0f..Constants.BBOX_RESIZE_MAX,
            steps = 0,
            thumbStyle = ThumbStyle.BAR,  // ROUND, BAR, or DOUBLE_BAR
            thumbColor = colorResource(R.color.std_purple),
            thumbWidth = 8.dp,
            thumbHeight = 55.dp,
            trackHeight = 20.dp,
            activeTrackColor = Color.White,
            inactiveTrackColor = Color.White,
            trackShadow = 5.dp,
            modifier = Modifier.fillMaxWidth(0.9f)
        )
    }
}

@SuppressLint("DefaultLocale")
@Composable
fun TextSizeSlider(
    textSizeRatio: Float,
    onTextChange: (Float) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth(0.85f)
            .padding(horizontal = 16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Current value display
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "+${((textSizeRatio - Constants.TEXT_SIZE_WIDTH_SCREEN) / Constants.TEXT_RESIZE_MAX * 100).toInt()} %",
                color = colorResource(R.color.std_cyan),
                fontSize = Constants.STD_SUBTITLE_SIZE.sp,
                fontFamily = robotoExtraBold
            )
        }

        CustomSlider(
            value = textSizeRatio,
            onValueChange = onTextChange,
            valueRange = Constants.TEXT_SIZE_WIDTH_SCREEN..Constants.TEXT_RESIZE_MAX,
            steps = 4,
            thumbStyle = ThumbStyle.BAR,  // ROUND, BAR, or DOUBLE_BAR
            thumbColor = colorResource(R.color.std_purple),
            thumbWidth = 8.dp,
            thumbHeight = 55.dp,
            trackHeight = 20.dp,
            activeTrackColor = Color.White,
            inactiveTrackColor = Color.White,
            trackShadow = 5.dp,
            modifier = Modifier.fillMaxWidth(0.75f),
            stepsColor = colorResource(R.color.purple_light)
        )
    }
}
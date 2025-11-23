@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.tabs.home.findmyobjects

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.KeyEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.tween
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.main.HomeActivity
import com.visionassist.appspace.activities.tabs.reports.EnvironmentReportsManagerKt
import com.visionassist.appspace.jetpack.design.BackArrowLargeFab
import com.visionassist.appspace.jetpack.design.CustomSlider
import com.visionassist.appspace.jetpack.design.NextArrowLargeFab
import com.visionassist.appspace.jetpack.design.ThumbStyle
import com.visionassist.appspace.models.classifier.YOLOClassifier
import com.visionassist.appspace.models.detector.DetectionResult
import com.visionassist.appspace.models.detector.YOLODetector
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.BackgroundTaskExecutor
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.ImageUtils
import com.visionassist.appspace.utils.PermissionChecker
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.robotoBold
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.robotoSemibold
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
    private val monitor = PhoneStatusMonitor.getInstance()
    private val detector: YOLODetector= monitor.modelManager.detector
    private val classifier: YOLOClassifier = monitor.modelManager.classifier

    // Detection data
    private lateinit var objectsToFind: MutableMap<Int, String>
    private lateinit var remainingClassIndices: MutableList<Int>

    // Camera
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService

    // State
    private val stopDetection = AtomicBoolean(false)
    private val resultsReady = AtomicBoolean(false)
    private val displayReady = AtomicBoolean(false)
    private var frameDelayMs = (1000/ Constants.DEFAULT_FPS) // Default 10 FPS

    // Results
    private var resultBitmap: Bitmap? = null
    private val threadCount = AtomicInteger(0)
    private var avgDetectorLatency = 0L
    private var avgClassifierLatency = 0L

    // Handlers
    private val mainHandler = Handler(Looper.getMainLooper())

    // Compose states
    private val showResult = mutableStateOf(false)
    private val currentFPS = mutableStateOf(Constants.DEFAULT_FPS)
    private val showBatteryWarning = mutableStateOf(false)
    private val avgBatteryMoreUsed = mutableStateOf(0f)
    private val batteryCheckRunning = mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        extractIntentData()
        cameraExecutor = Executors.newSingleThreadExecutor()

        setContent {
            FindMyObjectScreen(
                showResult = showResult.value,
                resultBitmap = resultBitmap,
                hasMoreObjects = remainingClassIndices.isNotEmpty(),
                currentFPS = currentFPS.value,
                showBatteryWarning = showBatteryWarning.value,
                onFPSChange = { fps ->
                    currentFPS.value = fps
                    if (AppConfig.haptics) {
                        vibrate(haptic_model0())
                    }
                    frameDelayMs = 1000 / fps
                },
                onBackClick = ::handleBackClick,
                onNextClick = if (remainingClassIndices.isNotEmpty()) ::handleNextClick else null,
                onCameraReady = { previewView ->
                        startCameraX(previewView)
                }
            )
        }
    }

    private fun extractIntentData() {
        val classIndices = intent.getIntArrayExtra(Constants.EXTRA_MATCHED_INDICES) ?: intArrayOf()
        val matchedWords = intent.getStringArrayExtra(Constants.EXTRA_SYNONYMS_WORDS) ?: arrayOf()

        objectsToFind = mutableMapOf()
        remainingClassIndices = mutableListOf()

        for (i in classIndices.indices) {
            objectsToFind.put(classIndices[i], matchedWords[i])
            remainingClassIndices.add(classIndices[i])
        }

        Log.d(TAG, "Objects to find (detector_class:synonym): $objectsToFind")
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

        imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .build()

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            cameraProvider?.unbindAll()
            cameraProvider?.bindToLifecycle(
                this, cameraSelector, preview, imageCapture
            )
            Log.d(TAG, "Camera bound")
        } catch (e: Exception) {
            Log.e(TAG, "Error binding camera", e)
        }
    }

    private fun startDetectionProcess() {
        // Start phone status monitoring
        monitor.startMonitoring(mainHandler)
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
        mainHandler.post(object : Runnable {
            override fun run() {
                if (stopDetection.get() ) {
                    // Stop loop and show results
                    showResults()
                    return
                }

                // Launch detection thread
                launchDetectionThread()

                // Schedule next iteration
                mainHandler.postDelayed(this, frameDelayMs.toLong())
            }
        })
    }

    private fun launchDetectionThread() {
        threadCount.incrementAndGet()

        BackgroundTaskExecutor.getInstance().executeAsync(
            {
                // Capture image
                val bitmap = captureImageSync() ?: return@executeAsync null

                var sceneClassId = -1
                var classifierLatency = 0L

                val classifierLatch = CountDownLatch(1)

                if (AppConfig.env_reports) {
                    BackgroundTaskExecutor.getInstance().executeAsync(
                        {
                            val classifierStart = System.currentTimeMillis()
                            sceneClassId = classifier.detectScene(bitmap)
                            classifierLatency = System.currentTimeMillis() - classifierStart
                            null
                        },
                        object : BackgroundTaskExecutor.TaskCallback<Any?> {
                            override fun onSuccess(result: Any?) {
                                classifierLatch.countDown()
                            }

                            override fun onError(e: Exception) {
                                Log.e(TAG, "Classifier error", e)
                                classifierLatch.countDown()
                            }
                        }
                    )
                } else {
                    classifierLatch.countDown() // Skip waiting if disabled
                }

                // Run detector
                val detectorStart = System.currentTimeMillis()
                val detectionResult = detector.detectObjects(bitmap)
                val detectorLatency = System.currentTimeMillis() - detectorStart

                // Check if found target objects
                val detectedClasses = detectionResult.classIndices
                val foundClasses = remainingClassIndices.filter { it in detectedClasses }

                if (foundClasses.isEmpty()) {
                    return@executeAsync ThreadResult(null, null, -1, detectorLatency, 0)
                }

                // Filter detection result
                val filteredResult = filterDetectionResult(detectionResult, foundClasses)

                // Draw bounding boxes
                val resultBmp = detector.drawDetections(bitmap, filteredResult)

                try {
                    val finished = classifierLatch.await(1, TimeUnit.SECONDS)
                    if (!finished) {
                        Log.w(TAG, "Classifier timeout after 3 seconds")
                    }
                } catch (e: InterruptedException) {
                    Log.e(TAG, "Wait interrupted", e)
                }

                // Now classifier is done, return result
                ThreadResult(
                    filteredResult,
                    resultBmp,
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

    private fun captureImageSync(): Bitmap? {
        return try {
            val captureResult = CompletableFuture<Bitmap?>()

            imageCapture?.takePicture(
                ContextCompat.getMainExecutor(this),
                object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(image: ImageProxy) {
                        try {
                            // Convert ImageProxy to Bitmap using ImageUtils
                            val bitmap = ImageUtils.imageProxyToBitmap(image)
                            captureResult.complete(bitmap)

                            if (bitmap != null) {
                                Log.d(TAG, "Image captured: ${bitmap.width}x${bitmap.height}")
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Error converting ImageProxy to Bitmap", e)
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
            null
        }
    }

    private fun filterDetectionResult(
        original: DetectionResult,
        foundClasses: List<Int>
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

                Log.d(TAG, "Filtered detection: $synonym (class $classIdx) with confidence ${originalConfidences[i]}")
            }
        }

        Log.d(TAG, "Filtered ${filteredLabels.size} detections from ${originalClassIndices.size} total")

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
                avgClassifierLatency = (avgClassifierLatency  + classifierLatency) /
                            (2)
            }
        }
    }

    private fun processFoundObjects(result: ThreadResult) {
        // Signal stop
        stopDetection.set(true)
        monitor.stopMonitoring()
        batteryCheckRunning.value = false

        // Remove found classes
        val foundSynonyms = mutableListOf<String>()
        result.detectionResult?.classIndices?.forEach { classIdx ->
            if (classIdx in remainingClassIndices) {
                foundSynonyms.add(objectsToFind[classIdx] ?: "")
                remainingClassIndices.remove(classIdx)
            }
        }

        // Write environment report
        if (AppConfig.env_reports) {
            val batteryUsageIncrease = 1f - avgBatteryMoreUsed.value
            EnvironmentReportsManagerKt.writeDetectionReport(
                this,
                result.sceneClassId,
                foundSynonyms,
                threadCount.get(),
                avgDetectorLatency,
                avgClassifierLatency,
                batteryUsageIncrease
            )
        }

        // Set result bitmap
        resultBitmap = result.bitmap

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
                mainHandler.postDelayed(this, 1000)
            }
        })
    }

    private fun handleBackClick() {
        finish()
        startActivity(Intent(this, HomeActivity::class.java))
    }

    private fun reloadDetectionPhase(){
        // Reset states
        showResult.value = false
        resultBitmap = null
        currentFPS.value= Constants.DEFAULT_FPS
        frameDelayMs=1000/currentFPS.value
    }

    private fun handleNextClick() {
        PermissionChecker.checkAndRequestPermissions(this, false) {
            reloadDetectionPhase()
            startDetectionProcess()
        }
    }

    override fun onResume() {
        super.onResume()
        if (PhoneStatusMonitor.getInstance().isReturningFromPermissions) {
            PermissionChecker.checkAndRequestPermissions(this, false) {
                reloadDetectionPhase()
                startDetectionProcess()
            }
        }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        return when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                handleBackClick()
                true
            }

            KeyEvent.KEYCODE_VOLUME_UP -> {
                if (remainingClassIndices.isNotEmpty()) {
                    handleNextClick()
                }
                true
            }

            else -> super.onKeyDown(keyCode, event)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopDetection.set(true)
        batteryCheckRunning.value = false
        monitor.stopMonitoring()
        cameraProvider?.unbindAll()
        cameraExecutor.shutdown()
    }

    data class ThreadResult(
        val detectionResult: DetectionResult?,
        val bitmap: Bitmap?,
        val sceneClassId: Int,
        val detectorLatency: Long,
        val classifierLatency: Long
    )
}

@Composable
fun FindMyObjectScreen(
    showResult: Boolean,
    resultBitmap: Bitmap?,
    hasMoreObjects: Boolean,
    currentFPS: Int,
    showBatteryWarning: Boolean,
    onFPSChange: (Int) -> Unit,
    onBackClick: () -> Unit,
    onNextClick: (() -> Unit)?,
    onCameraReady: (PreviewView) -> Unit
) {
    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val screenHeight=maxHeight
        val screenWidth=maxWidth

        if (!showResult) {
            // Detection phase - Camera + FPS Slider
            DetectionPhase(
                screenHeight=screenHeight,screenWidth=screenWidth,
                currentFPS = currentFPS,
                onFPSChange = onFPSChange,
                onCameraReady = onCameraReady
            )
        } else {
            // Result phase - Result image + Navigation
            ResultPhase(
                screenHeight=screenHeight,screenWidth=screenWidth,
                resultBitmap = resultBitmap,
                hasMoreObjects = hasMoreObjects,
                onBackClick = onBackClick,
                onNextClick = onNextClick
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
    screenHeight: Dp,screenWidth: Dp,
    currentFPS: Int,
    onFPSChange: (Int) -> Unit,
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

        // FPS Slider at bottom
        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
        ) {
            FPSSliderWithStops(
                screenHeight=screenHeight,
                currentFPS = currentFPS,
                onFPSChange = onFPSChange
            )
        }
    }
}

@Composable
fun FPSSliderWithStops(
    screenHeight: Dp,
    currentFPS: Int,
    onFPSChange: (Int) -> Unit
) {
    var fps by remember { mutableIntStateOf(currentFPS) }
    var isExpanded by remember { mutableStateOf(false) }

    // Animated height for smooth expansion
    val containerHeight by animateDpAsState(
        targetValue = if (isExpanded) screenHeight*0.18f else screenHeight*0.05f,
        animationSpec = tween(durationMillis = 250),
        label = "fps_container_height"
    )

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(containerHeight)
            .background(
                color = colorResource(R.color.notification_button_white), // Semi-transparent black
                shape = RoundedCornerShape(topStart = 28.dp, topEnd = 28.dp)
            )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // ============================================
            // CLICKABLE BAR (Always visible)
            // ============================================
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(screenHeight*0.05f)
                    .clickable { isExpanded = !isExpanded }
                    .padding(horizontal = 20.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Left side: FPS value
                Row(
                    horizontalArrangement = Arrangement.Start,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "$fps",
                        fontSize = Constants.STD_SUBTITLE_SIZE.sp,
                        color = colorResource(R.color.std_purple),
                        fontFamily = robotoExtraBold
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "FPS",
                        fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                        color = colorResource(R.color.std_purple),
                        fontFamily = robotoSemibold
                    )
                }

                // Right side: Label + Arrow icon
                Row(
                    horizontalArrangement = Arrangement.End,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Adjust the accuracy",
                        fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                        color = colorResource(R.color.std_cyan),
                        fontFamily = robotoSemibold
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Icon(
                        imageVector = if (isExpanded)
                            Icons.Default.KeyboardArrowDown
                        else
                            Icons.Default.KeyboardArrowUp,
                        contentDescription = if (isExpanded) "Collapse" else "Expand",
                        tint = colorResource(R.color.std_cyan),
                        modifier = Modifier.size(Constants.STD_INFO_BUTTON_SIZE.dp)
                    )
                }
            }

            var showText by remember { mutableStateOf(false) }
            // ============================================
            // EXPANDABLE SLIDER SECTION
            // ============================================
            AnimatedVisibility(
                visible = isExpanded,
                enter = slideInVertically(
                    initialOffsetY = { -it },
                    animationSpec = tween(250)
                ),
                exit = slideOutVertically(
                    targetOffsetY = { it },
                    animationSpec = tween(250)
                )
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 8.dp, bottom = 16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    // Slider with 6 stops: 10, 20, 30, 40, 50, 60
                    CustomSlider(
                        value = fps.toFloat(),
                        onValueChange = { newValue ->
                            val rounded = newValue.toInt()
                            if (rounded != fps && rounded in 10..60) {
                                fps = rounded
                                showText=fps>30
                                onFPSChange(fps)
                            }
                        },
                        valueRange = 10f..60f,
                        steps = 5, // 6 stops
                        thumbStyle = ThumbStyle.DOUBLE_BAR,
                        thumbColor = colorResource(R.color.std_purple),
                        thumbWidth = 8.dp,
                        thumbHeight = 55.dp,
                        thumbBarSpacing = 4.dp,
                        trackHeight = 25.dp,
                        activeTrackColor = Color.White,
                        inactiveTrackColor = Color.White,
                        trackShadow = 5.dp,
                        modifier = Modifier.fillMaxWidth(0.85f),
                        stepsColor = colorResource(R.color.purple_light)
                    )
                    // FPS range indicator
                    if(!showText) {
                        Text(
                            text = "Always prefer lower FPS for less battery consumption",
                            fontSize = Constants.STD_FONT_SIZE.sp,
                            color = colorResource(R.color.checked_green),
                            fontFamily = robotoBold
                        )
                    }
                    else
                        Text(
                            text = "Higher FPS will lead to higher battery consumption",
                            fontSize = Constants.STD_FONT_SIZE.sp,
                            color = colorResource(R.color.error_red),
                            fontFamily = robotoBold
                        )
                }
            }
        }
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
            .padding(horizontal=24.dp,vertical=12.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(
            imageVector = Icons.Filled.Warning,
            contentDescription = "Warning",
            modifier = Modifier.size(24.dp),
            tint = Color(0xFFFF9800)
        )

        Text(
            text = "The application detected an higher usage of battery than usual, you can solve this by lowering the FPS from the bottom section",
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
fun ResultPhase(
    screenHeight: Dp,screenWidth: Dp,
    resultBitmap: Bitmap?,
    hasMoreObjects: Boolean,
    onBackClick: () -> Unit,
    onNextClick: (() -> Unit)?
) {
    Box(modifier = Modifier.fillMaxSize()) {
        // Result image
        if (resultBitmap != null) {
            Image(
                bitmap = resultBitmap.asImageBitmap(),
                contentDescription = "Detection Result",
                modifier = Modifier.fillMaxSize()
            )
        }

        // Navigation buttons at bottom
        val bottomSpace = screenHeight * Constants.STD_NAV_MARGIN_BOTTOM
        if (hasMoreObjects) {
            // Navigation Buttons (not animated, always visible at bottom)
            Row(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = bottomSpace),
                horizontalArrangement = Arrangement.spacedBy(screenWidth * 0.08f),
            ) {
                BackArrowLargeFab(
                    onClick = onBackClick
                )

                NextArrowLargeFab(
                    onClick = onNextClick as () -> Unit,
                )
            }
        } else {
            // Back Button (not animated, always visible at bottom)
            Box(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = bottomSpace)
            ) {
                BackArrowLargeFab(
                    onClick = onBackClick
                )
            }
        }
    }
}

/*
@Preview(
    name = "FindMyObjectPreview Both Sections",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
@Composable
fun FindMyObjectPreview() {
    val imageWidth = 412f
    val imageHeight = 917f
    val mutableBitmap = createBitmap(imageWidth.toInt(), imageHeight.toInt())
    val canvas = Canvas(mutableBitmap)
    canvas.drawColor(android.graphics.Color.DKGRAY)
    FindMyObjectScreen(
        showResult = false,
        resultBitmap = null,
        hasMoreObjects = true,
        currentFPS = 30,
        showBatteryWarning = true,
        onFPSChange = {},
        onCameraReady = {},
        onNextClick = {},
        onBackClick = {}
    )
}
*/
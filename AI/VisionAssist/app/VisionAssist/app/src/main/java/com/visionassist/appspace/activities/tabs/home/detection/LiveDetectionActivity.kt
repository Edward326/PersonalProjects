@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.tabs.home.detection

import android.content.Intent
import android.graphics.Bitmap
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
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
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
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Warning
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
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.main.HomeActivity
import com.visionassist.appspace.activities.tabs.LightManager
import com.visionassist.appspace.activities.tabs.MotionManager
import com.visionassist.appspace.activities.tabs.home.findmyobjects.BBoxSizeSlider
import com.visionassist.appspace.activities.tabs.home.findmyobjects.TextSizeSlider
import com.visionassist.appspace.activities.tabs.reports.EnvironmentReportsManagerKt
import com.visionassist.appspace.activities.tabs.settings.BlockingOverlay
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

class LiveDetectionActivity : ComponentActivity() {
    private val TAG = "FindMyObjectActivity"

    // Models
    private lateinit var motionMonitor: MotionManager
    private lateinit var lightMonitor: LightManager
    private lateinit var currentDetectorModel: YOLODetectorPool
    private var preferNanoModel = false
    private lateinit var classifier: YOLOClassifier
    private var canSwitchModels = false

    // Results
    private val resultBitmap = mutableStateOf<Bitmap?>(null)
    private val threadCount = AtomicInteger(0)
    private val NOTHREADS_CLASSIFIER_DELAY = 3
    private var avgDetectorLatency = 0L
    private var avgClassifierLatency = 0L
    private val bitmapList = mutableListOf<Bitmap>()
    private val maxBitmapListSize = 10

    // Camera
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private var currentCamera: Camera? = null
    private var isFlashlightOn = false

    // Threads Control States
    private val stopDetection = AtomicBoolean(false)

    // Handlers
    private val mainHandler = Handler(Looper.getMainLooper())

    // Compose states
    private val showBatteryWarning = mutableStateOf(false)
    private val showMemoryWarning = mutableStateOf(false)

    // Monitor parameters
    private val avgBatteryMoreUsed = mutableStateOf(0f)
    private val batteryCheckRunning = mutableStateOf(false)

    // Resize bbox + text parameters
    private var currentBBoxOffset = 0f
    private var currentTextRatio = Constants.TEXT_SIZE_WIDTH_SCREEN

    data class ThreadResult(
        val classifierRan: Boolean,
        val detectionResult: DetectionResult?,
        val bitmap: Bitmap?,
        val sceneClassId: Int,
        val detectorLatency: Long,
        val classifierLatency: Long
    )

    private var quickActionIndex = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        WindowCompat.setDecorFitsSystemWindows(window, false)

        val intent = getIntent()
        quickActionIndex = intent.getIntExtra("QUICK_ACTION_INDEX", 0)

        classifier = if (AppConfig.env_reports)
            PhoneStatusMonitor.getInstance().modelManager.classifier
        else
            YOLOClassifier(this)

        cameraExecutor = Executors.newSingleThreadExecutor()

        currentDetectorModel = PhoneStatusMonitor.getInstance().modelManager.detector

        val tempAccModel = currentDetectorModel.acquireAccModel(5000)
        val tempNanoModel = currentDetectorModel.acquireNanoModel(5000)

        canSwitchModels= tempAccModel != null && tempNanoModel != null

        if(tempAccModel != null)
            currentDetectorModel.releaseAccModel(tempAccModel)
        if(tempNanoModel != null)
            currentDetectorModel.releaseNanoModel(tempNanoModel)

        motionMonitor = MotionManager(
            this,
            mainHandler
        ) {
            preferNanoModel =
                motionMonitor.linearSpeed > Constants.LINEAR_SPEED_THRESHOLD || motionMonitor.rotationSpeed > Constants.ROTATION_SPEED_THRESHOLD
        }

        lightMonitor = LightManager(
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
            LiveDetectionScreen(
                resultBitmap = resultBitmap.value,
                showBatteryWarning = showBatteryWarning.value,
                showMemoryWarning = showMemoryWarning.value,
                onCameraReady = { previewView ->
                    startCameraX(previewView)
                },
                onHomeClick = ::handleHomeClick,
                onBBoxResize = { newBBoxSize -> currentBBoxOffset = newBBoxSize },
                onTextResize = { newTextRatio -> currentTextRatio = newTextRatio }
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
            if (stopDetection.get()) {
                Log.d(TAG, "Activity resumed - restarting detection")
                reloadDetectionPhase()
                startDetectionProcess()
            }
        }
    }

    private fun turnFlashlightOn() {
        try {
            if (currentCamera?.cameraInfo?.hasFlashUnit() == true && !isFlashlightOn) {
                currentCamera?.cameraControl?.enableTorch(true)
                isFlashlightOn = true
                Log.d(TAG, "🔦 Flashlight turned ON via CameraX")
            } else {
                Log.w(
                    TAG,
                    "Cannot turn ON flashlight: hasFlash=${currentCamera?.cameraInfo?.hasFlashUnit()}, isOn=$isFlashlightOn"
                )
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

    private fun handleHomeClick() {
        if (AppConfig.haptics) {
            vibrate(haptic_model0())
        }

        if(quickActionIndex!=0){
            val intent = Intent(this, HomeActivity::class.java)
            startActivity(intent)
            finish()
        }
        else
            finish()
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
            currentCamera = cameraProvider?.bindToLifecycle(
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
            batteryCheckRunning.value = false; motionMonitor.stopMonitoring()
            lightMonitor.stopMonitoring(); turnFlashlightOff()
        }
        if (canSwitchModels && AppConfig.SoA)
            motionMonitor.startMonitoring()
        lightMonitor.startMonitoring()
        // Start battery level check
        batteryCheckRunning.value = true
        startBatteryLevelCheck(batteryCheckRunning, showBatteryWarning, avgBatteryMoreUsed)

        // Start detection loop
        stopDetection.set(false)
        startDetectionLoop()
    }

    private fun startDetectionLoop() {
        // Launch detection thread
        launchDetectionThread()

        mainHandler.post(object : Runnable {
            override fun run() {
                if (!bitmapList.isEmpty()) {
                    // Stop loop and show results
                    startDisplayLoop()
                    return
                }

                // Schedule next iteration
                mainHandler.postDelayed(this, 200)
            }
        })
    }

    private fun startDisplayLoop() {
        mainHandler.post(object : Runnable {
            override fun run() {
                if (stopDetection.get()) return

                val nextBitmap = getNextBitmapFromList()
                if (nextBitmap != null) {
                    resultBitmap.value?.recycle()
                    resultBitmap.value = nextBitmap
                }

                mainHandler.postDelayed(this, Constants.IMAGE_REFRESH_MS.toLong())
            }
        })
    }

    @Synchronized
    private fun addBitmapToList(bitmap: Bitmap) {
        bitmapList.add(bitmap)
        while (bitmapList.size > maxBitmapListSize) {
            val oldBitmap = bitmapList.removeAt(0)
            oldBitmap.recycle()
        }
        Log.d(TAG, "Bitmap added to list, size: ${bitmapList.size}")
    }

    private fun getNextBitmapFromList(): Bitmap? {
        return if (bitmapList.isNotEmpty()) {
            bitmapList.removeAt(0)
        } else {
            null
        }
    }

    private fun launchDetectionThread() {
        val toRun = {
            threadCount.incrementAndGet()
            if (!stopDetection.get()) {
                BackgroundTaskExecutor.getInstance().executeAsync(
                    {
                        // Capture image
                        val currentThreadNo = "Thread_" + threadCount.get()
                        val detectorWrapper = currentDetectorModel.acquireDetector(
                            preferNanoModel,  // Prefer nano if moving fast
                            5  // 5 second timeout
                        )
                        if (detectorWrapper == null)
                            return@executeAsync ThreadResult(false, null, null, -1, 0, 0)

                        val detector = detectorWrapper.detector

                        val bitmap = captureImageSync() ?: return@executeAsync null

                        var sceneClassId = -1
                        var classifierLatency = 0L
                        var classifierRan = false

                        val classifierLatch = CountDownLatch(1)

                        if (AppConfig.env_reports && (threadCount.get() % NOTHREADS_CLASSIFIER_DELAY == 0)) {
                            classifierRan = true
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
                        val foundObjects = detectionResult.classIndices.isNotEmpty()
                        currentDetectorModel.releaseDetector(detectorWrapper)

                        //val detectedClasses = detectionResult.classIndices
                        //val foundClasses = remainingClassIndices.filter { it in detectedClasses }

                        if (!foundObjects) {
                            return@executeAsync ThreadResult(
                                false,
                                null,
                                bitmap,
                                -1,
                                detectorLatency,
                                0
                            )
                        }

                        try {
                            val finished = classifierLatch.await(1, TimeUnit.SECONDS)
                            if (!finished) {
                                Log.w(TAG, currentThreadNo + "Classifier timeout after 3 seconds")
                            }
                        } catch (e: InterruptedException) {
                            Log.e(TAG, currentThreadNo + "Wait interrupted", e)
                        }

                        val bitmapWithDetections = YOLODetector.drawDetectionsWithSmartResize(
                            bitmap,
                            detectionResult,
                            currentBBoxOffset,
                            currentTextRatio,
                            resources.displayMetrics
                        )
                        bitmap.recycle()

                        // Now classifier is done, return result
                        ThreadResult(
                            classifierRan,
                            detectionResult,
                            bitmapWithDetections,
                            sceneClassId,
                            detectorLatency,
                            classifierLatency
                        )
                    },
                    object : BackgroundTaskExecutor.TaskCallback<ThreadResult?> {
                        override fun onSuccess(result: ThreadResult?) {
                            if (result == null || result.detectionResult == null) {
                                if (result?.bitmap != null) {
                                    addBitmapToList(result.bitmap)
                                }
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
                            addBitmapToList(result.bitmap!!)
                            // Write environment report
                            if (result.classifierRan) {
                                val listIndices =
                                    result.detectionResult.classIndices as List<Int>
                                val batteryUsageIncrease = if (avgBatteryMoreUsed.value > 0f) {
                                    1f - avgBatteryMoreUsed.value
                                } else {
                                    0f
                                }
                                EnvironmentReportsManagerKt.writeDetectionReport(
                                    PhoneStatusMonitor.getInstance().currentContext,
                                    result.sceneClassId, listIndices,
                                    threadCount.get(),
                                    avgDetectorLatency,
                                    avgClassifierLatency,
                                    batteryUsageIncrease
                                )
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
                    showMemoryWarning.value = false
                    mainHandler.post(toRun)
                } else {
                    showMemoryWarning.value = true
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
                            if (!stopDetection.get()) {
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

    private fun reloadDetectionPhase() {
        // Reset states
        resultBitmap.value?.recycle()
    }

    private fun showCameraError(reason: Int) {
        val monitor = PhoneStatusMonitor.getInstance()
        val errorDialog = ErrorDialogManager(monitor.currentActivity)
        errorDialog.setupDialog(reason)
        monitor.shutdownApp(errorDialog, monitor.currentContext)
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        return when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_UP -> {
                handleHomeClick()
                true
            }

            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                true
            }

            else -> super.onKeyDown(keyCode, event)
        }
    }

    override fun onPause() {
        super.onPause()
        PhoneStatusMonitor.getInstance().stopMonitoring()
        motionMonitor.stopMonitoring(); lightMonitor.stopMonitoring(); turnFlashlightOff()
        batteryCheckRunning.value = false
        mainHandler.removeCallbacksAndMessages(null)
        stopDetection.set(true)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraProvider?.unbindAll()
        cameraExecutor.shutdown()
    }
}

@Composable
fun LiveDetectionScreen(
    onCameraReady: (PreviewView) -> Unit,
    resultBitmap: Bitmap?,
    showBatteryWarning: Boolean,
    showMemoryWarning: Boolean,
    onHomeClick: () -> Unit,
    onBBoxResize: (Float) -> Unit,
    onTextResize: (Float) -> Unit
) {
    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val screenHeight = maxHeight

        Box(
            modifier = Modifier
                .fillMaxSize()
                .then(
                    if (showMemoryWarning) {
                        Modifier.clearAndSetSemantics { }  //  COMPLETELY REMOVE from tree!
                    } else {
                        Modifier
                    }
                )
        ) {
            if (resultBitmap == null) {
                // Detection phase - Camera + FPS Slider
                DetectionPhase(
                    onCameraReady = onCameraReady
                )
            } else {
                // Result phase - Result image + Navigation
                ResultPhaseWithSettings(
                    screenHeight = screenHeight,
                    resultBitmap = resultBitmap,
                    onHomeClick = onHomeClick,
                    onBBoxResize = onBBoxResize,
                    onTextResize = onTextResize
                )
            }

            // Battery warning notification
            AnimatedVisibility(
                visible = showBatteryWarning,
                enter = slideInVertically(initialOffsetY = { -it }),
                exit = slideOutVertically(targetOffsetY = { -it }),
                modifier = Modifier
                    .statusBarsPadding()
                    .align(Alignment.TopCenter)
            ) {
                BatteryWarningNotification()
            }
        }

        BlockingOverlay(showMemoryWarning)

        MemoryWarningNotification(showMemoryWarning)
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
fun MemoryWarningNotification(
    isVisible: Boolean
) {
    AnimatedVisibility(
        visible = isVisible,
        enter = fadeIn(
            initialAlpha = 0f,
            animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY)
        ),
        exit = fadeOut(
            targetAlpha = 0f,
            animationSpec = tween(durationMillis = 0)  // ← Instant exit, no glitch!
        )
    ) {
        // Full screen white overlay with 50% opacity
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Gray.copy(alpha = Constants.BACKGROUND_OPACITY)),
            contentAlignment = Alignment.Center
        ) {
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
                    text = if (AppConfig.mainLanguage.code == "en")
                        "At the moment memory usage is at maximum, this page will be paused, until sufficient memory is available again, you can wait or exit the page via the volume up button"
                    else
                        "În acest moment, utilizarea memoriei este la maxim, iar pagina curentă va fii întreruptă până când vor exista resurse disponibile suficiente.Puteți aștepta sau ieși din pagină folosind butonul de volume up",
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
    }
}

@Composable
fun ResultPhaseWithSettings(
    screenHeight: Dp,
    resultBitmap: Bitmap?,
    onHomeClick: () -> Unit,
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
                onHomeClick = onHomeClick,
                onBBoxResize = onBBoxResize,
                onTextResize = onTextResize
            )
        }
    }
}

@Composable
fun SettingsPanel(
    screenHeight: Dp,
    onHomeClick: () -> Unit,
    onBBoxResize: (Float) -> Unit,
    onTextResize: (Float) -> Unit
) {
    val navBarHeight = WindowInsets.navigationBars.getBottom(LocalDensity.current)
    val navBarHeightDp = with(LocalDensity.current) { navBarHeight.toDp() }

    var isExpanded by remember { mutableStateOf(false) }
    var currentSliderSection by remember { mutableIntStateOf(1) } // 1 = BBox, 2 = Text

    var bboxOffset by remember { mutableFloatStateOf(0f) }
    var textSizeRatio by remember { mutableFloatStateOf(Constants.TEXT_SIZE_WIDTH_SCREEN) }

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
        AnimatedVisibility(
            visible = !isExpanded,
            enter = slideInHorizontally(
                initialOffsetX = { it },
                animationSpec = tween(
                    durationMillis = 250,
                    delayMillis = 250,
                    easing = FastOutSlowInEasing
                )
            ),
            exit = slideOutHorizontally(
                targetOffsetX = { it },
                animationSpec = tween(
                    durationMillis = 250,
                    delayMillis = 0,
                    easing = FastOutSlowInEasing
                )
            ),
            modifier = Modifier
                .fillMaxWidth()
                .navigationBarsPadding()
                .align(Alignment.BottomEnd)  // ✅ Keep right alignment
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 43.dp),
                horizontalAlignment = Alignment.End
            ) {
                ButtonLive(
                    onHomeClick,
                    Icons.Filled.Home,
                    if (AppConfig.mainLanguage.code == "en") "Home" else "Αcasă"
                )

                Spacer(modifier = Modifier.height(15.dp))

                ButtonLive(
                    {
                        isExpanded = true
                        if (AppConfig.haptics) vibrate(haptic_model0())
                    },
                    Icons.Filled.Settings,
                    if (AppConfig.mainLanguage.code == "en") "Settings" else "Setări"
                )
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
                .padding(bottom = navBarHeightDp),
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
fun ButtonLive(onClick: () -> Unit, imageVector: ImageVector, contentDescription: String) {
    Box(
        modifier = Modifier
            .size(
                width = Constants.NAV_BUTTONS_WIDTH.dp * 0.7f,
                height = Constants.NAV_BUTTONS_HEIGHT.dp * 0.7f
            )
            .clip(RoundedCornerShape(16.dp))
            .background(colorResource(R.color.std_light_purple))
            .clickable {
                if (AppConfig.haptics) vibrate(haptic_model0())
                onClick()
            },
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Icon(
                imageVector = imageVector,
                contentDescription = contentDescription,
                tint = colorResource(R.color.std_purple),
                modifier = Modifier.size(40.dp)
            )
        }
    }
}
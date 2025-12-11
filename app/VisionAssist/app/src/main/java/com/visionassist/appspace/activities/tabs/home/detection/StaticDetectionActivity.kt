@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.tabs.home.detection

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.KeyEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.Settings
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
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import androidx.core.graphics.createBitmap
import androidx.core.net.toUri
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.main.HomeActivity
import com.visionassist.appspace.activities.tabs.home.findmyobjects.BBoxSizeSlider
import com.visionassist.appspace.activities.tabs.home.findmyobjects.TextSizeSlider
import com.visionassist.appspace.activities.tabs.reports.EnvironmentReportsManagerKt
import com.visionassist.appspace.jetpack.design.LoadingComponent
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager
import com.visionassist.appspace.jetpack.managers.InfoNotificationManager
import com.visionassist.appspace.models.classifier.YOLOClassifier
import com.visionassist.appspace.models.detector.DetectionResult
import com.visionassist.appspace.models.detector.YOLODetector
import com.visionassist.appspace.models.detector.YOLODetectorPool
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.BackgroundTaskExecutor
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.PermissionChecker
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.load_classificationSuccess
import com.visionassist.appspace.utils.load_noObjectsFound
import com.visionassist.appspace.utils.load_scanningScene
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.vibrate
import java.io.File
import java.io.IOException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import androidx.core.graphics.scale

class StaticDetectionActivity : ComponentActivity() {
    private val TAG = "StaticDetectionActivity"

    // Models
    private val currentDetectorModel: YOLODetectorPool =
        PhoneStatusMonitor.getInstance().modelManager.detector
    private lateinit var classifier: YOLOClassifier

    // Results
    private var originalBitmap: Bitmap? = null
    private val resultBitmap = mutableStateOf<Bitmap?>(null)
    private var detectionResult: DetectionResult? = null
    private var avgDetectorLatency = 0L
    private var avgClassifierLatency = 0L
    private var sceneClassId = -1

    // States
    private val displayReady = AtomicBoolean(false)

    // UI States
    private val showLoading = mutableStateOf(true)
    private val loadingText = mutableStateOf("")
    private val showResult = mutableStateOf(false)
    private val showClassificationDialog = mutableStateOf(false)
    private val classificationText = mutableStateOf("")

    // Handlers
    private val mainHandler = Handler(Looper.getMainLooper())
    private val updateHandler = Handler(Looper.getMainLooper())
    private var updateRunnable: Runnable? = null
    private lateinit var infoNotificationManager: InfoNotificationManager

    // Camera
    private lateinit var takePictureLauncher: ActivityResultLauncher<Uri>
    private var currentPhotoUri: Uri? = null

    private var currentBBoxOffset = 0f
    private var currentTextRatio = Constants.TEXT_SIZE_WIDTH_SCREEN

    // Thread result
    data class ThreadResult(
        val detectionResult: DetectionResult?,
        val bitmap: Bitmap?,
        val sceneClassId: Int,
        val detectorLatency: Long,
        val classifierLatency: Long,
        val errorType: Int
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        classifier = if (AppConfig.env_reports)
            PhoneStatusMonitor.getInstance().modelManager.classifier
        else
            YOLOClassifier(this)

        // Initialize notification manager
        infoNotificationManager = InfoNotificationManager(this)
        // Register camera launcher
        registerCameraLauncher()

        // Load initial loading text
        loadingText.value = load_scanningScene(this)

        setContent {
            StaticDetectionScreen(
                showLoading = showLoading.value,
                loadingText = loadingText.value,
                showResult = showResult.value,
                showClassificationDialog = showClassificationDialog.value,
                classificationText = classificationText.value,
                resultBitmap = resultBitmap.value,
                onHomeClick = ::handleHomeClick,
                onPhotoClick = ::handlePhotoClick,
                onBBoxResize = ::handleBBoxResize,
                onTextResize = ::handleTextResize
            )
        }

        mainHandler.postDelayed({
            // Start detection process
            val imageUriString = intent.getStringExtra(Constants.EXTRA_IMAGE_URI)
            if (imageUriString != null) {
                val imageUri = imageUriString.toUri()
                loadAndDetectImage(imageUri)
            } else {
                Log.e(TAG, "No image URI provided!")
                showError(Constants.INTENT_URI_IS_NULL)
            }
        }, 500)
    }

    @Suppress("DEPRECATION")
    private fun registerCameraLauncher() {
        takePictureLauncher = registerForActivityResult(
            ActivityResultContracts.TakePicture()
        ) { isSuccess ->
            if (isSuccess && currentPhotoUri != null) {
                try {
                    mainHandler.removeCallbacksAndMessages(null)

                    // Reset states
                    showResult.value = false
                    showLoading.value = true
                    displayReady.set(false)

                    originalBitmap?.recycle()
                    resultBitmap.value?.recycle()
                    originalBitmap = null
                    detectionResult = null

                    loadAndDetectImage(currentPhotoUri!!)
                } catch (e: IOException) {
                    Log.e(TAG, "Error loading new captured image", e)
                    showError(Constants.INTENT_URI_IS_NULL)
                }
            } else {
                Log.e(TAG, "Image capture failed or cancelled")
            }
        }
    }

    override fun onResume() {
        super.onResume()

        if (PhoneStatusMonitor.getInstance().isReturningFromPermissions) {
            PermissionChecker.checkAndRequestPermissions(this, false) {
                checkPhoneStatusAndNavigate {
                    launchCamera()
                }
            }
        }
    }

    private fun loadAndDetectImage(imageUri: Uri) {
        BackgroundTaskExecutor.getInstance().executeAsync(
            {
                try {
                    // Load bitmap from URI
                    val inputStream = contentResolver.openInputStream(imageUri)
                    val initBitmap = BitmapFactory.decodeStream(inputStream)
                    inputStream?.close()

                    if (initBitmap == null) {
                        Log.e(TAG, "Failed to decode bitmap from URI")
                        return@executeAsync null
                    }

                    Log.d(TAG, "Init image resolution: ${initBitmap.width}x${initBitmap.height}")
                    val bitmap = initBitmap.scale(
                        resources.displayMetrics.widthPixels,
                        resources.displayMetrics.heightPixels
                    )
                    Log.d(TAG, "After rescale image resolution: ${bitmap.width}x${bitmap.height}")

                    // Run detection
                    runDetection(bitmap)
                } catch (e: Exception) {
                    Log.e(TAG, "Error loading image", e)
                    return@executeAsync null
                }
            },
            object : BackgroundTaskExecutor.TaskCallback<ThreadResult?> {
                override fun onSuccess(result: ThreadResult?) {
                    if (result == null) {
                        showError(Constants.INTENT_URI_IS_NULL)
                        return
                    }

                    when (result.errorType) {
                        Constants.DETECTOR_AQUIRE_FAILED -> {
                            showError(Constants.DETECTOR_AQUIRE_FAILED)
                        }

                        Constants.DETECTOR_NO_OBJECTS_FOUND -> {
                            showNoObjectsFound()
                        }

                        else -> processFoundObjects(result)
                    }
                }

                override fun onError(e: Exception) {
                    Log.e(TAG, "Detection error", e)
                    showError(Constants.INTENT_URI_IS_NULL)
                }
            }
        )
        showResults()
    }

    private fun runDetection(bitmap: Bitmap): ThreadResult {
        // Acquire detector
        val detectorWrapper = currentDetectorModel.acquireDetector(
            false,  // Don't prefer nano for static detection
            5  // 5 second timeout
        )

        if (detectorWrapper == null) {
            Log.e(TAG, "No detector available")
            bitmap.recycle()
            return ThreadResult(
                null, null, -1, 0, 0,
                Constants.DETECTOR_AQUIRE_FAILED
            )
        }

        val detector = detectorWrapper.detector

        var sceneId = -1
        var classifierLatency = 0L

        // Run classifier if enabled
        val classifierLatch = CountDownLatch(1)

        if (AppConfig.env_reports) {
            BackgroundTaskExecutor.getInstance().executeAsync(
                {
                    val classifierStart = System.currentTimeMillis()
                    sceneId = classifier.detectScene(bitmap, "StaticDetectionActivity")
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
            classifierLatch.countDown()
        }

        // Run detector
        val detectorStart = System.currentTimeMillis()
        val detectionResult = detector.detectObjects(bitmap, "StaticDetection")
        val detectorLatency = System.currentTimeMillis() - detectorStart

        // Check if found any objects
        val foundObjects = detectionResult.classIndices.isNotEmpty()

        // Release detector
        currentDetectorModel.releaseDetector(detectorWrapper)

        if (!foundObjects) {
            bitmap.recycle()

            // Wait for classifier to finish
            try {
                classifierLatch.await(1, TimeUnit.SECONDS)
            } catch (e: InterruptedException) {
                Log.e(TAG, "Wait interrupted", e)
            }

            return ThreadResult(
                null, null, -1, detectorLatency, classifierLatency,
                Constants.DETECTOR_NO_OBJECTS_FOUND
            )
        }

        // Wait for classifier
        try {
            val finished = classifierLatch.await(1, TimeUnit.SECONDS)
            if (!finished) {
                Log.w(TAG, "Classifier timeout after 1 second")
            }
        } catch (e: InterruptedException) {
            Log.e(TAG, "Wait interrupted", e)
        }

        return ThreadResult(
            detectionResult,
            bitmap,
            sceneId,
            detectorLatency,
            classifierLatency,
            Constants.ANIMATION_DELAY
        )
    }

    private fun processFoundObjects(result: ThreadResult) {
        // Update latency stats
        updateLatencyStats(result.detectorLatency, result.classifierLatency)

        // Write environment report
        if (AppConfig.env_reports) {
            EnvironmentReportsManagerKt.writeDetectionReport(
                this,
                result.sceneClassId,
                result.detectionResult!!.classIndices as List<Int>,
                1,  // Single thread for static detection
                avgDetectorLatency,
                avgClassifierLatency,
                0f  // No battery usage tracking for static
            )
        }

        // Save results
        originalBitmap = result.bitmap
        detectionResult = result.detectionResult
        sceneClassId = result.sceneClassId

        resultBitmap.value = YOLODetector.drawDetections(
            originalBitmap,
            detectionResult
        )


        currentBBoxOffset = 0f
        currentTextRatio = Constants.TEXT_SIZE_WIDTH_SCREEN
        // Signal ready
        displayReady.set(true)
    }

    private fun updateLatencyStats(detectorLatency: Long, classifierLatency: Long) {
        avgDetectorLatency = if (avgDetectorLatency == 0L) {
            detectorLatency
        } else {
            (avgDetectorLatency + detectorLatency) / 2
        }

        if (classifierLatency > 0) {
            avgClassifierLatency = if (avgClassifierLatency == 0L) {
                classifierLatency
            } else {
                (avgClassifierLatency + classifierLatency) / 2
            }
        }
    }

    private fun showResults() {
        mainHandler.post(object : Runnable {
            override fun run() {
                if (displayReady.get()) {
                    // Success - show results
                    showLoading.value = false
                    mainHandler.postDelayed({
                        showResult.value = true
                        if (AppConfig.haptics) {
                            vibrate(haptic_model0())
                        }

                        // Show classification notification
                        if (AppConfig.env_reports && sceneClassId >= 0) {
                            val sceneName = classifier.getClassName(sceneClassId)
                            mainHandler.postDelayed({
                                showClassificationNotification(sceneName)
                            }, 500)
                        }
                    }, Constants.ANIMATION_DELAY.toLong())
                } else
                // Schedule next iteration
                {
                    mainHandler.postDelayed(this, 500)
                    Log.w(TAG, "Result not ready, waiting...")
                }
            }
        })
    }

    private fun showError(errorType: Int) {
        //displayReady.set(true)
        showLoading.value = false

        mainHandler.postDelayed(
            {
                showErrorDialog(errorType)
            }, Constants.ANIMATION_DELAY.toLong()
        )
    }

    private fun showErrorDialog(errorType: Int) {
        val monitor = PhoneStatusMonitor.getInstance()
        val errorDialog = ErrorDialogManager(monitor.currentActivity)
        errorDialog.setupDialog(errorType)
        monitor.shutdownApp(errorDialog, monitor.currentContext)
    }

    private fun showNoObjectsFound() {
        showLoading.value = false
        mainHandler.postDelayed({
            val message = load_noObjectsFound(this)

            // Show info dialog and exit after delay
            infoNotificationManager.showNotificationTwoButtons(
                message,
                if (AppConfig.mainLanguage.code == "en") "Retry" else "Reîncearcă",
                "OK",
                {
                    if (AppConfig.haptics) {
                        vibrate(haptic_model0())
                    }
                    infoNotificationManager.hideNotification()
                    PermissionChecker.checkAndRequestPermissions(this, false) {
                        // Check phone status
                        checkPhoneStatusAndNavigate {
                            launchCamera()
                        }
                    }
                },
                {
                    if (AppConfig.haptics) {
                        vibrate(haptic_model0())
                    }
                    infoNotificationManager.hideNotification()
                    val intent = Intent(this, HomeActivity::class.java)
                    startActivity(intent)
                    finish()
                },
            )
        }, Constants.ANIMATION_DELAY.toLong())
    }

    private fun showClassificationNotification(sceneName: String) {
        classificationText.value = load_classificationSuccess(sceneName)
        showClassificationDialog.value = true
        // Auto-hide after delay
        mainHandler.postDelayed({
            showClassificationDialog.value = false
        }, 4500)
    }

    private fun handleHomeClick() {
        if (AppConfig.haptics) {
            vibrate(haptic_model0())
        }

        val intent = Intent(this, HomeActivity::class.java)
        startActivity(intent)
        finish()
    }

    private fun handlePhotoClick() {
        if (AppConfig.haptics) {
            vibrate(haptic_model0())
        }

        // Check permissions first
        PermissionChecker.checkAndRequestPermissions(this, false) {
            // Check phone status
            checkPhoneStatusAndNavigate {
                launchCamera()
            }
        }
    }

    private fun checkPhoneStatusAndNavigate(onSuccess: () -> Unit) {
        PhoneStatusMonitor.getInstance().checkPhoneStatus()
        // If check passes, execute success callback
        onSuccess()
    }

    private fun launchCamera() {
        try {
            val photoFile = File.createTempFile(
                "temp_visionassist",
                ".jpg",
                cacheDir
            )

            currentPhotoUri = FileProvider.getUriForFile(
                this,
                "$packageName.fileprovider",
                photoFile
            )

            Log.d(TAG, "Launching camera with URI: $currentPhotoUri")

            takePictureLauncher.launch(currentPhotoUri!!)
        } catch (e: IOException) {
            Log.e(TAG, "Error creating temp file", e)
            showCameraError()
        }
    }

    private fun showCameraError() {
        val monitor = PhoneStatusMonitor.getInstance()
        val errorDialog = ErrorDialogManager(this)
        errorDialog.setupDialog(Constants.CAMERA_MAKE_PHOTO)
        monitor.shutdownApp(errorDialog, this)
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

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        return when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_UP -> {
                handleHomeClick()
                true
            }

            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                if (showResult.value) {
                    handlePhotoClick()
                }
                true
            }

            else -> super.onKeyDown(keyCode, event)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        originalBitmap?.recycle()
        resultBitmap.value?.recycle()
        mainHandler.removeCallbacksAndMessages(null)
        updateHandler.removeCallbacksAndMessages(null)
    }
}

@Composable
fun StaticDetectionScreen(
    showLoading: Boolean,
    loadingText: String,
    showResult: Boolean,
    showClassificationDialog: Boolean,
    classificationText: String,
    resultBitmap: Bitmap?,
    onHomeClick: () -> Unit,
    onPhotoClick: () -> Unit,
    onBBoxResize: (Float) -> Unit,
    onTextResize: (Float) -> Unit
) {
    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val screenHeight = maxHeight

        // Background: Black or Result Image
        Box(
            modifier = Modifier
                .fillMaxSize()
        ) {
            Image(
                painter = painterResource(R.drawable.app_background),
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
            if (showResult) {
                if (resultBitmap != null)
                    Image(
                        bitmap = resultBitmap.asImageBitmap(),
                        contentDescription = "Detection Result",
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Fit
                    )
            }
            // Loading overlay
            LoadingComponent(
                isVisible = showLoading,
                loadingText = loadingText,
                animSpec = Pair(
                    fadeIn(
                        initialAlpha = 0f,
                        animationSpec = tween(durationMillis = 0)
                    ),
                    fadeOut(
                        targetAlpha = 0f,
                        animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY)
                    )
                )
            )

            AnimatedVisibility(
                visible = showClassificationDialog,
                enter = slideInVertically(initialOffsetY = { -it }),
                exit = slideOutVertically(targetOffsetY = { -it }),
                modifier = Modifier.align(Alignment.TopCenter)
            ) {
                SceneClassifiedNotification(classificationText)
            }
        }

        // Bottom panel (navigation + settings)
        if (showResult) {
            Box(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
            ) {
                SettingsPanel(
                    screenHeight = screenHeight,
                    onHomeClick = onHomeClick,
                    onPhotoClick = onPhotoClick,
                    onBBoxResize = onBBoxResize,
                    onTextResize = onTextResize
                )
            }
        }
    }
}

@Composable
fun SceneClassifiedNotification(
    text: String
) {
    Column(
        modifier = Modifier
            .fillMaxWidth(0.8f)
            .background(
                color = colorResource(R.color.notification_white), // Semi-transparent black
                shape = RoundedCornerShape(28.dp)
            )
            .padding(horizontal = 24.dp, vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Icon(
            imageVector = Icons.Filled.Info,
            contentDescription = "Info",
            modifier = Modifier.size(24.dp),
            tint = colorResource(R.color.std_purple)
        )

        Text(
            text = buildAnnotatedString {
                val parts = text.split("~")
                withStyle(
                    style = SpanStyle(
                        fontSize = Constants.STD_FONT_SIZE.sp
                    )
                ) {
                    append(parts[0])
                }

                if (parts.size > 1) {
                    withStyle(
                        style = SpanStyle(
                            fontWeight = FontWeight.Bold,
                            fontSize = Constants.STD_FONT_SIZE.sp,
                            color = Color.Black
                        )
                    ) {
                        append(parts[1])
                    }
                }
            },
            fontSize = Constants.STD_FONT_SIZE.sp,
            color = colorResource(R.color.notification_text_gray),
            textAlign = TextAlign.Center,
            lineHeight = 20.sp
        )
    }
}

@Composable
fun SettingsPanel(
    screenHeight: Dp,
    onHomeClick: () -> Unit,
    onPhotoClick: () -> Unit,
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
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
                .offset(y = navigationOffsetY)
                .padding(bottom = 43.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Home + Photo buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Home button (left)
                HomeButton(onClick = onHomeClick, imageVector = Icons.Filled.Home)

                Spacer(Modifier.width(22.5.dp))

                // Photo button (center, larger)
                PhotoButton(onClick = onPhotoClick)

                Spacer(Modifier.width(22.5.dp))

                // Settings button (right)
                HomeButton(
                    onClick = {
                        isExpanded = true
                        if (AppConfig.haptics) vibrate(haptic_model0())
                    },
                    imageVector = Icons.Filled.Settings
                )
            }
        }

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
fun HomeButton(onClick: () -> Unit, imageVector: ImageVector) {
    Box(
        modifier = Modifier
            .size(
                width = Constants.NAV_BUTTONS_WIDTH.dp * 0.6f,
                height = Constants.NAV_BUTTONS_HEIGHT.dp * 0.8f
            )
            .clip(RoundedCornerShape(100))
            .background(colorResource(R.color.std_light_purple))
            .clickable {
                if (AppConfig.haptics) vibrate(haptic_model0())
                onClick()
            },
        contentAlignment = Alignment.Center
    ) {
        Icon(
            imageVector = imageVector,
            contentDescription = "Home",
            tint = colorResource(R.color.std_purple),
            modifier = Modifier.size(42.dp)
        )
    }
}

@Composable
fun PhotoButton(onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .size(
                width = Constants.NAV_BUTTONS_WIDTH.dp,
                height = Constants.NAV_BUTTONS_HEIGHT.dp
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
                imageVector = Icons.Filled.PhotoCamera,
                contentDescription = "Photo",
                tint = colorResource(R.color.std_purple),
                modifier = Modifier.size(58.dp)
            )
        }
    }
}

@Preview(name = "Static Detection Result", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun StaticDetectionScreenPreview() {
    val imageWidth = 412f
    val imageHeight = 917f
    val mutableBitmap = createBitmap(imageWidth.toInt(), imageHeight.toInt())
    val canvas = Canvas(mutableBitmap)
    canvas.drawColor(android.graphics.Color.BLACK)

    StaticDetectionScreen(
        showLoading = false,
        loadingText = "Scanning the scene...",
        showResult = true,
        resultBitmap = mutableBitmap,
        onHomeClick = {},
        onPhotoClick = {},
        onBBoxResize = {},
        onTextResize = {},
        classificationText = "",
        showClassificationDialog = false
    )
}
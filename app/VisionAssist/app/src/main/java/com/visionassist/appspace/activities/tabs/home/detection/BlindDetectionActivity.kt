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
import androidx.compose.foundation.Image
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.Dp
import androidx.core.content.ContextCompat
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.activities.main.BlindHomeActivity
import com.visionassist.appspace.activities.tabs.LightManager
import com.visionassist.appspace.activities.tabs.MotionManager
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager
import com.visionassist.appspace.models.detector.DetectionResult
import com.visionassist.appspace.models.detector.YOLODetector
import com.visionassist.appspace.models.detector.YOLODetectorPool
import com.visionassist.appspace.sound.SoundConstants
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.BackgroundTaskExecutor
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.ImageUtils
import com.visionassist.appspace.utils.PermissionChecker
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.load_speakNoObjectsFound
import com.visionassist.appspace.utils.startBatteryLevelCheck
import com.visionassist.appspace.utils.vibrate
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

class BlindDetectionActivity : ComponentActivity() {
    private val TAG = "FindMyObjectActivity"

    // Models
    private lateinit var motionMonitor: MotionManager
    private lateinit var lightMonitor: LightManager
    private lateinit var currentDetectorModel: YOLODetectorPool
    private var preferNanoModel = false
    private var canSwitchModels = false

    // Detection data
    private lateinit var objectsToFind: MutableMap<Int, String>
    private lateinit var remainingClassIndices: MutableList<Int>

    // Results
    private val resultBitmap = mutableStateOf<Bitmap?>(null)
    private var resultInText = mutableListOf<String>()
    private val threadCount = AtomicInteger(0)

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

    // Compose states
    private val showResult = mutableStateOf(false)
    private val showBatteryWarning = mutableStateOf(false)
    private var showMemoryWarning = false

    // Monitor parameters
    private val avgBatteryMoreUsed = mutableStateOf(0f)
    private val batteryCheckRunning = mutableStateOf(false)

    // Managers
    private val soundManager = PhoneStatusMonitor.getInstance()
        .soundManager
    private val ttsManager = PhoneStatusMonitor.getInstance()
        .ttsManager
    private var isSpeakingPhase = false
    private var currentSentenceIndex = 0
    private var locked = false

    data class ThreadResult(
        val detectionResult: DetectionResult?,
        val bitmap: Bitmap?
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
            BlindDetectionScreen(
                showResult = showResult.value,
                resultBitmap = resultBitmap.value,
                onResultReadyText = ::onResultReadyText,
                showBatteryWarning = showBatteryWarning.value,
                batteryWarningTrigger = ::handleBatteryWarning,
                onBackClick = ::handleBackClick,
                onCameraReady = { previewView ->
                    startCameraX(previewView)
                }
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

    private fun waitForTTSSpeech(afterTTSSpeech: Runnable) {
        val checkRunnable = object : Runnable {
            override fun run() {
                if (ttsManager.isDoneSpeaking) {
                    mainHandler.post(afterTTSSpeech)
                } else {
                    mainHandler.postDelayed(this, 350)
                }
            }
        }
        mainHandler.post(checkRunnable)
    }

    private fun onResultReadyText() {
        locked = true
        vibrate(haptic_model0())
        soundManager.play(SoundConstants.FIND_MY_OBJECT_DONE_ID, 0.7f, 0.7f) {
            val textToSpeak1 = load_speakNoObjectsFound(resultInText.lastIndex)
            isSpeakingPhase = true
            ttsManager.speak(
                textToSpeak1,
                AppConfig.tts_pitch,
                AppConfig.tts_speech_rate,
                true,
                null
            )

            waitForTTSSpeech {
                iterateAllSentences()
            }
        }
    }

    private fun iterateAllSentences() {
        ttsManager.speak(
            resultInText[currentSentenceIndex],
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            true,
            null
        )
        waitForTTSSpeech {
            currentSentenceIndex++
            if (currentSentenceIndex < resultInText.size)
                iterateAllSentences()
            else
                handleEndSpeakSentences()
        }
    }

    private fun handleEndSpeakSentences() {
        vibrate(haptic_model0())
        soundManager.play(SoundConstants.FIND_MY_OBJECT_DONE_ID, 0.7f, 0.7f) {
            currentSentenceIndex = 0
            isSpeakingPhase = false
            handleNextClick()
        }
    }

    private fun handleBatteryWarning() {
        ttsManager.speak(
            if (AppConfig.mainLanguage.code == "en")
                "The application detected an higher usage of battery than usual, to solve this you could return to home page"
            else
                "Aplicația a detectat o utilizare a bateriei mai mare decât de obicei, pentru a rezolva acest lucru puteți reveni la pagina principală",
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            false,
            haptic_model0()
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
        soundManager.play(SoundConstants.FIND_MY_OBJECT_STARTED_ID, 0.7f, 0.7f) {
            // Start phone status monitoring
            PhoneStatusMonitor.getInstance().startMonitoring(mainHandler) {
                resultsReady.set(true); batteryCheckRunning.value = false
                motionMonitor.stopMonitoring();lightMonitor.stopMonitoring();turnFlashlightOff()
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
    }

    private fun startDetectionLoop() {
        // Launch detection thread
        launchDetectionThread()

        mainHandler.post(object : Runnable {
            override fun run() {
                if (displayReady.get()) {
                    // Stop loop and show results
                    showResult.value = true
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
            if (!stopDetection.get()) {
                BackgroundTaskExecutor.getInstance().executeAsync(
                    {
                        val currentThreadNo = "Thread_" + threadCount.get()
                        val detectorWrapper = currentDetectorModel.acquireDetector(
                            preferNanoModel,  // Prefer nano if moving fast
                            5  // 5 second timeout
                        )
                        if (detectorWrapper == null)
                            return@executeAsync ThreadResult(null, null)

                        val detector = detectorWrapper.detector

                        val bitmap = captureImageSync() ?: return@executeAsync null

                        // Run detector
                        val detectionResult = detector.detectObjects(bitmap, currentThreadNo)

                        // Check if found target objects
                        val foundObjects = detectionResult.classIndices.isNotEmpty()
                        currentDetectorModel.releaseDetector(detectorWrapper)

                        if (!foundObjects) {
                            bitmap.recycle()
                            return@executeAsync ThreadResult(
                                null, null
                            )
                        }

                        val bitmapWithDetections = YOLODetector.drawDetections(
                            bitmap,
                            detectionResult,
                        )
                        bitmap.recycle()

                        // Now classifier is done, return result
                        ThreadResult(
                            detectionResult,
                            bitmapWithDetections
                        )
                    },
                    object : BackgroundTaskExecutor.TaskCallback<ThreadResult?> {
                        override fun onSuccess(result: ThreadResult?) {
                            if (result == null || result.bitmap == null)
                                return

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
                    if (showMemoryWarning) {
                        showMemoryWarning = false
                        val checkRunnable = object : Runnable {
                            override fun run() {
                                if (ttsManager.isDoneSpeaking) {
                                    mainHandler.post(toRun)
                                } else {
                                    Log.i(TAG, "Speaking process isn't finished")
                                    mainHandler.postDelayed(
                                        this,
                                        Constants.RETRY_TTS_DELAY_MS.toLong()
                                    )
                                }
                            }
                        }
                        val checkRunnable2 = object : Runnable {
                            override fun run() {
                                if (ttsManager.isDoneSpeaking) {
                                    val speakText =
                                        if (ttsManager.currentLocale.language == "en")
                                            "App has now sufficient memory to run, resuming detection process"
                                        else
                                            "Aplicația are acum suficientă memorie pentru a rula, se reia procesul de detecție"
                                    ttsManager.speak(
                                        speakText,
                                        AppConfig.tts_pitch,
                                        AppConfig.tts_speech_rate,
                                        false,
                                        null
                                    )
                                    mainHandler.post(checkRunnable)
                                } else {
                                    Log.i(TAG, "Speaking process isn't finished")
                                    mainHandler.postDelayed(
                                        this,
                                        Constants.RETRY_TTS_DELAY_MS.toLong()
                                    )
                                }
                            }
                        }
                        mainHandler.post(checkRunnable2)
                    } else
                        mainHandler.post(toRun)
                } else {
                    if (!showMemoryWarning) {
                        ttsManager.stopSpeaking()
                        showMemoryWarning = true
                        isSpeakingPhase = true
                        val speakText =
                            if (ttsManager.currentLocale.language == "en")
                                "At the moment memory usage is at maximum, this page will be paused, until sufficient memory is available again, you can wait or exit the page via the volume up button"
                            else
                                "În acest moment, utilizarea memoriei este la maxim, iar pagina curentă va fii întreruptă până când vor exista resurse disponibile suficiente.Puteți aștepta sau ieși din pagină folosind butonul de volume up"
                        ttsManager.speak(
                            speakText,
                            AppConfig.tts_pitch,
                            AppConfig.tts_speech_rate,
                            true,
                            null
                        )
                    }
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

    private fun processFoundObjects(result: ThreadResult) {
        // Signal stop
        stopDetection.set(true)
        PhoneStatusMonitor.getInstance().stopMonitoring()
        motionMonitor.stopMonitoring();lightMonitor.stopMonitoring();turnFlashlightOff()
        batteryCheckRunning.value = false

        // Set result bitmap
        resultBitmap.value = result.bitmap
        resultInText = generateSpatialDescriptions(
            result.bitmap!!.width,
            result.bitmap.height,
            result.detectionResult!!
        )

        // Signal display ready
        displayReady.set(true)
    }

    private fun handleBackClick() {
        soundManager.releaseCallback()
        ttsManager.stopSpeaking()
        ttsManager.speak(
            if(ttsManager.currentLocale.language=="en")
                "Returning to home page"
            else
                "Se revine în pagina principală"
            ,
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            false,
            haptic_model0()
        )
        waitForTTSSpeech {
            startActivity(Intent(this, BlindHomeActivity::class.java))
            finish()
        }
    }

    private fun reloadDetectionPhase() {
        // Reset states
        showResult.value = false
        resultBitmap.value?.recycle()
    }

    private fun handleNextClick() {
        //soundManager.releaseCallback()
        //ttsManager.stopSpeaking()

        reloadDetectionPhase()
        mainHandler.postDelayed({
            startDetectionProcess()
        }, 500)
    }

    private fun showCameraError(reason: Int) {
        val monitor = PhoneStatusMonitor.getInstance()
        val errorDialog = ErrorDialogManager(monitor.currentActivity)
        errorDialog.setupDialog(reason)
        monitor.shutdownApp(errorDialog, monitor.currentContext)
    }

    fun generateSpatialDescriptions(
        bitmapWidth: Int,
        bitmapHeight: Int,
        detectionResult: DetectionResult
    ): MutableList<String> {
        val descriptions = mutableListOf<String>()

        val boundingBoxes = detectionResult.boundingBoxes
        val labels = detectionResult.labels

        if (boundingBoxes.isEmpty() || labels.isEmpty()) {
            return descriptions
        }

        for (i in boundingBoxes.indices) {
            val bbox = boundingBoxes[i]
            val label = labels[i]

            // Calculate center point of bounding box
            val centerX = (bbox.left + bbox.right) / 2f
            val centerY = (bbox.top + bbox.bottom) / 2f

            // Generate spatial description
            val spatialDesc = getSpatialDescription(centerX, centerY, bitmapWidth, bitmapHeight)

            // Build complete sentence
            val description = buildDescriptionSentence(label, spatialDesc)
            descriptions.add(description)
        }
        return descriptions
    }

    private fun getSpatialDescription(
        centerX: Float,
        centerY: Float,
        width: Int,
        height: Int
    ): SpatialPosition {
        // Divide frame into thirds horizontally
        val horizontalPosition = when {
            centerX < width / 3f -> HorizontalPosition.LEFT
            centerX > 2 * width / 3f -> HorizontalPosition.RIGHT
            else -> HorizontalPosition.CENTER
        }

        // Divide frame into thirds vertically
        val verticalPosition = when {
            centerY < height / 3f -> VerticalPosition.TOP
            centerY > 2 * height / 3f -> VerticalPosition.BOTTOM
            else -> VerticalPosition.MIDDLE
        }

        return SpatialPosition(horizontalPosition, verticalPosition)
    }

    private fun buildDescriptionSentence(
        label: String,
        position: SpatialPosition,
    ): String {
        // Check current language
        return if (AppConfig.mainLanguage.code == "en") {
            // English
            val article = getArticleEnglish(label)
            val spatialPhrase = position.toPhrase()
            buildEnglishSentence(article, label, spatialPhrase)
        } else {
            // Romanian
            val article = getArticleRomanian(label)
            val spatialPhrase = position.toPhrase()
            buildRomanianSentence(article, label, spatialPhrase)
        }
    }

    private fun buildEnglishSentence(
        article: String,
        label: String,
        spatialPhrase: String
    ): String {
        // Format: "There is a bottle in the center" or "A person on your left"
        return when {
            spatialPhrase.contains("center") || spatialPhrase.contains("middle") -> {
                "There is $article $label $spatialPhrase"
            }

            else -> {
                "$article $label $spatialPhrase".capitalize()
            }
        }
    }

    private fun buildRomanianSentence(
        article: String,
        label: String,
        spatialPhrase: String
    ): String {
        // Format: "Este o sticlă în centru" or "O persoană în stânga ta"
        return when {
            spatialPhrase.contains("centru") || spatialPhrase.contains("centrală") -> {
                "Este $article $label $spatialPhrase"
            }

            else -> {
                "$article $label $spatialPhrase".capitalize()
            }
        }
    }

    private fun getArticleEnglish(label: String): String {
        // Check if word starts with vowel sound
        val vowels = setOf('a', 'e', 'i', 'o', 'u')
        return if (label.isNotEmpty() && label[0].lowercase()[0] in vowels) {
            "an"
        } else {
            "a"
        }
    }

    private fun getArticleRomanian(label: String): String {
        // Detect gender from Romanian word ending
        if (label.isEmpty()) return "un"

        val lastChar = label.last().lowercaseChar()
        val lastTwoChars = if (label.length >= 2) {
            label.takeLast(2).lowercase()
        } else {
            ""
        }

        // Check for feminine endings
        return when {
            // Feminine endings
            lastChar == 'ă' || lastTwoChars == "ie" || lastChar == 'e' -> "o"

            // Masculine/Neuter (default)
            else -> "un"
        }
    }

    private data class SpatialPosition(
        val horizontal: HorizontalPosition,
        val vertical: VerticalPosition
    ) {
        fun toPhrase(): String {
            return if (PhoneStatusMonitor.getInstance().ttsManager.currentLocale.language == "en")
                toPhraseEnglish()
            else
                toPhraseRomanian()
        }

        private fun toPhraseEnglish(): String {
            return when (horizontal) {
                HorizontalPosition.CENTER if vertical == VerticalPosition.MIDDLE ->
                    "in the center"

                HorizontalPosition.CENTER if vertical == VerticalPosition.TOP ->
                    "at the top center"

                HorizontalPosition.CENTER if vertical == VerticalPosition.BOTTOM ->
                    "at the bottom center"

                // Left positions
                HorizontalPosition.LEFT if vertical == VerticalPosition.MIDDLE ->
                    "on your left"

                HorizontalPosition.LEFT if vertical == VerticalPosition.TOP ->
                    "on your top left"

                HorizontalPosition.LEFT if vertical == VerticalPosition.BOTTOM ->
                    "on your bottom left"

                // Right positions
                HorizontalPosition.RIGHT if vertical == VerticalPosition.MIDDLE ->
                    "on your right"

                HorizontalPosition.RIGHT if vertical == VerticalPosition.TOP ->
                    "on your top right"

                HorizontalPosition.RIGHT if vertical == VerticalPosition.BOTTOM ->
                    "on your bottom right"

                else -> "in front of you"
            }
        }

        private fun toPhraseRomanian(): String {
            return when (horizontal) {
                HorizontalPosition.CENTER if vertical == VerticalPosition.MIDDLE ->
                    "în centru"

                HorizontalPosition.CENTER if vertical == VerticalPosition.TOP ->
                    "în partea de sus centrală"

                HorizontalPosition.CENTER if vertical == VerticalPosition.BOTTOM ->
                    "în partea de jos centrală"

                // Left positions
                HorizontalPosition.LEFT if vertical == VerticalPosition.MIDDLE ->
                    "în stânga ta"

                HorizontalPosition.LEFT if vertical == VerticalPosition.TOP ->
                    "în stânga sus"

                HorizontalPosition.LEFT if vertical == VerticalPosition.BOTTOM ->
                    "în stânga jos"

                // Right positions
                HorizontalPosition.RIGHT if vertical == VerticalPosition.MIDDLE ->
                    "în dreapta ta"

                HorizontalPosition.RIGHT if vertical == VerticalPosition.TOP ->
                    "în dreapta sus"

                HorizontalPosition.RIGHT if vertical == VerticalPosition.BOTTOM ->
                    "în dreapta jos"

                else -> "în fața ta"
            }
        }
    }

    private enum class HorizontalPosition {
        LEFT, CENTER, RIGHT
    }

    private enum class VerticalPosition {
        TOP, MIDDLE, BOTTOM
    }

    private fun String.capitalize(): String {
        return this.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        return when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_UP -> {
                handleBackClick()
                true
            }

            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                if (isSpeakingPhase)
                    ttsManager.onVolumeDownPressed()
                true
            }

            else -> super.onKeyDown(keyCode, event)
        }
    }

    override fun onPause() {
        super.onPause()
        soundManager.releaseCallback()
        ttsManager.stopSpeaking()
        PhoneStatusMonitor.getInstance()
            .stopMonitoring();lightMonitor.stopMonitoring();turnFlashlightOff()
        motionMonitor.stopMonitoring()
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
fun BlindDetectionScreen(
    showResult: Boolean,
    resultBitmap: Bitmap?,
    showBatteryWarning: Boolean,
    batteryWarningTrigger: () -> Unit,
    onResultReadyText: () -> Unit,
    onBackClick: () -> Unit,
    onCameraReady: (PreviewView) -> Unit
) {
    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val screenWidth = maxWidth

        if (!showResult) {
            // Detection phase - Camera + FPS Slider
            DetectionPhase(
                onCameraReady = onCameraReady
            )
        } else {
            // Result phase - Result image
            ResultPhaseWithoutSettings(
                screenWidth = screenWidth,
                resultBitmap = resultBitmap,
                onResultReadyText = onResultReadyText,
                onBackClick = onBackClick
            )
        }

        if (showBatteryWarning)
            batteryWarningTrigger()
    }
}

@Composable
fun ResultPhaseWithoutSettings(
    screenWidth: Dp,
    resultBitmap: Bitmap?,
    onResultReadyText: () -> Unit,
    onBackClick: () -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                var swipeStartX = 0f
                detectHorizontalDragGestures(
                    onDragStart = {
                        swipeStartX = 0f
                    },
                    onDragEnd = {
                        val threshold = (screenWidth * Constants.MIN_HDISTANCE_THRESHOLD).toPx()
                        when {
                            swipeStartX >= threshold -> {
                                onBackClick()
                            }
                        }
                        swipeStartX = 0f
                    },
                    onHorizontalDrag = { _, dragAmount ->
                        swipeStartX += dragAmount
                    }
                )
            }

    ) {
        // Result image
        if (resultBitmap != null) {
            Image(
                bitmap = resultBitmap.asImageBitmap(),
                contentDescription = "Detection Result",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit
            )
            onResultReadyText()
        }
    }
}
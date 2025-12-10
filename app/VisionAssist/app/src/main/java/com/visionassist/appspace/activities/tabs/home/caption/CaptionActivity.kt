@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.tabs.home.caption

import android.content.Intent
import android.graphics.Bitmap
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
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.main.HomeActivity
import com.visionassist.appspace.activities.tabs.reports.EnvironmentReportsManagerKt
import com.visionassist.appspace.jetpack.design.LoadingComponent
import com.visionassist.appspace.jetpack.managers.InfoNotificationManager
import com.visionassist.appspace.models.captioner.BLIPModel
import com.visionassist.appspace.models.captioner.Tokenizer
import com.visionassist.appspace.models.classifier.YOLOClassifier
import com.visionassist.appspace.models.translator.CaptionTranslator
import com.visionassist.appspace.sound.SoundConstants
import com.visionassist.appspace.utils.*
import java.io.File

class CaptionActivity : ComponentActivity() {
    private val TAG = "CaptionActivity"

    // Models
    private lateinit var captioner: BLIPModel
    private lateinit var tokenizer: Tokenizer
    private lateinit var classifier: YOLOClassifier
    private var translator: CaptionTranslator? = null

    // Image data
    private var originalBitmap: Bitmap? = null
    private var imageHash: String = ""

    // Results
    private var tokenIds: List<Int>? = null
    private var captionText: String = ""
    private var captionerLatency: Long = 0
    private var classifierLatency: Long = 0
    private var sceneClassId: Int = -1
    private var foundInCache: Boolean = false

    // UI States
    private val showLoading = mutableStateOf(true)
    private val loadingText = mutableStateOf("")
    private val showResult = mutableStateOf(false)
    private val showClassificationDialog = mutableStateOf(false)
    private val classificationText = mutableStateOf("")

    // Text size control
    private var currentTextSize = mutableFloatStateOf(24f)
    private val textSizes = listOf(16f, 20f, 24f, 28f, 32f, 36f, 40f, 44f)

    // Handlers
    private val mainHandler = Handler(Looper.getMainLooper())
    private lateinit var infoNotificationManager: InfoNotificationManager

    // Camera
    private lateinit var takePictureLauncher: ActivityResultLauncher<Uri>
    private var currentPhotoUri: Uri? = null

    // Settings panel
    private val showSettings = mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize models
        try {
            captioner = PhoneStatusMonitor.getInstance().modelManager.captioner
            tokenizer = captioner.tokenizer
            classifier = PhoneStatusMonitor.getInstance().modelManager.classifier

            // Initialize translator if Romanian
            if (AppConfig.mainLanguage.code == "ro") {
                translator = PhoneStatusMonitor.getInstance().modelManager.translator
            }

            Log.d(TAG, "Models initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize models", e)
            finish()
            return
        }

        // Initialize camera launcher
        takePictureLauncher = registerForActivityResult(
            ActivityResultContracts.TakePicture()
        ) { success ->
            if (success && currentPhotoUri != null) {
                processCapturedImage(currentPhotoUri!!)
            }
        }

        // Initialize info notification manager
        infoNotificationManager = InfoNotificationManager(this)

        // Set up UI
        setContent {
            CaptionScreen(
                showLoading = showLoading.value,
                loadingText = loadingText.value,
                showResult = showResult.value,
                showSettings = showSettings.value,
                showClassificationDialog = showClassificationDialog.value,
                classificationText = classificationText.value,
                captionText = captionText,
                captionerLatency = captionerLatency,
                foundInCache = foundInCache,
                currentTextSize = currentTextSize.floatValue,
                textSizes = textSizes,
                onTextSizeChange = { size -> currentTextSize.floatValue = size },
                onHomeClick = { handleHomeClick() },
                onCameraClick = { handleCameraClick() },
                onSpeakClick = { handleSpeakClick() },
                onSettingsClick = { showSettings.value = !showSettings.value },
                onClassificationDismiss = { showClassificationDialog.value = false }
            )
        }

        // Get image URI from intent
        val imageUriString = intent.getStringExtra(Constants.EXTRA_IMAGE_URI)
        if (imageUriString != null) {
            val imageUri = Uri.parse(imageUriString)
            processImageUri(imageUri)
        } else {
            Log.e(TAG, "No image URI provided")
            finish()
        }
    }

    private fun processImageUri(imageUri: Uri) {
        BackgroundTaskExecutor.getInstance().executeAsync(
            {
                // Load bitmap from URI
                val inputStream = contentResolver.openInputStream(imageUri)
                val bitmap = android.graphics.BitmapFactory.decodeStream(inputStream)
                inputStream?.close()

                if (bitmap == null) {
                    Log.e(TAG, "Failed to decode bitmap")
                    return@executeAsync null
                }

                Log.d(TAG, "Bitmap loaded: ${bitmap.width}x${bitmap.height}")
                bitmap
            },
            object : BackgroundTaskExecutor.TaskCallback<Bitmap?> {
                override fun onSuccess(result: Bitmap?) {
                    if (result != null) {
                        originalBitmap = result
                        startCaptionProcess(result)
                    } else {
                        finish()
                    }
                }

                override fun onError(e: Exception) {
                    Log.e(TAG, "Error loading bitmap", e)
                    finish()
                }
            }
        )
    }

    private fun processCapturedImage(imageUri: Uri) {
        showLoading.value = true
        loadingText.value = "Processing image..."
        showResult.value = false

        processImageUri(imageUri)
    }

    private fun startCaptionProcess(bitmap: Bitmap) {
        loadingText.value = "Computing image hash..."

        BackgroundTaskExecutor.getInstance().executeAsync(
            {
                // Step 1: Compute perceptual hash
                val hash = computePerceptualHash(bitmap)
                Log.d(TAG, "Image hash: $hash")
                imageHash = hash

                // Step 2: Search in hash cache file
                loadingText.value = "Searching cache..."
                val cachedTokens = searchHashCache(hash)

                cachedTokens
            },
            object : BackgroundTaskExecutor.TaskCallback<List<Int>?> {
                override fun onSuccess(result: List<Int>?) {
                    if (result != null) {
                        // Found in cache!
                        Log.d(TAG, "Caption found in cache: ${result.size} tokens")
                        foundInCache = true
                        tokenIds = result

                        // Decode tokens
                        val tokenArray = result.map { it.toLong() }.toLongArray()
                        captionText = tokenizer.decode(tokenArray)

                        // Run classifier if needed (for env reports)
                        if (AppConfig.env_reports) {
                            runClassifier(bitmap, onCacheHit = true)
                        } else {
                            // Translate if needed and show result
                            translateAndShowResult()
                        }
                    } else {
                        // Not found in cache - generate caption
                        Log.d(TAG, "Caption not found in cache, generating...")
                        foundInCache = false
                        generateCaption(bitmap)
                    }
                }

                override fun onError(e: Exception) {
                    Log.e(TAG, "Error in cache search", e)
                    // Fallback to generation
                    foundInCache = false
                    generateCaption(bitmap)
                }
            }
        )
    }

    private fun generateCaption(bitmap: Bitmap) {
        loadingText.value = "Generating caption..."

        // Launch captioner task
        BackgroundTaskExecutor.getInstance().executeAsync(
            {
                val startTime = System.currentTimeMillis()
                val tokens = captioner.generateCaption(bitmap)
                val latency = System.currentTimeMillis() - startTime

                Log.d(TAG, "Caption generated: ${tokens.size} tokens in ${latency}ms")

                Pair(tokens, latency)
            },
            object : BackgroundTaskExecutor.TaskCallback<Pair<List<Int>, Long>> {
                override fun onSuccess(result: Pair<List<Int>, Long>) {
                    tokenIds = result.first
                    captionerLatency = result.second

                    // Decode tokens
                    val tokenArray = result.first.map { it.toLong() }.toLongArray()
                    captionText = tokenizer.decode(tokenArray)

                    Log.d(TAG, "Caption text: $captionText")

                    // Save to hash cache
                    saveToHashCache(imageHash, result.first)

                    // Run classifier if needed
                    if (AppConfig.env_reports) {
                        runClassifier(bitmap!!, onCacheHit = false)
                    } else {
                        // Translate if needed and show result
                        translateAndShowResult()
                    }
                }

                override fun onError(e: Exception) {
                    Log.e(TAG, "Error generating caption", e)
                    captionText = "Failed to generate caption."
                    showResult.value = true
                    showLoading.value = false
                }
            }
        )
    }

    private fun runClassifier(bitmap: Bitmap, onCacheHit: Boolean) {
        loadingText.value = "Classifying scene..."

        BackgroundTaskExecutor.getInstance().executeAsync(
            {
                val startTime = System.currentTimeMillis()
                val classId = classifier.classifyEnvironment(bitmap)
                val latency = System.currentTimeMillis() - startTime

                Log.d(TAG, "Scene classified: $classId in ${latency}ms")

                Pair(classId, latency)
            },
            object : BackgroundTaskExecutor.TaskCallback<Pair<Int, Long>> {
                override fun onSuccess(result: Pair<Int, Long>) {
                    sceneClassId = result.first
                    classifierLatency = result.second

                    // Write to env reports
                    if (!onCacheHit) {
                        // Only write captioner latency if generated (not from cache)
                        EnvironmentReportsManagerKt.writeCaptionReport(
                            this@CaptionActivity,
                            sceneClassId,
                            captionerLatency,
                            classifierLatency
                        )
                    } else {
                        // Only write classifier latency if from cache
                        EnvironmentReportsManagerKt.writeCaptionReportCacheHit(
                            this@CaptionActivity,
                            sceneClassId,
                            classifierLatency
                        )
                    }

                    // Translate if needed and show result
                    translateAndShowResult()
                }

                override fun onError(e: Exception) {
                    Log.e(TAG, "Error classifying scene", e)
                    // Continue anyway
                    translateAndShowResult()
                }
            }
        )
    }

    private fun translateAndShowResult() {
        if (AppConfig.mainLanguage.code == "ro" && translator != null) {
            loadingText.value = "Translating..."

            BackgroundTaskExecutor.getInstance().executeAsync(
                {
                    translator!!.translate(captionText)
                },
                object : BackgroundTaskExecutor.TaskCallback<String?> {
                    override fun onSuccess(result: String?) {
                        if (result != null) {
                            captionText = result
                            Log.d(TAG, "Translated caption: $result")
                        }
                        showResultScreen()
                    }

                    override fun onError(e: Exception) {
                        Log.e(TAG, "Error translating", e)
                        // Show English caption anyway
                        showResultScreen()
                    }
                }
            )
        } else {
            showResultScreen()
        }
    }

    private fun showResultScreen() {
        showLoading.value = false
        showResult.value = true

        // Show classification notification if scene was classified
        if (sceneClassId >= 0) {
            mainHandler.postDelayed({
                classificationText.value = "Scene: ${classifier.getClassName(sceneClassId)}"
                showClassificationDialog.value = true

                // Auto-hide after 3 seconds
                mainHandler.postDelayed({
                    showClassificationDialog.value = false
                }, 3000)
            }, 500)
        }
    }

    // Navigation handlers
    private fun handleHomeClick() {
        if (AppConfig.haptics) vibrate(haptic_model0())
        soundManager.play(SoundConstants.BACK_ID, 0.7f, 0.7f) {
            finish()
            startActivity(Intent(this, HomeActivity::class.java))
        }
    }

    private fun handleCameraClick() {
        if (AppConfig.haptics) vibrate(haptic_model0())

        try {
            val photoFile = File(cacheDir, "caption_${System.currentTimeMillis()}.jpg")
            currentPhotoUri = FileProvider.getUriForFile(
                this,
                "${packageName}.fileprovider",
                photoFile
            )
            takePictureLauncher.launch(currentPhotoUri!!)
        } catch (e: Exception) {
            Log.e(TAG, "Error launching camera", e)
        }
    }

    private fun handleSpeakClick() {
        if (AppConfig.haptics) vibrate(haptic_model0())
        ttsManager.speak(
            captionText,
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            false,
            haptic_model0()
        )
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        return when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_UP -> {
                // Go home
                handleHomeClick()
                true
            }
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                // Take another photo (only when result is shown)
                if (showResult.value) {
                    handleCameraClick()
                }
                true
            }
            else -> super.onKeyDown(keyCode, event)
        }
    }

    override fun onPause() {
        super.onPause()
        soundManager.releaseCallback()
        ttsManager.stopSpeaking()
        mainHandler.removeCallbacksAndMessages(null)
    }

    override fun onDestroy() {
        super.onDestroy()
        originalBitmap?.recycle()
    }
}

@Composable
fun CaptionScreen(
    showLoading: Boolean,
    loadingText: String,
    showResult: Boolean,
    showSettings: Boolean,
    showClassificationDialog: Boolean,
    classificationText: String,
    captionText: String,
    captionerLatency: Long,
    foundInCache: Boolean,
    currentTextSize: Float,
    textSizes: List<Float>,
    onTextSizeChange: (Float) -> Unit,
    onHomeClick: () -> Unit,
    onCameraClick: () -> Unit,
    onSpeakClick: () -> Unit,
    onSettingsClick: () -> Unit,
    onClassificationDismiss: () -> Unit
) {
    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val screenWidth = maxWidth
        val screenHeight = maxHeight

        // Background
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(colorResource(R.color.app_background))
        )

        // Loading screen
        if (showLoading) {
            LoadingComponent(
                text = loadingText,
                showText = true
            )
        }

        // Result screen
        if (showResult) {
            Column(
                modifier = Modifier.fillMaxSize(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Caption section
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                        .background(Color.White)
                ) {
                    // Purple overlay (60% alpha)
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(colorResource(R.color.std_purple).copy(alpha = 0.6f))
                    )

                    // Caption content
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(24.dp),
                        verticalArrangement = Arrangement.Center,
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        // Caption text
                        Text(
                            text = captionText,
                            fontSize = currentTextSize.sp,
                            color = Color.White,
                            textAlign = TextAlign.Center,
                            fontFamily = robotoExtraBold,
                            modifier = Modifier
                                .fillMaxWidth()
                                .weight(1f)
                                .padding(16.dp),
                            lineHeight = (currentTextSize * 1.3f).sp
                        )

                        // Inference time and cache indicator
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 16.dp, vertical = 8.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            // Inference time box
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(16.dp))
                                    .background(Color.White.copy(alpha = 0.3f))
                                    .padding(horizontal = 16.dp, vertical = 8.dp)
                            ) {
                                Text(
                                    text = "${captionerLatency}ms",
                                    fontSize = 16.sp,
                                    color = Color.White,
                                    fontFamily = robotoExtraBold
                                )
                            }

                            // Lightning bolt icon (cache hit indicator)
                            if (foundInCache) {
                                Box(
                                    modifier = Modifier
                                        .size(40.dp)
                                        .clip(CircleShape)
                                        .background(colorResource(R.color.std_green)),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.Bolt,
                                        contentDescription = "Found in cache",
                                        tint = Color.White,
                                        modifier = Modifier.size(24.dp)
                                    )
                                }
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                // Text size slider
                TextSizeSlider(
                    screenWidth = screenWidth,
                    textSizes = textSizes,
                    currentTextSize = currentTextSize,
                    onTextSizeChange = onTextSizeChange
                )

                Spacer(modifier = Modifier.height(16.dp))

                // Navigation buttons
                NavigationSection(
                    onHomeClick = onHomeClick,
                    onCameraClick = onCameraClick,
                    onSpeakClick = onSpeakClick,
                    onSettingsClick = onSettingsClick
                )

                Spacer(modifier = Modifier.height(32.dp))
            }
        }

        // Classification notification
        AnimatedVisibility(
            visible = showClassificationDialog,
            enter = fadeIn() + slideInVertically(initialOffsetY = { -it }),
            exit = fadeOut() + slideOutVertically(targetOffsetY = { -it })
        ) {
            ClassificationNotification(
                text = classificationText,
                onDismiss = onClassificationDismiss
            )
        }

        // Settings panel
        AnimatedVisibility(
            visible = showSettings,
            enter = slideInHorizontally(initialOffsetX = { it }),
            exit = slideOutHorizontally(targetOffsetX = { it })
        ) {
            SettingsPanel(
                onClose = { onSettingsClick() }
            )
        }
    }
}

@Composable
fun NavigationSection(
    onHomeClick: () -> Unit,
    onCameraClick: () -> Unit,
    onSpeakClick: () -> Unit,
    onSettingsClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 32.dp),
        horizontalArrangement = Arrangement.SpaceEvenly,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Home button (left)
        NavigationButton(
            icon = Icons.Default.Home,
            contentDescription = "Home",
            onClick = onHomeClick,
            size = 56.dp,
            iconSize = 28.dp
        )

        // Camera button (center - larger)
        NavigationButton(
            icon = Icons.Default.PhotoCamera,
            contentDescription = "Camera",
            onClick = onCameraClick,
            size = 72.dp,
            iconSize = 36.dp,
            backgroundColor = colorResource(R.color.std_purple)
        )

        // Speak button (right)
        NavigationButton(
            icon = Icons.Default.VolumeUp,
            contentDescription = "Speak",
            onClick = onSpeakClick,
            size = 56.dp,
            iconSize = 28.dp
        )
    }
}

@Composable
fun NavigationButton(
    icon: ImageVector,
    contentDescription: String,
    onClick: () -> Unit,
    size: Dp,
    iconSize: Dp,
    backgroundColor: Color = colorResource(R.color.std_green)
) {
    Box(
        modifier = Modifier
            .size(size)
            .clip(CircleShape)
            .background(backgroundColor)
            .pointerInput(Unit) {
                detectTapGestures(
                    onTap = { onClick() }
                )
            },
        contentAlignment = Alignment.Center
    ) {
        Icon(
            imageVector = icon,
            contentDescription = contentDescription,
            tint = Color.White,
            modifier = Modifier.size(iconSize)
        )
    }
}

@Composable
fun ClassificationNotification(
    text: String,
    onDismiss: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(colorResource(R.color.std_green))
            .padding(16.dp)
            .pointerInput(Unit) {
                detectTapGestures(onTap = { onDismiss() })
            }
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Info,
                contentDescription = "Info",
                tint = Color.White,
                modifier = Modifier.size(24.dp)
            )

            Text(
                text = text,
                fontSize = 16.sp,
                color = Color.White,
                fontFamily = robotoExtraBold,
                modifier = Modifier.weight(1f)
            )
        }
    }
}

@Composable
fun SettingsPanel(
    onClose: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxHeight()
            .width(300.dp)
            .background(colorResource(R.color.app_background))
            .padding(16.dp)
    ) {
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Settings",
                    fontSize = 24.sp,
                    color = Color.White,
                    fontFamily = robotoExtraBold
                )

                Box(
                    modifier = Modifier
                        .size(40.dp)
                        .clip(CircleShape)
                        .background(colorResource(R.color.std_purple))
                        .pointerInput(Unit) {
                            detectTapGestures(onTap = { onClose() })
                        },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Close,
                        contentDescription = "Close",
                        tint = Color.White,
                        modifier = Modifier.size(24.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Settings content
            Text(
                text = "Caption settings coming soon...",
                fontSize = 16.sp,
                color = Color.White.copy(alpha = 0.7f),
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )
        }
    }
}

// Text Size Slider (reuse from StaticDetectionActivity or create simple version)
@Composable
fun TextSizeSlider(
    screenWidth: Dp,
    textSizes: List<Float>,
    currentTextSize: Float,
    onTextSizeChange: (Float) -> Unit
) {
    val currentIndex = textSizes.indexOf(currentTextSize).coerceAtLeast(0)

    Box(
        modifier = Modifier
            .width(screenWidth * 0.8f)
            .height(60.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(Color.White.copy(alpha = 0.1f))
            .padding(8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxSize(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            textSizes.forEachIndexed { index, size ->
                val isSelected = index == currentIndex

                Box(
                    modifier = Modifier
                        .size(if (isSelected) 40.dp else 32.dp)
                        .clip(CircleShape)
                        .background(
                            if (isSelected)
                                colorResource(R.color.std_purple)
                            else
                                Color.White.copy(alpha = 0.3f)
                        )
                        .pointerInput(Unit) {
                            detectTapGestures(
                                onTap = {
                                    if (AppConfig.haptics) vibrate(haptic_model0())
                                    onTextSizeChange(size)
                                }
                            )
                        },
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "A",
                        fontSize = (size / 2).sp,
                        color = if (isSelected) Color.White else Color.Gray,
                        fontFamily = robotoExtraBold
                    )
                }
            }
        }
    }
}
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
import androidx.activity.compose.setContent
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import androidx.core.net.toUri
import com.visionassist.appspace.BaseActivity
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.main.HomeActivity
import com.visionassist.appspace.activities.newprofile.LoadProfileActivity
import com.visionassist.appspace.activities.tabs.home.caption.PerceptualHash.computeHash
import com.visionassist.appspace.activities.tabs.home.detection.SceneClassifiedNotification
import com.visionassist.appspace.activities.tabs.reports.EnvironmentReportsManagerKt
import com.visionassist.appspace.jetpack.design.CustomSlider
import com.visionassist.appspace.jetpack.design.LoadingComponent
import com.visionassist.appspace.jetpack.design.NotificationDialog
import com.visionassist.appspace.jetpack.design.ThumbStyle
import com.visionassist.appspace.models.classifier.YOLOClassifier
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.BackgroundTaskExecutor
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.load_classificationSuccess
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.saveToHashCache
import com.visionassist.appspace.utils.searchHashCache
import com.visionassist.appspace.utils.vibrate
import java.io.File

class CaptionActivity : BaseActivity() {
    private val TAG = "CaptionActivity"

    // Models
    private val captioner = PhoneStatusMonitor.getInstance().modelManager.captioner
    private val tokenizer = captioner.tokenizer
    private lateinit var classifier: YOLOClassifier
    private var translator = PhoneStatusMonitor.getInstance().modelManager.translator
    private val ttsManager = PhoneStatusMonitor.getInstance()
        .ttsManager

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
    private val showErrorDialog = mutableStateOf(false)
    private val showClassificationDialog = mutableStateOf(false)
    private val errorMessage = mutableStateOf("")
    private val classificationText = mutableStateOf("")

    // Text size control (8 stops)
    private var currentTextSize = mutableFloatStateOf(28f)
    private val textSizes = listOf(20f, 24f, 28f, 32f, 36f, 40f, 44f)

    // Handlers
    private val mainHandler = Handler(Looper.getMainLooper())

    // Camera
    private lateinit var takePictureLauncher: ActivityResultLauncher<Uri>
    private var currentPhotoUri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        classifier = if (AppConfig.env_reports)
            PhoneStatusMonitor.getInstance().modelManager.classifier
        else
            YOLOClassifier(this)

        // Initialize camera launcher
        takePictureLauncher = registerForActivityResult(
            ActivityResultContracts.TakePicture()
        ) { success ->
            if (success && currentPhotoUri != null) {
                processCapturedImage(currentPhotoUri!!)
            }
        }

        // Set up UI
        setContent {
            CaptionScreen(
                showLoading = showLoading.value,
                loadingText = loadingText.value,
                showResult = showResult.value,
                showErrorDialog = showErrorDialog.value,
                errorMessage = errorMessage.value,
                showClassificationDialog = showClassificationDialog.value,
                classificationText = classificationText.value,
                captionText = captionText,
                captionerLatency = captionerLatency,
                foundInCache = foundInCache,
                currentTextSize = currentTextSize.floatValue,
                textSizes = textSizes,
                onTextSizeChange = { size ->
                    currentTextSize.floatValue = size
                    if (AppConfig.haptics) vibrate(haptic_model0())
                },
                onHomeClick = { handleHomeClick() },
                onCameraClick = { handleCameraClick() },
                onSpeakClick = { handleSpeakClick() },
                onErrorRetry = { handleRetry() },
                onErrorOk = { handleHomeClick() }
            )
        }


        mainHandler.postDelayed({
            // Get image URI from intent
            val imageUriString = intent.getStringExtra(Constants.EXTRA_IMAGE_URI)
            if (imageUriString != null) {
                val imageUri = imageUriString.toUri()
                processImageUri(imageUri)
            } else {
                Log.e(TAG, "❌ No image URI provided")
                finish()
            }
        }, 500)
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
                        errorMessage.value = if (AppConfig.mainLanguage.code == "en")
                            "Failed to generate caption. You can try again or return to home page"
                        else
                            "Nu s-a reușit generarea descrierii textuale. Poți încearca din nou sau poți să revii în pagina principala"
                        showLoading.value = false
                        mainHandler.postDelayed({
                            showErrorDialog.value = true
                        }, Constants.ANIMATION_DELAY.toLong())
                    }
                }

                override fun onError(e: Exception) {
                    Log.e(TAG, "Error loading bitmap", e)
                    errorMessage.value = if (AppConfig.mainLanguage.code == "en")
                        "Failed to generate caption. You can try again or return to home page"
                    else
                        "Nu s-a reușit generarea descrierii textuale. Poți încearca din nou sau poți să revii în pagina principala"
                    showLoading.value = false
                    mainHandler.postDelayed({
                        showErrorDialog.value = true
                    }, Constants.ANIMATION_DELAY.toLong())
                }
            }
        )
    }

    private fun processCapturedImage(imageUri: Uri) {
        // Reset states
        showLoading.value = true
        showResult.value = false
        showErrorDialog.value = false
        loadingText.value = if (AppConfig.mainLanguage.code == "en")
            "Processing image..."
        else
            "Procesez imaginea..."

        processImageUri(imageUri)
    }

    private fun startCaptionProcess(bitmap: Bitmap) {
        BackgroundTaskExecutor.getInstance().executeAsync(
            {
                // Step 1: Compute perceptual hash
                val hash = computeHash(bitmap)
                Log.d(TAG, "Image hash: $hash")
                imageHash = hash

                val cachedTokens = searchHashCache(hash)

                cachedTokens
            },
            object : BackgroundTaskExecutor.TaskCallback<List<Int>?> {
                override fun onSuccess(result: List<Int>?) {
                    if (result != null) {
                        // ⚡ Found in cache!
                        Log.d(TAG, "⚡ Caption found in cache: ${result.size} tokens")
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
                        // ❌ Not found in cache - generate caption
                        Log.d(TAG, "❌ Caption not found in cache, generating...")
                        foundInCache = false
                        generateCaption(bitmap)
                    }
                }

                override fun onError(e: Exception) {
                    Log.e(TAG, "❌ Error in cache search", e)
                    // Fallback to generation
                    foundInCache = false
                    generateCaption(bitmap)
                }
            }
        )
    }

    private fun generateCaption(bitmap: Bitmap) {
        loadingText.value = if (AppConfig.mainLanguage.code == "en")
            "Generating caption..."
        else
            "Generez descriere..."

        // Launch captioner task
        BackgroundTaskExecutor.getInstance().executeAsync(
            {
                val startTime = System.currentTimeMillis()
                val tokens = captioner.generateCaption(bitmap)
                val latency = System.currentTimeMillis() - startTime

                Log.d(TAG, "✅ Caption generated: ${tokens.size} tokens in ${latency}ms")

                Pair(tokens.toList(), latency)
            },
            object : BackgroundTaskExecutor.TaskCallback<Pair<List<Int>, Long>> {
                override fun onSuccess(result: Pair<List<Int>, Long>) {
                    tokenIds = result.first
                    captionerLatency = result.second

                    // Decode tokens
                    val tokenArray = result.first.map { it.toLong() }.toLongArray()
                    captionText = tokenizer.decode(tokenArray)

                    Log.d(TAG, "📝 Caption text: $captionText")

                    // Save to hash cache
                    saveToHashCache(imageHash, result.first)

                    // Run classifier if needed
                    if (AppConfig.env_reports) {
                        runClassifier(bitmap, onCacheHit = false)
                    } else {
                        // Translate if needed and show result
                        translateAndShowResult()
                    }
                }

                override fun onError(e: Exception) {
                    Log.e(TAG, "❌ Error generating caption", e)
                    showErrorDialog.value = true
                    showLoading.value = false
                }
            }
        )
    }

    private fun runClassifier(bitmap: Bitmap, onCacheHit: Boolean) {
        loadingText.value = if (AppConfig.mainLanguage.code == "en")
            "Classifying scene..."
        else
            "Clasificare scena..."

        BackgroundTaskExecutor.getInstance().executeAsync(
            {
                val startTime = System.currentTimeMillis()
                val classId = classifier.detectScene(bitmap, "CaptionActivity")
                val latency = System.currentTimeMillis() - startTime

                Log.d(TAG, "✅ Scene classified: $classId in ${latency}ms")

                Pair(classId, latency)
            },
            object : BackgroundTaskExecutor.TaskCallback<Pair<Int, Long>> {
                override fun onSuccess(result: Pair<Int, Long>) {
                    sceneClassId = result.first
                    classifierLatency = result.second

                    // Write to env reports
                    if (!onCacheHit) {
                        // Caption generated (not from cache)
                        EnvironmentReportsManagerKt.writeCaptionReport(
                            this@CaptionActivity,
                            sceneClassId,
                            captionerLatency,
                            classifierLatency
                        )
                    } else {
                        // Caption from cache
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
                    Log.e(TAG, "❌ Error classifying scene", e)
                    // Continue anyway
                    translateAndShowResult()
                }
            }
        )
    }

    private fun translateAndShowResult() {
        if (AppConfig.mainLanguage.code == "ro" && translator != null) {
            loadingText.value = "Traduc..."

            BackgroundTaskExecutor.getInstance().executeAsync(
                {
                    translator!!.translate(captionText)
                },
                object : BackgroundTaskExecutor.TaskCallback<String?> {
                    override fun onSuccess(result: String?) {
                        if (result != null) {
                            captionText = result
                            Log.d(TAG, "✅ Translated caption: $result")
                        }
                        showResultScreen()
                    }

                    override fun onError(e: Exception) {
                        Log.e(TAG, "❌ Error translating", e)
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
                val className = classifier.getClassName(sceneClassId)
                classificationText.value = load_classificationSuccess(className)

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
        finish()
        startActivity(Intent(this, HomeActivity::class.java))
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
            Log.e(TAG, "❌ Error launching camera", e)
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

    private fun handleRetry() {
        if (AppConfig.haptics) vibrate(haptic_model0())
        showErrorDialog.value = false
        handleCameraClick()
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        return when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_UP -> {
                // Go home (always)
                handleHomeClick()
                true
            }

            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                // Take photo (only when result shown)
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
    showErrorDialog: Boolean,
    errorMessage: String,
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
    onErrorRetry: () -> Unit,
    onErrorOk: () -> Unit
) {
    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val screenHeight = maxHeight

        // Background
        Box(modifier = Modifier.fillMaxSize()) {
            Image(
                painter = painterResource(R.drawable.app_background),
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )

            // Main caption screen (when result ready)
            if (showResult) {
                MainCaptionScreen(
                    screenHeight = screenHeight,
                    captionText = captionText,
                    captionerLatency = captionerLatency,
                    foundInCache = foundInCache,
                    currentTextSize = currentTextSize,
                    textSizes = textSizes,
                    onTextSizeChange = onTextSizeChange,
                    onHomeClick = onHomeClick,
                    onCameraClick = onCameraClick,
                    onSpeakClick = onSpeakClick
                )
            }

            // Loading overlay (instant enter, fade out exit)
            LoadingComponent(
                isVisible = showLoading,
                loadingText = loadingText,
                animSpec = Pair(
                    fadeIn(
                        initialAlpha = 0f,
                        animationSpec = tween(durationMillis = 0)  // Instant enter
                    ),
                    fadeOut(
                        targetAlpha = 0f,
                        animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY)  // Fade out
                    )
                )
            )

            // Classification notification (top)
            AnimatedVisibility(
                visible = showClassificationDialog,
                enter = slideInVertically(initialOffsetY = { -it }),
                exit = slideOutVertically(targetOffsetY = { -it }),
                modifier = Modifier.align(Alignment.TopCenter)
            ) {
                SceneClassifiedNotification(classificationText)
            }

            // Error dialog
            NotificationDialog(
                isVisible = showErrorDialog,
                type = LoadProfileActivity.NotificationType.ERROR,
                message = errorMessage,
                showTwoButtons = true,
                firstButtonLabel = if (AppConfig.mainLanguage.code == "en") "Retry" else "Reîncearcă",
                secondButtonLabel = if (AppConfig.mainLanguage.code == "en") "OK" else "OK",
                firstButtonClick = onErrorRetry,
                secondButtonClick = onErrorOk
            )
        }
    }
}

@Composable
fun MainCaptionScreen(
    screenHeight: Dp,
    captionText: String,
    captionerLatency: Long,
    foundInCache: Boolean,
    currentTextSize: Float,
    textSizes: List<Float>,
    onTextSizeChange: (Float) -> Unit,
    onHomeClick: () -> Unit,
    onCameraClick: () -> Unit,
    onSpeakClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(colorResource(R.color.std_purple).copy(alpha = 0.6f))
        ) {
            Column(
                modifier = Modifier.fillMaxSize(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Caption section
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(screenHeight * 0.69f)
                        .clip(
                            RoundedCornerShape(
                                bottomStart = 16.dp,
                                bottomEnd = 16.dp
                            )
                        )
                        .background(colorResource(R.color.std_purple)),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.SpaceBetween
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(30.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Spacer(modifier = Modifier.height(screenHeight * 0.075f))

                        // Caption text (centered, scrollable if needed)
                        Text(
                            text = captionText,
                            fontSize = currentTextSize.sp,
                            color = Color.White,
                            textAlign = TextAlign.Center,
                            fontFamily = robotoExtraBold,
                            lineHeight = (currentTextSize * 1.5f).sp
                        )
                    }
                    // Bottom info section
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(5.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        // Lightning bolt icon (cache hit indicator)
                        if (!foundInCache) {
                            Box(
                                modifier = Modifier
                                    .size(40.dp)
                                    .clip(CircleShape)
                                    .background(colorResource(R.color.std_light_purple)),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Bolt,
                                    contentDescription = "Found in cache",
                                    tint = colorResource(R.color.error_red),
                                    modifier = Modifier.size(30.dp)
                                )
                            }
                        } else
                        // Inference time box
                            Box(
                                modifier = Modifier
                                    .height(40.dp)
                                    .clip(RoundedCornerShape(16.dp))
                                    .background(colorResource(R.color.std_light_purple))
                                    .padding(horizontal = 16.dp, vertical = 8.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = "${captionerLatency}ms",
                                    fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                                    color = colorResource(R.color.std_purple),
                                    fontFamily = robotoExtraBold
                                )
                            }
                    }
                }

                Column(
                    modifier = Modifier
                        .fillMaxSize(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Bottom
                ) {
                    TextSizeSlider(
                        textSizes = textSizes,
                        currentTextSize = currentTextSize,
                        onTextSizeChange = onTextSizeChange
                    )

                    Spacer(Modifier.height(20.dp))

                    // Navigation section (bottom)
                    NavigationSection(
                        onHomeClick = onHomeClick,
                        onCameraClick = onCameraClick,
                        onSpeakClick = onSpeakClick
                    )
                }
            }
        }
    }
}

@Composable
fun NavigationSection(
    onHomeClick: () -> Unit,
    onCameraClick: () -> Unit,
    onSpeakClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(bottom = 43.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Home button (left, smaller, rounded)
        NavigationButton(
            modifier = Modifier
                .size(
                    width = Constants.NAV_BUTTONS_WIDTH.dp * 0.6f,
                    height = Constants.NAV_BUTTONS_HEIGHT.dp * 0.8f
                )
                .clip(RoundedCornerShape(100)),
            icon = Icons.Filled.Home,
            contentDescription = if (AppConfig.mainLanguage.code == "en") "Home" else "Acasă",
            onClick = onHomeClick,
            iconSize = 42.dp
        )

        Spacer(Modifier.width(22.5.dp))

        // Camera button (center, larger, rounded corners)
        NavigationButton(
            modifier = Modifier
                .size(
                    width = Constants.NAV_BUTTONS_WIDTH.dp,
                    height = Constants.NAV_BUTTONS_HEIGHT.dp
                )
                .clip(RoundedCornerShape(16.dp)),
            icon = Icons.Filled.PhotoCamera,
            contentDescription = "Camera",
            onClick = onCameraClick,
            iconSize = 58.dp
        )

        Spacer(Modifier.width(22.5.dp))

        // Speak button (right, smaller, rounded)
        NavigationButton(
            modifier = Modifier
                .size(
                    width = Constants.NAV_BUTTONS_WIDTH.dp * 0.6f,
                    height = Constants.NAV_BUTTONS_HEIGHT.dp * 0.8f
                )
                .clip(RoundedCornerShape(100)),
            icon = Icons.AutoMirrored.Filled.VolumeUp,
            contentDescription = if (AppConfig.mainLanguage.code == "en") "Speak text" else "Vorbeste textul",
            onClick = onSpeakClick,
            iconSize = 42.dp
        )
    }
}

@Composable
fun NavigationButton(
    modifier: Modifier,
    icon: ImageVector,
    contentDescription: String,
    onClick: () -> Unit,
    iconSize: Dp,
) {
    Box(
        modifier = modifier
            .background(colorResource(R.color.std_purple))
            .clickable(onClick = onClick),
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
fun TextSizeSlider(
    textSizes: List<Float>,
    currentTextSize: Float,
    onTextSizeChange: (Float) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth(0.65f),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = if (AppConfig.mainLanguage.code == "en") "Text size" else "Dimensiunea textului",
            color = Color.White,
            fontSize = Constants.STD_FONT_SIZE.sp,
            fontFamily = robotoExtraBold
        )

        CustomSlider(
            value = currentTextSize,
            onValueChange = onTextSizeChange,
            valueRange = textSizes.first()..textSizes.last(),
            steps = 0,
            thumbStyle = ThumbStyle.BAR,  // ROUND, BAR, or DOUBLE_BAR
            thumbColor = Color.Black,
            thumbWidth = 8.dp,
            thumbHeight = 55.dp,
            trackHeight = 40.dp,
            activeTrackColor = colorResource(R.color.std_purple),
            inactiveTrackColor = Color.White,
            trackShadow = 5.dp,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

@Preview(name = "Caption Screen Preview", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun CaptionScreenPreview() {
    CaptionScreen(
        showLoading = false,
        loadingText = "",
        showResult = true,
        showErrorDialog = false,
        showClassificationDialog = false,
        classificationText = load_classificationSuccess("kitchen"),
        captionText = "A modern kitchen with stainless steel appliances and granite countertops, featuring a large island and pendant lighting",
        captionerLatency = 4500,
        foundInCache = true,
        currentTextSize = 28f,
        textSizes = listOf(20f, 24f, 28f, 32f, 36f, 40f, 44f),
        onTextSizeChange = {},
        onHomeClick = {},
        onCameraClick = {},
        onSpeakClick = {},
        onErrorRetry = {},
        onErrorOk = {},
        errorMessage = ""
    )
}
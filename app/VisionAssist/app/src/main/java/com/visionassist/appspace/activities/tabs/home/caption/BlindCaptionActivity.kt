@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.tabs.home.caption

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.KeyEvent
import androidx.activity.compose.setContent
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.hideFromAccessibility
import androidx.compose.ui.semantics.invisibleToUser
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
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
import com.visionassist.appspace.activities.newprofile.LoadProfileActivity
import com.visionassist.appspace.activities.tabs.home.caption.SemanticHash.computeFromDetections
import com.visionassist.appspace.jetpack.design.LoadingComponent
import com.visionassist.appspace.jetpack.design.NotificationDialog
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager
import com.visionassist.appspace.sound.SoundConstants
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.BackgroundTaskExecutor
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.load_captionError2
import com.visionassist.appspace.utils.load_captioningScene
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.saveToHashCache
import com.visionassist.appspace.utils.searchHashCache
import java.io.File

class BlindCaptionActivity : BaseActivity() {
    private val TAG = "CaptionActivity"

    // Models
    private val captioner = PhoneStatusMonitor.getInstance().modelManager.captioner
    private val tokenizer = captioner.tokenizer
    private var translator = PhoneStatusMonitor.getInstance().modelManager.translator
    private val ttsManager = PhoneStatusMonitor.getInstance()
        .ttsManager
    private val soundManager = PhoneStatusMonitor.getInstance().soundManager

    // Image data
    private var originalBitmap: Bitmap? = null
    private var imageHash: String = ""

    // Results
    private var tokenIds: List<Int>? = null
    private var captionText: String = ""
    private var captionerLatency: Long = 0
    private var foundInCache: Boolean = false
    private var isSpeakingPhase: Boolean = false
    private var isDestroyed = false

    // UI States
    private val showLoading = mutableStateOf(true)
    private val loadingText = mutableStateOf("")
    private val showResult = mutableStateOf(false)
    private val showErrorDialog = mutableStateOf(false)
    private val errorMessage = mutableStateOf("")

    // Text size control (8 stops)
    private val currentTextSize = 40f

    // Handlers
    private val mainHandler = Handler(Looper.getMainLooper())

    // Camera
    private lateinit var takePictureLauncher: ActivityResultLauncher<Uri>
    private var currentPhotoUri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

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
            BlindCaptionScreen(
                showLoading = showLoading.value,
                loadingText = loadingText.value,
                showResult = showResult.value,
                showErrorDialog = showErrorDialog.value,
                errorMessage = errorMessage.value,
                captionText = captionText,
                currentTextSize = currentTextSize,
                onHomeClick = {
                    handleHomeClick()
                },
                onCameraClick = {
                    if (!isSpeakingPhase && showResult.value)
                        handleCameraClick()
                },
                onErrorRetry = { handleRetry() },
                onErrorOk = {
                    showErrorDialog.value = false
                    handleHomeClick()
                },
                onMainScreenTap = {
                    captionSpeak()
                }
            )
        }

        loadingText.value = load_captioningScene(this)
        mainHandler.postDelayed({
            // Get image URI from intent
            val imageUriString = intent.getStringExtra(Constants.EXTRA_IMAGE_URI)
            if (imageUriString != null) {
                val imageUri = imageUriString.toUri()
                processImageUri(imageUri)
            } else {
                Log.e(TAG, "No image URI provided")
                finish()
            }
        }, 500)
    }

    private fun processImageUri(imageUri: Uri) {
        BackgroundTaskExecutor.getInstance().executeAsync(
            {
                // Load bitmap from URI
                val inputStream = contentResolver.openInputStream(imageUri)
                val bitmap = BitmapFactory.decodeStream(inputStream)
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
                    if(isDestroyed)return
                    if (result != null) {
                        originalBitmap = result
                        startCaptionProcess(result)
                    } else {
                        errorMessage.value = load_captionError2(
                            PhoneStatusMonitor.getInstance().currentContext
                        )
                        showLoading.value = false
                        mainHandler.postDelayed({
                            showErrorDialog.value = true
                            ttsManager.speak(
                                errorMessage.value,
                                AppConfig.tts_pitch,
                                AppConfig.tts_speech_rate,
                                true,
                                null
                            )
                            isSpeakingPhase = true
                        }, Constants.ANIMATION_DELAY.toLong())
                    }
                }

                override fun onError(e: Exception) {
                    Log.e(TAG, "Error loading bitmap", e)
                    errorMessage.value = load_captionError2(
                        PhoneStatusMonitor.getInstance().currentContext
                    )
                    showLoading.value = false
                    mainHandler.postDelayed({
                        showErrorDialog.value = true
                        ttsManager.speak(
                            errorMessage.value,
                            AppConfig.tts_pitch,
                            AppConfig.tts_speech_rate,
                            true,
                            null
                        )
                        isSpeakingPhase = true
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
        loadingText.value = load_captioningScene(this)

        processImageUri(imageUri)
    }

    private fun startCaptionProcess(bitmap: Bitmap) {
        BackgroundTaskExecutor.getInstance().executeAsync(
            {
                if (AppConfig.hash_caching == null)
                    return@executeAsync null

                val detectorModel = PhoneStatusMonitor.getInstance().modelManager.detector
                val detectorWrapper = detectorModel
                    .acquireDetector(
                        false,
                        5
                    )
                if (detectorWrapper == null)
                    return@executeAsync null

                val detector = detectorWrapper.detector
                val detectionResult = detector.detectObjects(bitmap, "CaptionActivity")
                detectorModel.releaseDetector(detectorWrapper)

                if (detectionResult.classIndices.isEmpty())
                    return@executeAsync null

                imageHash = computeFromDetections(detectionResult, bitmap.width, bitmap.height)
                    ?: return@executeAsync null

                return@executeAsync searchHashCache(imageHash)
            },
            object : BackgroundTaskExecutor.TaskCallback<List<Int>?> {
                override fun onSuccess(result: List<Int>?) {
                    if(isDestroyed)return
                    if (result != null) {
                        Log.d(TAG, "Caption found in cache: ${result.size} tokens")
                        foundInCache = true
                        tokenIds = result
                        decodeAndTranslate()
                    } else {
                        Log.d(TAG, "Caption not found in cache, generating...")
                        generateCaption(bitmap)
                    }
                }

                override fun onError(e: Exception) {
                    Log.e(TAG, "Error in cache search", e)
                    errorMessage.value = load_captionError2(
                        PhoneStatusMonitor.getInstance().currentContext
                    )
                    showLoading.value = false
                    mainHandler.postDelayed({
                        showErrorDialog.value = true
                        ttsManager.speak(
                            errorMessage.value,
                            AppConfig.tts_pitch,
                            AppConfig.tts_speech_rate,
                            true,
                            null
                        )
                        isSpeakingPhase = true
                    }, Constants.ANIMATION_DELAY.toLong())
                }
            }
        )
    }

    private fun generateCaption(bitmap: Bitmap) {
        // Launch captioner task
        BackgroundTaskExecutor.getInstance().executeAsync(
            {
                val startTime = System.currentTimeMillis()
                tokenIds = captioner.generateCaption(bitmap).toList()
                captionerLatency = System.currentTimeMillis() - startTime

                if (AppConfig.hash_caching != null) {
                    PhoneStatusMonitor.getInstance().writingToHCFinished = false
                    saveToHashCache(imageHash, tokenIds!!.toList())
                }

                captionText = tokenizer.decode(tokenIds?.toIntArray())

                if (AppConfig.mainLanguage.code == "ro" && translator != null)
                    captionText = translator.translate(captionText)
            },
            {
                foundInCache = false
                showResult()
            },
            {
                errorMessage.value = load_captionError2(
                    PhoneStatusMonitor.getInstance().currentContext
                )
                showLoading.value = false
                mainHandler.postDelayed({
                    showErrorDialog.value = true
                    ttsManager.speak(
                        errorMessage.value,
                        AppConfig.tts_pitch,
                        AppConfig.tts_speech_rate,
                        true,
                        null
                    )
                    isSpeakingPhase = true
                }, Constants.ANIMATION_DELAY.toLong())
            }
        )
    }

    private fun decodeAndTranslate() {
        BackgroundTaskExecutor.getInstance().executeAsync(
            {
                captionText = tokenizer.decode(tokenIds?.toIntArray())

                if (AppConfig.mainLanguage.code == "ro" && translator != null)
                    captionText = translator.translate(captionText)
            },
            {
                showResult()
            },
            {
                errorMessage.value = load_captionError2(
                    PhoneStatusMonitor.getInstance().currentContext
                )
                showLoading.value = false
                mainHandler.postDelayed({
                    showErrorDialog.value = true
                    ttsManager.speak(
                        errorMessage.value,
                        AppConfig.tts_pitch,
                        AppConfig.tts_speech_rate,
                        true,
                        null
                    )
                    isSpeakingPhase = true
                }, Constants.ANIMATION_DELAY.toLong())
            }
        )
    }

    private fun showResult() {
        if(isDestroyed)return

        showLoading.value = false

        mainHandler.postDelayed({
            showResult.value = true
            soundManager.play(
                if (foundInCache)
                    SoundConstants.CAPTION_DONE_WITH_HC_ID
                else
                    SoundConstants.CAPTION_DONE_ID,
                0.35f,
                0.35f
            ) {
                captionSpeak()
            }
        }, Constants.ANIMATION_DELAY.toLong())
    }

    private fun captionSpeak() {
        ttsManager.speak(
            captionText,
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            true,
            null
        )
        isSpeakingPhase = true
        val checkRunnable = object : Runnable {
            override fun run() {
                if (ttsManager.isDoneSpeaking) {
                    isSpeakingPhase = false
                } else {
                    Log.i(TAG, "TTS isn't stopped at the moment")
                    mainHandler.postDelayed(this, 100)
                }
            }
        }
        mainHandler.post(checkRunnable)
    }

    // Navigation handlers
    private fun handleHomeClick() {
        soundManager.releaseCallback()
        ttsManager.stopSpeaking()
        ttsManager.speak(
            if (ttsManager.currentLocale.language == "en")
                "Returning to home page"
            else
                "Se revine în pagina principală",
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            false,
            haptic_model0()
        )

        val checkRunnable = object : Runnable {
            override fun run() {
                if (ttsManager.isDoneSpeaking) {
                    finish()
                } else {
                    mainHandler.postDelayed(this, 350)
                }
            }
        }
        mainHandler.post(checkRunnable)
    }

    private fun handleCameraClick() {
        if (!PhoneStatusMonitor.getInstance().writingToHCFinished) return

        ttsManager.speak(
            if (ttsManager.currentLocale.language == "en")
                "Opening camera app"
            else
                "Se deschide aplicația cameră",
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            false,
            haptic_model0()
        )

        val checkRunnable = object : Runnable {
            override fun run() {
                if (ttsManager.isDoneSpeaking) {
                    try {
                        val photoFile = File(cacheDir, "caption_${System.currentTimeMillis()}.jpg")
                        currentPhotoUri = FileProvider.getUriForFile(
                            this@BlindCaptionActivity,
                            "${packageName}.fileprovider",
                            photoFile
                        )
                        takePictureLauncher.launch(currentPhotoUri!!)
                    } catch (e: Exception) {
                        Log.e(TAG, "Error launching camera", e)
                        showCameraError()
                    }
                } else {
                    mainHandler.postDelayed(this, 350)
                }
            }
        }
        mainHandler.post(checkRunnable)
    }

    private fun showCameraError() {
        val monitor = PhoneStatusMonitor.getInstance()
        val errorDialog = ErrorDialogManager(this)
        errorDialog.setupDialog(Constants.CAMERA_MAKE_PHOTO)
        monitor.shutdownApp(errorDialog, this)
    }

    private fun handleRetry() {
        ttsManager.stopSpeaking(); isSpeakingPhase = false
        if (!PhoneStatusMonitor.getInstance().writingToHCFinished) return
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
                if (isSpeakingPhase)
                    ttsManager.onVolumeDownPressed()
                else
                    if (showResult.value)
                        handleCameraClick()
                true
            }

            else -> super.onKeyDown(keyCode, event)
        }
    }

    override fun onPause() {
        super.onPause()
        soundManager.releaseCallback()
        ttsManager.stopSpeaking()
    }

    override fun onDestroy() {
        super.onDestroy()
        isDestroyed=true
        mainHandler.removeCallbacksAndMessages(null)
        originalBitmap?.recycle()
    }
}

@Composable
fun BlindCaptionScreen(
    showLoading: Boolean,
    loadingText: String,
    showResult: Boolean,
    showErrorDialog: Boolean,
    errorMessage: String,
    captionText: String,
    currentTextSize: Float,
    onHomeClick: () -> Unit,
    onCameraClick: () -> Unit,
    onErrorRetry: () -> Unit,
    onErrorOk: () -> Unit,
    onMainScreenTap: () -> Unit
) {
    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val screenHeight = maxHeight
        val screenWidth = maxHeight

        // Background
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
                                swipeStartX <= -threshold -> {
                                    onCameraClick()
                                }

                                swipeStartX >= threshold -> {
                                    onHomeClick()
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
            Image(
                painter = painterResource(R.drawable.app_background),
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )

            // Main caption screen (when result ready)
            if (showResult) {
                BlindMainCaptionScreen(
                    screenHeight = screenHeight,
                    captionText = captionText,
                    currentTextSize = currentTextSize,
                    onMainScreenTap = onMainScreenTap
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

            if (showErrorDialog) {
                Row(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color.Transparent)  // Invisible
                ) {
                    // Left half - Retry
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxHeight()
                            .semantics {
                                contentDescription = if (AppConfig.mainLanguage.code == "en") {
                                    "Retry button"
                                } else {
                                    "Buton de reîncercare"
                                }
                                role = Role.Button
                            }
                            .clickable(
                                indication = null,  // No ripple effect
                                interactionSource = remember { MutableInteractionSource() }
                            ) {
                                onErrorRetry()
                            }
                    )

                    // Right half - OK
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxHeight()
                            .clickable(
                                indication = null,  // No ripple effect
                                interactionSource = remember { MutableInteractionSource() }
                            ) {
                                onErrorOk()
                            }
                            .semantics {
                                contentDescription = if (AppConfig.mainLanguage.code == "en") {
                                    "OK button"
                                } else {
                                    "Buton ok"
                                }
                                role = Role.Button
                            }
                    )
                }
            }
        }
    }
}

@Composable
fun BlindMainCaptionScreen(
    screenHeight: Dp,
    captionText: String,
    currentTextSize: Float,
    onMainScreenTap: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
            .clickable(
                indication = null,  // No ripple effect
                interactionSource = remember { MutableInteractionSource() }
            ) {
                onMainScreenTap()
            }
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(colorResource(R.color.std_purple).copy(alpha = 0.6f)),
            contentAlignment = Alignment.Center
        ) {
            // Caption section
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(screenHeight * 0.8f)
                    .clip(
                        RoundedCornerShape(16.dp)
                    )
                    .background(colorResource(R.color.std_purple)),
                contentAlignment = Alignment.Center
            ) {
                // Caption text (centered, scrollable if needed)
                Text(
                    modifier = Modifier
                        .padding(30.dp)
                        .semantics {
                            // ✅ Hide from TalkBack
                            hideFromAccessibility()
                        }
                    ,
                    text = captionText,
                    fontSize = currentTextSize.sp,
                    color = Color.White,
                    textAlign = TextAlign.Center,
                    fontFamily = robotoExtraBold,
                    lineHeight = (currentTextSize * 1.5f).sp
                )
            }
        }
    }
}

@Preview(
    name = "Blind Caption Screen Preview",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
@Composable
fun BlindCaptionScreenPreview() {
    BlindCaptionScreen(
        showLoading = false,
        loadingText = "",
        showResult = true,
        showErrorDialog = false,
        captionText = "A modern kitchen with stainless steel appliances and granite countertops, featuring a large island and pendant lighting",
        currentTextSize = 28f,
        onHomeClick = {},
        onCameraClick = {},
        onErrorRetry = {},
        onErrorOk = {},
        errorMessage = "",
        onMainScreenTap = {}
    )
}
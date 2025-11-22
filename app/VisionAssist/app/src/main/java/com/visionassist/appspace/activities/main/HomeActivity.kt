@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.main

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.KeyEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts.TakePicture
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
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
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Sync
import androidx.compose.material.icons.filled.SyncProblem
import androidx.compose.material.icons.filled.TextFields
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemColors
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.tabs.home.caption.CaptionActivity
import com.visionassist.appspace.activities.tabs.home.detection.LiveDetectionActivity
import com.visionassist.appspace.activities.tabs.home.detection.StaticDetectionActivity
import com.visionassist.appspace.activities.tabs.home.findmyobjects.FindMyObjectActivity
import com.visionassist.appspace.activities.tabs.reports.EnvironmentReportsActivity
import com.visionassist.appspace.activities.tabs.settings.SettingsActivity
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager
import com.visionassist.appspace.models.sttengine.SpeechRecognizer
import com.visionassist.appspace.sound.SoundConstants
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.BackgroundTaskExecutor
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.PermissionChecker
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.load_captionTutorial
import com.visionassist.appspace.utils.load_detectionTutorial
import com.visionassist.appspace.utils.load_errorSTT
import com.visionassist.appspace.utils.load_errorSTTRuntime
import com.visionassist.appspace.utils.load_homeTitle
import com.visionassist.appspace.utils.load_speakTutorial
import com.visionassist.appspace.utils.load_syncErrorText
import com.visionassist.appspace.utils.load_syncStatusText
import com.visionassist.appspace.utils.load_unavailableSTT
import com.visionassist.appspace.utils.robotoBold
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.robotoExtraBoldItalic
import com.visionassist.appspace.utils.robotoSemibold
import com.visionassist.appspace.utils.vibrate
import kotlinx.coroutines.delay
import java.io.File
import java.io.IOException

class HomeActivity : ComponentActivity() {
    private val TAG = "HomeActivity"

    // State variables
    private val titleText = mutableStateOf("")
    private val syncStatus = mutableStateOf(0) // 0=hidden, 1=show days, 2=error
    private val syncDays = mutableStateOf(0)

    // Tutorial info button states
    private val speakInfoClickCount = mutableStateOf(1)
    private var basicInfoClickCount = 1
    private var basicInfoClickCount2 = 1

    // Detection button states
    private val showDetectionOptions = mutableStateOf(false)
    private val selectedDetectionOption = mutableStateOf<DetectionOption?>(null)
    private val detectionIconColor = mutableStateOf(R.color.std_purple_dark)

    // Speech recognition parameters
    private val speechRecognizer = PhoneStatusMonitor.getInstance()
        .modelManager.speechRecognizer
    private val soundManager = PhoneStatusMonitor.getInstance()
        .soundManager
    private val ttsManager = PhoneStatusMonitor.getInstance()
        .ttsManager
    private val showSpeechDialog = mutableStateOf(false)
    private val speechText = mutableStateOf("")
    private val speechProcessText = mutableStateOf("")
    private val isSpeaking = mutableStateOf(true)
    private val retrySpeech = mutableStateOf(false)
    private val sendSpeech = mutableStateOf(false)
    private lateinit var matchedIndices: List<SpeechRecognizer.MatchedObject>
    private lateinit var classNames: List<String>

    // Volume button tracking
    private var locked = false
    private var uiLocked = false
    private var handleVolumeDownControl = false

    // Runnable for navigation to an activity
    private var onPermissionGranted = {}
    private var classOpt = 0

    // Camera intent parameters
    private lateinit var takePictureLauncher: ActivityResultLauncher<Uri>
    private lateinit var currentPhotoUri: Uri

    // Main handler
    private val handler = Handler(Looper.getMainLooper())
    private lateinit var afterResumeRunnable: Runnable
    private var hasToExecAfterResume = false

    enum class DetectionOption {
        LIVE, STATIC
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        registerCameraLauncher()

        // Load initial title
        titleText.value = load_homeTitle()

        // Get sync status
        val dbManager = PhoneStatusMonitor.getInstance().dbManager
        syncStatus.value = dbManager.statusOverview
        if (syncStatus.value == 1) {
            syncDays.value = dbManager.diffDays.toInt()
        }

        uiLocked = true
        PhoneStatusMonitor.getInstance().soundManager.play(SoundConstants.OPEN_UP_ID, 1f, 1f) {
            uiLocked = false
        }

        setContent {
            HomeScreen(
                titleText = titleText.value,
                syncStatus = syncStatus.value,
                syncDays = syncDays.value,
                showDetectionOptions = showDetectionOptions.value,
                detectionIconColor = detectionIconColor.value,
                selectedDetectionOption = selectedDetectionOption.value,
                showSpeechDialog = showSpeechDialog.value,
                speechText = speechText.value,
                speechProcessText = speechProcessText.value,
                isSpeaking = isSpeaking.value,
                onDetectionClick = ::handleDetectionClick,
                onDetectionOptionSelected = ::handleDetectionOptionSelected,
                onCaptionClick = ::handleCaptionClick,
                onDetectionInfoClick = ::handleDetectionInfoClick,
                onCaptionInfoClick = ::handleCaptionInfoClick,
                onSpeakInfoClick = ::handleSpeakInfoClick,
                onNavigateHome = ::handleNavigateHome,
                onNavigateReports = ::handleNavigateReports,
                onNavigateSettings = ::handleNavigateSettings,
                onSpeechDialogTap = ::handleSpeechDialogTap,
                navigateFun = ::navigateToLiveOrStatic,
                onSwipeToLeft = ::handleSwipeToLeft
            )
        }
    }

    @Suppress("DEPRECATION")
    private fun registerCameraLauncher() {
        takePictureLauncher = registerForActivityResult(
            TakePicture()
        ) { isSuccess ->
            if (isSuccess) {
                try {
                    navigateToFindMyObjectWithBitmap()
                } catch (e: IOException) {
                    Log.e(TAG, "Error loading captured image", e)
                    showCameraError()
                }
            } else {
                Log.e(TAG, "Image capture failed or cancelled")
                uiLocked = false
                locked = false
            }
        }

        Log.d(TAG, "Camera launcher registered")
    }

    override fun onResume() {
        super.onResume()

        if (PhoneStatusMonitor.getInstance().isReturningFromPermissions)
            PermissionChecker.checkAndRequestPermissions(
                this,
                AppConfig.blindness,
                onPermissionGranted
            )

        if (hasToExecAfterResume) {
            hasToExecAfterResume = false
            handler.post(afterResumeRunnable)
        }
    }

    private fun handleDetectionClick() {
        if (!uiLocked) {
            vibrateIfEnabled()
            showDetectionOptions.value = !showDetectionOptions.value
            detectionIconColor.value = if (showDetectionOptions.value) {
                R.color.std_cyan
            } else {
                selectedDetectionOption.value = null
                R.color.std_purple_dark
            }
        }
    }

    private fun handleDetectionOptionSelected(option: DetectionOption?, navigate: Boolean) {
        if (option != null) {
            if (selectedDetectionOption.value != option) {
                vibrateIfEnabled()
                selectedDetectionOption.value = option
            }
        } else
            selectedDetectionOption.value = null
        if (navigate) {
            navigateToLiveOrStatic()
        }
    }

    private fun navigateToLiveOrStatic() {
        when (selectedDetectionOption.value) {
            DetectionOption.LIVE -> launchLiveDetection()
            DetectionOption.STATIC -> launchStaticDetection()
            else -> {
                handleDetectionClick()
            }
        }
    }

    private fun launchLiveDetection() {
        onPermissionGranted = {
            checkPhoneStatusAndNavigate {
                // Navigate to LiveDetectionActivity
                val intent = Intent(this, LiveDetectionActivity::class.java)
                startActivity(intent)
                finish()
            }
        }
        PermissionChecker.checkAndRequestPermissions(this, AppConfig.blindness, onPermissionGranted)
    }

    private fun launchStaticDetection() {
        classOpt = 0
        onPermissionGranted = {
            checkPhoneStatusAndNavigate {
                launchCamera()
            }
        }
        PermissionChecker.checkAndRequestPermissions(this, AppConfig.blindness, onPermissionGranted)
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

            takePictureLauncher.launch(currentPhotoUri)
        } catch (e: IOException) {
            Log.e(TAG, "Error creating temp file", e)
            showCameraError()
        }
    }

    private fun navigateToFindMyObjectWithBitmap() {
        try {
            val intent = Intent(
                this,
                when (classOpt) {
                    0 -> StaticDetectionActivity::class.java
                    else -> CaptionActivity::class.java
                }
            )
            intent.putExtra(Constants.EXTRA_IMAGE_URI, currentPhotoUri.toString())
            startActivity(intent)
            finish()
        } catch (e: Exception) {
            Log.e(TAG, "Error navigating to FindMyObjectActivity", e)
            showCameraError()
        }
    }

    private fun showCameraError() {
        val monitor = PhoneStatusMonitor.getInstance()
        val errorDialog = ErrorDialogManager(monitor.currentActivity)
        errorDialog.setupDialog(Constants.CAMERA_MAKE_PHOTO)
        monitor.shutdownApp(errorDialog, monitor.currentContext)
    }

    private fun handleCaptionClick() {
        if (!uiLocked) {
            vibrateIfEnabled()
            launchCaptionActivity()
        }
    }

    private fun launchCaptionActivity() {
        classOpt = 1
        onPermissionGranted = {
            checkPhoneStatusAndNavigate {
                launchCamera()
            }
        }
        PermissionChecker.checkAndRequestPermissions(this, AppConfig.blindness, onPermissionGranted)
    }

    private fun handleDetectionInfoClick() {
        if (!uiLocked) {
            vibrateIfEnabled()
            titleText.value = load_detectionTutorial(this, getNextInfoStep())
        }
    }

    private fun handleCaptionInfoClick() {
        if (!uiLocked) {
            vibrateIfEnabled()
            titleText.value = load_captionTutorial(this, getNextInfoStep2())
        }
    }

    private fun handleSpeakInfoClick() {
        if (!uiLocked) {
            vibrateIfEnabled()

            titleText.value = load_speakTutorial(this, speakInfoClickCount.value)

            // Reset after showing all messages
            if (speakInfoClickCount.value == 7) {
                speakInfoClickCount.value = 0
            } else
                speakInfoClickCount.value++
        }
    }

    private fun getNextInfoStep(): Int {
        // Simple counter for tutorial steps
        if (++basicInfoClickCount == 4) {
            basicInfoClickCount = 1
            return 0
        } else {
            return basicInfoClickCount - 1
        }
    }

    private fun getNextInfoStep2(): Int {
        // Simple counter for tutorial steps
        if (++basicInfoClickCount2 == 4) {
            basicInfoClickCount2 = 1
            return 0
        } else {
            return basicInfoClickCount2 - 1
        }
    }

    private fun handleNavigateHome() {
        if (!uiLocked) {
            vibrateIfEnabled()
            // Already on home, do nothing
        }
    }

    private fun handleNavigateReports() {
        if (!uiLocked) {
            vibrateIfEnabled()

            val intent = Intent(this, EnvironmentReportsActivity::class.java)
            startActivity(intent)
            finish()
        }
    }

    private fun handleNavigateSettings() {
        if (!uiLocked) {
            vibrateIfEnabled()
            val intent = Intent(this, SettingsActivity::class.java)
            startActivity(intent)
            finish()
        }
    }

    private fun handleSwipeToLeft() {
        if (!uiLocked) {
            val intent = Intent(
                this, if (AppConfig.env_reports)
                    EnvironmentReportsActivity::class.java
                else
                    SettingsActivity::class.java
            )
            startActivity(intent)
            finish()
        }
    }

    private fun launchSpeechRecognition(firstTimeSpeak: Boolean) {
        if (firstTimeSpeak) {
            if (AppConfig.mainLanguage.code == "ro") {
                hasToExecAfterResume = true
                soundManager.play(SoundConstants.STT_ERROR_ID, 0.7f, 0.7f) {
                    ttsManager.speak(
                        load_unavailableSTT(this),
                        AppConfig.tts_pitch,
                        AppConfig.tts_speech_rate,
                        false,
                        null
                    )
                }
                afterResumeRunnable = object : Runnable {
                    override fun run() {
                        if (ttsManager.isDoneSpeaking) {
                            uiLocked = false
                            locked = false
                        } else {
                            handler.postDelayed(this, Constants.LOAD_CHECK_DELAY_MS.toLong())
                        }
                    }
                }
                handler.postDelayed(
                    afterResumeRunnable,
                    SoundConstants.getDuration(SoundConstants.STT_ERROR_ID).toLong() + 500
                )
            } else
                if (speechRecognizer == null) {
                    soundManager.play(SoundConstants.STT_ERROR_ID, 0.7f, 0.7f) {
                        ttsManager.speak(
                            load_errorSTT(this),
                            AppConfig.tts_pitch,
                            AppConfig.tts_speech_rate,
                            false,
                            null
                        )
                    }
                    afterResumeRunnable = object : Runnable {
                        override fun run() {
                            if (ttsManager.isDoneSpeaking) {
                                uiLocked = false
                                locked = false
                            } else {
                                handler.postDelayed(this, Constants.LOAD_CHECK_DELAY_MS.toLong())
                            }
                        }
                    }
                    handler.postDelayed(
                        afterResumeRunnable,
                        SoundConstants.getDuration(SoundConstants.STT_ERROR_ID).toLong() + 500
                    )
                } else {
                    retrySpeech.value = false
                    sendSpeech.value = false
                    showSpeechDialog.value = true
                    speakingProcess()
                }
        } else {
            speakingProcess()
        }
    }

    private fun formatRecognizedText(text: String): String {
        if (text.isBlank()) return text

        // Replace [unk] with ??
        var formatted = text.replace("[unk]", "??")

        // Capitalize first letter if it's not ??
        if (formatted.isNotEmpty() && !formatted.startsWith("??")) {
            formatted = formatted.replaceFirstChar {
                if (it.isLowerCase()) it.titlecase() else it.toString()
            }
        }

        return formatted
    }

    private fun speakingProcess() {
        isSpeaking.value = true
        speechText.value = "Listening..."
        soundManager.play(SoundConstants.STT_SPEAK_OPEN_ID, 0.7f, 0.7f) {
            speechRecognizer.startListening(object :
                SpeechRecognizer.RecognitionCallback {
                override fun onResult(recognizedText: String, isFinalResult: Boolean) {
                    if (isFinalResult) {
                        isSpeaking.value = false
                        speechText.value = formatRecognizedText(recognizedText)
                        handler.postDelayed({ processRecognizedSpeech() }, 1000)

                        /*
                        sendSpeech.value = true
                        retrySpeech.value = true
                        locked = false
                        vibrateIfEnabled()*/
                    } else {
                        speechText.value = formatRecognizedText(recognizedText)
                    }
                }

                override fun onError(error: String) {
                    hasToExecAfterResume = true
                    soundManager.play(SoundConstants.STT_ERROR_ID, 0.7f, 0.7f) {
                        ttsManager.speak(
                            load_errorSTTRuntime(PhoneStatusMonitor.getInstance().currentContext),
                            AppConfig.tts_pitch,
                            AppConfig.tts_speech_rate,
                            false,
                            null
                        )
                    }
                    afterResumeRunnable = object : Runnable {
                        override fun run() {
                            if (ttsManager.isDoneSpeaking) {
                                handleSpeechDialogTap()
                            } else {
                                handler.postDelayed(this, Constants.LOAD_CHECK_DELAY_MS.toLong())
                            }
                        }
                    }
                    handler.postDelayed(
                        afterResumeRunnable,
                        SoundConstants.getDuration(SoundConstants.STT_ERROR_ID).toLong() + 500
                    )
                }
            })
        }
    }

    private fun processRecognizedSpeech() {
        speechProcessText.value = "Matching known objects..."

        var finishedLoading = false
        val backgroundTask = BackgroundTaskExecutor.getInstance()
        backgroundTask.executeAsync(
            {
                matchedIndices = speechRecognizer.processRecognizedText(speechText.value)

                if (!matchedIndices.isEmpty()) {
                    classNames = matchedIndices.map { it.matchedWord }
                }
                return@executeAsync 0
            },
            object : BackgroundTaskExecutor.TaskCallback<Int> {
                override fun onSuccess(result: Int) {
                    finishedLoading = true
                }

                override fun onError(e: Exception) {
                    finishedLoading = true
                }
            }
        )

        val checkRunnable = object : Runnable {
            override fun run() {
                if (finishedLoading) {
                    processTextOutput()
                } else {
                    handler.postDelayed(this, 500)
                }
            }
        }
        handler.postDelayed(checkRunnable, 2000)
    }

    private fun processTextOutput() {
        if (matchedIndices.isEmpty()) {
            speechProcessText.value = "No matched known classes"
            retrySpeech.value = true
            sendSpeech.value = false
            locked = false
            vibrateIfEnabled()
        } else {
            speechProcessText.value = "Matched: ${classNames.joinToString(", ")}"
            retrySpeech.value = true
            sendSpeech.value = true
            locked = false
            vibrateIfEnabled()
        }
    }

    private fun sendProcessedSpeech() {
        val classIndices = IntArray(matchedIndices.size) { matchedIndices[it].classIndex }
        val matchedWords = Array(matchedIndices.size) { matchedIndices[it].matchedWord }

        val intent = Intent(this, FindMyObjectActivity::class.java).apply {
            putExtra(Constants.EXTRA_MATCHED_INDICES, classIndices)
            putExtra(Constants.EXTRA_SYNONYMS_WORDS, matchedWords)
        }
        startActivity(intent)
        finish()
    }

    private fun handleSpeechDialogTap() {
        // Cancel speech recognition
        handler.removeCallbacksAndMessages(null)
        soundManager.releaseCallback()
        speechRecognizer.stopListening()
        locked = true
        showSpeechDialog.value = false
        handler.postDelayed({
            uiLocked = false
            retrySpeech.value = false
            sendSpeech.value = false
            isSpeaking.value = false
            speechText.value = ""
            speechProcessText.value = ""
            locked = false
        }, Constants.ANIMATION_DELAY.toLong())
    }

    private fun resetSpeechStates() {
        speechProcessText.value = ""
        speechText.value = ""
        //cancelSpeech.value = true
        retrySpeech.value = false
        sendSpeech.value = false
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_UP -> {
                if (!locked) {
                    //uiLocked=true
                    if (retrySpeech.value) {
                        // Retry speech recognition
                        locked = true
                        resetSpeechStates()
                        launchSpeechRecognition(false)
                    } else {
                        uiLocked = true
                        locked = true
                        // Launch static detection
                        launchStaticDetection()
                    }
                }
                return true
            }

            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                if (!locked) {
                    uiLocked = true
                    if (!handleVolumeDownControl) {
                        if (!showSpeechDialog.value)
                            handleVolumeDownControl = true
                        handler.removeCallbacksAndMessages(null)
                        handler.postDelayed(
                            { handleSingleVolumeDown() }, Constants.VOLUME_DOWN_DELAY_MS.toLong()
                        )
                    } else {
                        handler.removeCallbacksAndMessages(null)
                        locked = true
                        handleVolumeDownControl = false
                        launchSpeechRecognition(true)
                    }
                }
                return true
            }
        }
        return super.onKeyDown(keyCode, event)
    }

    private fun handleSingleVolumeDown() {
        when {
            sendSpeech.value -> {
                locked = true
                // Process recognized text
                sendProcessedSpeech()
            }

            else -> {
                // Launch caption activity
                launchCaptionActivity()
            }
        }
    }

    private fun vibrateIfEnabled() {
        if (AppConfig.haptics) {
            vibrate(haptic_model0())
        }
    }

    private fun checkPhoneStatusAndNavigate(onSuccess: () -> Unit) {
        PhoneStatusMonitor.getInstance().checkPhoneStatus()
        // If check passes, execute success callback
        onSuccess()
    }

    override fun onPause() {
        super.onPause()
        handler.removeCallbacksAndMessages(null)
        soundManager.releaseCallback()
        ttsManager.stopSpeaking()
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    titleText: String,
    syncStatus: Int,
    syncDays: Int,
    showDetectionOptions: Boolean,
    detectionIconColor: Int,
    selectedDetectionOption: HomeActivity.DetectionOption?,
    showSpeechDialog: Boolean,
    speechText: String,
    speechProcessText: String,
    isSpeaking: Boolean,
    onDetectionClick: () -> Unit,
    onDetectionOptionSelected: (HomeActivity.DetectionOption?, navigate: Boolean) -> Unit,
    onCaptionClick: () -> Unit,
    onDetectionInfoClick: () -> Unit,
    onCaptionInfoClick: () -> Unit,
    onSpeakInfoClick: () -> Unit,
    onNavigateHome: () -> Unit,
    onNavigateReports: () -> Unit,
    onNavigateSettings: () -> Unit,
    onSpeechDialogTap: () -> Unit,
    navigateFun: () -> Unit,
    onSwipeToLeft: () -> Unit
) {
    var swipeStartX by remember { mutableStateOf(0f) }

    BoxWithConstraints(
        modifier = Modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                detectDragGestures(
                    onDragStart = { offset ->
                        swipeStartX = offset.x
                    },
                    onDrag = { _, _ -> },
                    onDragEnd = {
                        // Drag ended - calculate swipe
                    },
                    onDragCancel = {
                        swipeStartX = 0f
                    }
                )
            }
            .pointerInput(Unit) {
                detectHorizontalDragGestures { change, dragAmount ->
                    change.consume()

                    val swipeEndX = change.position.x
                    val swipeDistance = swipeEndX - swipeStartX
                    val swipeThreshold = 50f // Minimum swipe distance

                    when {
                        // ✅ Swipe LEFT (right to left)
                        swipeDistance < -swipeThreshold -> {
                            Log.d("HomeScreen", "Swipe LEFT detected")
                            onSwipeToLeft() // Navigate to Settings
                            swipeStartX = 0f
                        }
                    }
                }
            }
    ) {
        val screenHeight = maxHeight
        val screenWidth = maxWidth
        val navbarHeight = 90.dp / maxHeight
        val sectionMain = 1.0f - navbarHeight

        // Background image
        Image(
            painter = painterResource(R.drawable.app_background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        // Main content
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .fillMaxHeight(sectionMain)
        ) {
            Box(modifier = Modifier.height(screenHeight * 0.045f))

            // Logo
            Image(
                painter = painterResource(R.drawable.vision_assist_logo),
                contentDescription = "app logo",
                modifier = Modifier.size(Constants.LOGO_SIZE.dp)
            )

            // Title with typewriter animation
            TypewriterText(
                text = titleText,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp)
            )

            Box(modifier = Modifier.height(screenWidth * 0.05f))

            // Detection Button with options
            DetectionButtonSection(
                showOptions = showDetectionOptions,
                iconColor = detectionIconColor,
                selectedOption = selectedDetectionOption,
                screenWidth = screenWidth,
                showInfoButton = AppConfig.showTutorial,
                onDetectionClick = onDetectionClick,
                onIconPress = onDetectionClick,
                onOptionSelected = onDetectionOptionSelected,
                onInfoClick = onDetectionInfoClick,
                navigate = navigateFun
            )


            Box(modifier = Modifier.height(screenHeight * 0.04f))

            // Caption Button
            CaptionButtonSection(
                screenWidth = screenWidth,
                showInfoButton = AppConfig.showTutorial,
                onCaptionClick = onCaptionClick,
                onInfoClick = onCaptionInfoClick
            )

            //Box(modifier = Modifier.height(screenHeight * 0.21f))

            Spacer(modifier = Modifier.weight(1f))

            SyncStatusSection(
                syncStatus = syncStatus,
                syncDays = syncDays,
                showInfoButton = AppConfig.showTutorial && AppConfig.mainLanguage.code == "en",
                onInfoClick = onSpeakInfoClick
            )
        }


        // Bottom Navigation Bar
        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .fillMaxHeight(navbarHeight),
        ) {
            BottomNavigationBar(
                onNavigateHome = onNavigateHome,
                onNavigateReports = onNavigateReports,
                onNavigateSettings = onNavigateSettings,
                showReports = AppConfig.env_reports
            )
        }

        // Speech Recognition Dialog
        SpeechRecognitionDialog(
            isVisible = showSpeechDialog,
            speechText = speechText,
            processText = speechProcessText,
            isSpeaking = isSpeaking,
            onTap = onSpeechDialogTap
        )
    }
}

@Composable
fun TypewriterText(
    text: String,
    modifier: Modifier = Modifier
) {
    var displayedText by remember { mutableStateOf("") }

    LaunchedEffect(text) {
        displayedText = ""
        text.forEachIndexed { index, _ ->
            delay(70) // Typing speed
            displayedText = text.take(index + 1)
        }
    }

    // Parse text with ~ separator
    val parts = displayedText.split("~")

    Text(
        text = buildAnnotatedString {
            when (parts.size) {
                1 -> {
                    withStyle(
                        style = SpanStyle(
                            color = colorResource(R.color.std_purple),
                            fontFamily = robotoSemibold,
                            fontSize = Constants.STD_SUBTITLE_SIZE.sp
                        )
                    ) {
                        append(parts[0])
                    }
                }

                3 -> {
                    withStyle(
                        style = SpanStyle(
                            color = colorResource(R.color.std_purple),
                            fontFamily = robotoSemibold,
                            fontSize = Constants.STD_SUBTITLE_SIZE.sp
                        )
                    ) {
                        append(parts[0])
                    }
                    withStyle(
                        style = SpanStyle(
                            color = colorResource(R.color.std_purple_dark),
                            fontFamily = robotoExtraBold,
                            fontSize = Constants.STD_SUBTITLE_SIZE.sp
                        )
                    ) {
                        append(parts[1])
                    }
                    withStyle(
                        style = SpanStyle(
                            color = colorResource(R.color.std_purple),
                            fontFamily = robotoSemibold,
                            fontSize = Constants.STD_SUBTITLE_SIZE.sp
                        )
                    ) {
                        append(parts[2])
                    }
                }
            }
        },
        textAlign = TextAlign.Center,
        modifier = modifier
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DetectionButtonSection(
    showOptions: Boolean,
    iconColor: Int,
    selectedOption: HomeActivity.DetectionOption?,
    screenWidth: Dp,
    showInfoButton: Boolean,
    onDetectionClick: () -> Unit,
    onIconPress: () -> Unit,
    onOptionSelected: (HomeActivity.DetectionOption?, navigate: Boolean) -> Unit,
    onInfoClick: () -> Unit,
    navigate: () -> Unit
) {
    // Button dimensions
    val buttonHeight = Constants.STD_BUTTON_PAGE_HEIGHT.dp
    val optionWidth = screenWidth * 0.21f
    val detectionX = screenWidth * 0.23f

    // ✅ NEW: Drag detection state
    val density = LocalDensity.current

    // Helper function to determine which button is being hovered
    fun getHoveredOption(position: Offset): HomeActivity.DetectionOption? {
        if (!showOptions) return null

        with(density) {
            // Static button bounds (left of detection)
            val staticLeft = (detectionX - optionWidth - 5.dp).toPx()
            val staticRight = (detectionX - 5.dp).toPx()
            val staticTop = buttonHeight.toPx()
            val staticBottom = (buttonHeight + buttonHeight).toPx()

            if (position.x in staticLeft..staticRight &&
                position.y in staticTop..staticBottom
            ) {
                return HomeActivity.DetectionOption.STATIC
            }

            // Live button bounds (above detection, rotated)
            val liveLeft = detectionX.toPx()
            val liveRight = (detectionX + buttonHeight).toPx() // Rotated: width is height
            val liveTop = (buttonHeight - optionWidth - 4.dp).toPx()
            val liveBottom = (buttonHeight - 4.dp).toPx()

            if (position.x in liveLeft..liveRight &&
                position.y in liveTop..liveBottom
            ) {
                return HomeActivity.DetectionOption.LIVE
            }
        }

        return null
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height((Constants.STD_BUTTON_PAGE_HEIGHT).dp + optionWidth)
            .pointerInput(showOptions) {
                // ✅ Detect drag gestures when options are visible
                if (showOptions) {
                    detectDragGestures(
                        onDragStart = { offset ->
                            onOptionSelected(getHoveredOption(offset), false)
                        },
                        onDrag = { change, _ ->
                            onOptionSelected(getHoveredOption(change.position), false)
                        },
                        onDragEnd = {
                            navigate()
                        },
                        onDragCancel = {
                            navigate()
                        }
                    )
                }
            }
    ) {
        // === STATIC BUTTON (Slides LEFT) ===
        AnimatedVisibility(
            visible = showOptions,
            enter = slideInHorizontally(
                initialOffsetX = { it },
                animationSpec = tween(250)
            ),
            exit = slideOutHorizontally(
                targetOffsetX = { it },
                animationSpec = tween(250)
            ),
            modifier = Modifier
                .offset(
                    x = detectionX - optionWidth - 5.dp,
                    y = buttonHeight
                )
        ) {
            OptionButton(
                text = "Static",
                isSelected = selectedOption == HomeActivity.DetectionOption.STATIC,
                onClick = { onOptionSelected(HomeActivity.DetectionOption.STATIC, true) },
                width = optionWidth,
                height = buttonHeight,
                isRotated = false
            )
        }

        // === LIVE BUTTON (Slides UP, rotated 90°) ===
        AnimatedVisibility(
            visible = showOptions,
            enter = slideInVertically(
                initialOffsetY = { it },
                animationSpec = tween(250)
            ),
            exit = slideOutVertically(
                targetOffsetY = { it },
                animationSpec = tween(250)
            ),
            modifier = Modifier
                .offset(
                    x = detectionX,
                    y = buttonHeight - optionWidth - 8.dp
                )
        ) {
            OptionButton(
                text = "Live",
                isSelected = selectedOption == HomeActivity.DetectionOption.LIVE,
                onClick = { onOptionSelected(HomeActivity.DetectionOption.LIVE, true) },
                width = optionWidth,
                height = buttonHeight,
                isRotated = true
            )
        }

        // === DETECTION BUTTON (Fixed position) ===
        Row(
            modifier = Modifier
                .offset(x = detectionX, y = buttonHeight)
                .fillMaxWidth(0.77f),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            MainActionButton(
                shape = RoundedCornerShape(
                    topStart = 5.dp,
                    topEnd = 31.dp,
                    bottomStart = 5.dp,
                    bottomEnd = 5.dp
                ),
                text = "Detection",
                iconRes = R.drawable.detection_icon,
                iconColor = iconColor,
                screenWidth = screenWidth,
                onClick = onDetectionClick,
                onIconPress = onIconPress
            )

            if (showInfoButton) {
                InfoButtonWithPulse(
                    onClick = onInfoClick,
                    isPulsing = true
                )
            }
        }
    }
}

// === UNIFIED OPTION BUTTON (handles both regular and rotated) ===
@Composable
fun OptionButton(
    text: String,
    isSelected: Boolean,
    onClick: () -> Unit,
    width: Dp,
    height: Dp,
    isRotated: Boolean
) {
    if (isRotated) {
        // Rotated button (for "Live" - vertical)
        Button(
            onClick = onClick,
            modifier = Modifier
                .width(height) // Swap dimensions
                .height(height)
                .shadow(
                    3.dp, RoundedCornerShape(
                        topStart = 31.dp,
                        topEnd = 31.dp,
                        bottomStart = 5.dp,
                        bottomEnd = 5.dp
                    )
                ),
            shape = RoundedCornerShape(
                topStart = 31.dp,
                topEnd = 31.dp,
                bottomStart = 5.dp,
                bottomEnd = 5.dp
            ),
            colors = ButtonDefaults.buttonColors(
                containerColor = if (isSelected) {
                    colorResource(R.color.std_purple_dark)
                } else {
                    colorResource(R.color.std_cyan)
                },
                contentColor = Color.White
            ),
            contentPadding = PaddingValues(0.dp)
        ) {
            Text(
                text = text,
                fontSize = 22.sp,
                fontFamily = robotoExtraBoldItalic
            )
        }
    } else {
        // Regular button (for "Static" - horizontal)
        Button(
            onClick = onClick,
            modifier = Modifier
                .width(width)
                .height(height)
                .shadow(
                    3.dp, RoundedCornerShape(
                        topStart = 31.dp,
                        topEnd = 5.dp,
                        bottomStart = 31.dp,
                        bottomEnd = 5.dp
                    )
                ),
            shape = RoundedCornerShape(
                topStart = 31.dp,
                topEnd = 5.dp,
                bottomStart = 31.dp,
                bottomEnd = 5.dp
            ),
            colors = ButtonDefaults.buttonColors(
                containerColor = if (isSelected) {
                    colorResource(R.color.std_purple_dark)
                } else {
                    colorResource(R.color.std_cyan)
                },
                contentColor = Color.White
            ),
            contentPadding = PaddingValues(0.dp)
        ) {
            Text(
                text = text,
                fontSize = 22.sp,
                fontFamily = robotoExtraBoldItalic
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainActionButton(
    shape: Shape,
    predefinedIcon: ImageVector? = null,
    text: String,
    iconRes: Int = 0,
    iconColor: Int,
    screenWidth: Dp,
    onClick: () -> Unit,
    onIconPress: () -> Unit
) {
    val buttonWidth = screenWidth * 0.6f
    // Main button
    Button(
        onClick = onClick,
        modifier = Modifier
            .shadow(
                3.dp,
                shape
            )
            .width(buttonWidth)
            .height(Constants.STD_BUTTON_PAGE_HEIGHT.dp),
        shape = shape,
        colors = ButtonDefaults.buttonColors(
            containerColor = Color.White,
            contentColor = colorResource(R.color.std_purple)
        ),
        contentPadding = PaddingValues(0.dp)
    ) {
        // Icon button (circle)
        Box(
            modifier = Modifier
                .size(56.dp)
                .shadow(3.dp, CircleShape)
                .clip(CircleShape)
                .background(colorResource(iconColor))
                .pointerInput(Unit) {
                    detectTapGestures(
                        onTap = {
                            onIconPress()
                        }
                    )
                },
            contentAlignment = Alignment.Center
        ) {
            if (predefinedIcon == null) {
                Icon(
                    painter = painterResource(iconRes),
                    contentDescription = "Action icon",
                    tint = Color.White.copy(alpha = 0.7f),
                    modifier = Modifier.size(28.dp)
                )
            } else {
                Icon(
                    imageVector = predefinedIcon,
                    contentDescription = "Action icon",
                    tint = Color.White.copy(alpha = 0.7f),
                    modifier = Modifier.size(28.dp)
                )
            }
        }
        Spacer(modifier = Modifier.width(buttonWidth * 0.05f))
        Text(
            text = text,
            fontSize = Constants.STD_SUBTITLE_SIZE.sp,
            fontFamily = robotoBold,
            color = colorResource(R.color.std_purple_dark)
        )
    }
}

@Composable
fun CaptionButtonSection(
    screenWidth: Dp,
    showInfoButton: Boolean,
    onCaptionClick: () -> Unit,
    onInfoClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height((Constants.STD_BUTTON_PAGE_HEIGHT).dp)
    ) {
        Row(
            modifier = Modifier
                .offset(x = screenWidth * 0.23f)
                .fillMaxWidth(0.77f),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Caption Button
            MainActionButton(
                shape = RoundedCornerShape(
                    topStart = 5.dp,
                    topEnd = 5.dp,
                    bottomStart = 31.dp,
                    bottomEnd = 31.dp
                ),
                predefinedIcon = Icons.Filled.TextFields,
                text = "Caption",
                iconColor = R.color.std_purple_dark,
                screenWidth = screenWidth,
                onClick = onCaptionClick,
                onIconPress = {}, // No special behavior
            )

            // Info button
            if (showInfoButton) {
                Spacer(modifier = Modifier.width(15.dp))
                InfoButtonWithPulse(
                    onClick = onInfoClick,
                    isPulsing = true
                )
            }
        }
    }
}

@Composable
fun InfoButtonWithPulse(
    onClick: () -> Unit,
    isPulsing: Boolean,
    modifier: Modifier = Modifier
) {
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")

    // Scale animation for the pulse circle (grows from 1.0 to 1.4)
    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 1.3f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 1500,
                easing = FastOutSlowInEasing
            ),
            repeatMode = RepeatMode.Restart
        ),
        label = "pulse_scale"
    )

    // Alpha animation for the pulse circle (fades from 0.4 to 0.0)
    val pulseAlpha by infiniteTransition.animateFloat(
        initialValue = 0.4f,
        targetValue = 0f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = 1500,
                easing = FastOutSlowInEasing
            ),
            repeatMode = RepeatMode.Restart
        ),
        label = "pulse_alpha"
    )

    Box(
        modifier = modifier.size(Constants.STD_INFO_BUTTON_SIZE.dp * 1.3f),
        contentAlignment = Alignment.Center
    ) {
        // Pulse circle (behind the button) - only visible when pulsing
        if (isPulsing) {
            Box(
                modifier = Modifier
                    .size(Constants.STD_INFO_BUTTON_SIZE.dp)
                    .scale(pulseScale)
                    .background(
                        color = colorResource(R.color.std_purple).copy(alpha = pulseAlpha),
                        shape = CircleShape
                    )
            )
        }

        // Actual button (ALWAYS constant size)
        IconButton(
            onClick = onClick,
            modifier = Modifier
                .size(Constants.STD_INFO_BUTTON_SIZE.dp)
                .background(
                    color = colorResource(R.color.std_purple).copy(alpha = 0.0f),
                    shape = CircleShape
                )
        ) {
            Icon(
                imageVector = Icons.Filled.Info,
                contentDescription = "Info",
                tint = colorResource(R.color.std_purple),
                modifier = Modifier.size((Constants.STD_INFO_BUTTON_SIZE).dp)
            )
        }
    }
}

@Composable
fun SyncStatusSection(
    syncStatus: Int,
    syncDays: Int,
    showInfoButton: Boolean,
    onInfoClick: () -> Unit
) {
    // Main row ALWAYS exists
    Row(
        modifier = Modifier
            .fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // === FIRST ROW: 0.7 weight (Sync on LEFT) ===
        Row(
            modifier = Modifier
                .weight(0.7f)
                .padding(start = 10.dp),
            horizontalArrangement = Arrangement.Start,  // Align LEFT
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (syncStatus > 0) {
                // Sync circle icon
                Box(
                    modifier = Modifier
                        .size(24.dp)
                        .clip(CircleShape)
                        .background(
                            if (syncStatus == 1) colorResource(R.color.checked_green) else colorResource(
                                R.color.error_red
                            )
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = if (syncStatus == 1) Icons.Filled.Sync else Icons.Filled.SyncProblem,
                        contentDescription = if (AppConfig.mainLanguage.code == "en") "Sync status" else "Status sincronizare",
                        tint = Color.White,
                        modifier = Modifier.size(21.dp)
                    )
                }

                Spacer(modifier = Modifier.width(5.dp))

                // Sync text
                Text(
                    text = if (syncStatus == 1) {
                        load_syncStatusText(syncDays)
                    } else {
                        load_syncErrorText(LocalContext.current)
                    },
                    fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                    fontFamily = robotoSemibold,
                    color = colorResource(if (syncStatus == 1) R.color.std_cyan_dark else R.color.error_red)
                )
            }
        }

        // === SECOND ROW: Remaining weight (Info on RIGHT) ===
        Row(
            modifier = Modifier.weight(1f - 0.7f), // Rest of the weight
            horizontalArrangement = Arrangement.End,  // Align RIGHT
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (showInfoButton) {
                InfoButtonWithPulse(
                    onClick = onInfoClick,
                    isPulsing = true
                )
            }
        }
    }
}

@Composable
fun BottomNavigationBar(
    onNavigateHome: () -> Unit,
    onNavigateReports: () -> Unit,
    onNavigateSettings: () -> Unit,
    showReports: Boolean
) {
    NavigationBar(
        containerColor = colorResource(R.color.std_light_purple)
    ) {
        NavigationBarItem(
            icon = {
                Icon(
                    imageVector = Icons.Filled.Home,
                    contentDescription = "Home"
                )
            },
            label = {
                Text(
                    text = "Home",
                    fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                    fontFamily = robotoExtraBold
                )

            },
            selected = true,
            colors = NavigationBarItemColors(
                selectedIconColor = Color.White,
                unselectedIconColor = Color(0xFF828188),
                selectedIndicatorColor = colorResource(R.color.std_purple),
                selectedTextColor = Color.Black,
                unselectedTextColor = Color(0xFF828188),
                disabledIconColor = Color.White,
                disabledTextColor = Color.White
            ),
            onClick = onNavigateHome
        )

        if (showReports) {
            NavigationBarItem(
                icon = {
                    Icon(
                        imageVector = Icons.Filled.Description,
                        contentDescription = "Reports"
                    )
                },
                label = {
                    Text(
                        "Reports",
                        fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                        fontFamily = robotoExtraBold
                    )
                },
                selected = false,
                onClick = onNavigateReports,
                colors = NavigationBarItemColors(
                    selectedIconColor = Color.White,
                    unselectedIconColor = Color(0xFF828188),
                    selectedIndicatorColor = colorResource(R.color.std_purple),
                    selectedTextColor = Color.Black,
                    unselectedTextColor = Color(0xFF828188),
                    disabledIconColor = Color.White,
                    disabledTextColor = Color.White
                )
            )
        }

        NavigationBarItem(
            icon = {
                Icon(
                    imageVector = Icons.Filled.Settings,
                    contentDescription = "Settings"
                )
            },
            label = {
                Text(
                    "Settings",
                    fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                    fontFamily = robotoExtraBold
                )
            },
            selected = false,
            onClick = onNavigateSettings,
            colors = NavigationBarItemColors(
                selectedIconColor = Color.White,
                unselectedIconColor = Color(0xFF615E65),
                selectedIndicatorColor = colorResource(R.color.std_purple),
                selectedTextColor = Color.Black,
                unselectedTextColor = Color(0xFF828188),
                disabledIconColor = Color.White,
                disabledTextColor = Color.White
            )
        )
    }
}

@Composable
fun SpeechRecognitionDialog(
    isVisible: Boolean,
    speechText: String,
    processText: String,
    isSpeaking: Boolean,
    onTap: () -> Unit
) {
    AnimatedVisibility(
        visible = isVisible,
        enter = fadeIn(animationSpec = tween(Constants.ANIMATION_DELAY)),
        exit = fadeOut(animationSpec = tween(Constants.ANIMATION_DELAY))
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Gray.copy(alpha = 0.8f))
                .pointerInput(Unit) {
                    detectTapGestures {
                        onTap()
                    }
                },
            contentAlignment = Alignment.Center
        ) {
            val hasProcessText = processText.isNotEmpty()

            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center,
                modifier = Modifier
                    .fillMaxWidth(0.85f)
                    .animateContentSize()
            ) {
                // === TOP TEXT (processText when exists, speechText when it doesn't) ===
                AnimatedContent(
                    targetState = hasProcessText,
                    transitionSpec = {
                        if (targetState) {
                            // ProcessText appearing: fade in
                            fadeIn(animationSpec = tween(Constants.ANIMATION_DELAY))
                                .togetherWith(fadeOut(animationSpec = tween(0)))
                        } else {
                            // ProcessText disappearing: instant reset (no animation)
                            fadeIn(animationSpec = tween(0))
                                .togetherWith(fadeOut(animationSpec = tween(0)))
                        }
                    },
                    label = "top_text_animation"
                ) { showingProcess ->
                    if (showingProcess) {
                        // Show processText at top (title size, white)
                        Text(
                            text = processText,
                            fontSize = Constants.STD_TITLE_SIZE.sp,
                            fontFamily = robotoSemibold,
                            color = Color.White,
                            textAlign = TextAlign.Center
                        )
                    } else {
                        // Show speechText at top (title size, colored)
                        Text(
                            text = speechText,
                            fontSize = Constants.STD_TITLE_SIZE.sp,
                            fontFamily = robotoSemibold,
                            color = if (isSpeaking) {
                                Color.White
                            } else {
                                colorResource(R.color.std_light_purple)
                            },
                            textAlign = TextAlign.Center
                        )
                    }
                }

                // === BOTTOM TEXT (speechText slides down when processText appears) ===
                if (hasProcessText) {
                    Spacer(modifier = Modifier.height(24.dp))

                    // Animated appearance of bottom speechText
                    AnimatedVisibility(
                        visible = true,
                        enter = slideInVertically(
                            initialOffsetY = { -it / 2 }, // Slide from above
                            animationSpec = tween(Constants.ANIMATION_DELAY)
                        ) + fadeIn(animationSpec = tween(Constants.ANIMATION_DELAY))
                    ) {
                        Text(
                            text = speechText,
                            fontSize = Constants.STD_SLIDER_INFO_SIZE.sp, // Small font
                            fontFamily = robotoSemibold,
                            color = if (isSpeaking) {
                                Color.White
                            } else {
                                colorResource(R.color.std_light_purple)
                            },
                            textAlign = TextAlign.Center
                        )
                    }
                }
            }
        }
    }
}

@Preview(name = "Home Activity", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun HomeActivityPreview() {
    HomeScreen(
        titleText = "Hello ~Eduard~,\nwhat can I do for you?",
        syncStatus = 1,
        syncDays = 5,
        showDetectionOptions = false,
        detectionIconColor = R.color.std_purple_dark,
        selectedDetectionOption = null,
        showSpeechDialog = false,
        speechText = "",
        speechProcessText = "",
        isSpeaking = true,
        onDetectionClick = {},
        onDetectionOptionSelected = { _, _ -> },
        onCaptionClick = {},
        onDetectionInfoClick = {},
        onCaptionInfoClick = {},
        onSpeakInfoClick = {},
        onNavigateHome = {},
        onNavigateReports = {},
        onNavigateSettings = {},
        onSpeechDialogTap = {},
        navigateFun = {},
        onSwipeToLeft = {}
    )
}

@Preview(
    name = "Home Activity - Detection Options Visible",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
@Composable
fun HomeActivityWithOptionsPreview() {
    HomeScreen(
        titleText = "Hello ~Eduard~,\nwhat can I do for you?",
        syncStatus = 1,
        syncDays = 5,
        showDetectionOptions = true,
        detectionIconColor = R.color.std_cyan,
        selectedDetectionOption = HomeActivity.DetectionOption.LIVE,
        showSpeechDialog = false,
        speechText = "",
        speechProcessText = "",
        isSpeaking = true,
        onDetectionClick = {},
        onDetectionOptionSelected = { _, _ -> },
        onCaptionClick = {},
        onDetectionInfoClick = {},
        onCaptionInfoClick = {},
        onSpeakInfoClick = {},
        onNavigateHome = {},
        onNavigateReports = {},
        onNavigateSettings = {},
        onSpeechDialogTap = {},
        navigateFun = {},
        onSwipeToLeft = {}
    )
}

@Preview(
    name = "Home Activity - Speaking Dialog Enabled",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
@Composable
fun HomeActivityWithSpeakingDialogPreview() {
    HomeScreen(
        titleText = "Hello ~Eduard~,\nwhat can I do for you?",
        syncStatus = 1,
        syncDays = 5,
        showDetectionOptions = false,
        detectionIconColor = R.color.std_purple_dark,
        selectedDetectionOption = HomeActivity.DetectionOption.LIVE,
        showSpeechDialog = true,
        speechText = "Where is my phone, tablet, apple, and laptop",
        speechProcessText = "ddd",
        isSpeaking = false,
        onDetectionClick = {},
        onDetectionOptionSelected = { _, _ -> },
        onCaptionClick = {},
        onDetectionInfoClick = {},
        onCaptionInfoClick = {},
        onSpeakInfoClick = {},
        onNavigateHome = {},
        onNavigateReports = {},
        onNavigateSettings = {},
        onSpeechDialogTap = {},
        navigateFun = {},
        onSwipeToLeft = {}
    )
}
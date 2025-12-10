@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.main

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.util.Log
import android.view.KeyEvent
import android.view.accessibility.AccessibilityManager
import androidx.activity.compose.setContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.BaseActivity
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.tabs.home.caption.BlindCaptionActivity
import com.visionassist.appspace.activities.tabs.home.detection.BlindDetectionActivity
import com.visionassist.appspace.activities.tabs.home.findmyobjects.BlindFindMyObjectActivity
import com.visionassist.appspace.activities.tabs.settings.BlindSettingsActivity
import com.visionassist.appspace.models.sttengine.SpeechRecognizer
import com.visionassist.appspace.sound.SoundConstants
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.BackgroundTaskExecutor
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.CustomColorSchema
import com.visionassist.appspace.utils.PermissionChecker
import com.visionassist.appspace.utils.TypewriterColorSchema
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.load_captionTutorialSpeak
import com.visionassist.appspace.utils.load_detectionTutorialSpeak
import com.visionassist.appspace.utils.load_errorSTT
import com.visionassist.appspace.utils.load_errorSTTRuntime
import com.visionassist.appspace.utils.load_homePageIntro
import com.visionassist.appspace.utils.load_navigateToSettings
import com.visionassist.appspace.utils.load_speakTutorialSpeak
import com.visionassist.appspace.utils.load_syncErrorSpeech
import com.visionassist.appspace.utils.load_syncStatusTextSpeech
import com.visionassist.appspace.utils.load_talkbackError
import com.visionassist.appspace.utils.load_tutorialDialog
import com.visionassist.appspace.utils.load_unavailableSTT
import com.visionassist.appspace.utils.robotoSemibold
import com.visionassist.appspace.utils.vibrate

class BlindHomeActivity : BaseActivity() {
    private val TAG = "BlindHomeActivity"

    private val enableTutorial = mutableStateOf(false)
    private val currentTutorialText = mutableStateOf("")

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
    private var requestToStop = false
    private var locked = false
    private var lockedVolumeDown = false
    private var uiLocked = false
    private var handleVolumeDownControl = false

    // Runnable for navigation to an activity
    private var onPermissionGranted = {}

    // Main handler
    private val mainHandler = Handler(Looper.getMainLooper())
    private lateinit var afterResumeRunnable: Runnable
    private var hasToExecAfterResume = false
    private var returnFromFindMyObject = false

    // State management
    private var waitForTalkback = false
    private var tutorialStep = -1
    private var alreadyMenu = false
    private var playSound = true

    private var syncStatus: Int = 0
    private var syncDays: Int = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Get sync status
        val dbManager = PhoneStatusMonitor.getInstance().dbManager
        syncStatus = dbManager.statusOverview
        syncDays = if (syncStatus > 0) dbManager.diffDays.toInt() else 0

        uiLocked = true
        locked = true
        PhoneStatusMonitor.getInstance().soundManager.play(SoundConstants.OPEN_UP_ID, 1f, 1f) {
            checkTalkBackAndProceed()
        }

        setContent {
            BlindHomeScreen(
                enableTutorial = enableTutorial.value,
                tutorialText = currentTutorialText.value,
                showSpeechDialog = showSpeechDialog.value,
                speechText = speechText.value,
                speechProcessText = speechProcessText.value,
                isSpeaking = isSpeaking.value,
                onDetectionClick = ::handleDetectionClick,
                onCaptionClick = ::handleCaptionClick,
                onNavigationHomeClick = ::handleNavigationHomeClick,
                onNavigationSettingsClick = ::handleNavigationSettingsClick,
                onTutorialClick = ::handleTutorialClick,
                onTutorialSwipeUp = ::closeTutorial,
                onSpeechDialogTap = ::handleSpeechDialogTap
            )
        }
    }

    override fun onResume() {
        super.onResume()

        if (PhoneStatusMonitor.getInstance().isReturningFromPermissions)
            PermissionChecker.checkAndRequestPermissions(
                this,
                AppConfig.blindness,
                onPermissionGranted
            )

        if (waitForTalkback) {
            waitForTalkback = false
            checkTutorialAndProceed()
        }

        if (hasToExecAfterResume) {
            hasToExecAfterResume = false
            mainHandler.post(afterResumeRunnable)
        }

        if (returnFromFindMyObject) {
            returnFromFindMyObject = false
            playSound = false
            handleSpeechDialogTap()
        }
    }

    private fun handleSpeechDialogTap() {
        // Cancel speech recognition
        requestToStop = true
        cancelAllHandlers()
        mainHandler.removeCallbacksAndMessages(null)
        soundManager.releaseCallback()
        speechRecognizer.stopListening()
        locked = true
        lockedVolumeDown = false

        val waitTime =
            if (playSound)
                Constants.ANIMATION_DELAY.toLong() + SoundConstants.STT_SPEAK_CLOSE_MS.toLong() - Constants.ANIMATION_DELAY.toLong()
            else
                Constants.ANIMATION_DELAY.toLong()

        if (playSound)
            soundManager.play(SoundConstants.STT_SPEAK_CLOSE_ID, 0.7f, 0.7f) {}
        showSpeechDialog.value = false
        mainHandler.postDelayed(
            {
                uiLocked = false
                retrySpeech.value = false
                sendSpeech.value = false
                isSpeaking.value = false
                speechText.value = ""
                speechProcessText.value = ""
                locked = false
                playSound = true
            },
            waitTime
        )
    }

    private fun checkTalkBackAndProceed() {
        if (!isTalkBackEnabled()) {
            ttsManager.speak(
                load_talkbackError(this),
                AppConfig.tts_pitch,
                AppConfig.tts_speech_rate,
                true,
                null
            )
            waitForTTSSpeech { openTalkBackSettings() }
        } else {
            checkTutorialAndProceed()
        }
    }

    private fun waitForTTSSpeech(afterTTSSpeech: Runnable) {
        val checkRunnable = object : Runnable {
            override fun run() {
                if (ttsManager.isDoneSpeaking) {
                    mainHandler.post(afterTTSSpeech)
                } else {
                    mainHandler.postDelayed(this, 500)
                }
            }
        }
        mainHandler.post(checkRunnable)
    }

    private fun checkTutorialAndProceed() {
        if (AppConfig.showTutorial) {
            showTutorialDialog()
        } else {
            homePageDescription()
        }
    }

    private fun showTutorialDialog() {
        enableTutorial.value = true
        mainHandler.postDelayed({ speakTutorialStep() }, Constants.ANIMATION_DELAY.toLong())
    }

    private fun speakTutorialStep() {
        tutorialStep++
        if (tutorialStep == 5)
            if (AppConfig.mainLanguage.code != "en") {
                tutorialStep--
                if (!alreadyMenu)
                    speakMenuAndCloseTutorial()
                else
                    closeTutorial()
                return
            }
        if (tutorialStep == 12) {
            tutorialStep--
            if (!alreadyMenu)
                speakMenuAndCloseTutorial()
            else
                closeTutorial()
            return
        }
        val textToSpeak: String = when (tutorialStep) {
            0 -> {
                load_tutorialDialog(this)
            }

            1, 2 -> {
                load_detectionTutorialSpeak(this, tutorialStep).replace("~", "")
            }

            3, 4 -> {
                load_captionTutorialSpeak(this, tutorialStep).replace("~", "")
            }

            else -> load_speakTutorialSpeak(this, tutorialStep).replace("~", "")
        }

        ttsManager.speak(textToSpeak, AppConfig.tts_pitch, AppConfig.tts_speech_rate, true, null)
        mainHandler.postDelayed({
            currentTutorialText.value = textToSpeak
        }, SoundConstants.TTS_REPEAT_MS.toLong())
    }

    private fun handleTutorialClick() {
        cancelAllHandlers()
        speakTutorialStep()
    }

    private fun speakMenuAndCloseTutorial() {
        cancelAllHandlers()
        alreadyMenu = true
        val textToSpeak = load_homePageIntro(this)
        ttsManager.speak(textToSpeak, AppConfig.tts_pitch, AppConfig.tts_speech_rate, true, null)
        currentTutorialText.value = textToSpeak
    }

    private fun closeTutorial() {
        cancelAllHandlers()
        enableTutorial.value = false
        mainHandler.postDelayed({ homePageDescription() }, Constants.ANIMATION_DELAY.toLong())
    }

    private fun homePageDescription() {
        val textBefore =
            if (ttsManager.currentLocale.language == "en") "Account status: " else "Status cont: "
        when (syncStatus) {
            0 -> unlockUI()
            1 -> {
                ttsManager.speak(
                    textBefore + load_syncStatusTextSpeech(syncDays),
                    AppConfig.tts_pitch,
                    AppConfig.tts_speech_rate,
                    true,
                    null
                )
                waitForTTSSpeech { unlockUI() }
            }

            2 -> {
                ttsManager.speak(
                    textBefore + load_syncErrorSpeech(this),
                    AppConfig.tts_pitch,
                    AppConfig.tts_speech_rate,
                    true,
                    null
                )
                waitForTTSSpeech { unlockUI() }
            }
        }
    }

    private fun unlockUI() {
        uiLocked = false
        locked = false
        Log.i(TAG, "UI unlocked")
    }

    private fun checkPhoneStatusAndNavigate(onSuccess: () -> Unit) {
        PhoneStatusMonitor.getInstance().checkPhoneStatus()
        // If check passes, execute success callback
        onSuccess()
    }

    private fun handleDetectionClick() {
        if (uiLocked) return

        onPermissionGranted = {
            checkPhoneStatusAndNavigate {
                val intent = Intent(this, BlindDetectionActivity::class.java)
                startActivity(intent)
                finish()
            }
        }
        PermissionChecker.checkAndRequestPermissions(this, AppConfig.blindness, onPermissionGranted)
    }

    private fun handleCaptionClick() {
        if (uiLocked) return

        onPermissionGranted = {
            checkPhoneStatusAndNavigate {
                val intent = Intent(this, BlindCaptionActivity::class.java)
                startActivity(intent)
                finish()
            }
        }
        PermissionChecker.checkAndRequestPermissions(this, AppConfig.blindness, onPermissionGranted)
    }

    private fun handleNavigationHomeClick() {
        if (uiLocked) return

        if (AppConfig.haptics) vibrate(haptic_model0())
    }

    private fun handleNavigationSettingsClick() {
        if (uiLocked) return

        if (AppConfig.haptics) vibrate(haptic_model0())

        val navText = load_navigateToSettings(this)
        ttsManager.speak(navText, AppConfig.tts_pitch, AppConfig.tts_speech_rate, false, null)
        val intent = Intent(this, BlindSettingsActivity::class.java)
        startActivity(intent)

    }

    private fun isTalkBackEnabled(): Boolean {
        val am = getSystemService(ACCESSIBILITY_SERVICE) as AccessibilityManager
        return am.isEnabled
    }

    private fun openTalkBackSettings() {
        waitForTalkback = true
        val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
        startActivity(intent)
    }

    private fun cancelAllHandlers() {
        mainHandler.removeCallbacksAndMessages(null)
        ttsManager.stopSpeaking()
    }

    private fun sendProcessedSpeech() {
        val classIndices = IntArray(matchedIndices.size) { matchedIndices[it].classIndex }
        val matchedWords = Array(matchedIndices.size) { matchedIndices[it].matchedWord }

        val intent = Intent(this, BlindFindMyObjectActivity::class.java).apply {
            putExtra(Constants.EXTRA_MATCHED_INDICES, classIndices)
            putExtra(Constants.EXTRA_SYNONYMS_WORDS, matchedWords)
        }
        returnFromFindMyObject = true
        startActivity(intent)

        //finish()
    }

    private fun handleSingleVolumeDown() {
        when {
            sendSpeech.value -> {
                locked = true
                cancelAllHandlers()
                ttsManager.speak(
                    "Navigating to find my objects page",
                    AppConfig.tts_pitch,
                    AppConfig.tts_speech_rate,
                    false,
                    null
                )

                // Process recognized text
                waitForTTSSpeech { sendProcessedSpeech() }
            }

            else -> {
                if (!lockedVolumeDown) {
                    uiLocked = false
                    // Launch caption activity
                    ttsManager.speak(
                        if (ttsManager.currentLanguage == "en")
                            "Opening caption page" else "Se deschide pagina de descriere textuală",
                        AppConfig.tts_pitch,
                        AppConfig.tts_speech_rate,
                        false,
                        haptic_model0()
                    )
                    waitForTTSSpeech { handleCaptionClick() }
                }
            }
        }
    }

    private fun launchSpeechRecognition(firstTimeSpeak: Boolean) {
        requestToStop = false
        if (firstTimeSpeak) {
            if (AppConfig.mainLanguage.code == "ro") {
                hasToExecAfterResume = true
                soundManager.play(SoundConstants.STT_ERROR_ID, 0.7f, 0.7f) {
                    ttsManager.speak(
                        load_unavailableSTT(this),
                        AppConfig.tts_pitch,
                        AppConfig.tts_speech_rate,
                        true,
                        null
                    )
                }
                afterResumeRunnable = object : Runnable {
                    override fun run() {
                        if (ttsManager.isDoneSpeaking) {
                            uiLocked = false
                            locked = false
                        } else {
                            mainHandler.postDelayed(this, Constants.LOAD_CHECK_DELAY_MS.toLong())
                        }
                    }
                }
                mainHandler.postDelayed(
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
                            true,
                            null
                        )
                    }
                    afterResumeRunnable = object : Runnable {
                        override fun run() {
                            if (ttsManager.isDoneSpeaking) {
                                uiLocked = false
                                locked = false
                            } else {
                                mainHandler.postDelayed(
                                    this,
                                    Constants.LOAD_CHECK_DELAY_MS.toLong()
                                )
                            }
                        }
                    }
                    mainHandler.postDelayed(
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
        lockedVolumeDown = false
        isSpeaking.value = true
        speechText.value = "Listening..."
        soundManager.play(SoundConstants.STT_SPEAK_OPEN_ID, 0.7f, 0.7f) {
            speechRecognizer.startListening(object :
                SpeechRecognizer.RecognitionCallback {
                override fun onResult(recognizedText: String, isFinalResult: Boolean) {
                    if (isFinalResult && !requestToStop) {
                        isSpeaking.value = false
                        speechText.value = formatRecognizedText(recognizedText)
                        //vibrate(haptic_model0())
                        mainHandler.postDelayed({ processRecognizedSpeech() }, 1000)
                        /*
                        sendSpeech.value = true
                        retrySpeech.value = true
                        locked = false
                        vibrateIfEnabled()
                        */
                    } else {
                        if (!isFinalResult)
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
                                mainHandler.postDelayed(
                                    this,
                                    Constants.LOAD_CHECK_DELAY_MS.toLong()
                                )
                            }
                        }
                    }
                    mainHandler.postDelayed(
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
                    mainHandler.postDelayed(this, 500)
                }
            }
        }
        mainHandler.postDelayed(checkRunnable, 2000)
    }

    private fun processTextOutput() {
        if (matchedIndices.isEmpty()) {
            speechProcessText.value = "No matched known classes"
            ttsManager.speak(
                speechProcessText.value,
                AppConfig.tts_pitch,
                AppConfig.tts_speech_rate,
                false,
                haptic_model0()
            )
            retrySpeech.value = true
            sendSpeech.value = false
            lockedVolumeDown = true
            locked = false
        } else {
            speechProcessText.value = "Matched: ${classNames.joinToString(", ")}"
            ttsManager.speak(
                speechProcessText.value,
                AppConfig.tts_pitch,
                AppConfig.tts_speech_rate,
                false,
                haptic_model0()
            )
            retrySpeech.value = true
            sendSpeech.value = true
            locked = false
        }
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
                    cancelAllHandlers()
                    //uiLocked=true
                    if (retrySpeech.value) {
                        // Retry speech recognition
                        locked = true
                        resetSpeechStates()
                        launchSpeechRecognition(false)
                    } else {
                        //uiLocked = true
                        locked = true
                        // Launch static detection
                        ttsManager.speak(
                            if (ttsManager.currentLanguage == "en")
                                "Opening detection page" else "Se deschide pagina de detecție",
                            AppConfig.tts_pitch,
                            AppConfig.tts_speech_rate,
                            false,
                            haptic_model0()
                        )
                        waitForTTSSpeech { handleDetectionClick() }
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
                        cancelAllHandlers()
                        mainHandler.removeCallbacksAndMessages(null)
                        mainHandler.postDelayed(
                            { handleSingleVolumeDown() }, Constants.VOLUME_DOWN_DELAY_MS.toLong()
                        )
                    } else {
                        mainHandler.removeCallbacksAndMessages(null)
                        locked = true
                        handleVolumeDownControl = false
                        launchSpeechRecognition(true)
                    }
                } else {
                    if (!ttsManager.isDoneSpeaking)
                        ttsManager.onVolumeDownPressed()
                }
                return true
            }
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onPause() {
        super.onPause()
        mainHandler.removeCallbacksAndMessages(null)
        soundManager.releaseCallback()
        ttsManager.stopSpeaking()
    }
}

@Composable
fun BlindHomeScreen(
    enableTutorial: Boolean,
    tutorialText: String,
    showSpeechDialog: Boolean,
    speechText: String,
    speechProcessText: String,
    isSpeaking: Boolean,
    onDetectionClick: () -> Unit,
    onCaptionClick: () -> Unit,
    onNavigationHomeClick: () -> Unit,
    onNavigationSettingsClick: () -> Unit,
    onTutorialClick: () -> Unit,
    onTutorialSwipeUp: () -> Unit,
    onSpeechDialogTap: () -> Unit,
) {
    BoxWithConstraints(
        modifier = Modifier
            .fillMaxSize()
    ) {
        val screenHeight = maxHeight
        val screenWidth = maxWidth
        val navbarHeight = 90.dp / maxHeight
        val sectionMain = 1.0f - navbarHeight

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
                                    onNavigationSettingsClick()
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

            // Background
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
                Box(modifier = Modifier.height(screenHeight * 0.09f))

                // Logo at top (250dp instead of 200dp)
                Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                    Image(
                        painter = painterResource(R.drawable.vision_assist_logo),
                        contentDescription = "app logo",
                        modifier = Modifier.size(Constants.BLIND_LOGO_SIZE.dp)
                    )
                }

                // Buttons column
                Column(
                    modifier = Modifier.weight(1f),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    // Detection row (full width clickable)
                    DetectionRow(
                        onClick = onDetectionClick,
                        screenWidth = screenWidth,
                        screenHeight = screenHeight
                    )

                    //Box(modifier = Modifier.height(screenHeight * 0.04f))

                    // Caption row (full width clickable)
                    CaptionRow(
                        onClick = onCaptionClick,
                        screenWidth = screenWidth,
                        screenHeight = screenHeight
                    )

                    // Navigation areas (two clickable columns)
                    NavigationAreas(
                        onHomeClick = onNavigationHomeClick,
                        onSettingsClick = onNavigationSettingsClick
                    )
                }
            }

            Box(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .fillMaxHeight(navbarHeight),
            ) {
                BottomNavigationBar(
                    onNavigateHome = onNavigationHomeClick,
                    onNavigateReports = {},
                    onNavigateSettings = onNavigationSettingsClick,
                    showReports = AppConfig.env_reports
                )
            }

            // Tutorial overlay
            TutorialOverlay(
                screenHeight = screenHeight,
                text = tutorialText,
                textShowSpeed =
                if (PhoneStatusMonitor.getInstance().ttsManager.currentLocale.language == "en")
                    (Constants.TTS_CHAR_DELAY_EN / AppConfig.tts_speech_rate).toLong()
                else
                    (Constants.TTS_CHAR_DELAY_RO / AppConfig.tts_speech_rate).toLong(),
                onClick = onTutorialClick,
                onSwipeUp = onTutorialSwipeUp,
                enabled = enableTutorial
            )

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
}

@Composable
fun DetectionRow(
    onClick: () -> Unit,
    screenWidth: Dp,
    screenHeight: Dp,
) {
    val rowHeight = Constants.STD_BUTTON_PAGE_HEIGHT.dp + screenHeight * 0.075f

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(rowHeight)
            .clickable(onClick = onClick)
            .semantics {
                contentDescription =
                    if (AppConfig.mainLanguage.code == "en") "Detection button" else "Buton de detecție"
            },
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.Bottom
    ) {
        MainActionButton(
            shape = RoundedCornerShape(
                topStart = 5.dp,
                topEnd = 31.dp,
                bottomStart = 5.dp,
                bottomEnd = 5.dp
            ),
            text = if (AppConfig.mainLanguage.code == "en") "Detection" else "Detecție",
            iconRes = R.drawable.detection_icon,
            iconColor = R.color.std_purple_dark,
            screenWidth = screenWidth,
            onClick = onClick,
            onIconPress = onClick
        )
    }
}

@Composable
fun CaptionRow(
    onClick: () -> Unit,
    screenWidth: Dp,
    screenHeight: Dp,
) {
    val rowHeight = Constants.STD_BUTTON_PAGE_HEIGHT.dp + screenHeight * 0.075f

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(rowHeight)
            .clickable(onClick = onClick)
            .semantics {
                contentDescription =
                    if (AppConfig.mainLanguage.code == "en") "Caption button" else "Buton descriere textuala"
            },
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.Bottom
    ) {
        MainActionButton(
            shape = RoundedCornerShape(
                topStart = 5.dp,
                topEnd = 31.dp,
                bottomStart = 5.dp,
                bottomEnd = 5.dp
            ),
            text = if (AppConfig.mainLanguage.code == "en") "Caption" else "Descriere textuală",
            iconRes = R.drawable.detection_icon,
            iconColor = R.color.std_purple_dark,
            screenWidth = screenWidth,
            onClick = onClick,
            onIconPress = onClick
        )
    }
}

@Composable
fun NavigationAreas(
    onHomeClick: () -> Unit,
    onSettingsClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
    ) {
        Row(
            modifier = Modifier.fillMaxSize()
        ) {
            // Home column (left 50%)
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .clickable(onClick = onHomeClick)
                    .semantics { contentDescription = "Home button" },
                contentAlignment = Alignment.Center
            ) {}

            // Settings column (right 50%)
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .clickable(onClick = onSettingsClick)
                    .semantics { contentDescription = "Settings button" },
                contentAlignment = Alignment.Center
            ) {}
        }
    }
}

@Composable
fun TutorialOverlay(
    screenHeight: Dp,
    text: String,
    textShowSpeed: Long,
    onClick: () -> Unit,
    onSwipeUp: () -> Unit,
    enabled: Boolean
) {
    AnimatedVisibility(
        visible = enabled,
        enter = fadeIn(animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY)),
        exit = fadeOut(animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY))
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Gray.copy(alpha = 0.8f))
                .pointerInput(Unit) {
                    var swipeStartY = 0f
                    detectVerticalDragGestures(
                        onDragStart = {
                            swipeStartY = 0f
                        },
                        onDragEnd = {
                            val threshold =
                                (screenHeight * Constants.MIN_VDISTANCE_THRESHOLD).toPx()
                            when {
                                swipeStartY <= -threshold -> {
                                    onSwipeUp()
                                }
                            }
                            swipeStartY = 0f
                        },
                        onVerticalDrag = { _, dragAmount ->
                            swipeStartY += dragAmount
                        }
                    )
                }
                .clickable(onClick = onClick),
            contentAlignment = Alignment.Center
        ) {
            TypewriterText(
                text = text,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp),
                colorSchema = TypewriterColorSchema(
                    wordColorSchema = CustomColorSchema(
                        color = Color.White,
                        fontFamily = robotoSemibold,
                        fontSize = Constants.STD_SUBTITLE_SIZE.sp
                    ),
                    outlinedWordColorSchema = CustomColorSchema(
                        color = Color.White,
                        fontFamily = robotoSemibold,
                        fontSize = Constants.STD_SUBTITLE_SIZE.sp
                    )
                ),
                textShowSpeed = textShowSpeed
            )
        }
    }
}

@Preview(
    name = "Blind Home - No Tutorial",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
@Composable
fun BlindHomeScreenPreview() {
    BlindHomeScreen(
        enableTutorial = false,
        tutorialText = "",
        showSpeechDialog = false,
        speechText = "",
        speechProcessText = "",
        isSpeaking = true,
        onDetectionClick = {},
        onCaptionClick = {},
        onNavigationHomeClick = {},
        onNavigationSettingsClick = {},
        onTutorialClick = {},
        onTutorialSwipeUp = {},
        onSpeechDialogTap = {}
    )
}

@Preview(
    name = "Blind Home - W/ Tutorial",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
@Composable
fun BlindHomeScreenWithTutorialPreview() {
    BlindHomeScreen(
        enableTutorial = true,
        tutorialText = "Welcome to VisionAssist application. You are now on the home page and the tutorial display is now on the screen. If you want to just exit the tutorial, slide up. But if you want to hear every step, you will press anywhere on the screen.",
        showSpeechDialog = false,
        speechText = "",
        speechProcessText = "",
        isSpeaking = true,
        onDetectionClick = {},
        onCaptionClick = {},
        onNavigationHomeClick = {},
        onNavigationSettingsClick = {},
        onTutorialClick = {},
        onTutorialSwipeUp = {},
        onSpeechDialogTap = {}
    )
}
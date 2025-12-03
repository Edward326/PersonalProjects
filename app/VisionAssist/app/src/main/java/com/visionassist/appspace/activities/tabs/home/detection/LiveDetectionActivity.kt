@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.tabs.home.detection

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.KeyEvent
import androidx.activity.compose.setContent
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts.TakePicture
import androidx.compose.runtime.mutableStateOf
import androidx.core.content.FileProvider
import com.visionassist.appspace.BaseActivity
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.tabs.home.caption.CaptionActivity
import com.visionassist.appspace.activities.tabs.home.caption.HomeScreen
import com.visionassist.appspace.activities.tabs.home.findmyobjects.FindMyObjectActivity
import com.visionassist.appspace.activities.tabs.reports.EnvironmentReportsActivity
import com.visionassist.appspace.activities.tabs.settings.SettingsActivity
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager
import com.visionassist.appspace.models.sttengine.SpeechRecognizer
import com.visionassist.appspace.sound.SoundConstants
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.BackgroundTaskExecutor
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.DetectionOption
import com.visionassist.appspace.utils.PermissionChecker
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.load_captionTutorial
import com.visionassist.appspace.utils.load_detectionTutorial
import com.visionassist.appspace.utils.load_errorSTT
import com.visionassist.appspace.utils.load_errorSTTRuntime
import com.visionassist.appspace.utils.load_homeTitle
import com.visionassist.appspace.utils.load_speakTutorial
import com.visionassist.appspace.utils.load_unavailableSTT
import com.visionassist.appspace.utils.vibrate
import java.io.File
import java.io.IOException

class LiveDetectionActivity : BaseActivity() {
    private val TAG = "HomeActivity"

    // State variables
    private val titleText = mutableStateOf("")

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
    private var lockedVolumeDown = false
    private var uiLocked = false
    private var handleVolumeDownControl = false

    // Runnable for navigation to an activity
    private var onPermissionGranted = {}
    private var classOpt = 0

    // Camera intent parameters
    private lateinit var takePictureLauncher: ActivityResultLauncher<Uri>
    private lateinit var currentPhotoUri: Uri

    // Main handler
    private val mainHandler = Handler(Looper.getMainLooper())
    private lateinit var afterResumeRunnable: Runnable
    private var hasToExecAfterResume = false
    private var returnFromFindMyObject = false

    private var syncStatus: Int = 0
    private var syncDays: Int = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        registerCameraLauncher()

        // Load initial title
        titleText.value = load_homeTitle()

        // Get sync status
        val dbManager = PhoneStatusMonitor.getInstance().dbManager
        syncStatus = dbManager.statusOverview
        syncDays = if (syncStatus > 0) dbManager.diffDays.toInt() else 0

        uiLocked = true
        locked = true
        PhoneStatusMonitor.getInstance().soundManager.play(SoundConstants.OPEN_UP_ID, 1f, 1f) {
            uiLocked = false
            locked = false
        }

        //Log.d(TAG, FileUtils.readProfileFileAsString(this, Constants.ENV_REPORTS_FILE_NAME))

        setContent {
            HomeScreen(
                titleText = titleText.value,
                syncStatus = syncStatus,
                syncDays = syncDays,
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
            mainHandler.post(afterResumeRunnable)
        }

        if (returnFromFindMyObject) {
            returnFromFindMyObject = false
            handleSpeechDialogTap()
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
            vibrateIfEnabled()
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
                                mainHandler.postDelayed(this, Constants.LOAD_CHECK_DELAY_MS.toLong())
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
                    if (isFinalResult) {
                        isSpeaking.value = false
                        speechText.value = formatRecognizedText(recognizedText)
                        mainHandler.postDelayed({ processRecognizedSpeech() }, 1000)

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
                                mainHandler.postDelayed(this, Constants.LOAD_CHECK_DELAY_MS.toLong())
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
            retrySpeech.value = true
            sendSpeech.value = false
            lockedVolumeDown = true
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
        returnFromFindMyObject = true
        startActivity(intent)

        //finish()
    }

    private fun handleSpeechDialogTap() {
        // Cancel speech recognition
        mainHandler.removeCallbacksAndMessages(null)
        soundManager.releaseCallback()
        speechRecognizer.stopListening()
        locked = true
        lockedVolumeDown = false
        showSpeechDialog.value = false
        mainHandler.postDelayed({
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
                if (!lockedVolumeDown)
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
        mainHandler.removeCallbacksAndMessages(null)
        soundManager.releaseCallback()
        ttsManager.stopSpeaking()
    }
}
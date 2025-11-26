@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.main

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.KeyEvent
import android.view.accessibility.AccessibilityManager
import androidx.activity.compose.setContent
import androidx.compose.animation.*
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectVerticalDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.BaseActivity
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.tabs.home.caption.CaptionCameraXActivity
import com.visionassist.appspace.activities.tabs.home.detection.DetectionCameraXActivity
import com.visionassist.appspace.activities.tabs.settings.BlindSettingsActivity
import com.visionassist.appspace.models.ttsengine.TTSManager
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.PermissionChecker
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.robotoSemibold
import com.visionassist.appspace.utils.vibrate
import kotlin.math.abs

class BlindHomeActivity : BaseActivity() {
    private val TAG = "BlindHomeActivity"

    private val ttsManager: TTSManager = PhoneStatusMonitor.getInstance().ttsManager
    private val dbManager = PhoneStatusMonitor.getInstance().dbManager
    private val mainHandler = Handler(Looper.getMainLooper())

    // State management
    private var talkBackEnabled = false
    private var uiLocked = true
    private var volumeButtonsLocked = true
    private var tutorialStep = 0
    private val tutorialTexts = mutableListOf<Pair<String, String>>()

    // Compose states
    private val showTutorial = mutableStateOf(false)
    private val currentTutorialText = mutableStateOf("")
    private val uiEnabled = mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            BlindHomeScreen(
                showTutorial = showTutorial.value,
                tutorialText = currentTutorialText.value,
                uiEnabled = uiEnabled.value,
                onDetectionClick = ::handleDetectionClick,
                onCaptionClick = ::handleCaptionClick,
                onNavigationHomeClick = ::handleNavigationHomeClick,
                onNavigationSettingsClick = ::handleNavigationSettingsClick,
                onTutorialClick = ::handleTutorialClick,
                onTutorialSwipeUp = ::handleTutorialSwipeUp
            )
        }

        lockUI()
        playOpenSoundAndInitialize()
    }

    private fun playOpenSoundAndInitialize() {
        mainHandler.postDelayed({
            checkTalkBackAndProceed()
        }, 1000)
    }

    private fun checkTalkBackAndProceed() {
        if (!isTalkBackEnabled()) {
            ttsManager.speak(
                loadString(this, "talkback_error"),
                Constants.TTS_PITCH,
                Constants.TTS_SPEECH_RATE,
                false,
                null
            )

            mainHandler.postDelayed({
                openTalkBackSettings()
            }, 3000)

        } else {
            talkBackEnabled = true
            waitForTalkBackConfirmation()
        }
    }

    override fun onResume() {
        super.onResume()

        if (!talkBackEnabled && isTalkBackEnabled()) {
            talkBackEnabled = true
            waitForTalkBackConfirmation()
        }
    }

    private fun waitForTalkBackConfirmation() {
        val checkRunnable = object : Runnable {
            override fun run() {
                if (talkBackEnabled) {
                    checkTutorialAndProceed()
                } else {
                    mainHandler.postDelayed(this, Constants.WAIT_CHECK)
                }
            }
        }
        mainHandler.post(checkRunnable)
    }

    private fun checkTutorialAndProceed() {
        if (AppConfig.showTutorial) {
            prepareTutorialTexts()
            showTutorialDialog()
        } else {
            homePageDescription()
        }
    }

    private fun prepareTutorialTexts() {
        tutorialTexts.clear()
        tutorialTexts.add(Pair(loadString(this, "tutorial_intro"), "tutorial_intro"))
        tutorialTexts.add(Pair(loadString(this, "tutorial_detection"), "tutorial_detection"))
        tutorialTexts.add(Pair(loadString(this, "tutorial_caption"), "tutorial_caption"))

        if (AppConfig.mainLanguage.code == "en") {
            tutorialTexts.add(Pair(loadString(this, "tutorial_findmyobject"), "tutorial_findmyobject"))
        }

        tutorialTexts.add(Pair(loadString(this, "tutorial_homepage_nav"), "tutorial_homepage_nav"))
        tutorialStep = 0
    }

    private fun showTutorialDialog() {
        uiEnabled.value = false
        showTutorial.value = true
        speakTutorialStep()
    }

    private fun speakTutorialStep() {
        if (tutorialStep >= tutorialTexts.size) {
            closeTutorial()
            return
        }

        val (text, _) = tutorialTexts[tutorialStep]
        currentTutorialText.value = ""
        displayTextWithTypewriter(text)

        ttsManager.speak(text, Constants.TTS_PITCH, Constants.TTS_SPEECH_RATE, true, null)
    }

    private fun displayTextWithTypewriter(text: String) {
        mainHandler.post {
            var displayedText = ""
            val delay = calculateTypingDelay()

            text.forEachIndexed { index, char ->
                mainHandler.postDelayed({
                    displayedText += char
                    currentTutorialText.value = displayedText
                }, (index * delay))
            }
        }
    }

    private fun calculateTypingDelay(): Long {
        val baseDelay = 70L
        return (baseDelay / Constants.TTS_SPEECH_RATE).toLong()
    }

    private fun handleTutorialClick() {
        cancelAllHandlers()
        tutorialStep++
        speakTutorialStep()
    }

    private fun handleTutorialSwipeUp() {
        closeTutorial()
    }

    private fun closeTutorial() {
        cancelAllHandlers()
        showTutorial.value = false
        homePageDescription()
    }

    private fun homePageDescription() {
        val syncStatus = dbManager.statusOverview

        when (syncStatus) {
            0 -> unlockUI()
            1 -> {
                val syncText = loadString(this, "load_syncStatusText")
                ttsManager.speak(syncText, Constants.TTS_PITCH, Constants.TTS_SPEECH_RATE, true) {
                    unlockUI()
                }
            }
            2 -> {
                val errorText = loadString(this, "load_syncErrorText")
                ttsManager.speak(errorText, Constants.TTS_PITCH, Constants.TTS_SPEECH_RATE, true) {
                    unlockUI()
                }
            }
        }
    }

    private fun lockUI() {
        uiLocked = true
        volumeButtonsLocked = true
        uiEnabled.value = false
    }

    private fun unlockUI() {
        uiLocked = false
        volumeButtonsLocked = false
        uiEnabled.value = true
    }

    private fun handleDetectionClick() {
        if (uiLocked) return
        if (AppConfig.haptics) vibrate(haptic_model0())

        PermissionChecker.checkAndRequestPermissions(this, false) {
            val intent = Intent(this, DetectionCameraXActivity::class.java)
            startActivity(intent)
        }
    }

    private fun handleCaptionClick() {
        if (uiLocked) return
        if (AppConfig.haptics) vibrate(haptic_model0())

        PermissionChecker.checkAndRequestPermissions(this, false) {
            val intent = Intent(this, CaptionCameraXActivity::class.java)
            startActivity(intent)
        }
    }

    private fun handleNavigationHomeClick() {
        if (AppConfig.haptics) vibrate(haptic_model0())
    }

    private fun handleNavigationSettingsClick() {
        if (uiLocked) return
        if (AppConfig.haptics) vibrate(haptic_model0())

        val navText = loadString(this, "navigating_to_settings")
        ttsManager.speak(navText, Constants.TTS_PITCH, Constants.TTS_SPEECH_RATE, false) {
            val intent = Intent(this, BlindSettingsActivity::class.java)
            startActivity(intent)
        }
    }

    private fun isTalkBackEnabled(): Boolean {
        val am = getSystemService(ACCESSIBILITY_SERVICE) as AccessibilityManager
        return am.isEnabled && am.isTouchExplorationEnabled
    }

    private fun openTalkBackSettings() {
        val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
        startActivity(intent)
    }

    private fun cancelAllHandlers() {
        mainHandler.removeCallbacksAndMessages(null)
        ttsManager.stopSpeaking()
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if (volumeButtonsLocked) return true

        when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                ttsManager.onVolumeDownPressed()
                return true
            }
            KeyEvent.KEYCODE_VOLUME_UP -> return true
        }

        return super.onKeyDown(keyCode, event)
    }

    override fun onPause() {
        super.onPause()
        cancelAllHandlers()
    }

    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacksAndMessages(null)
    }
}

@Composable
fun BlindHomeScreen(
    showTutorial: Boolean,
    tutorialText: String,
    uiEnabled: Boolean,
    onDetectionClick: () -> Unit,
    onCaptionClick: () -> Unit,
    onNavigationHomeClick: () -> Unit,
    onNavigationSettingsClick: () -> Unit,
    onTutorialClick: () -> Unit,
    onTutorialSwipeUp: () -> Unit
) {
    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val screenHeight = maxHeight
        val screenWidth = maxWidth

        // Background
        Image(
            painter = painterResource(R.drawable.app_background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        // Main content
        if (uiEnabled) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.SpaceBetween
            ) {
                // Logo at top (250dp instead of 200dp)
                Image(
                    painter = painterResource(R.drawable.vision_assist_logo),
                    contentDescription = "VisionAssist Logo",
                    modifier = Modifier
                        .size(250.dp)
                        .padding(top = 32.dp)
                )

                // Buttons column
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.Center
                ) {
                    // Detection row (full width clickable)
                    DetectionRow(onClick = onDetectionClick)

                    Spacer(modifier = Modifier.height(24.dp))

                    // Caption row (full width clickable)
                    CaptionRow(onClick = onCaptionClick)

                    // Navigation areas (two clickable columns)
                    NavigationAreas(
                        onHomeClick = onNavigationHomeClick,
                        onSettingsClick = onNavigationSettingsClick
                    )
                }
            }
        }

        // Tutorial overlay
        AnimatedVisibility(
            visible = showTutorial,
            enter = fadeIn(animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY)),
            exit = fadeOut(animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY))
        ) {
            TutorialOverlay(
                text = tutorialText,
                onClick = onTutorialClick,
                onSwipeUp = onTutorialSwipeUp
            )
        }
    }
}

@Composable
fun DetectionRow(onClick: () -> Unit) {
    val detectionDesc = loadString(null, "detection_button_desc")

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(Constants.STD_BUTTON_PAGE_HEIGHT.dp)
            .shadow(elevation = 3.dp, shape = RoundedCornerShape(16.dp))
            .background(
                color = colorResource(R.color.std_purple),
                shape = RoundedCornerShape(16.dp)
            )
            .clickable(onClick = onClick)
            .semantics { contentDescription = detectionDesc },
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = "Detection",
            fontSize = Constants.STD_FONT_SIZE.sp,
            color = Color.White,
            fontFamily = robotoSemibold,
            textAlign = TextAlign.Center
        )
    }
}

@Composable
fun CaptionRow(onClick: () -> Unit) {
    val captionDesc = loadString(null, "caption_button_desc")

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(Constants.STD_BUTTON_PAGE_HEIGHT.dp)
            .shadow(elevation = 3.dp, shape = RoundedCornerShape(16.dp))
            .background(
                color = colorResource(R.color.std_cyan),
                shape = RoundedCornerShape(16.dp)
            )
            .clickable(onClick = onClick)
            .semantics { contentDescription = captionDesc },
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = "Caption",
            fontSize = Constants.STD_FONT_SIZE.sp,
            color = Color.White,
            fontFamily = robotoSemibold,
            textAlign = TextAlign.Center
        )
    }
}

@Composable
fun NavigationAreas(
    onHomeClick: () -> Unit,
    onSettingsClick: () -> Unit
) {
    val homeDesc = loadString(null, "navigation_home_desc")
    val settingsDesc = loadString(null, "navigation_settings_desc")

    var swipeOffsetX by remember { mutableStateOf(0f) }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(200.dp)
            .pointerInput(Unit) {
                detectHorizontalDragGestures(
                    onDragEnd = {
                        if (abs(swipeOffsetX) > 100) {
                            // Swipe detected - go to settings
                            onSettingsClick()
                        }
                        swipeOffsetX = 0f
                    },
                    onHorizontalDrag = { _, dragAmount ->
                        swipeOffsetX += dragAmount
                    }
                )
            }
    ) {
        Row(
            modifier = Modifier.fillMaxSize(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            // Home column (left 50%)
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .clickable(onClick = onHomeClick)
                    .semantics { contentDescription = homeDesc },
                contentAlignment = Alignment.Center
            ) {
                // Empty - just for clicking
            }

            // Settings column (right 50%)
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .clickable(onClick = onSettingsClick)
                    .semantics { contentDescription = settingsDesc },
                contentAlignment = Alignment.Center
            ) {
                // Empty - just for clicking
            }
        }
    }
}

@Composable
fun TutorialOverlay(
    text: String,
    onClick: () -> Unit,
    onSwipeUp: () -> Unit
) {
    var swipeOffsetY by remember { mutableStateOf(0f) }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.85f))
            .pointerInput(Unit) {
                detectVerticalDragGestures(
                    onDragEnd = {
                        if (swipeOffsetY < -200) {
                            // Swipe up detected
                            onSwipeUp()
                        }
                        swipeOffsetY = 0f
                    },
                    onVerticalDrag = { _, dragAmount ->
                        swipeOffsetY += dragAmount
                    }
                )
            }
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth(0.85f)
                .padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = text,
                fontSize = Constants.STD_FONT_SIZE.sp,
                color = Color.White,
                fontFamily = robotoExtraBold,
                textAlign = TextAlign.Center,
                lineHeight = 32.sp
            )
        }
    }
}
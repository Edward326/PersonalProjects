@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.main

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.KeyEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
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
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.tabs.home.caption.CaptionActivity
import com.visionassist.appspace.activities.tabs.home.detection.LiveDetectionActivity
import com.visionassist.appspace.activities.tabs.home.detection.StaticDetectionActivity
import com.visionassist.appspace.activities.tabs.reports.EnvironmentReportsActivity
import com.visionassist.appspace.activities.tabs.settings.SettingsActivity
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.PermissionChecker
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.load_captionTutorial
import com.visionassist.appspace.utils.load_detectionTutorial
import com.visionassist.appspace.utils.load_homeTitle
import com.visionassist.appspace.utils.load_speakTutorial
import com.visionassist.appspace.utils.load_syncErrorText
import com.visionassist.appspace.utils.load_syncStatusText
import com.visionassist.appspace.utils.robotoBold
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.robotoExtraBoldItalic
import com.visionassist.appspace.utils.robotoSemibold
import com.visionassist.appspace.utils.vibrate
import kotlinx.coroutines.delay

class HomeActivity : ComponentActivity() {
    private val TAG = "HomeActivity"

    // State variables
    private val titleText = mutableStateOf("")
    private val syncStatus = mutableStateOf(0) // 0=hidden, 1=show days, 2=error
    private val syncDays = mutableStateOf(0)

    // Tutorial info button states
    private val speakInfoClickCount = mutableStateOf(1)

    // Detection button states
    private val showDetectionOptions = mutableStateOf(false)
    private val selectedDetectionOption = mutableStateOf<DetectionOption?>(null)
    private val selectedDetectionOptionPrevious = mutableStateOf<DetectionOption?>(null)
    private val detectionIconColor = mutableStateOf(R.color.std_purple_dark)

    // Speech recognition states
    private val showSpeechDialog = mutableStateOf(false)
    private val speechText = mutableStateOf("")
    private val speechProcessText = mutableStateOf("")
    private val isSpeaking = mutableStateOf(true)
    private val retrySpeech = mutableStateOf(false)
    private val cancelSpeech = mutableStateOf(false)
    private val sendSpeech = mutableStateOf(false)

    // Volume button tracking
    private var lastVolumeDownPress = 0L
    private var volumeDownPressCount = 0
    private var basicInfoClickCount = 1

    private val handler = Handler(Looper.getMainLooper())

    enum class DetectionOption {
        LIVE, STATIC
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Load initial title
        titleText.value = load_homeTitle(this)

        // Get sync status
        val dbManager = PhoneStatusMonitor.getInstance().dbManager
        syncStatus.value = dbManager.statusOverview
        if (syncStatus.value == 1) {
            syncDays.value = dbManager.diffDays.toInt()
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
                navigateFun = ::navigateToLiveOrStatic
            )
        }
    }

    override fun onResume() {
        super.onResume()

        val phoneMonitor = PhoneStatusMonitor.getInstance()
        if (phoneMonitor.isReturningFromPermissions) {
            // Determine what action we were doing before permissions
            // This is simplified - you may need additional state tracking
            handler.postDelayed({
                // Re-check what action to resume
            }, 500)
        }
    }

    private fun handleDetectionClick() {
        vibrateIfEnabled()
        showDetectionOptions.value = !showDetectionOptions.value
        detectionIconColor.value = if (showDetectionOptions.value) {
            R.color.std_cyan
        } else {
            selectedDetectionOption.value = null
            R.color.std_purple
        }

    }

    private fun handleDetectionOptionSelected(option: DetectionOption?, navigate: Boolean) {
        if(option!=null) {
            if (selectedDetectionOption.value != option) {
                vibrateIfEnabled()
                selectedDetectionOption.value = option
            }
        }
        else
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
        PermissionChecker.checkAndRequestPermissions(this, AppConfig.blindness) {
            checkPhoneStatusAndNavigate {
                // Navigate to LiveDetectionActivity
                val intent = Intent(this, LiveDetectionActivity::class.java)
                startActivity(intent)
                finish()
            }
        }
    }

    private fun launchStaticDetection() {
        PermissionChecker.checkAndRequestPermissions(this, AppConfig.blindness) {
            checkPhoneStatusAndNavigate {
                // Take photo and navigate to StaticDetectionActivity
                // TODO: Implement camera capture
                // For now, just navigate
                val intent = Intent(this, StaticDetectionActivity::class.java)
                startActivity(intent)
                finish()
            }
        }
    }

    private fun handleCaptionClick() {
        vibrateIfEnabled()
        launchCaptionActivity()
    }

    private fun launchCaptionActivity() {
        PermissionChecker.checkAndRequestPermissions(this, AppConfig.blindness) {
            checkPhoneStatusAndNavigate {
                // Take photo and navigate to CaptionActivity
                // TODO: Implement camera capture
                val intent = Intent(this, CaptionActivity::class.java)
                startActivity(intent)
                finish()
            }
        }
    }

    private fun handleDetectionInfoClick() {
        vibrateIfEnabled()

        // Cycle through tutorial messages
        titleText.value = load_detectionTutorial(this, getNextInfoStep())
    }

    private fun handleCaptionInfoClick() {
        vibrateIfEnabled()

        titleText.value = load_captionTutorial(this, getNextInfoStep())
    }

    private fun handleSpeakInfoClick() {
        vibrateIfEnabled()

        titleText.value = load_speakTutorial(this, speakInfoClickCount.value)

        // Reset after showing all messages
        if (speakInfoClickCount.value == 7) {
            speakInfoClickCount.value = 0
        }
        else
            speakInfoClickCount.value++
    }

    private fun getNextInfoStep(): Int {
        // Simple counter for tutorial steps
        if(basicInfoClickCount ==3) {
            basicInfoClickCount = 0
            return basicInfoClickCount
        }
        else {
            basicInfoClickCount++
            return basicInfoClickCount-1
        }
    }

    private fun handleNavigateHome() {
        vibrateIfEnabled()
        // Already on home, do nothing
    }

    private fun handleNavigateReports() {
        vibrateIfEnabled()
        if (AppConfig.env_reports) {
            val intent = Intent(this, EnvironmentReportsActivity::class.java)
            startActivity(intent)
            finish()
        }
    }

    private fun handleNavigateSettings() {
        vibrateIfEnabled()
        val intent = Intent(this, SettingsActivity::class.java)
        startActivity(intent)
        finish()
    }

    private fun launchSpeechRecognition() {
        showSpeechDialog.value = true
        isSpeaking.value = true
        speechText.value = ""
        speechProcessText.value = ""

        // Start speech recognition
        val speechRecognizer = PhoneStatusMonitor.getInstance()
            .modelManager.speechRecognizer

        speechRecognizer.startListening(object :
            com.visionassist.appspace.models.sttengine.SpeechRecognizer.RecognitionCallback {
            override fun onResult(recognizedText: String) {
                handler.post {
                    speechText.value = recognizedText
                    isSpeaking.value = false
                    sendSpeech.value = true
                }
            }

            override fun onError(error: String) {
                handler.post {
                    Log.e(TAG, "Speech recognition error: $error")
                    showSpeechDialog.value = false
                }
            }
        })
    }

    private fun processRecognizedSpeech() {
        speechProcessText.value = "Matching detector classes..."

        val speechRecognizer = PhoneStatusMonitor.getInstance()
            .modelManager.speechRecognizer

        val matchedIndices = speechRecognizer.processRecognizedText(speechText.value)

        if (matchedIndices.isEmpty()) {
            speechProcessText.value = "No matched known classes"
            retrySpeech.value = true
        } else {
            val yoloDetector = PhoneStatusMonitor.getInstance()
                .modelManager.detector

            val classNames = matchedIndices.map { yoloDetector.getClassName(it) }
            speechProcessText.value = "Matched: ${classNames.joinToString(", ")}"

            // Now allow user to proceed to FindMyObjectActivity
            // TODO: Navigate with matched indices
        }
    }

    private fun handleSpeechDialogTap() {
        // Cancel speech recognition
        showSpeechDialog.value = false
        resetSpeechStates()
    }

    private fun resetSpeechStates() {
        isSpeaking.value = true
        speechText.value = ""
        speechProcessText.value = ""
        retrySpeech.value = false
        cancelSpeech.value = false
        sendSpeech.value = false
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_UP -> {
                if (retrySpeech.value) {
                    // Retry speech recognition
                    resetSpeechStates()
                    launchSpeechRecognition()
                } else {
                    // Launch static detection
                    launchStaticDetection()
                }
                return true
            }

            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                val currentTime = System.currentTimeMillis()

                if (currentTime - lastVolumeDownPress < 500) {
                    // Double press - launch speech recognition
                    volumeDownPressCount = 0
                    launchSpeechRecognition()
                } else {
                    volumeDownPressCount = 1
                    lastVolumeDownPress = currentTime

                    // Wait to see if second press comes
                    handler.postDelayed({
                        if (volumeDownPressCount == 1) {
                            // Single press
                            handleSingleVolumeDown()
                        }
                        volumeDownPressCount = 0
                    }, 500)
                }
                return true
            }
        }
        return super.onKeyDown(keyCode, event)
    }

    private fun handleSingleVolumeDown() {
        when {
            cancelSpeech.value -> {
                // Stop speech recognition
                showSpeechDialog.value = false
                resetSpeechStates()
            }

            sendSpeech.value -> {
                // Process recognized text
                processRecognizedSpeech()
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
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacksAndMessages(null)
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
) {
    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenHeight = maxHeight
        val screenWidth = maxWidth

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
                .fillMaxHeight(0.913f)
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

            Box(modifier = Modifier.height(screenWidth * 0.10f))

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
                .fillMaxHeight(0.087f),
        ) {
            BottomNavigationBar(
                onNavigateHome = onNavigateHome,
                onNavigateReports = onNavigateReports,
                onNavigateSettings = onNavigateSettings,
                showReports = AppConfig.env_reports
            )
        }

        // Speech Recognition Dialog
        if (showSpeechDialog) {
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
                animationSpec = tween(Constants.ANIMATION_DELAY)
            ),
            exit = slideOutHorizontally(
                targetOffsetX = { it },
                animationSpec = tween(Constants.ANIMATION_DELAY)
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
                animationSpec = tween(Constants.ANIMATION_DELAY)
            ),
            exit = slideOutVertically(
                targetOffsetY = { it },
                animationSpec = tween(Constants.ANIMATION_DELAY)
            ),
            modifier = Modifier
                .offset(
                    x = detectionX,
                    y = buttonHeight - optionWidth - 4.dp
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
                    colorResource(R.color.std_purple)
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
                        onPress = {
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
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {

        Box(modifier = Modifier.width(screenWidth * 0.2f))

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

@Composable
fun InfoButtonWithPulse(
    onClick: () -> Unit,
    isPulsing: Boolean,
    modifier: Modifier = Modifier
) {
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")

    val scale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = if (isPulsing) 1.2f else 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scale"
    )

    IconButton(
        onClick = onClick,
        modifier = modifier
            .scale(if (isPulsing) scale else 1f)
            .size(Constants.STD_INFO_BUTTON_SIZE.dp)
    ) {
        Icon(
            imageVector = Icons.Filled.Info,
            contentDescription = "Info",
            tint = colorResource(R.color.std_purple),
            modifier = Modifier.size(Constants.STD_INFO_BUTTON_SIZE.dp)
        )
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
                        contentDescription = "Sync status",
                        tint = Color.White,
                        modifier = Modifier.size(21.dp)
                    )
                }

                Spacer(modifier = Modifier.width(4.dp))

                // Sync text
                Text(
                    text = if (syncStatus == 1) {
                        load_syncStatusText(LocalContext.current, syncDays)
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
        containerColor = Color.White
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
                    fontSize = Constants.STD_FONT_SIZE_LW.sp,
                    fontFamily = robotoSemibold
                )

            },
            selected = true,
            colors = NavigationBarItemColors(
                selectedIconColor = Color.White.copy(0.7f),
                unselectedIconColor = Color(0xFF49454F),
                selectedIndicatorColor = colorResource(R.color.std_purple),
                selectedTextColor = Color.Black,
                unselectedTextColor = Color(0xFF49454F),
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
                label = { Text("Reports") },
                selected = false,
                onClick = onNavigateReports
            )
        }

        NavigationBarItem(
            icon = {
                Icon(
                    imageVector = Icons.Filled.Settings,
                    contentDescription = "Settings"
                )
            },
            label = { Text("Settings") },
            selected = false,
            onClick = onNavigateSettings
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
                .background(Color.Gray.copy(alpha = Constants.BACKGROUND_OPACITY))
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
                                colorResource(R.color.std_purple_dark)
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
                            fontSize = Constants.STD_FONT_SIZE.sp, // Small font
                            fontFamily = robotoSemibold,
                            color = if (isSpeaking) {
                                Color.White
                            } else {
                                colorResource(R.color.std_purple_dark)
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
        navigateFun = {}
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
        navigateFun = {}
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
        speechProcessText = "Processing the speech",
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
        navigateFun = {}
    )
}
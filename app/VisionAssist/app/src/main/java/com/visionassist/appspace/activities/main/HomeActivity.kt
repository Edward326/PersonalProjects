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
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
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
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.robotoExtraBoldItalic
import com.visionassist.appspace.utils.robotoLight
import com.visionassist.appspace.utils.robotoSemibold
import kotlinx.coroutines.delay

class HomeActivity : ComponentActivity() {
    private val TAG = "HomeActivity"

    // State variables
    private val titleText = mutableStateOf("")
    private val syncStatus = mutableStateOf(0) // 0=hidden, 1=show days, 2=error
    private val syncDays = mutableStateOf(0)

    // Tutorial info button states
    private val detectionInfoPressed = mutableStateOf(false)
    private val captionInfoPressed = mutableStateOf(false)
    private val speakInfoPressed = mutableStateOf(false)
    private val speakInfoClickCount = mutableStateOf(0)

    // Detection button states
    private val showDetectionOptions = mutableStateOf(false)
    private val selectedDetectionOption = mutableStateOf<DetectionOption?>(null)
    private val detectionIconColor = mutableStateOf(R.color.std_purple)

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
                detectionInfoPressed = detectionInfoPressed.value,
                captionInfoPressed = captionInfoPressed.value,
                speakInfoPressed = speakInfoPressed.value,
                showSpeechDialog = showSpeechDialog.value,
                speechText = speechText.value,
                speechProcessText = speechProcessText.value,
                isSpeaking = isSpeaking.value,
                onDetectionClick = ::handleDetectionClick,
                onDetectionIconPress = ::handleDetectionIconPress,
                onDetectionIconRelease = ::handleDetectionIconRelease,
                onDetectionOptionSelected = ::handleDetectionOptionSelected,
                onCaptionClick = ::handleCaptionClick,
                onDetectionInfoClick = ::handleDetectionInfoClick,
                onCaptionInfoClick = ::handleCaptionInfoClick,
                onSpeakInfoClick = ::handleSpeakInfoClick,
                onNavigateHome = ::handleNavigateHome,
                onNavigateReports = ::handleNavigateReports,
                onNavigateSettings = ::handleNavigateSettings,
                onSpeechDialogTap = ::handleSpeechDialogTap
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
            R.color.std_purple
        }
    }

    private fun handleDetectionIconPress() {
        vibrateIfEnabled()
        detectionIconColor.value = R.color.std_cyan
        selectedDetectionOption.value = DetectionOption.LIVE
    }

    private fun handleDetectionIconRelease() {
        val selected = selectedDetectionOption.value

        detectionIconColor.value = R.color.std_purple
        showDetectionOptions.value = false

        if (selected != null) {
            when (selected) {
                DetectionOption.LIVE -> launchLiveDetection()
                DetectionOption.STATIC -> launchStaticDetection()
            }
        } else {
            vibrateIfEnabled()
        }

        selectedDetectionOption.value = null
    }

    private fun handleDetectionOptionSelected(option: DetectionOption) {
        vibrateIfEnabled()
        selectedDetectionOption.value = option
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
        if (!detectionInfoPressed.value) {
            detectionInfoPressed.value = true
        }

        // Cycle through tutorial messages
        titleText.value = load_detectionTutorial(this, getNextInfoStep())
    }

    private fun handleCaptionInfoClick() {
        vibrateIfEnabled()
        if (!captionInfoPressed.value) {
            captionInfoPressed.value = true
        }

        titleText.value = load_captionTutorial(this, getNextInfoStep())
    }

    private fun handleSpeakInfoClick() {
        vibrateIfEnabled()
        if (!speakInfoPressed.value) {
            speakInfoPressed.value = true
        }

        speakInfoClickCount.value++
        titleText.value = load_speakTutorial(this, speakInfoClickCount.value)

        // Reset after showing all messages
        if (speakInfoClickCount.value >= 5) {
            speakInfoClickCount.value = 0
            titleText.value = load_homeTitle(this)
        }
    }

    private fun getNextInfoStep(): Int {
        // Simple counter for tutorial steps
        return 1 // Implement cycling logic if needed
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
            haptic_model0()
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
    detectionInfoPressed: Boolean,
    captionInfoPressed: Boolean,
    speakInfoPressed: Boolean,
    showSpeechDialog: Boolean,
    speechText: String,
    speechProcessText: String,
    isSpeaking: Boolean,
    onDetectionClick: () -> Unit,
    onDetectionIconPress: () -> Unit,
    onDetectionIconRelease: () -> Unit,
    onDetectionOptionSelected: (HomeActivity.DetectionOption) -> Unit,
    onCaptionClick: () -> Unit,
    onDetectionInfoClick: () -> Unit,
    onCaptionInfoClick: () -> Unit,
    onSpeakInfoClick: () -> Unit,
    onNavigateHome: () -> Unit,
    onNavigateReports: () -> Unit,
    onNavigateSettings: () -> Unit,
    onSpeechDialogTap: () -> Unit
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
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(modifier = Modifier.height(screenHeight * 0.045f))

            // Logo
            Image(
                painter = painterResource(R.drawable.vision_assist_logo),
                contentDescription = "app logo",
                modifier = Modifier.size(Constants.LOGO_SIZE.dp)
            )

            Box(modifier = Modifier.height(screenHeight * 0.05f))

            // Title with typewriter animation
            TypewriterText(
                text = titleText,
                fontSize = Constants.STD_TITLE_SIZE.sp,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp)
            )

            Box(modifier = Modifier.height(screenHeight * 0.08f))

            // Detection Button with options
            DetectionButtonSection(
                showOptions = showDetectionOptions,
                iconColor = detectionIconColor,
                selectedOption = selectedDetectionOption,
                screenWidth = screenWidth,
                showInfoButton = AppConfig.showTutorial && !detectionInfoPressed,
                onDetectionClick = onDetectionClick,
                onIconPress = onDetectionIconPress,
                onIconRelease = onDetectionIconRelease,
                onOptionSelected = onDetectionOptionSelected,
                onInfoClick = onDetectionInfoClick
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Caption Button
            CaptionButtonSection(
                screenWidth = screenWidth,
                showInfoButton = AppConfig.showTutorial && !captionInfoPressed,
                onCaptionClick = onCaptionClick,
                onInfoClick = onCaptionInfoClick
            )

            Spacer(modifier = Modifier.weight(1f))

            // Sync status section
            if (syncStatus > 0) {
                SyncStatusSection(
                    syncStatus = syncStatus,
                    syncDays = syncDays
                )
                Spacer(modifier = Modifier.height(16.dp))
            }

            // Speak info button (only in English and if tutorial enabled)
            if (AppConfig.showTutorial &&
                AppConfig.mainLanguage.code == "en" &&
                !speakInfoPressed
            ) {
                InfoButtonWithPulse(
                    onClick = onSpeakInfoClick,
                    isPulsing = !speakInfoPressed
                )
                Spacer(modifier = Modifier.height(16.dp))
            }
        }

        // Bottom Navigation Bar
        Box(
            modifier = Modifier.align(Alignment.BottomCenter)
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
    fontSize: androidx.compose.ui.unit.TextUnit,
    modifier: Modifier = Modifier
) {
    var displayedText by remember { mutableStateOf("") }

    LaunchedEffect(text) {
        displayedText = ""
        text.forEachIndexed { index, _ ->
            delay(50) // Typing speed
            displayedText = text.take(index + 1)
        }
    }

    // Parse text with ~ separator
    val parts = displayedText.split("~")

    Text(
        text = buildAnnotatedString {
            when {
                parts.size == 1 -> {
                    withStyle(
                        style = SpanStyle(
                            color = colorResource(R.color.std_cyan),
                            fontFamily = robotoExtraBold,
                            fontSize = fontSize
                        )
                    ) {
                        append(parts[0])
                    }
                }

                parts.size > 1 -> {
                    withStyle(
                        style = SpanStyle(
                            color = colorResource(R.color.std_cyan),
                            fontFamily = robotoExtraBold,
                            fontSize = fontSize
                        )
                    ) {
                        append(parts[0])
                    }
                    append("\n")
                    withStyle(
                        style = SpanStyle(
                            color = colorResource(R.color.std_purple),
                            fontFamily = robotoLight,
                            fontSize = fontSize
                        )
                    ) {
                        append(parts[1])
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
    onIconRelease: () -> Unit,
    onOptionSelected: (HomeActivity.DetectionOption) -> Unit,
    onInfoClick: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Detection options (slide from left)
        AnimatedVisibility(
            visible = showOptions,
            enter = slideInHorizontally(initialOffsetX = { -it }) + fadeIn(),
            exit = slideOutHorizontally(targetOffsetX = { -it }) + fadeOut()
        ) {
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                DetectionOptionButton(
                    text = "Static",
                    isSelected = selectedOption == HomeActivity.DetectionOption.STATIC,
                    onClick = { onOptionSelected(HomeActivity.DetectionOption.STATIC) }
                )

                DetectionOptionButton(
                    text = "Live",
                    isSelected = selectedOption == HomeActivity.DetectionOption.LIVE,
                    onClick = { onOptionSelected(HomeActivity.DetectionOption.LIVE) }
                )
            }
        }

        Spacer(modifier = Modifier.width(8.dp))

        // Main Detection Button
        MainActionButton(
            text = "Detection",
            iconRes = R.drawable.detection_icon, // Replace with actual icon
            iconColor = iconColor,
            screenWidth = screenWidth,
            onClick = onDetectionClick,
            onIconPress = onIconPress,
            onIconRelease = onIconRelease
        )

        // Info button
        if (showInfoButton) {
            Spacer(modifier = Modifier.width(8.dp))
            InfoButtonWithPulse(
                onClick = onInfoClick,
                isPulsing = true
            )
        }
    }
}

@Composable
fun DetectionOptionButton(
    text: String,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Button(
        onClick = onClick,
        modifier = Modifier
            .width((0.2f * LocalContext.current.resources.displayMetrics.widthPixels / LocalContext.current.resources.displayMetrics.density).dp)
            .height(Constants.STD_BUTTON_PAGE_HEIGHT.dp),
        shape = RoundedCornerShape(16.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = if (isSelected) {
                colorResource(R.color.std_purple)
            } else {
                colorResource(R.color.std_cyan)
            },
            contentColor = Color.White
        )
    ) {
        Text(
            text = text,
            fontSize = 22.sp,
            fontFamily = robotoExtraBoldItalic
        )
    }
}



@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainActionButton(
    predefinedIcon: ImageVector? = null,
    text: String,
    iconRes: Int=0,
    iconColor: Int,
    screenWidth: Dp,
    onClick: () -> Unit,
    onIconPress: () -> Unit,
    onIconRelease: () -> Unit
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(0.dp)
    ) {
        // Main button
        Button(
            onClick = onClick,
            modifier = Modifier
                .shadow(
                    3.dp,
                    RoundedCornerShape(
                        topStart = 5.dp,
                        topEnd = 31.dp,
                        bottomStart = 5.dp,
                        bottomEnd = 5.dp
                    )
                )
                .width(screenWidth * 0.6f)
                .height(Constants.STD_BUTTON_PAGE_HEIGHT.dp),
            shape = RoundedCornerShape(
                topStart = 5.dp,
                topEnd = 31.dp,
                bottomStart = 5.dp,
                bottomEnd = 5.dp
            ),
            colors = ButtonDefaults.buttonColors(
                containerColor = Color.White,
                contentColor = colorResource(R.color.std_purple)
            )
        ) {
            Text(
                text = text,
                fontSize = Constants.STD_SUBTITLE_SIZE.sp,
                fontFamily = robotoSemibold
            )
        }

        // Icon button (circle)
        Box(
            modifier = Modifier
                .offset(x = (-10).dp) // Overlap with main button
                .size(56.dp)
                .shadow(3.dp, CircleShape)
                .clip(CircleShape)
                .background(colorResource(iconColor))
                .pointerInput(Unit) {
                    detectTapGestures(
                        onPress = {
                            onIconPress()
                            tryAwaitRelease()
                            onIconRelease()
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
                    imageVector =predefinedIcon,
                    contentDescription = "Action icon",
                    tint = Color.White.copy(alpha = 0.7f),
                    modifier = Modifier.size(28.dp)
                )
            }
        }
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
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Caption Button
        MainActionButton(
            predefinedIcon = Icons.Filled.TextFields,
            text = "Caption",
            iconColor = R.color.std_purple,
            screenWidth = screenWidth,
            onClick = onCaptionClick,
            onIconPress = {}, // No special behavior
            onIconRelease = {}
        )

        // Info button
        if (showInfoButton) {
            Spacer(modifier = Modifier.width(8.dp))
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
    syncDays: Int
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center,
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp)
    ) {
        Box(
            modifier = Modifier
                .size(24.dp)
                .clip(CircleShape)
                .background(if (syncStatus == 1) Color.Green else Color.Red),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = if (syncStatus == 1) Icons.Filled.Sync else Icons.Filled.SyncProblem,
                contentDescription = "Sync status",
                tint = Color.White,
                modifier = Modifier.size(16.dp)
            )
        }

        Spacer(modifier = Modifier.width(8.dp))

        Text(
            text = if (syncStatus == 1) {
                load_syncStatusText(LocalContext.current, syncDays)
            } else {
                load_syncErrorText(LocalContext.current)
            },
            fontSize = Constants.STD_FONT_SIZE.sp,
            fontFamily = robotoSemibold,
            color = colorResource(if (syncStatus == 1) R.color.std_cyan else R.color.error_red)
        )
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
        containerColor = Color.White.copy(alpha = 0.9f),
        contentColor = colorResource(R.color.std_purple)
    ) {
        NavigationBarItem(
            icon = {
                Icon(
                    imageVector = Icons.Filled.Home,
                    contentDescription = "Home"
                )
            },
            label = { Text("Home") },
            selected = true,
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
        exit = fadeOut(animationSpec = tween(0))
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
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center,
                modifier = Modifier
                    .fillMaxWidth(0.8f)
                    .animateContentSize()
            ) {
                // Speaking section
                AnimatedContent(
                    targetState = isSpeaking,
                    transitionSpec = {
                        fadeIn() togetherWith fadeOut()
                    },
                    label = "speaking_animation"
                ) { speaking ->
                    if (speaking) {
                        Text(
                            text = speechText.ifEmpty { "Listening..." },
                            fontSize = Constants.STD_SUBTITLE_SIZE.sp,
                            fontFamily = robotoSemibold,
                            color = Color.White,
                            textAlign = TextAlign.Center
                        )
                    } else {
                        Text(
                            text = speechText,
                            fontSize = Constants.STD_TITLE_SIZE.sp,
                            fontFamily = robotoExtraBoldItalic,
                            color = colorResource(R.color.std_cyan),
                            textAlign = TextAlign.Center
                        )
                    }
                }

                // Process text section
                if (processText.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(24.dp))

                    Text(
                        text = processText,
                        fontSize = Constants.STD_FONT_SIZE.sp,
                        fontFamily = robotoSemibold,
                        color = colorResource(R.color.std_purple),
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
    }
}

@Preview(name = "Home Activity", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun HomeActivityPreview() {
    HomeScreen(
        titleText = "Hello ~Eduard, what can I do for you?",
        syncStatus = 1,
        syncDays = 5,
        showDetectionOptions = false,
        detectionIconColor = R.color.std_purple,
        selectedDetectionOption = null,
        detectionInfoPressed = false,
        captionInfoPressed = false,
        speakInfoPressed = false,
        showSpeechDialog = false,
        speechText = "",
        speechProcessText = "",
        isSpeaking = true,
        onDetectionClick = {},
        onDetectionIconPress = {},
        onDetectionIconRelease = {},
        onDetectionOptionSelected = {},
        onCaptionClick = {},
        onDetectionInfoClick = {},
        onCaptionInfoClick = {},
        onSpeakInfoClick = {},
        onNavigateHome = {},
        onNavigateReports = {},
        onNavigateSettings = {},
        onSpeechDialogTap = {}
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
        titleText = "Hello ~Eduard, what can I do for you?",
        syncStatus = 1,
        syncDays = 5,
        showDetectionOptions = true,
        detectionIconColor = R.color.std_cyan,
        selectedDetectionOption = HomeActivity.DetectionOption.LIVE,
        detectionInfoPressed = false,
        captionInfoPressed = false,
        speakInfoPressed = false,
        showSpeechDialog = false,
        speechText = "",
        speechProcessText = "",
        isSpeaking = true,
        onDetectionClick = {},
        onDetectionIconPress = {},
        onDetectionIconRelease = {},
        onDetectionOptionSelected = {},
        onCaptionClick = {},
        onDetectionInfoClick = {},
        onCaptionInfoClick = {},
        onSpeakInfoClick = {},
        onNavigateHome = {},
        onNavigateReports = {},
        onNavigateSettings = {},
        onSpeechDialogTap = {}
    )
}

@Preview(name = "Speech Recognition Dialog", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun SpeechDialogPreview() {
    SpeechRecognitionDialog(
        isVisible = true,
        speechText = "Find my apple",
        processText = "Matched: apple",
        isSpeaking = false,
        onTap = {}
    )
}
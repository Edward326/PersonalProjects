package com.visionassist.appspace.activities.newprofile

import android.annotation.SuppressLint
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.KeyEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale.Companion.Crop
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.newprofile.jsonCollection.ProfileFileCollection
import com.visionassist.appspace.jetpack.design.BlindnessNotificationDialog
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager
import com.visionassist.appspace.models.ttsengine.TTSManager
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.robotoLight
import com.visionassist.appspace.utils.robotoSemibold

class ConfigurationActivity : ComponentActivity() {
    private val TAG = "ConfigurationActivity"

    private var ttsManager: TTSManager = PhoneStatusMonitor.getInstance().ttsManager
    private val mainHandler = Handler(Looper.getMainLooper())

    // State for notification visibility
    private val showNotification = mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            ConfigurationScreen(
                showNotification = showNotification.value,
                onBlindnessClick = ::handleBlindnessClick,
                onLowEyesightClick = ::handleLowEyesightClick,
                onNotificationOkClick = ::handleNotificationOk
            )
        }
        speakInitialCaption()
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                Log.d(TAG, "Volume button down for repeat pressed")
                ttsManager.onVolumeDownPressed()
                return true
            }

            KeyEvent.KEYCODE_VOLUME_UP -> {
                Log.d(TAG, "Volume button up pressed")
                return true
            }
        }

        return super.onKeyDown(keyCode, event)
    }

    private fun speakInitialCaption() {
        val retryRunnable = object : Runnable {
            override fun run() {
                if (ttsManager.isReady) {
                    val captionTutorial = getString(R.string.tutorial_speak)
                    val caption = getString(R.string.config_activity)
                    ttsManager.speak(
                        captionTutorial,
                        Constants.TTS_PITCH,
                        Constants.TTS_SPEECH_RATE,
                        true,
                        null
                    )

                    val checkSpeakingRunnable = object : Runnable {
                        override fun run() {
                            if (ttsManager.isDoneSpeaking) {
                                ttsManager.speak(
                                    caption,
                                    Constants.TTS_PITCH,
                                    Constants.TTS_SPEECH_RATE,
                                    true,
                                    null
                                )
                            } else {
                                mainHandler.postDelayed(this, 500)
                            }
                        }
                    }
                    mainHandler.post(checkSpeakingRunnable)
                } else {
                    Log.w(TAG, "TTS not ready, retrying...")
                    mainHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS.toLong())
                }
            }
        }
        mainHandler.post(retryRunnable)
    }

    private fun handleBlindnessClick() {
        // Cancel any pending handlers
        cancelAllHandlers()
        showNotification.value = true
        // Speak the notification text
        speakNotificationWarning()
    }

    private fun speakNotificationWarning() {
        val retryRunnable = object : Runnable {
            override fun run() {
                if (ttsManager.isReady) {
                    val vw =
                        ".This dialog box is central on the screen, please press the button OK below the dialog, to configure the profile"
                    val warningText = getString(R.string.initial_blindness_notification) + vw
                    ttsManager.speak(
                        warningText,
                        Constants.TTS_PITCH,
                        Constants.TTS_SPEECH_RATE,
                        true,
                        null
                    )
                    Log.d(TAG, "Speaking blindness warning")
                } else {
                    Log.w(TAG, "TTS not ready, retrying...")
                    mainHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS.toLong())
                }
            }
        }
        mainHandler.post(retryRunnable)
    }

    private fun handleLowEyesightClick() {
        // Cancel any pending handlers
        cancelAllHandlers()
        writeToProfileAndNavigate(false)
    }

    private fun handleNotificationOk() {
        Log.d(TAG, "OK button clicked in notification")

        cancelAllHandlers()
        showNotification.value = false
        // Write blindness:true to profile and navigate to WelcomeActivity
        writeToProfileAndNavigate(true)
    }

    private fun writeToProfileAndNavigate(blindness: Boolean) {
        try {
            ProfileFileCollection.configurationActivityWrite(blindness)
            // Update AppConfig
            AppConfig.blindness = blindness
            // Navigate to WelcomeActivity
            val intent = Intent(this, WelcomeActivity::class.java)
            startActivity(intent)
            finish()
        } catch (_: Exception) {
            val phoneMonitor = PhoneStatusMonitor.getInstance()
            val errorDialog = ErrorDialogManager(this)
            errorDialog.setupDialog(Constants.FILE_WRITE_ERROR)
            phoneMonitor.shutdownApp(errorDialog, this)
        }
    }

    override fun onPause() {
        super.onPause()
        cancelAllHandlers()
    }

    private fun cancelAllHandlers() {
        mainHandler.removeCallbacksAndMessages(null)
        if (!ttsManager.isDoneSpeaking)
            ttsManager.stopSpeaking()
        Log.d(TAG, "All handlers cancelled")
    }
}

@SuppressLint("InflateParams")
@Composable
fun ConfigurationScreen(
    showNotification: Boolean,
    onBlindnessClick: () -> Unit,
    onLowEyesightClick: () -> Unit,
    onNotificationOkClick: () -> Unit
) {
    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenHeight = maxHeight
        val screenWidth=maxWidth
        // Background image
        Image(
            painter = painterResource(R.drawable.welcome_background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = Crop
        )

        // Main content
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.SpaceAround
        ) {
            Box(modifier = Modifier.height(screenHeight * 0.045f))

            // Logo
            Image(
                painter = painterResource(R.drawable.vision_assist_logo),
                contentDescription = "app logo",
                modifier = Modifier
                    .size(screenWidth*0.47f)
            )

            Box(modifier = Modifier.height(screenHeight * 0.1f))

            // Welcome text
            Text(
                text = "Welcome to VisionAssist",
                fontSize = 40.sp,
                color = colorResource(R.color.std_cyan),
                fontFamily = robotoLight,
                letterSpacing = 6.sp,
                modifier = Modifier.fillMaxWidth(),
                textAlign = TextAlign.Center,
                lineHeight = 60.sp
            )

            Box(modifier = Modifier.height(screenHeight * Constants.STD_TITLE_SUBTITLE_MARGIN_TOP))

            // Problem text
            Text(
                text = "What is your\nvisual problem?",
                fontSize = 32.sp,
                color = colorResource(R.color.std_cyan),
                fontFamily = robotoSemibold,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )

            Box(modifier = Modifier.height(screenHeight * 0.07f))

            // Buttons row
            Row(
                modifier = Modifier
                    .fillMaxWidth(),
                verticalAlignment = Alignment.Bottom,
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                // Blindness button
                VisualProblemButton(
                    text = "Blindness",
                    contentDescription = "Blindness button pressed",
                    iconResId = R.drawable.blindness,
                    onClick = onBlindnessClick,
                    screenWidth=screenWidth,
                    screenHeight=screenHeight
                )

                // Low eyesight button
                VisualProblemButton(
                    text = "Low eyesight",
                    contentDescription = "Low eyesight button pressed",
                    iconResId = R.drawable.eyesight,
                    onClick = onLowEyesightClick,
                    screenWidth=screenWidth,
                    screenHeight=screenHeight
                )
            }
            Box(modifier = Modifier.height(screenHeight * 0.21f))
        }

        // Notification Dialog - Only shows when showNotification is true
        // This doesn't block touches when hidden because AnimatedVisibility handles it
        BlindnessNotificationDialog(
            isVisible = showNotification,
            onOkClick = onNotificationOkClick
        )
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun VisualProblemButton(
    text: String,
    contentDescription: String,
    iconResId: Int,
    onClick: () -> Unit,
    screenWidth: Dp,
    screenHeight: Dp
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = text,
            fontSize = Constants.STD_FONT_SIZE.sp,
            color = colorResource(R.color.std_cyan),
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 8.dp),
            fontFamily = robotoSemibold,
        )

        Button(
            onClick = onClick,
            modifier = Modifier
                .shadow(
                    elevation = 3.dp, shape = MaterialTheme.shapes.large
                )
                .width(screenWidth * 0.35f)
                .height(screenHeight * 0.093f),
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = Color(0xFFEADDFF),        // purple background
                contentColor = Color(0xFF6750A4)
            )
        ) {
            Icon(
                painter = painterResource(iconResId),
                contentDescription = contentDescription,
                tint = Color.Black,
                modifier = Modifier.size(34.dp)
            )

        }

    }
}

@Preview(name = "Configuration Activity", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun ConfigurationActivityPreview() {
    ConfigurationScreen(
        showNotification = false,
        onBlindnessClick = {},
        onLowEyesightClick = {},
        onNotificationOkClick = {}
    )
}
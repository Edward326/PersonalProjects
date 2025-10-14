package com.visionassist.appspace.activities.newprofile

import android.annotation.SuppressLint
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.LayoutInflater
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.jetpack.managers.BlindnessNotificationManager
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager
import com.visionassist.appspace.models.ttsengine.TTSManager
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.FileUtils
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.robotoSemibold
import org.json.JSONObject

class ConfigurationActivity : ComponentActivity() {
    private val TAG = "ConfigurationActivity"

    private var ttsManager: TTSManager = PhoneStatusMonitor.getInstance().ttsManager
    private lateinit var notificationManager: BlindnessNotificationManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Setup notification manager
        val notificationBox = ComposeView(this)
        notificationManager = BlindnessNotificationManager(
            notificationBox = notificationBox,
            context = this,
            onOkPressed = ::handleNotificationOk
        )
        notificationManager.setupNotification()

        setContent {
            ConfigurationScreen(
                notificationBox = notificationBox,
                onBlindnessClick = ::handleBlindnessClick,
                onLowEyesightClick = ::handleLowEyesightClick
            )
        }
        speakInitialCaption()
    }

    private fun speakInitialCaption() {
        val handler = Handler(Looper.getMainLooper())
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

                    val handler2 = Handler(Looper.getMainLooper())
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
                                handler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS.toLong())
                            }
                        }
                    }
                    handler2.post(checkSpeakingRunnable)
                } else {
                    Log.w(TAG, "TTS not ready, retrying...")
                    handler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS.toLong())
                }
            }
        }
        handler.post(retryRunnable)
    }

    private fun handleBlindnessClick() {
        // Stop any current speaking
        ttsManager.stopSpeaking()

        // Speak "Blindness" with vibration
        val blindnessText = "Blindness button pressed"
        ttsManager.speak(
            blindnessText,
            Constants.TTS_PITCH,
            Constants.TTS_SPEECH_RATE,
            false,
            haptic_model0()
        )

        // Wait until done speaking, then show notification
        val handler = Handler(Looper.getMainLooper())
        val checkSpeakingRunnable = object : Runnable {
            override fun run() {
                if (ttsManager.isDoneSpeaking) {
                    // Show blindness warning notification
                    notificationManager.showNotification()
                } else {
                    handler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS.toLong())
                }
            }
        }
        handler.post(checkSpeakingRunnable)
    }

    private fun handleLowEyesightClick() {
        // Stop any current speaking
        ttsManager.stopSpeaking()

        // Speak "Low eyesight" with vibration
        val lowEyesightText = "Low eyesight button pressed"
        ttsManager.speak(
            lowEyesightText,
            Constants.TTS_PITCH,
            Constants.TTS_SPEECH_RATE,
            false,
            haptic_model0()
        )

        // Wait until done speaking, then write to profile and navigate
        val handler = Handler(Looper.getMainLooper())
        val checkSpeakingRunnable = object : Runnable {
            override fun run() {
                if (ttsManager.isDoneSpeaking) {
                    writeToProfileAndNavigate(false)
                } else {
                    handler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS.toLong())
                }
            }
        }
        handler.post(checkSpeakingRunnable)
    }

    private fun handleNotificationOk() {
        // This is called when OK button in notification is pressed
        // Write blindness:true to profile and navigate to WelcomeActivity
        writeToProfileAndNavigate(true)
    }

    private fun writeToProfileAndNavigate(blindness: Boolean) {
        try {
            // Read existing profile
            val profileFile = FileUtils.getProfileFile(this)
            val jsonObject: JSONObject = if (profileFile.exists() && profileFile.length() > 0) {
                val content = profileFile.readText()
                JSONObject(content)
            } else {
                JSONObject()
            }

            // Update blindness field
            jsonObject.put("blindness", blindness)
            // Write back to file
            FileUtils.writeProfileFile(this, this, jsonObject.toString())

            // Update AppConfig
            AppConfig.blindness = blindness

            // Navigate to WelcomeActivity
            val intent = Intent(this, WelcomeActivity::class.java)
            startActivity(intent)
            finish()
        } catch (_: Exception) {
            val phoneMonitor = PhoneStatusMonitor.getInstance()
            val errorDialog = ErrorDialogManager(this)
            errorDialog.setupDialog(Constants.FILE_WRITE_ERROR, getString(R.string.exit_error_en))
            phoneMonitor.shutdownApp(errorDialog, this)
        }
    }
}

@SuppressLint("InflateParams")
@Composable
fun ConfigurationScreen(
    notificationBox: ComposeView,
    onBlindnessClick: () -> Unit,
    onLowEyesightClick: () -> Unit
) {
    Box(
        modifier = Modifier.fillMaxSize()
    ) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { context ->
                LayoutInflater.from(context).inflate(R.layout.activity_configuration, null)
            }
        )

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
                .padding(bottom = 150.dp)
        ) {
            // Blindness button
            VisualProblemButton(
                text = "Blindness",
                iconResId = R.drawable.blindness,
                modifier = Modifier.weight(1f),
                onClick = onBlindnessClick
            )

            // Low eyesight button
            VisualProblemButton(
                text = "Low eyesight",
                iconResId = R.drawable.eyesight,
                modifier = Modifier.weight(1f),
                onClick = onLowEyesightClick
            )
        }

        androidx.compose.ui.viewinterop.AndroidView(
            factory = { notificationBox },
            modifier = Modifier.fillMaxSize()
        )
    }
}

@Composable
fun VisualProblemButton(
    text: String,
    iconResId: Int,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = text,
            fontSize = 14.sp, // Font mai mic
            color = colorResource(R.color.std_cyan),
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 8.dp),
            fontFamily = robotoSemibold,
        )

        // 2. The Icon Button (Surface-ul)
        Surface(
            modifier = Modifier
                .width(144.dp)
                .height(86.dp)
                .clickable(onClick = onClick),
            shape = RoundedCornerShape(16.dp),
            color = Color(0xFFEADDFF),
            shadowElevation = 10.dp,

            ) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Image(
                    painter = painterResource(iconResId),
                    contentDescription = text,
                    colorFilter = ColorFilter.tint(Color.Black),
                    modifier = Modifier.size(34.dp)
                )
            }
        }
    }
}

@Preview(name = "Configuration Activity", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun ConfigurationActivityPreview() {
    ConfigurationScreen(
        ComposeView(LocalContext.current),
        {},
        {})
}
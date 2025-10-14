package com.visionassist.appspace.jetpack.managers

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.platform.ComposeView
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.jetpack.design.BlindnessNotificationDialog
import com.visionassist.appspace.models.ttsengine.TTSManager
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.haptic_model0

class BlindnessNotificationManager(
    private val notificationBox: ComposeView,
    private val context: Context,
    private val onOkPressed: () -> Unit
) {
    private val TAG = "BlindnessNotificationManager"

    private var isVisibleState = mutableStateOf(false)
    private var ttsManager: TTSManager = PhoneStatusMonitor.getInstance().ttsManager

    fun setupNotification() {
        notificationBox.setContent {
            BlindnessNotificationDialog(
                isVisible = isVisibleState.value,
                onOkClick = ::handleOkClick
            )
        }
    }

    fun showNotification() {
        isVisibleState.value = true

        // Speak the blindness warning with repeat option
        val handler = Handler(Looper.getMainLooper())
        val retryRunnable = object : Runnable {
            override fun run() {
                if (ttsManager.isReady) {
                    val vw =
                        ".This dialog box is central on the screen, please press the button OK below the dialog, to configure the profile"
                    val warningText =
                        context.getString(R.string.initial_blindness_notification) + vw
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
                    handler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS.toLong())
                }
            }
        }
        handler.post(retryRunnable)
    }

    fun hideNotification() {
        isVisibleState.value = false
    }

    private fun handleOkClick() {
        Log.d(TAG, "OK button clicked in notification")

        ttsManager.stopSpeaking()
        val okText = "OK button pressed"
        ttsManager.speak(
            okText,
            Constants.TTS_PITCH,
            Constants.TTS_SPEECH_RATE,
            false,
            haptic_model0()
        )

        // Wait until done speaking, then proceed
        val handler = Handler(Looper.getMainLooper())
        val checkSpeakingRunnable = object : Runnable {
            override fun run() {
                if (ttsManager.isDoneSpeaking) {
                    Log.d(TAG, "Done speaking OK confirmation, hiding notification")
                    hideNotification()

                    // Call the callback provided by the activity
                    onOkPressed.invoke()
                } else {
                    handler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS.toLong())
                }
            }
        }
        handler.post(checkSpeakingRunnable)
    }
}
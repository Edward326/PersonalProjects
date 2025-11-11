@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.newprofile

import android.annotation.SuppressLint
import android.content.Intent
import android.os.Bundle
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
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.newprofile.jsonCollection.ProfileFileCollection
import com.visionassist.appspace.jetpack.design.BackArrowLargeFab
import com.visionassist.appspace.jetpack.design.CustomSlider
import com.visionassist.appspace.jetpack.design.NextArrowLargeFab
import com.visionassist.appspace.jetpack.design.ThumbStyle
import com.visionassist.appspace.models.ttsengine.TTSManager
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.load_pitchSpeed
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.robotoSemibold

class UserInfoE3Activity : ComponentActivity() {
    private val TAG = "UserInfoE3Activity"

    private val ttsManager: TTSManager = PhoneStatusMonitor.getInstance().ttsManager

    // State for pitch and speed
    private val pitchValue = mutableFloatStateOf(Constants.TTS_PITCH)
    private val speedValue = mutableFloatStateOf(Constants.TTS_SPEECH_RATE)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            UserInfoE3Screen(
                pitchValue = pitchValue.floatValue,
                speedValue = speedValue.floatValue,
                onPitchChange = ::handlePitchChange,
                onSpeedChange = ::handleSpeedChange,
                onBackClick = ::handleBackClick,
                onNextClick = ::handleNextClick
            )
        }
    }

    @SuppressLint("DefaultLocale")
    private fun handlePitchChange(newPitch: Float) {
        AppConfig.tts_pitch = newPitch
        pitchValue.floatValue = newPitch
        ttsManager.stopSpeaking()
        // Speak the current pitch with current settings
        val text = if(AppConfig.mainLanguage.code=="en")
            "The current pitch is $newPitch"
        else
            "Tonalitatea vocii este $newPitch"
        ttsManager.speak(text, newPitch, speedValue.floatValue, false, haptic_model0())
    }

    @SuppressLint("DefaultLocale")
    private fun handleSpeedChange(newSpeed: Float) {
        AppConfig.tts_speech_rate = newSpeed
        speedValue.floatValue = newSpeed

        ttsManager.stopSpeaking()
        // Speak the current pitch with current settings
        val text = if(AppConfig.mainLanguage.code=="en")
            "The speed of assistant is $newSpeed"
        else
            "Tonalitatea vocii este $newSpeed"
        ttsManager.speak(text, pitchValue.floatValue, newSpeed, false, haptic_model0())
    }

    private fun handleBackClick() {
        // Check if contributor
        if (AppConfig.isContributor) {
            ProfileFileCollection.deleteUserInfoActivity(1)
            val intent = Intent(this, UserInfoActivity::class.java)
            intent.putExtra(Constants.EXTRA_USERINFO_OPTION, 2)
            startActivity(intent)
            finish()
        }
        else {
            ProfileFileCollection.deleteUserInfoActivity(0)
            val intent = Intent(this, UserInfoActivity::class.java)
            intent.putExtra(Constants.EXTRA_USERINFO_OPTION, 1)
            startActivity(intent)
            finish()
        }
    }

    private fun handleNextClick() {
        // Write TTS pitch and speed to profile
        ProfileFileCollection.writeUserInfoE3Activity(pitchValue.floatValue, speedValue.floatValue)

        // Navigate to UserHashCachingActivity
        val intent = Intent(this, UserHashCachingActivity::class.java)
        intent.putExtra(Constants.EXTRA_HCACHING_OPTION, 1)
        startActivity(intent)
        finish()
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                Log.d(TAG, "Volume button down pressed")
                return true
            }

            KeyEvent.KEYCODE_VOLUME_UP -> {
                Log.d(TAG, "Volume button up pressed")
                return true
            }
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onPause() {
        super.onPause()
        if (!ttsManager.isDoneSpeaking) {
            ttsManager.stopSpeaking()
        }
    }
}

@SuppressLint("DefaultLocale")
@Composable
fun UserInfoE3Screen(
    pitchValue: Float,
    speedValue: Float,
    onPitchChange: (Float) -> Unit,
    onSpeedChange: (Float) -> Unit,
    onBackClick: () -> Unit,
    onNextClick: () -> Unit
) {
    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenHeight = maxHeight
        val screenWidth = maxWidth

        // Background image
        Image(
            painter = painterResource(R.drawable.welcome_background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        // Content
        Column(
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // First slider - Pitch
            Box(modifier = Modifier.height(screenHeight * Constants.STD_TITLE_MARGIN_TOP))

            Text(
                text = load_pitchSpeed(androidx.compose.ui.platform.LocalContext.current, true),
                fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                color = colorResource(R.color.std_cyan),
                fontFamily = robotoExtraBold,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Pitch Slider
            CustomSlider(
                value = if(AppConfig.tts_pitch!=0.0f)AppConfig.tts_pitch else pitchValue,
                onValueChange = onPitchChange,
                valueRange = 0f..2f,
                steps = 8,  // 8 stops
                onSliderMove = { newValue ->
                    onPitchChange(newValue)
                },
                thumbStyle = ThumbStyle.DOUBLE_BAR,
                thumbColor = colorResource(R.color.std_purple),
                thumbWidth = 8.dp,
                thumbHeight = 55.dp,
                thumbBarSpacing = 4.dp,
                trackHeight = 25.dp,
                activeTrackColor = Color.White,
                inactiveTrackColor = Color.White,
                trackShadow = 5.dp,
                modifier = Modifier.fillMaxWidth(0.8f),
                stepsColor = colorResource(R.color.purple_light)
            )

            Spacer(modifier = Modifier.height(12.dp))

            Text(
                text = String.format("%.2f", if(AppConfig.tts_pitch!=0.0f)AppConfig.tts_pitch else pitchValue),
                fontSize = Constants.STD_SLIDER_INFO_SIZE.sp,
                color = colorResource(R.color.std_purple),
                fontFamily = robotoSemibold
            )

            // Second slider - Speed
            Box(modifier = Modifier.height(screenHeight * 0.06f))

            Text(
                text = load_pitchSpeed(androidx.compose.ui.platform.LocalContext.current, false),
                fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                color = colorResource(R.color.std_cyan),
                fontFamily = robotoExtraBold,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Speed Slider
            CustomSlider(
                value = if(AppConfig.tts_speech_rate!=0.0f)AppConfig.tts_speech_rate else speedValue,
                onValueChange = onSpeedChange,
                valueRange = 0f..2f,
                steps = 8,  // 8 stops
                thumbStyle = ThumbStyle.DOUBLE_BAR,
                thumbColor = colorResource(R.color.std_purple),
                thumbWidth = 8.dp,
                thumbHeight = 55.dp,
                thumbBarSpacing = 4.dp,
                trackHeight = 25.dp,
                activeTrackColor = Color.White,
                inactiveTrackColor = Color.White,
                trackShadow = 5.dp,
                modifier = Modifier.fillMaxWidth(0.8f),
                stepsColor = colorResource(R.color.purple_light)
            )

            Spacer(modifier = Modifier.height(12.dp))

            Text(
                text = String.format("%.2f", if(AppConfig.tts_speech_rate!=0.0f)AppConfig.tts_speech_rate else speedValue),
                fontSize = Constants.STD_SLIDER_INFO_SIZE.sp,
                color = colorResource(R.color.std_purple),
                fontFamily = robotoSemibold
            )
        }

        // Navigation buttons
        val bottomSpace = screenHeight * Constants.STD_NAV_MARGIN_BOTTOM
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = bottomSpace),
            horizontalArrangement = Arrangement.spacedBy(screenWidth * 0.08f),
        ) {
            BackArrowLargeFab(
                onClick = onBackClick
            )

            NextArrowLargeFab(
                onClick = onNextClick,
            )
        }
    }
}

@Preview(name = "UserInfoE3 Activity", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun UserInfoE3ActivityPreview() {
    UserInfoE3Screen(
        pitchValue = 0f,
        speedValue = 1.0f,
        onPitchChange = {},
        onSpeedChange = {},
        onBackClick = {},
        onNextClick = {}
    )
}
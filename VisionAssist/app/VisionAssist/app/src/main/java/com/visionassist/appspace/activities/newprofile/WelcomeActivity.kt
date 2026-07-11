@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

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
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.togetherWith
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
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddCircleOutline
import androidx.compose.material.icons.filled.ArrowCircleDown
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.view.WindowCompat
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.newprofile.WelcomeActivity.Section
import com.visionassist.appspace.activities.newprofile.jsonCollection.ProfileFileCollection
import com.visionassist.appspace.activities.tabs.settings.BlockingOverlay
import com.visionassist.appspace.jetpack.design.BackArrowLargeFab
import com.visionassist.appspace.jetpack.design.LanguageSelector
import com.visionassist.appspace.jetpack.design.LoadingComponent
import com.visionassist.appspace.jetpack.design.NextArrowLargeFab
import com.visionassist.appspace.jetpack.managers.InfoNotificationManager
import com.visionassist.appspace.models.ttsengine.TTSManager
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.Language
import com.visionassist.appspace.utils.load_loadProfileText
import com.visionassist.appspace.utils.load_newProfileText
import com.visionassist.appspace.utils.load_profileSelectionButton
import com.visionassist.appspace.utils.robotoRegular
import com.visionassist.appspace.utils.robotoSemibold

class WelcomeActivity : ComponentActivity() {
    private val TAG = "WelcomeActivity"

    private var currentSection by mutableStateOf(Section.LANGUAGE)
    private var selectedLanguage by mutableStateOf(
        if (AppConfig.mainLanguage != null) AppConfig.mainLanguage else
            Language("en", "English", "US")
    )
    private var startWithProfileSelection = false
    private val showLoading = mutableStateOf(false)
    private val loadingText = mutableStateOf("")
    private lateinit var loadProfileInfoManager: InfoNotificationManager
    private lateinit var newProfileInfoManager: InfoNotificationManager
    private var ttsManager: TTSManager = PhoneStatusMonitor.getInstance().ttsManager
    private val ttsHandler = Handler(Looper.getMainLooper())
    private var waitingForTTSLanguage = false

    enum class Section {
        LANGUAGE, PROFILE_SELECTION
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        WindowCompat.setDecorFitsSystemWindows(window, false)

        // Check if we should start with profile selection
        startWithProfileSelection = intent.getBooleanExtra(Constants.EXTRA_WELCOME_OPTION, false)

        // Initialize info notification managers
        loadProfileInfoManager = InfoNotificationManager(this)
        newProfileInfoManager = InfoNotificationManager(this)

        if (startWithProfileSelection) {
            currentSection = Section.PROFILE_SELECTION
        }

        setContent {
            MaterialTheme {
                WelcomeScreen(
                    loadProfileInfoManagerValue= loadProfileInfoManager.isVisibleState.value,
                    newProfileInfoManagerValue = newProfileInfoManager.isVisibleState.value,
                    showLoading = showLoading.value,
                    loadingText = loadingText.value,
                    selectedLanguage = selectedLanguage,
                    currentSection = currentSection,
                    onLanguageSelected = { language -> selectedLanguage = language },
                    onLanguageBackPressed = ::onLanguageBackPressed,
                    onLanguageNextPressed = ::onLanguageNextPressed,
                    onProfileSelectionBackPressed = ::onProfileSelectionBackPressed,
                    onLoadProfileClicked = ::onLoadProfileClicked,
                    onNewProfileClicked = ::onNewProfileClicked,
                    onShowLoadProfileInfo = ::showLoadProfileInfo,
                    onShowNewProfileInfo = ::showNewProfileInfo
                )
            }
        }
    }

    private fun onLanguageBackPressed() {
        ProfileFileCollection.deleteConfigurationActivity()
        Log.d(TAG, "Language section: Back pressed")
        val intent = Intent(this, ConfigurationActivity::class.java)
        startActivity(intent)
        finish()
    }

    override fun onResume() {
        super.onResume()
        // Handle return from TTS language installation
        if (waitingForTTSLanguage) {
            ttsHandler.removeCallbacksAndMessages(null)
            ttsManager.recheckPendingLanguage()
            waitForTTSAndNavigate(true)
        }
    }

    private fun setTTSLanguage() {
        loadingText.value =
            if (selectedLanguage.code == "en") "TTS model is configuring, please wait"
            else
                "Modelul TTS se configurează, vă rugăm așteptați"
        showLoading.value = true
        val handler = Handler(Looper.getMainLooper())
        val checkTTS: Runnable = Runnable {
            waitingForTTSLanguage = true
            ttsManager.changeLanguage(selectedLanguage, this@WelcomeActivity)
            waitForTTSAndNavigate(true)
        }
        handler.postDelayed(checkTTS, Constants.ANIMATION_DELAY.toLong() + 1000)
    }

    private fun waitForTTSAndNavigate(wait: Boolean) {
        val checkTTS: Runnable = object : Runnable {
            override fun run() {
                if (ttsManager.isReady) {
                    Log.d(TAG, "TTS is ready, navigating to home")
                    waitingForTTSLanguage = false
                    onLanguageNextPressedNavigate(wait)
                } else {
                    Log.w(TAG, "TTS not ready, retrying...")
                    ttsHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS.toLong())
                }
            }
        }
        ttsHandler.post(checkTTS)
    }

    private fun onLanguageNextPressed() {
        if (selectedLanguage.code != ttsManager.currentLocale.language) {
            Log.d(TAG, "TTS is not init on the lang selected")
            setTTSLanguage()
        } else {
            Log.d(TAG, "TTS is already init, navigating to 2nd section")
            waitForTTSAndNavigate(false)
        }
    }

    private fun onLanguageNextPressedNavigate(isLoadingActive: Boolean) {
        if (isLoadingActive) showLoading.value = false
        Log.d(TAG, "Language section: Next pressed, selected language: ${selectedLanguage.name}")
        ProfileFileCollection.writeWelcomeActivity(false, selectedLanguage, null)
        AppConfig.mainLanguage = selectedLanguage
        currentSection = Section.PROFILE_SELECTION
    }

    private fun onProfileSelectionBackPressed() {
        Log.d(TAG, "Profile selection: Back pressed")
        // Animate back to language section
        ProfileFileCollection.deleteWelcomeActivity(false)
        currentSection = Section.LANGUAGE
    }

    private fun onLoadProfileClicked() {
        Log.d(TAG, "Load Profile clicked")

        ProfileFileCollection.writeWelcomeActivity(true, null, false)

        // Navigate to LoadProfileActivity
        val intent = Intent(this, LoadProfileActivity::class.java)
        startActivity(intent)
        finish()
    }

    private fun onNewProfileClicked() {
        Log.d(TAG, "New Profile clicked")

        ProfileFileCollection.writeWelcomeActivity(true, null, true)

        // Navigate to next activity (UserInfoActivity or similar)
        val intent = Intent(this, NewProfileActivity::class.java)
        startActivity(intent)
        finish()
    }

    private fun showLoadProfileInfo() {
        Log.d(TAG, "Show Load Profile info")
        val message = load_loadProfileText(this)
        loadProfileInfoManager.showNotification(
            message, {
                loadProfileInfoManager.hideNotification()
            }, "OK"
        )
    }

    private fun showNewProfileInfo() {
        Log.d(TAG, "Show New Profile info")
        val message = load_newProfileText(this)
        loadProfileInfoManager.showNotification(
            message, {
                loadProfileInfoManager.hideNotification()
            }, "OK"
        )
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                Log.d(TAG, "Volume button down for repeat pressed")
                return true
            }

            KeyEvent.KEYCODE_VOLUME_UP -> {
                Log.d(TAG, "Volume button up pressed")
                return true
            }
        }
        return super.onKeyDown(keyCode, event)
    }
}

@Composable
fun WelcomeScreen(
    loadProfileInfoManagerValue: Boolean,
    newProfileInfoManagerValue: Boolean,
    showLoading: Boolean,
    loadingText: String,
    selectedLanguage: Language,
    currentSection: Section,
    onLanguageSelected: (Language) -> Unit,
    onLanguageBackPressed: () -> Unit,
    onLanguageNextPressed: () -> Unit,
    onProfileSelectionBackPressed: () -> Unit,
    onLoadProfileClicked: () -> Unit,
    onNewProfileClicked: () -> Unit,
    onShowLoadProfileInfo: () -> Unit,
    onShowNewProfileInfo: () -> Unit
) {
    val blockMainUI= loadProfileInfoManagerValue || newProfileInfoManagerValue || showLoading

    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenHeight = maxHeight
        val screenWidth = maxWidth
        // Background gradient image
        Image(
            painter = painterResource(id = R.drawable.welcome_background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        // Animated content with slide transitions
        AnimatedContent(
            modifier= Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
                .then(
                    if (blockMainUI) {
                        Modifier.clearAndSetSemantics { }  //  COMPLETELY REMOVE from tree!
                    } else {
                        Modifier
                    }
                )
            ,
            targetState = currentSection, transitionSpec = {
                if (targetState == Section.PROFILE_SELECTION) {
                    // Sliding to profile selection (right to left)
                    slideInHorizontally(
                        initialOffsetX = { fullWidth -> fullWidth },
                        animationSpec = tween(Constants.ANIMATION_DELAY)
                    ) togetherWith slideOutHorizontally(
                        targetOffsetX = { fullWidth -> -fullWidth },
                        animationSpec = tween(Constants.ANIMATION_DELAY)
                    )
                } else {
                    // Sliding back to language selection (left to right)
                    slideInHorizontally(
                        initialOffsetX = { fullWidth -> -fullWidth },
                        animationSpec = tween(Constants.ANIMATION_DELAY)
                    ) togetherWith slideOutHorizontally(
                        targetOffsetX = { fullWidth -> fullWidth },
                        animationSpec = tween(Constants.ANIMATION_DELAY)
                    )
                }
            }, label = "section_animation"
        ) { section ->
            when (section) {
                Section.LANGUAGE -> LanguageSelectionSection(
                    selectedLanguage = selectedLanguage,
                    onLanguageSelected = onLanguageSelected
                )

                Section.PROFILE_SELECTION -> ProfileSelectionSection(
                    onLoadProfileClicked = onLoadProfileClicked,
                    onNewProfileClicked = onNewProfileClicked,
                    onShowLoadProfileInfo = onShowLoadProfileInfo,
                    onShowNewProfileInfo = onShowNewProfileInfo
                )
            }
        }

        val bottomSpace = screenHeight * Constants.STD_NAV_MARGIN_BOTTOM
        if (currentSection == Section.LANGUAGE) {
            // Navigation Buttons (not animated, always visible at bottom)
            Row(
                modifier = Modifier
                    .navigationBarsPadding()
                    .align(Alignment.BottomCenter)
                    .padding(bottom = bottomSpace),
                horizontalArrangement = Arrangement.spacedBy(screenWidth * 0.08f),
            ) {
                BackArrowLargeFab(
                    onClick = onLanguageBackPressed
                )

                NextArrowLargeFab(
                    onClick = onLanguageNextPressed,
                )
            }
        } else {
            // Back Button (not animated, always visible at bottom)
            Box(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = bottomSpace)
            ) {
                BackArrowLargeFab(
                    onClick = onProfileSelectionBackPressed
                )
            }
        }

        BlockingOverlay(blockMainUI)

        LoadingComponent(
            isVisible = showLoading,
            loadingText = loadingText
        )
    }
}

@Composable
fun LanguageSelectionSection(
    selectedLanguage: Language,
    onLanguageSelected: (Language) -> Unit
) {
    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenHeight = maxHeight
        // Animated content (Title + Language Selector)
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.SpaceAround,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(modifier = Modifier.height(screenHeight * Constants.STD_SUBTITLE_MARGIN_TOP))

            // Title
            Text(
                text = "What language would\nyou prefer?",
                fontSize = Constants.STD_SUBTITLE_SIZE.sp,
                color = colorResource(R.color.std_cyan),
                fontFamily = robotoSemibold,
                textAlign = TextAlign.Center,
                lineHeight = 36.sp,
                modifier = Modifier.fillMaxWidth()
            )

            Box(modifier = Modifier.height(screenHeight * Constants.STD_SUBTITLE_BODY_MARGIN_TOP))

            // Language Selector
            LanguageSelector(
                selectedLanguage = selectedLanguage,
                onLanguageSelected = { language ->
                    onLanguageSelected(language)
                    Log.d("WelcomeActivity", "Language selected: ${language.code}")
                })

            Box(modifier = Modifier.height(screenHeight * 0.33f))
        }
    }
}

@Composable
fun ProfileSelectionSection(
    onLoadProfileClicked: () -> Unit,
    onNewProfileClicked: () -> Unit,
    onShowLoadProfileInfo: () -> Unit,
    onShowNewProfileInfo: () -> Unit,
    loadProfileText: String = load_profileSelectionButton(
        PhoneStatusMonitor.getInstance().currentContext,
        true
    ),
    newProfileText: String = load_profileSelectionButton(
        PhoneStatusMonitor.getInstance().currentContext,
        false
    )
) {
    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenHeight = maxHeight
        // Animated content (Profile buttons)
        Column(
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Box(modifier = Modifier.height(screenHeight * 0.4f))

            // Load Profile Button
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.Bottom,
                horizontalArrangement = Arrangement.Center
            ) {
                ProfileButton(
                    text = loadProfileText,
                    icon = Icons.Filled.ArrowCircleDown,
                    onClick = onLoadProfileClicked
                )

                InfoIconButton(
                    onClick = onShowLoadProfileInfo
                )
            }

            Box(modifier = Modifier.height(screenHeight * 0.1f))

            // New Profile Button
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.Bottom,
                horizontalArrangement = Arrangement.Center
            ) {
                ProfileButton(
                    text = newProfileText,
                    icon = Icons.Filled.AddCircleOutline,
                    onClick = onNewProfileClicked
                )

                InfoIconButton(
                    onClick = onShowNewProfileInfo
                )
            }

            Box(modifier = Modifier.height(screenHeight * 0.4f))
        }
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun ProfileButton(
    text: String, icon: ImageVector, onClick: () -> Unit,
) {
    Button(
        onClick = onClick,
        modifier = Modifier
            .shadow(
                elevation = 3.dp, shape = MaterialTheme.shapes.extraExtraLarge
            )
            .fillMaxWidth(0.75f)
            .height(Constants.STD_BUTTON_PAGE_HEIGHT.dp),
        shape = RoundedCornerShape(100.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = colorResource(R.color.notification_button_white),
            contentColor = colorResource(R.color.std_purple)
        ),
    ) {
        Icon(
            imageVector = icon,
            modifier = Modifier.size(30.dp),
            tint = Color.Black,
            contentDescription = ""
        )

        Spacer(modifier = Modifier.width(12.dp))

        Text(
            text = text, fontSize = 20.sp,
            fontFamily = robotoRegular
        )
    }
}

@Composable
fun InfoIconButton(onClick: () -> Unit) {
    IconButton(
        onClick = onClick, modifier = Modifier.size(35.dp)
    ) {
        Icon(
            imageVector = Icons.Filled.Info,
            contentDescription = "Information",
            modifier = Modifier.size(Constants.STD_INFO_BUTTON_SIZE.dp),
            tint = colorResource(R.color.std_purple)
        )
    }
}

@SuppressLint("UnrememberedMutableState")
@Preview(
    name = "Welcome Activity/LanguageSection", showBackground = true, widthDp = 412, heightDp = 917
)
@Composable
fun WelcomeActivity1Preview() {
    MaterialTheme {
        var selectedLanguage by mutableStateOf(Language("en", "English", "US"))
        Image(
            painter = painterResource(id = R.drawable.welcome_background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )
        LanguageSelectionSection(onLanguageSelected = {}, selectedLanguage = selectedLanguage)
    }
}

@Preview(
    name = "Welcome Activity/ProfileSection", showBackground = true, widthDp = 412, heightDp = 917
)
@Composable
fun WelcomeActivity2Preview() {
    MaterialTheme {
        Image(
            painter = painterResource(id = R.drawable.welcome_background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        ProfileSelectionSection(
            loadProfileText = "Load Profile",
            newProfileText = "New Profile",
            onLoadProfileClicked = {},
            onNewProfileClicked = {},
            onShowLoadProfileInfo = {},
            onShowNewProfileInfo = {})
    }
}

@SuppressLint("UnrememberedMutableState")
@Preview(
    name = "Welcome Activity", showBackground = true, widthDp = 412, heightDp = 917
)
@Composable
fun WelcomeActivityPreview() {
    MaterialTheme {
        var currentSection by mutableStateOf(Section.LANGUAGE)
        var selectedLanguage by mutableStateOf(Language("en", "English", "US"))
        WelcomeScreen(
            showLoading = false,
            loadingText = "",
            selectedLanguage = selectedLanguage,
            currentSection = currentSection,
            onLanguageSelected = {},
            onLanguageBackPressed = {},
            onLanguageNextPressed = {},
            onProfileSelectionBackPressed = {},
            onLoadProfileClicked = {},
            onNewProfileClicked = {},
            onShowLoadProfileInfo = {},
            onShowNewProfileInfo = {},
            loadProfileInfoManagerValue = false,
            newProfileInfoManagerValue = false
        )
    }
}
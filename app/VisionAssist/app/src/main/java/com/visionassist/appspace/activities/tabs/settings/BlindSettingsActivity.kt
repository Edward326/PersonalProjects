@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.tabs.settings

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.DocumentsContract
import android.util.Log
import android.view.KeyEvent
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.hideFromAccessibility
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.view.WindowCompat
import com.visionassist.appspace.BaseActivity
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.main.BlindHomeActivity
import com.visionassist.appspace.activities.main.BottomNavigationBar
import com.visionassist.appspace.activities.main.MainActivity
import com.visionassist.appspace.activities.main.SyncStatusSection
import com.visionassist.appspace.activities.newprofile.LoadProfileActivity
import com.visionassist.appspace.activities.newprofile.UserInfoE3Activity
import com.visionassist.appspace.activities.newprofile.jsonCollection.ProfileFileCollection.writeSoA
import com.visionassist.appspace.activities.newprofile.jsonCollection.ProfileFileCollection.writeWelcomeActivity
import com.visionassist.appspace.database.DBConstants
import com.visionassist.appspace.database.DBManager
import com.visionassist.appspace.database.NetworkUtils
import com.visionassist.appspace.jetpack.design.LanguageSelector
import com.visionassist.appspace.jetpack.design.LoadingComponent
import com.visionassist.appspace.jetpack.design.NotificationDialog
import com.visionassist.appspace.jetpack.design.QuickActionSelector
import com.visionassist.appspace.jetpack.managers.InfoNotificationManager
import com.visionassist.appspace.models.ttsengine.TTSManager
import com.visionassist.appspace.services.LockScreenService
import com.visionassist.appspace.services.LockScreenService.Companion.getCurrentQuickActionIndex
import com.visionassist.appspace.sound.SoundConstants
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.BackgroundTaskExecutor
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.FileUtils
import com.visionassist.appspace.utils.Language
import com.visionassist.appspace.utils.calculateHashCacheSize
import com.visionassist.appspace.utils.formatPercentage
import com.visionassist.appspace.utils.formatSizeKB
import com.visionassist.appspace.utils.getAccountSectionText
import com.visionassist.appspace.utils.getAccountSectionTextTTS
import com.visionassist.appspace.utils.getApplyingSettingsText
import com.visionassist.appspace.utils.getApplyingSettingsTextTTS
import com.visionassist.appspace.utils.getCacheClearedMessage
import com.visionassist.appspace.utils.getChangeTTSParametersText
import com.visionassist.appspace.utils.getClearCacheConfirmMessage
import com.visionassist.appspace.utils.getClearCacheText
import com.visionassist.appspace.utils.getCurrentEnvReportsSize
import com.visionassist.appspace.utils.getCurrentHashCacheSize
import com.visionassist.appspace.utils.getDeleteAccountConfirmMessage
import com.visionassist.appspace.utils.getDeleteAccountText
import com.visionassist.appspace.utils.getDeletingAccountText
import com.visionassist.appspace.utils.getExportProfileText
import com.visionassist.appspace.utils.getGeneralSectionText
import com.visionassist.appspace.utils.getGeneralSectionTextTTS
import com.visionassist.appspace.utils.getLanguageText
import com.visionassist.appspace.utils.getLogOutText
import com.visionassist.appspace.utils.getLoggingOffText
import com.visionassist.appspace.utils.getLoggingOffTextTTS
import com.visionassist.appspace.utils.getLogoutConfirmMessage
import com.visionassist.appspace.utils.getPasswordTitle
import com.visionassist.appspace.utils.getPasswordTitleTTS
import com.visionassist.appspace.utils.getProfileExportErrorMessage
import com.visionassist.appspace.utils.getProfileExportedMessage
import com.visionassist.appspace.utils.getProfileExportedMessageTutorial
import com.visionassist.appspace.utils.getProfileSyncErrorMessage
import com.visionassist.appspace.utils.getProfileSyncedMessage
import com.visionassist.appspace.utils.getQuickAccessType
import com.visionassist.appspace.utils.getQuickActionDisabledMessageTTS
import com.visionassist.appspace.utils.getQuickActionEnabledMessageTTS
import com.visionassist.appspace.utils.getQuickActionInfoMessage
import com.visionassist.appspace.utils.getQuickActionText
import com.visionassist.appspace.utils.getSoAInfoMessage
import com.visionassist.appspace.utils.getSoAText
import com.visionassist.appspace.utils.getSoAToggle
import com.visionassist.appspace.utils.getSyncProfileTTS
import com.visionassist.appspace.utils.getSyncProfileText
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.load_genericErrorDelete
import com.visionassist.appspace.utils.load_genericErrorLogout
import com.visionassist.appspace.utils.load_noInternet
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.vibrate
import org.json.JSONObject
import java.io.File
import java.io.FileNotFoundException
import java.io.IOException

class BlindSettingsActivity : BaseActivity() {
    private val TAG = "BlindSettingsActivity"

    private val mainHandler = Handler(Looper.getMainLooper())
    private val ttsHandler = Handler(Looper.getMainLooper())

    // Managers
    private val ttsManager: TTSManager = PhoneStatusMonitor.getInstance().ttsManager
    private val dbManager: DBManager = PhoneStatusMonitor.getInstance().dbManager
    private val soundManager = PhoneStatusMonitor.getInstance().soundManager
    private lateinit var infoNotificationManager: InfoNotificationManager

    // UI States
    private val isSpeakingApplyingSettings = mutableStateOf(false)
    private val showErrorPassword = mutableStateOf(false)
    private val showPasswordDialog = mutableStateOf(false)
    private val passwordValue = mutableStateOf("")
    private val showLoading = mutableStateOf(false)
    private val loadingText = mutableStateOf("")
    private val showNotifDialog = mutableStateOf(false)
    private val notifMessage = mutableStateOf("")
    private val slideMessage = mutableStateOf("")
    private val currentSection = mutableStateOf(0)

    // Size displays
    private val hashCacheSize = mutableStateOf("")
    private val hashCachePercent = mutableStateOf("")
    private val envReportsSize = mutableStateOf("")

    // Settings states
    private val selectedLanguage = mutableStateOf(AppConfig.mainLanguage)
    private val selectedQuickAction = mutableStateOf(0)
    private val soAEnabled = mutableStateOf(AppConfig.SoA)

    // Flags
    private var repeatTTS = false
    private var waitingForTTSLanguage = false
    private var exportErrorCode = 0
    private var exportFilesCompleted = 0
    private var profileData: JSONObject? = null

    // File picker launcher
    private val exportProfileLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocumentTree()
    ) { uri ->
        uri?.let { handleExportProfileResult(it) }
    }
    private var exportedFolderUri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        WindowCompat.setDecorFitsSystemWindows(window, false)

        selectedQuickAction.value = getCurrentQuickActionIndex(this)

        // Calculate initial sizes
        calculateSizes()

        // Setup info notification manager
        infoNotificationManager = InfoNotificationManager(this)

        val dbManager = PhoneStatusMonitor.getInstance().dbManager

        setContent {
            BlindSettingsScreen(
                isSpeakingApplyingSettings = isSpeakingApplyingSettings.value,
                infoNotificationManagerValue = infoNotificationManager.isVisibleState.value,
                showPasswordDialog = showPasswordDialog.value,
                showLoading = showLoading.value,
                loadingText = loadingText.value,
                showNotification = showNotifDialog.value,
                notificationMessage = notifMessage.value,
                currentSection = currentSection.value,
                selectedLanguage = selectedLanguage.value,
                selectedQuickAction = selectedQuickAction.value,
                soAEnabled = soAEnabled.value,
                hasRemoteProfile = dbManager.statusOverview == 1 || dbManager.statusOverview == 2,
                onLanguageChange = ::handleLanguageChange,
                onQuickActionChange = ::handleQuickActionChange,
                onQuickActionInfoClick = ::handleQuickActionInfo,
                onSoAToggle = ::handleSoAToggle,
                onSoAInfoClick = ::handleSoAInfo,
                onChangeTTSParameters = ::handleChangeTTSParameters,
                onClearCache = ::handleClearCache,
                onSyncProfile = ::handleSyncProfile,
                onLogout = ::handleLogout,
                onDeleteAccount = ::handleDeleteAccount,
                onExportProfile = ::handleExportProfile,
                onNavigateHome = ::handleNavigateHome,
                onNavigateSettings = ::handleNavigateSettings,
                onSectionChange = { section -> currentSection.value = section },
                firstButtonClick = ::handleNotifFirstButton,
                syncDays = dbManager.diffDays,
                onPasswordChange = { password ->
                    showErrorPassword.value = false
                    passwordValue.value = password
                },
                firstButtonClickPassword = ::handleBackPassword,
                secondButtonClickPassword = ::handleNextPassword,
                showErrorPassword = showErrorPassword.value,
                passwordValue = passwordValue.value,
            )
        }
    }

    private fun cancelAllHandlers() {
        repeatTTS = false
        mainHandler.removeCallbacksAndMessages(null)
        soundManager.releaseCallback()
        ttsManager.stopSpeaking()
    }

    private fun waitForTTSSpeech(afterTTSSpeech: Runnable) {
        val checkRunnable = object : Runnable {
            override fun run() {
                if (ttsManager.isDoneSpeaking) {
                    mainHandler.post(afterTTSSpeech)
                } else {
                    mainHandler.postDelayed(this, 350)
                }
            }
        }
        mainHandler.post(checkRunnable)
    }

    override fun onResume() {
        super.onResume()

        if (waitingForTTSLanguage) {
            ttsHandler.removeCallbacksAndMessages(null)
            Log.d(TAG, "Returned from TTS settings, rechecking language")
            ttsManager.recheckPendingLanguage()
            waitForTTSAndReload()
        }
    }

    private fun calculateSizes() {
        val currentHashSize = getCurrentHashCacheSize(this)
        val totalHashSize = calculateHashCacheSize()

        hashCacheSize.value = formatSizeKB(currentHashSize)
        hashCachePercent.value = formatPercentage(currentHashSize, totalHashSize)

        val currentReportsSize = getCurrentEnvReportsSize(this)
        envReportsSize.value = formatSizeKB(currentReportsSize)
    }

    private fun handleBackPassword() {
        cancelAllHandlers()
        showPasswordDialog.value = false
    }

    private fun handleNextPassword() {
        cancelAllHandlers()

        if (!NetworkUtils.isNetworkConnected(this)) {
            showPasswordDialog.value = false
            repeatTTS = true
            ttsManager.speak(
                load_noInternet(this), AppConfig.tts_pitch, AppConfig.tts_speech_rate, true, null
            )
            waitForTTSSpeech { repeatTTS = false }

            notifMessage.value = load_noInternet(this)
            showNotifDialog.value = true
            return
        }

        val profileFile = FileUtils.getProfileFile(this)
        profileData = JSONObject(profileFile.readText())
        val email = profileData?.getString("email") ?: ""

        BackgroundTaskExecutor.getInstance().executeAsync({
            return@executeAsync dbManager.verifyPassword(email, passwordValue.value)
        }, object : BackgroundTaskExecutor.TaskCallback<Boolean> {
            override fun onSuccess(result: Boolean) {
                if (!result) {
                    cancelAllHandlers()
                    showErrorPassword.value = true
                    ttsManager.speak(
                        if (AppConfig.mainLanguage.code == "en") "Incorrect password, please try again"
                        else "Parolă greșită, vă rugăm să încercați din nou",
                        AppConfig.tts_pitch,
                        AppConfig.tts_speech_rate,
                        false,
                        null
                    )
                } else {
                    cancelAllHandlers()
                    ttsManager.speak(
                        if (AppConfig.mainLanguage.code == "en") "Password is correct, deleting account now"
                        else "Parolă este corectă, se șterge contul acum",
                        AppConfig.tts_pitch,
                        AppConfig.tts_speech_rate,
                        false,
                        null
                    )
                    waitForTTSSpeech {
                        showPasswordDialog.value = false
                        executeDeleteAccount(email)
                    }
                }
            }

            override fun onError(e: Exception) {
                slideMessage.value = load_genericErrorDelete(this@BlindSettingsActivity)
                showPasswordDialog.value = false

                soundManager.play(
                    SoundConstants.SETTINGS_APPLIED_ID, 0.7f, 0.7f
                ) {
                    ttsManager.speak(
                        slideMessage.value,
                        AppConfig.tts_pitch,
                        AppConfig.tts_speech_rate,
                        false,
                        null
                    )
                }
            }
        })
    }

    private fun handleLanguageChange(language: Language) {
        cancelAllHandlers()

        if (language.code == selectedLanguage.value.code) return

        loadingText.value = getApplyingSettingsText(this)
        showLoading.value = true

        ttsManager.speak(
            getApplyingSettingsTextTTS(this),
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            false,
            null
        )

        waitForTTSSpeech {
            selectedLanguage.value = language
            AppConfig.mainLanguage = language
            writeWelcomeActivity(false, AppConfig.mainLanguage, false)
            setTTSLanguage()
        }
    }

    private fun setTTSLanguage() {
        Log.d(TAG, "Changing TTS language")
        waitingForTTSLanguage = true
        ttsManager.changeLanguage(AppConfig.mainLanguage, this)
        waitForTTSAndReload()
    }

    private fun waitForTTSAndReload() {
        val checkTTS = object : Runnable {
            override fun run() {
                if (ttsManager.isReady) {
                    Log.d(TAG, "TTS ready, reloading UI")
                    waitingForTTSLanguage = false

                    showLoading.value = false
                    soundManager.play(
                        SoundConstants.SETTINGS_APPLIED_ID, 0.7f, 0.7f
                    ) {
                        val intent =
                            Intent(this@BlindSettingsActivity, BlindSettingsActivity::class.java)
                        startActivity(intent)
                        finish()
                    }
                } else {
                    Log.w(TAG, "TTS not ready, retrying...")
                    ttsHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS.toLong())
                }
            }
        }
        ttsHandler.post(checkTTS)
    }

    private fun handleQuickActionChange(action: Int) {
        cancelAllHandlers()

        if (action == selectedQuickAction.value) return

        loadingText.value = getApplyingSettingsText(this)
        showLoading.value = true

        ttsManager.speak(
            getApplyingSettingsTextTTS(this),
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            false,
            null
        )

        waitForTTSSpeech {
            selectedQuickAction.value = action

            if (action == 0) {
                LockScreenService.stopService(this)
                slideMessage.value = getQuickActionDisabledMessageTTS(this)
            } else {
                LockScreenService.startService(this, action)
                slideMessage.value = getQuickActionEnabledMessageTTS(this)
            }

            showLoading.value = false
            isSpeakingApplyingSettings.value = true
            soundManager.play(
                SoundConstants.SETTINGS_APPLIED_ID, 0.7f, 0.7f
            ) {
                ttsManager.speak(
                    slideMessage.value, AppConfig.tts_pitch, AppConfig.tts_speech_rate, false, null
                )
            }
            waitForTTSSpeech {
                isSpeakingApplyingSettings.value = false
            }
        }
    }

    private fun handleQuickActionInfo() {
        cancelAllHandlers()

        repeatTTS = true
        ttsManager.speak(
            getQuickActionInfoMessage(this),
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            true,
            null
        )
        waitForTTSSpeech {
            repeatTTS = false
        }

        infoNotificationManager.showNotification(
            getQuickActionInfoMessage(this),
            { cancelAllHandlers(); infoNotificationManager.hideNotification() },
            "OK"
        )
    }

    private fun handleSoAToggle(enabled: Boolean) {
        cancelAllHandlers()

        soAEnabled.value = enabled
        AppConfig.SoA = enabled
        writeSoA(AppConfig.SoA)

        isSpeakingApplyingSettings.value = true
        soundManager.play(
            SoundConstants.SETTINGS_APPLIED_ID, 0.7f, 0.7f
        ) {
            ttsManager.speak(
                getSoAToggle(enabled),
                AppConfig.tts_pitch,
                AppConfig.tts_speech_rate,
                false,
                null
            )
        }
        waitForTTSSpeech {
            isSpeakingApplyingSettings.value = false
        }
    }

    private fun handleSoAInfo() {
        cancelAllHandlers()

        repeatTTS = true
        ttsManager.speak(
            getSoAInfoMessage(this), AppConfig.tts_pitch, AppConfig.tts_speech_rate, true, null
        )
        waitForTTSSpeech {
            repeatTTS = false
        }

        vibrateIfNeeded()
        infoNotificationManager.showNotification(
            getSoAInfoMessage(this),
            { cancelAllHandlers(); infoNotificationManager.hideNotification() },
            "OK"
        )
    }

    private fun handleChangeTTSParameters() {
        cancelAllHandlers()

        val intent = Intent(this, UserInfoE3Activity::class.java).apply {
            putExtra(Constants.EXTRA_USERACC_OPTION2, true)
        }
        startActivity(intent)
    }

    private fun handleClearCache() {
        cancelAllHandlers()

        repeatTTS = true
        ttsManager.speak(
            getClearCacheConfirmMessage(this),
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            true,
            null
        )
        waitForTTSSpeech {
            repeatTTS = false
        }

        infoNotificationManager.showNotificationTwoButtons(
            getClearCacheConfirmMessage(this),
            if (AppConfig.mainLanguage.code == "en") {
                getString(R.string.dont_button_en)
            } else {
                getString(R.string.dont_button_ro)
            },
            if (AppConfig.mainLanguage.code == "en") {
                getString(R.string.clear_button_en)
            } else {
                getString(R.string.clear_button_ro)
            },
            { cancelAllHandlers(); infoNotificationManager.hideNotification() },
            { infoNotificationManager.hideNotification(); executeClearCache() })
    }

    private fun executeClearCache() {
        cancelAllHandlers()
        loadingText.value = getApplyingSettingsText(this)
        showLoading.value = true

        ttsManager.speak(
            getApplyingSettingsTextTTS(this),
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            false,
            null
        )

        waitForTTSSpeech {
            try {
                FileUtils.createProfileDirFile(Constants.HASH_CACHE_FILE_NAME)
                slideMessage.value = getCacheClearedMessage(this)
            } catch (e: Exception) {
                Log.e(TAG, "Error clearing cache", e)
                slideMessage.value = if (AppConfig.mainLanguage.code == "en") "Error clearing cache"
                else "Eroare la ștergerea cache-ului"
            }

            showLoading.value = false
            isSpeakingApplyingSettings.value = true
            soundManager.play(
                SoundConstants.SETTINGS_APPLIED_ID, 0.7f, 0.7f
            ) {
                ttsManager.speak(
                    slideMessage.value, AppConfig.tts_pitch, AppConfig.tts_speech_rate, false, null
                )
            }
            waitForTTSSpeech {
                isSpeakingApplyingSettings.value = false
            }
        }
    }

    private fun handleSyncProfile() {
        cancelAllHandlers()

        if (!NetworkUtils.isNetworkConnected(this)) {
            repeatTTS = true
            ttsManager.speak(
                load_noInternet(this), AppConfig.tts_pitch, AppConfig.tts_speech_rate, true, null
            )
            waitForTTSSpeech { repeatTTS = false }

            notifMessage.value = load_noInternet(this)
            showNotifDialog.value = true
            return
        }

        loadingText.value = getSyncProfileTTS(this)
        showLoading.value = true

        ttsManager.speak(
            getSyncProfileTTS(this),
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            false,
            null
        )

        waitForTTSSpeech {
            BackgroundTaskExecutor.getInstance().executeAsync({
                // Load profile data
                try {
                    val profileFile = FileUtils.getProfileFile(this)
                    if (profileFile.exists()) {
                        profileData = JSONObject(profileFile.readText())
                    }
                } catch (_: Exception) {
                    return@executeAsync DBConstants.SYNC_ERROR
                }
                profileData?.let { dbManager.syncProfile(it) }
                return@executeAsync dbManager.status
            }, object : BackgroundTaskExecutor.TaskCallback<Int> {
                override fun onSuccess(result: Int?) {
                    slideMessage.value = when (result) {
                        DBConstants.SYNC_ERROR -> getProfileSyncErrorMessage(this@BlindSettingsActivity)
                        DBConstants.SYNC_OK -> getProfileSyncedMessage(this@BlindSettingsActivity)
                        else -> getProfileSyncedMessage(this@BlindSettingsActivity)
                    }
                    showLoading.value = false
                    isSpeakingApplyingSettings.value = true
                    soundManager.play(
                        SoundConstants.SETTINGS_APPLIED_ID, 0.7f, 0.7f
                    ) {
                        ttsManager.speak(
                            slideMessage.value,
                            AppConfig.tts_pitch,
                            AppConfig.tts_speech_rate,
                            false,
                            null
                        )
                    }
                    waitForTTSSpeech {
                        isSpeakingApplyingSettings.value = false
                    }
                }

                override fun onError(e: Exception?) {
                    mainHandler.post {
                        slideMessage.value = getProfileSyncErrorMessage(this@BlindSettingsActivity)
                        showLoading.value = false
                        isSpeakingApplyingSettings.value = true
                        soundManager.play(
                            SoundConstants.SETTINGS_APPLIED_ID, 0.7f, 0.7f
                        ) {
                            ttsManager.speak(
                                slideMessage.value,
                                AppConfig.tts_pitch,
                                AppConfig.tts_speech_rate,
                                false,
                                null
                            )
                        }
                        waitForTTSSpeech {
                            isSpeakingApplyingSettings.value = false
                        }
                    }
                }
            })
        }
    }

    private fun handleLogout() {
        cancelAllHandlers()

        repeatTTS = true
        ttsManager.speak(
            getLogoutConfirmMessage(this),
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            true,
            null
        )
        waitForTTSSpeech {
            repeatTTS = false
        }

        infoNotificationManager.showNotificationTwoButtons(
            getLogoutConfirmMessage(this),
            if (AppConfig.mainLanguage.code == "en") {
                getString(R.string.dont_button_en)
            } else {
                getString(R.string.dont_button_ro)
            },
            if (AppConfig.mainLanguage.code == "en") {
                getString(R.string.logout_button_en)
            } else {
                getString(R.string.logout_button_ro)
            },
            { cancelAllHandlers(); infoNotificationManager.hideNotification() },
            { infoNotificationManager.hideNotification(); executeLogout() })
    }

    private fun executeLogout() {
        cancelAllHandlers()
        loadingText.value = getLoggingOffText(this)
        showLoading.value = true

        ttsManager.speak(
            getLoggingOffTextTTS(this),
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            false,
            null
        )

        waitForTTSSpeech {
            try {
                // Delete all local files
                if (AppConfig.hash_caching.equals("light") || AppConfig.hash_caching.equals("heavy"))
                    if (!FileUtils.deleteProfileDirFile(Constants.HASH_CACHE_FILE_NAME)) throw Exception()

                if (!FileUtils.deleteProfileDirFile(Constants.PROFILE_FILE_NAME)) throw Exception()

                showLoading.value = false

                // Navigate to configuration
                val intent = Intent(this, MainActivity::class.java)
                startActivity(intent)
                finish()
            } catch (e: Exception) {
                Log.e(TAG, "Error during logout", e)
                slideMessage.value = load_genericErrorLogout(this@BlindSettingsActivity)
                showLoading.value = false
                soundManager.play(
                    SoundConstants.SETTINGS_APPLIED_ID, 0.7f, 0.7f
                ) {
                    ttsManager.speak(
                        slideMessage.value,
                        AppConfig.tts_pitch,
                        AppConfig.tts_speech_rate,
                        false,
                        null
                    )
                }
                waitForTTSSpeech {
                    val intent = Intent(this, MainActivity::class.java)
                    startActivity(intent)
                    finish()
                }
            }
        }
    }

    private fun handleDeleteAccount() {
        cancelAllHandlers()
        if (!NetworkUtils.isNetworkConnected(this)) {
            repeatTTS = true
            ttsManager.speak(
                load_noInternet(this), AppConfig.tts_pitch, AppConfig.tts_speech_rate, true, null
            )
            waitForTTSSpeech { repeatTTS = false }

            notifMessage.value = load_noInternet(this)
            showNotifDialog.value = true
            return
        }

        repeatTTS = true
        ttsManager.speak(
            getDeleteAccountConfirmMessage(this),
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            true,
            null
        )
        waitForTTSSpeech {
            repeatTTS = false
        }

        infoNotificationManager.showNotificationTwoButtons(
            getDeleteAccountConfirmMessage(this),
            if (AppConfig.mainLanguage.code == "en") {
                getString(R.string.dont_button_en)
            } else {
                getString(R.string.dont_button_ro)
            },
            if (AppConfig.mainLanguage.code == "en") {
                getString(R.string.delete_button_en)
            } else {
                getString(R.string.delete_button_ro)
            },
            { cancelAllHandlers(); infoNotificationManager.hideNotification() },
            {
                cancelAllHandlers()
                ttsManager.speak(
                    getPasswordTitleTTS(this),
                    AppConfig.tts_pitch,
                    AppConfig.tts_speech_rate,
                    false,
                    null
                )
                infoNotificationManager.hideNotification()
                showPasswordDialog.value = true
            })
    }

    private fun executeDeleteAccount(email: String) {
        cancelAllHandlers()
        loadingText.value = getDeletingAccountText(this)
        showLoading.value = true

        mainHandler.postDelayed({
            BackgroundTaskExecutor.getInstance().executeAsync({
                // Delete from Firebase
                dbManager.deleteAccount(email)

                if (AppConfig.hash_caching.equals("light") || AppConfig.hash_caching.equals("heavy"))
                    if (!FileUtils.deleteProfileDirFile(Constants.HASH_CACHE_FILE_NAME)) return@executeAsync -1

                if (!FileUtils.deleteProfileDirFile(Constants.PROFILE_FILE_NAME)) return@executeAsync -1

                return@executeAsync dbManager.status
            }, object : BackgroundTaskExecutor.TaskCallback<Int> {
                override fun onSuccess(result: Int) {
                    if (result == -1) {
                        Log.e(TAG, "Failed to delete local profile files")
                        val textTTS =
                            if (PhoneStatusMonitor.getInstance().ttsManager.currentLocale.language == "en") "An error was encountered while deleting the account, navigating to configuration page"
                            else "A aparut o eroare la ștergerea contului, se navighează către pagina de configurare"
                        soundManager.play(
                            SoundConstants.SETTINGS_APPLIED_ID, 0.7f, 0.7f
                        ) {
                            ttsManager.speak(
                                textTTS, AppConfig.tts_pitch, AppConfig.tts_speech_rate, false, null
                            )
                        }

                        waitForTTSSpeech {
                            showLoading.value = false
                            val intent =
                                Intent(this@BlindSettingsActivity, MainActivity::class.java)
                            startActivity(intent)
                            finish()
                        }
                    } else {
                        if (result == DBConstants.SYNC_OK) {
                            val textTTS =
                                if (PhoneStatusMonitor.getInstance().ttsManager.currentLocale.language == "en") "Account deleted successfully, navigating to configuration page"
                                else "Contul a fost șters cu succes, se navighează către pagina de configurare"

                            soundManager.play(
                                SoundConstants.SETTINGS_APPLIED_ID, 0.7f, 0.7f
                            ) {
                                ttsManager.speak(
                                    textTTS,
                                    AppConfig.tts_pitch,
                                    AppConfig.tts_speech_rate,
                                    false,
                                    null
                                )
                            }

                            waitForTTSSpeech {
                                showLoading.value = false
                                val intent =
                                    Intent(this@BlindSettingsActivity, MainActivity::class.java)
                                startActivity(intent)
                                finish()
                            }
                        } else {
                            val textTTS =
                                if (PhoneStatusMonitor.getInstance().ttsManager.currentLocale.language == "en") "An error was encountered while deleting the account, navigating to configuration page"
                                else "A aparut o eroare la ștergerea contului, se navighează către pagina de configurare"
                            soundManager.play(
                                SoundConstants.SETTINGS_APPLIED_ID, 0.7f, 0.7f
                            ) {
                                ttsManager.speak(
                                    textTTS,
                                    AppConfig.tts_pitch,
                                    AppConfig.tts_speech_rate,
                                    false,
                                    null
                                )
                            }

                            waitForTTSSpeech {
                                showLoading.value = false
                                val intent =
                                    Intent(this@BlindSettingsActivity, MainActivity::class.java)
                                startActivity(intent)
                                finish()
                            }
                        }
                    }
                }

                override fun onError(e: Exception?) {
                    val textTTS =
                        if (PhoneStatusMonitor.getInstance().ttsManager.currentLocale.language == "en") "An error was encountered while deleting the account, navigating to configuration page"
                        else "A aparut o eroare la ștergerea contului, se navighează către pagina de configurare"
                    soundManager.play(
                        SoundConstants.SETTINGS_APPLIED_ID, 0.7f, 0.7f
                    ) {
                        ttsManager.speak(
                            textTTS, AppConfig.tts_pitch, AppConfig.tts_speech_rate, false, null
                        )
                    }

                    waitForTTSSpeech {
                        showLoading.value = false
                        val intent = Intent(this@BlindSettingsActivity, MainActivity::class.java)
                        startActivity(intent)
                        finish()
                    }
                }
            })
        }, 1000)
    }

    private fun handleExportProfile() {
        cancelAllHandlers()

        repeatTTS = true
        ttsManager.speak(
            getProfileExportedMessageTutorial(this),
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            true,
            null
        )
        waitForTTSSpeech {
            repeatTTS = false
        }

        infoNotificationManager.showNotification(
            getProfileExportedMessageTutorial(this), {
                cancelAllHandlers(); infoNotificationManager.hideNotification()
                exportProfileLauncher.launch(null)
            }, if (AppConfig.mainLanguage.code == "en") "Browse" else "Răsfoire"
        )
    }

    private fun handleExportProfileResult(uri: Uri) {
        loadingText.value = if (AppConfig.mainLanguage.code == "en") {
            "Exporting profile..."
        } else {
            "Se exportă profilul..."
        }
        showLoading.value = true

        ttsManager.speak(
            if (PhoneStatusMonitor.getInstance().ttsManager.currentLanguage == "en") {
                "Exporting profile..."
            } else {
                "Se exportă profilul..."
            }, AppConfig.tts_pitch, AppConfig.tts_speech_rate, false, null
        )

        waitForTTSSpeech {
            exportErrorCode = 0
            exportFilesCompleted = 0
            exportedFolderUri = null

            try {
                val profileFolderUri = createProfileFolder(uri)
                exportedFolderUri = profileFolderUri

                // Thread 1: Copy profile.json (ALWAYS exists)
                BackgroundTaskExecutor.getInstance().executeAsync({
                    copyFileToUri(profileFolderUri, Constants.PROFILE_FILE_NAME)
                    return@executeAsync 0
                }, object : BackgroundTaskExecutor.TaskCallback<Int> {
                    override fun onSuccess(result: Int?) {
                        synchronized(this@BlindSettingsActivity) {
                            exportFilesCompleted++
                        }
                    }

                    override fun onError(e: Exception?) {
                        Log.e(TAG, "Error copying profile.json", e)
                        synchronized(this@BlindSettingsActivity) {
                            exportErrorCode = 1
                            exportFilesCompleted++
                        }
                    }
                })

                // Thread 2: Copy hash_cache (only if exists)
                val hashCacheFile = FileUtils.getHashCacheFile(this)
                if (hashCacheFile.exists()) {
                    BackgroundTaskExecutor.getInstance().executeAsync({
                        copyFileToUri(profileFolderUri, Constants.HASH_CACHE_FILE_NAME)
                        return@executeAsync 0
                    }, object : BackgroundTaskExecutor.TaskCallback<Int> {
                        override fun onSuccess(result: Int?) {
                            synchronized(this@BlindSettingsActivity) {
                                exportFilesCompleted++
                            }
                        }

                        override fun onError(e: Exception?) {
                            Log.e(TAG, "Error copying hash_cache", e)
                            synchronized(this@BlindSettingsActivity) {
                                exportErrorCode = 1
                                exportFilesCompleted++
                            }
                        }
                    })
                } else {
                    //  File doesn't exist, just increment counter
                    synchronized(this) {
                        exportFilesCompleted++
                    }
                }

                // Thread 3: Copy env_reports (only if exists)
                val envReportsFile = FileUtils.getEnvReportsFile(this)
                if (envReportsFile.exists()) {
                    BackgroundTaskExecutor.getInstance().executeAsync({
                        copyFileToUri(profileFolderUri, Constants.ENV_REPORTS_FILE_NAME)
                        return@executeAsync 0
                    }, object : BackgroundTaskExecutor.TaskCallback<Int> {
                        override fun onSuccess(result: Int?) {
                            synchronized(this@BlindSettingsActivity) {
                                exportFilesCompleted++
                            }
                        }

                        override fun onError(e: Exception?) {
                            Log.e(TAG, "Error copying env_reports", e)
                            synchronized(this@BlindSettingsActivity) {
                                exportErrorCode = 1
                                exportFilesCompleted++
                            }
                        }
                    })
                } else {
                    //  File doesn't exist, just increment counter
                    synchronized(this) {
                        exportFilesCompleted++
                    }
                }

                // Wait for all copies to complete
                waitForExportCompletion()

            } catch (e: Exception) {
                Log.e(TAG, "Error creating profile folder", e)
                slideMessage.value = getProfileExportErrorMessage(this)
                showLoading.value = false

                soundManager.play(
                    SoundConstants.SETTINGS_APPLIED_ID, 0.7f, 0.7f
                ) {
                    ttsManager.speak(
                        slideMessage.value,
                        AppConfig.tts_pitch,
                        AppConfig.tts_speech_rate,
                        false,
                        null
                    )
                }
            }
        }
    }

    private fun waitForExportCompletion() {
        val checkCompletion = object : Runnable {
            override fun run() {
                synchronized(this@BlindSettingsActivity) {
                    if (exportFilesCompleted >= 3) {
                        if (exportErrorCode == 0) {
                            //  SUCCESS
                            slideMessage.value =
                                getProfileExportedMessage(this@BlindSettingsActivity)
                            showLoading.value = false

                            soundManager.play(
                                SoundConstants.SETTINGS_APPLIED_ID, 0.7f, 0.7f
                            ) {
                                ttsManager.speak(
                                    slideMessage.value,
                                    AppConfig.tts_pitch,
                                    AppConfig.tts_speech_rate,
                                    false,
                                    null
                                )
                            }
                        } else {
                            //  ERROR - Delete created folder
                            deleteExportedFolder()

                            slideMessage.value =
                                getProfileExportErrorMessage(this@BlindSettingsActivity)
                            showLoading.value = false

                            soundManager.play(
                                SoundConstants.SETTINGS_APPLIED_ID, 0.7f, 0.7f
                            ) {
                                ttsManager.speak(
                                    slideMessage.value,
                                    AppConfig.tts_pitch,
                                    AppConfig.tts_speech_rate,
                                    false,
                                    null
                                )
                            }
                        }
                    } else {
                        mainHandler.postDelayed(this, Constants.WAIT_CHECK)
                    }
                }
            }
        }
        mainHandler.post(checkCompletion)
    }

    private fun deleteExportedFolder() {
        exportedFolderUri?.let { folderUri ->
            try {
                //  Delete folder (fast operation, can be done on main thread)
                DocumentsContract.deleteDocument(contentResolver, folderUri)
                Log.d(TAG, "Deleted exported folder after error")
            } catch (e: Exception) {
                Log.e(TAG, "Error deleting exported folder", e)
                // Continue anyway - error already reported to user
            }
        }
    }

    private fun copyFileToUri(targetUri: Uri, fileName: String) {
        val sourceFile = File(FileUtils.getProfileDirectory(this), fileName)

        //  Throw exception instead of silent return
        if (!sourceFile.exists()) {
            throw FileNotFoundException("Source file not found: $fileName")
        }

        val resolver = contentResolver

        val newFileUri = DocumentsContract.createDocument(
            resolver, targetUri, "application/octet-stream", fileName
        )

        newFileUri?.let { uri ->
            resolver.openOutputStream(uri)?.use { outputStream ->
                sourceFile.inputStream().use { inputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        } ?: throw IOException("Failed to create document: $fileName")
    }

    private fun createProfileFolder(parentUri: Uri): Uri {
        val resolver = contentResolver
        val parentFolderUri = DocumentsContract.buildDocumentUriUsingTree(
            parentUri, DocumentsContract.getTreeDocumentId(parentUri)
        )

        // Create VisionAssistProfile folder
        val folderUri = DocumentsContract.createDocument(
            resolver,
            parentFolderUri,
            DocumentsContract.Document.MIME_TYPE_DIR,
            Constants.PROFILE_FOLDER_NAME
        )

        return folderUri ?: throw IOException("Failed to create profile folder")
    }

    private fun handleNavigateHome() {
        cancelAllHandlers()

        ttsManager.speak(
            if (PhoneStatusMonitor.getInstance().ttsManager.currentLanguage == "en") "Navigating to home screen..."
            else "Se intoarce la ecranul principal...",
            AppConfig.tts_pitch,
            AppConfig.tts_speech_rate,
            false,
            null
        )

        waitForTTSSpeech {
            val intent = Intent(this, BlindHomeActivity::class.java)
            intent.putExtra(Constants.EXTRA_HOME_OPTION, true)
            startActivity(intent)
            finish()
        }
    }

    private fun handleNavigateSettings() {
    }

    private fun handleNotifFirstButton() {
        cancelAllHandlers()
        showNotifDialog.value = false
    }

    private fun vibrateIfNeeded() {
        if (AppConfig.haptics) {
            vibrate(haptic_model0())
        }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_UP -> {
                Log.d(TAG, "Volume button up pressed")
                return true
            }

            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                if (repeatTTS) ttsManager.onVolumeDownPressed()
                Log.d(TAG, "Volume button down pressed")
                return true
            }
        }
        return super.onKeyDown(keyCode, event)
    }
}

@Composable
fun BlindSettingsScreen(
    isSpeakingApplyingSettings: Boolean,
    infoNotificationManagerValue: Boolean,
    showPasswordDialog: Boolean,
    showLoading: Boolean,
    loadingText: String,
    showNotification: Boolean,
    notificationMessage: String,
    currentSection: Int,
    selectedLanguage: Language,
    selectedQuickAction: Int,
    soAEnabled: Boolean,
    hasRemoteProfile: Boolean,
    onLanguageChange: (Language) -> Unit = {},
    onQuickActionChange: (Int) -> Unit = {},
    onQuickActionInfoClick: () -> Unit = {},
    onSoAToggle: (Boolean) -> Unit = {},
    onSoAInfoClick: () -> Unit = {},
    onChangeTTSParameters: () -> Unit = {},
    onClearCache: () -> Unit = {},
    onSyncProfile: () -> Unit = {},
    onLogout: () -> Unit = {},
    onDeleteAccount: () -> Unit = {},
    onExportProfile: () -> Unit = {},
    onNavigateHome: () -> Unit = {},
    onNavigateSettings: () -> Unit = {},
    onSectionChange: (Int) -> Unit = {},
    firstButtonClick: () -> Unit = {},
    onPasswordChange: (String) -> Unit,
    firstButtonClickPassword: () -> Unit,
    secondButtonClickPassword: () -> Unit,
    syncDays: Long,
    showErrorPassword: Boolean,
    passwordValue: String
) {
    val blockMainUI =
        showLoading || infoNotificationManagerValue || showNotification || showPasswordDialog || isSpeakingApplyingSettings
    val navBarHeight = WindowInsets.navigationBars.getBottom(LocalDensity.current)
    val navBarHeightDp = with(LocalDensity.current) { navBarHeight.toDp() }

    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenWidth = maxWidth
        val navbarHeight = 80.dp / maxHeight
        val sectionMain = 1.0f - navbarHeight
        val context = LocalContext.current

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
                                swipeStartX >= threshold -> {
                                    onNavigateHome()
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
            // Background image
            Image(
                painter = painterResource(R.drawable.app_background),
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )

            Box(
                modifier = Modifier
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
            ) {
                // Main content
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .fillMaxHeight(sectionMain)
                ) {
                    Spacer(modifier = Modifier.weight(0.12f))

                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .weight(0.34f)
                    ) {
                        BlindTopSettingsSection(
                            selectedLanguage = selectedLanguage,
                            selectedQuickAction = selectedQuickAction,
                            soAEnabled = soAEnabled,
                            onLanguageChange = onLanguageChange,
                            onQuickActionChange = onQuickActionChange,
                            onQuickActionInfoClick = onQuickActionInfoClick,
                            onSoAToggle = onSoAToggle,
                            onSoAInfoClick = onSoAInfoClick,
                            context = context
                        )
                    }

                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .weight(0.50f),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        BlindSlideableSections(
                            currentSection = currentSection,
                            onChangeDetectionColors = onChangeTTSParameters,
                            onChangeCaptionColors = onClearCache,
                            onSyncProfile = onSyncProfile,
                            onLogout = onLogout,
                            onDeleteAccount = onDeleteAccount,
                            screenWidth = screenWidth
                        )

                        if (hasRemoteProfile) {
                            Row(
                                modifier = Modifier.fillMaxSize(),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Box(
                                    modifier = Modifier
                                        .weight(0.3f)
                                        .padding(start = 24.dp)
                                        .semantics {
                                            if (currentSection > 0) contentDescription =
                                                getGeneralSectionTextTTS(context)
                                            else hideFromAccessibility()
                                        }
                                        .clickable {
                                            if (currentSection > 0)
                                                onSectionChange(currentSection - 1)
                                        },
                                    contentAlignment = Alignment.CenterStart,
                                ) {
                                    if (currentSection > 0) {
                                        IconButton(
                                            onClick = {},
                                            modifier = Modifier
                                                .size(Constants.STD_BUTTON_HEIGHT.dp / 2)
                                                .background(
                                                    colorResource(R.color.std_purple), CircleShape
                                                )
                                        ) {
                                            Icon(
                                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                                contentDescription = "",
                                                tint = Color.White,
                                                modifier = Modifier.size(Constants.STD_INFO_BUTTON_SIZE.dp)
                                            )
                                        }
                                    }
                                }

                                BlindSectionIndicators(
                                    currentSection = currentSection, screenWidth = screenWidth
                                )

                                Box(
                                    modifier = Modifier
                                        .weight(0.3f)
                                        .padding(end = 24.dp)
                                        .semantics {
                                            if (currentSection < 1) contentDescription =
                                                getAccountSectionTextTTS(context)
                                            else hideFromAccessibility()
                                        }
                                        .clickable {
                                            if (currentSection < 1)
                                                onSectionChange(currentSection + 1)
                                        },
                                    contentAlignment = Alignment.CenterEnd
                                ) {
                                    if (currentSection < 1) {
                                        IconButton(
                                            onClick = {},
                                            modifier = Modifier
                                                .size(Constants.STD_BUTTON_HEIGHT.dp / 2)
                                                .background(
                                                    colorResource(R.color.std_purple), CircleShape
                                                )
                                        ) {
                                            Icon(
                                                imageVector = Icons.AutoMirrored.Filled.ArrowForward,
                                                contentDescription = "",
                                                tint = Color.White,
                                                modifier = Modifier.size(Constants.STD_INFO_BUTTON_SIZE.dp)
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable {
                                onExportProfile()
                            }
                            .semantics {
                                contentDescription =
                                    if (AppConfig.mainLanguage.code == "en") "Export profile button"
                                    else "Buton de exportare a profilului"
                            }
                            .weight(0.2f), verticalArrangement = Arrangement.Bottom) {
                        // Export Profile Button
                        Button(
                            onClick = onExportProfile,
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(Constants.STD_BUTTON_HEIGHT.dp)
                                .padding(horizontal = 24.dp),
                            shape = RoundedCornerShape(32.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = colorResource(R.color.std_purple)
                            )
                        ) {
                            Text(
                                text = getExportProfileText(context),
                                fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                                color = Color.White,
                                fontFamily = robotoExtraBold
                            )
                        }

                        Spacer(modifier = Modifier.height(8.dp))

                        // Sync Status Section
                        SyncStatusSection(
                            0, syncDays.toInt(), false, {}, true
                        )
                    }
                }

                // Bottom Navigation Bar
                Box(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth()
                        .fillMaxHeight(navbarHeight)
                ) {
                    // Bottom Navigation
                    BottomNavigationBar(
                        onNavigateHome, {}, onNavigateSettings, false, 2
                    )
                }
            }

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(navBarHeightDp)  // Takes the height of status bar
                    .background(colorResource(R.color.std_light_purple))  // Your color!
                    .align(Alignment.BottomCenter)
            )

            BlockingOverlay(blockMainUI)

            // Loading overlay
            LoadingComponent(
                isVisible = showLoading, loadingText = loadingText, animSpec = Pair(
                    fadeIn(initialAlpha = 0f, animationSpec = tween(durationMillis = 0)),
                    fadeOut(targetAlpha = 0f, animationSpec = tween(durationMillis = 0))
                )
            )

            NotificationDialog(
                isVisible = showNotification,
                type = LoadProfileActivity.NotificationType.NO_INTERNET,
                message = notificationMessage,
                firstButtonClick = firstButtonClick,
            )

            PasswordConfirmationDialog(
                showDialog = showPasswordDialog,
                title = getPasswordTitle(context),
                passwordValue = passwordValue,
                onPasswordChange = onPasswordChange,
                firstButtonLabel = if (AppConfig.mainLanguage.code == "en") "Cancel"
                else context.getString(R.string.dont_button_ro),
                secondButtonLabel = if (AppConfig.mainLanguage.code == "en") context.getString(R.string.delete_button_en)
                else context.getString(R.string.delete_button_ro),
                onFirstButtonClick = firstButtonClickPassword,
                onSecondButtonClick = secondButtonClickPassword,
                showErrorPassword = showErrorPassword
            )
        }
    }
}

@Composable
fun BlindTopSettingsSection(
    selectedLanguage: Language,
    selectedQuickAction: Int,
    soAEnabled: Boolean,
    onLanguageChange: (Language) -> Unit,
    onQuickActionChange: (Int) -> Unit,
    onQuickActionInfoClick: () -> Unit,
    onSoAToggle: (Boolean) -> Unit,
    onSoAInfoClick: () -> Unit,
    context: Context
) {
    val listOfQuickAction = remember(selectedLanguage) {
        listOf(
            getQuickAccessType(context, 0),
            getQuickAccessType(context, 1),
            getQuickAccessType(context, 2),
            getQuickAccessType(context, 3)
        )
    }

    val expandedLanguage = remember { mutableStateOf(false) }
    val expandedQuickAction = remember { mutableStateOf(false) }

    // Left Column
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(start = 24.dp),
        horizontalAlignment = Alignment.Start
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable {
                    expandedLanguage.value = !expandedLanguage.value
                }
                .semantics {
                    contentDescription =
                        if (AppConfig.mainLanguage.code == "en") "Language change button"
                        else "Buton de schimbare a limbii"
                },
            horizontalArrangement = Arrangement.Start,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                // Language Label
                Text(
                    modifier = Modifier
                        .semantics {
                            //  Hide from TalkBack
                            hideFromAccessibility()
                        },
                    text = getLanguageText(LocalContext.current),
                    fontSize = 12.sp,
                    color = colorResource(R.color.std_cyan_dark),
                    fontFamily = robotoExtraBold
                )

                Spacer(modifier = Modifier.height(3.dp))

                // Language Selector
                LanguageSelector(
                    selectedLanguage = selectedLanguage,
                    availableLanguages = listOf(
                        Language("en", "English", "US"), Language("ro", "Română", "RO")
                    ),
                    onLanguageSelected = { language ->
                        expandedLanguage.value = !expandedLanguage.value
                        onLanguageChange(language)
                    },
                    manualClickExpanded = true,
                    manualExpanded = expandedLanguage
                )
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable {
                    expandedQuickAction.value = !expandedQuickAction.value
                }
                .semantics {
                    contentDescription =
                        if (AppConfig.mainLanguage.code == "en") "Quick action change button"
                        else "Buton de schimbare Quick action"
                },
            horizontalArrangement = Arrangement.Start,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                // Quick Action Label
                Text(
                    modifier = Modifier
                        .semantics {
                            //  Hide from TalkBack
                            hideFromAccessibility()
                        },
                    text = getQuickActionText(LocalContext.current),
                    fontSize = 12.sp,
                    color = colorResource(R.color.std_cyan_dark),
                    fontFamily = robotoExtraBold
                )

                Spacer(modifier = Modifier.height(3.dp))

                // Quick Action Row (Selector + Info Button)
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Quick Action Selector
                    QuickActionSelector(
                        selectedOption = getQuickAccessType(
                            LocalContext.current, selectedQuickAction
                        ),
                        availableOptions = listOfQuickAction,
                        onOptionSelected = { quickAction ->
                            expandedQuickAction.value = !expandedQuickAction.value
                            onQuickActionChange(quickAction)
                        },
                        manualClickExpanded = true,
                        manualExpanded = expandedQuickAction
                    )

                    Spacer(modifier = Modifier.width(5.dp))

                    // Info Button
                    IconButton(
                        onClick = onQuickActionInfoClick,
                        modifier = Modifier
                            .size(Constants.STD_INFO_BUTTON_SIZE.dp)
                            .semantics {
                                contentDescription =
                                    if (AppConfig.mainLanguage.code == "en")
                                        "Quick action info button"
                                    else
                                        "Buton Quick action info"
                            }
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Info,
                            contentDescription = "Info",
                            tint = colorResource(R.color.std_purple),
                            modifier = Modifier
                                .size(Constants.STD_INFO_BUTTON_SIZE.dp)
                                .semantics {
                                    //  Hide from TalkBack
                                    hideFromAccessibility()
                                }
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable {
                    onSoAToggle(!soAEnabled)
                }
                .semantics {
                    contentDescription =
                        if (AppConfig.mainLanguage.code == "en") "Speed over Accuracy button"
                        else "Buton Speed over Accuracy"
                },
            horizontalArrangement = Arrangement.spacedBy(30.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // SoA Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Start
            ) {
                Text(
                    modifier = Modifier
                        .semantics {
                            //  Hide from TalkBack
                            hideFromAccessibility()
                        },
                    text = getSoAText(LocalContext.current),
                    fontSize = Constants.STD_FONT_SIZE.sp,
                    color = colorResource(R.color.std_cyan_dark),
                    fontFamily = robotoExtraBold
                )

                Spacer(modifier = Modifier.width(9.dp))

                Row(
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Switch(
                        checked = soAEnabled,
                        onCheckedChange = onSoAToggle,
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = Color.White,
                            checkedTrackColor = colorResource(R.color.std_purple),
                            uncheckedThumbColor = Color.White,
                            uncheckedTrackColor = Color.Gray
                        )
                    )

                    IconButton(
                        onClick = onSoAInfoClick,
                        modifier = Modifier
                            .size(Constants.STD_INFO_BUTTON_SIZE.dp)
                            .semantics {
                                contentDescription =
                                    if (AppConfig.mainLanguage.code == "en")
                                        "Speed over accuracy info button"
                                    else
                                        "Speed over accuracy info"
                            }
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Info,
                            contentDescription = "Info",
                            tint = colorResource(R.color.std_purple),
                            modifier = Modifier
                                .size(Constants.STD_INFO_BUTTON_SIZE.dp)
                                .semantics {
                                    //  Hide from TalkBack
                                    hideFromAccessibility()
                                }
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun BlindSectionIndicators(
    currentSection: Int, screenWidth: Dp
) {
    Row(
        modifier = Modifier
            .width(screenWidth * 0.17f)
            .height(Constants.STD_BUTTON_HEIGHT.dp / 2)
            .clip(RoundedCornerShape(32.dp))
            .background(Color(0x66808080)),
        horizontalArrangement = Arrangement.SpaceEvenly,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Appearance indicator
        Box(
            modifier = Modifier
                .size(7.dp)
                .clip(CircleShape)
                .background(if (currentSection == 0) Color.White else Color(0xB3808080))
        )

        Box(
            modifier = Modifier
                .size(7.dp)
                .clip(CircleShape)
                .background(if (currentSection == 1) Color.White else Color(0xB3808080))
        )
    }
}

@Composable
fun BlindSlideableSections(
    currentSection: Int,
    onChangeDetectionColors: () -> Unit,
    onChangeCaptionColors: () -> Unit,
    onSyncProfile: () -> Unit,
    onLogout: () -> Unit,
    onDeleteAccount: () -> Unit,
    screenWidth: Dp
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .fillMaxHeight(0.80f)
    ) {
        AnimatedVisibility(
            visible = currentSection == 0,
            enter = slideInHorizontally(
                initialOffsetX = { -it },
                animationSpec = tween(Constants.ANIMATION_DELAY)
            ),
            exit = slideOutHorizontally(
                targetOffsetX = { -it },
                animationSpec = tween(Constants.ANIMATION_DELAY)
            )
        ) {
            BlindAppearanceSection(
                onChangeDetectionColors = onChangeDetectionColors,
                onChangeCaptionColors = onChangeCaptionColors,
                screenWidth = screenWidth
            )
        }

        AnimatedVisibility(
            visible = currentSection == 1,
            enter = slideInHorizontally(
                initialOffsetX = { it },
                animationSpec = tween(Constants.ANIMATION_DELAY)
            ),
            exit = slideOutHorizontally(
                targetOffsetX = { it },
                animationSpec = tween(Constants.ANIMATION_DELAY)
            )
        ) {
            BlindAccountSection(
                onSyncProfile = onSyncProfile,
                onLogout = onLogout,
                onDeleteAccount = onDeleteAccount,
                screenWidth = screenWidth
            )
        }
    }
}

@Composable
fun BlindAppearanceSection(
    onChangeDetectionColors: () -> Unit, onChangeCaptionColors: () -> Unit, screenWidth: Dp
) {
    Column(
        modifier = Modifier
            .fillMaxHeight()
            .fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.SpaceEvenly
    ) {
        // Title
        Text(
            text = getGeneralSectionText(LocalContext.current),
            fontSize = Constants.STD_SUBTITLE_SIZE.sp,
            color = colorResource(R.color.std_cyan_dark),
            fontFamily = robotoExtraBold,
            textAlign = TextAlign.Center
        )

        // Options
        Column(
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        onChangeDetectionColors()
                    }
                    .semantics {
                        contentDescription =
                            if (AppConfig.mainLanguage.code == "en") "TTS parameters change button"
                            else "Buton de schimbare a parametrilor TTS"
                    },
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Button(
                    onClick = onChangeDetectionColors,
                    modifier = Modifier
                        .width(screenWidth * 0.6f)
                        .height(Constants.STD_BUTTON_HEIGHT.dp),
                    shape = RoundedCornerShape(32.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colorResource(R.color.std_cyan)
                    )
                ) {
                    Text(
                        text = getChangeTTSParametersText(LocalContext.current),
                        fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                        color = Color.White,
                        fontFamily = robotoExtraBold,
                        textAlign = TextAlign.Center
                    )
                }
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        onChangeCaptionColors()
                    }
                    .semantics {
                        contentDescription =
                            if (AppConfig.mainLanguage.code == "en") "Cache clear button"
                            else "Buton de ștergere a cache-ului"
                    },
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Button(
                    onClick = onChangeCaptionColors,
                    modifier = Modifier
                        .width(screenWidth * 0.6f)
                        .height(Constants.STD_BUTTON_HEIGHT.dp),
                    shape = RoundedCornerShape(32.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colorResource(R.color.std_cyan)
                    )
                ) {
                    Text(
                        text = getClearCacheText(LocalContext.current),
                        fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                        color = Color.White,
                        fontFamily = robotoExtraBold,
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
    }
}

@Composable
fun BlindAccountSection(
    onSyncProfile: () -> Unit, onLogout: () -> Unit, onDeleteAccount: () -> Unit, screenWidth: Dp
) {
    Column(
        modifier = Modifier
            .fillMaxHeight()
            .fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.SpaceEvenly
    ) {
        // Title
        Text(
            text = getAccountSectionText(LocalContext.current),
            fontSize = Constants.STD_SUBTITLE_SIZE.sp,
            color = colorResource(R.color.std_cyan_dark),
            fontFamily = robotoExtraBold,
            textAlign = TextAlign.Center
        )

        // Options
        Column(
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        onSyncProfile()
                    }
                    .semantics {
                        contentDescription =
                            if (AppConfig.mainLanguage.code == "en") "Sync profile button"
                            else "Buton de sincronizare a profilului"
                    },
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Button(
                    onClick = onSyncProfile,
                    modifier = Modifier
                        .width(screenWidth * 0.6f)
                        .height(Constants.STD_BUTTON_HEIGHT.dp),
                    shape = RoundedCornerShape(32.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colorResource(R.color.std_cyan)
                    )
                ) {
                    Text(
                        text = getSyncProfileText(LocalContext.current),
                        fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                        color = Color.White,
                        fontFamily = robotoExtraBold,
                        textAlign = TextAlign.Center
                    )
                }
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        onLogout()
                    }
                    .semantics {
                        contentDescription =
                            if (AppConfig.mainLanguage.code == "en") "Log out button"
                            else "Buton de deconectare"
                    },
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Button(
                    onClick = onLogout,
                    modifier = Modifier
                        .width(screenWidth * 0.6f)
                        .height(Constants.STD_BUTTON_HEIGHT.dp),
                    shape = RoundedCornerShape(32.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colorResource(R.color.std_cyan)
                    )
                ) {
                    Text(
                        text = getLogOutText(LocalContext.current),
                        fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                        color = Color.White,
                        fontFamily = robotoExtraBold,
                        textAlign = TextAlign.Center
                    )
                }
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        onDeleteAccount()
                    }
                    .semantics {
                        contentDescription =
                            if (AppConfig.mainLanguage.code == "en") "Delete account button"
                            else "Buton de ștergere a contului"
                    },
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Button(
                    onClick = onDeleteAccount,
                    modifier = Modifier
                        .width(screenWidth * 0.6f)
                        .height(Constants.STD_BUTTON_HEIGHT.dp),
                    shape = RoundedCornerShape(32.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color.Red
                    )
                ) {
                    Text(
                        text = getDeleteAccountText(LocalContext.current),
                        fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                        color = Color.White,
                        fontFamily = robotoExtraBold,
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
    }
}

@Composable
fun BlockingOverlay(isVisible: Boolean) {
    if (isVisible) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(Unit) {
                    awaitPointerEventScope {
                        while (true) {
                            val event = awaitPointerEvent()
                            event.changes.forEach { it.consume() }
                        }
                    }
                }
        )
    }
}

@Preview(
    name = "Blind Settings Screen - First Section",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
@Composable
fun BlindSettingsScreenPreview1() {
    BlindSettingsScreen(
        isSpeakingApplyingSettings = false,
        showLoading = false,
        loadingText = "",
        showNotification = false,
        notificationMessage = "",
        currentSection = 0,
        selectedLanguage = Language("en", "English", "US"),
        selectedQuickAction = 0,
        soAEnabled = true,
        hasRemoteProfile = true,
        onLanguageChange = {},
        onQuickActionChange = {},
        onQuickActionInfoClick = {},
        onSoAToggle = {},
        onSoAInfoClick = {},
        onChangeTTSParameters = {},
        onClearCache = {},
        onSyncProfile = {},
        onLogout = {},
        onDeleteAccount = {},
        onExportProfile = {},
        onNavigateHome = {},
        onNavigateSettings = {},
        onSectionChange = {},
        firstButtonClick = {},
        syncDays = 3,
        firstButtonClickPassword = {},
        secondButtonClickPassword = {},
        onPasswordChange = {},
        showPasswordDialog = false,
        showErrorPassword = false,
        passwordValue = "",
        infoNotificationManagerValue = false
    )
}

@Preview(
    name = "Blind Settings Screen - Second Section",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
@Composable
fun BlindSettingsScreenPreview2() {
    BlindSettingsScreen(
        isSpeakingApplyingSettings = false,
        showLoading = false,
        loadingText = "",
        showNotification = false,
        notificationMessage = "",
        currentSection = 1,
        selectedLanguage = Language("en", "English", "US"),
        selectedQuickAction = 0,
        soAEnabled = true,
        hasRemoteProfile = true,
        onLanguageChange = {},
        onQuickActionChange = {},
        onQuickActionInfoClick = {},
        onSoAToggle = {},
        onSoAInfoClick = {},
        onChangeTTSParameters = {},
        onClearCache = {},
        onSyncProfile = {},
        onLogout = {},
        onDeleteAccount = {},
        onExportProfile = {},
        onNavigateHome = {},
        onNavigateSettings = {},
        onSectionChange = {},
        firstButtonClick = {},
        syncDays = 3,
        firstButtonClickPassword = {},
        secondButtonClickPassword = {},
        onPasswordChange = {},
        showPasswordDialog = false,
        showErrorPassword = false,
        passwordValue = "",
        infoNotificationManagerValue = false
    )
}
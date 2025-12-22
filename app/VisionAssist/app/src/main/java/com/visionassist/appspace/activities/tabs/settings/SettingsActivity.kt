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
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.BaseActivity
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.main.BottomNavigationBar
import com.visionassist.appspace.activities.main.HomeActivity
import com.visionassist.appspace.activities.main.MainActivity
import com.visionassist.appspace.activities.main.SyncStatusSection
import com.visionassist.appspace.activities.newprofile.LoadProfileActivity
import com.visionassist.appspace.activities.newprofile.UserAccessibility1Activity
import com.visionassist.appspace.activities.newprofile.jsonCollection.ProfileFileCollection.writeSoA
import com.visionassist.appspace.activities.newprofile.jsonCollection.ProfileFileCollection.writeUserAccessibility1Caption
import com.visionassist.appspace.activities.newprofile.jsonCollection.ProfileFileCollection.writeWelcomeActivity
import com.visionassist.appspace.activities.tabs.home.detection.SceneClassifiedNotification
import com.visionassist.appspace.activities.tabs.reports.EnvironmentReportsActivity
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
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.BackgroundTaskExecutor
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.FileUtils
import com.visionassist.appspace.utils.Language
import com.visionassist.appspace.utils.calculateHashCacheSize
import com.visionassist.appspace.utils.formatPercentage
import com.visionassist.appspace.utils.formatSizeKB
import com.visionassist.appspace.utils.getAccountSectionText
import com.visionassist.appspace.utils.getAppearanceSectionText
import com.visionassist.appspace.utils.getApplyingSettingsText
import com.visionassist.appspace.utils.getCacheClearedMessage
import com.visionassist.appspace.utils.getChangeCaptionColorsText
import com.visionassist.appspace.utils.getChangeDetectionColorsText
import com.visionassist.appspace.utils.getClearCacheConfirmMessage
import com.visionassist.appspace.utils.getClearCacheText
import com.visionassist.appspace.utils.getClearReportsConfirmMessage
import com.visionassist.appspace.utils.getClearReportsText
import com.visionassist.appspace.utils.getCurrentEnvReportsSize
import com.visionassist.appspace.utils.getCurrentHashCacheSize
import com.visionassist.appspace.utils.getDeleteAccountConfirmMessage
import com.visionassist.appspace.utils.getDeleteAccountText
import com.visionassist.appspace.utils.getDeletingAccountText
import com.visionassist.appspace.utils.getExportProfileText
import com.visionassist.appspace.utils.getHapticsText
import com.visionassist.appspace.utils.getLanguageText
import com.visionassist.appspace.utils.getLogOutText
import com.visionassist.appspace.utils.getLoggingOffText
import com.visionassist.appspace.utils.getLogoutConfirmMessage
import com.visionassist.appspace.utils.getPasswordTitle
import com.visionassist.appspace.utils.getProfileExportErrorMessage
import com.visionassist.appspace.utils.getProfileExportedMessage
import com.visionassist.appspace.utils.getProfileExportedMessageTutorial
import com.visionassist.appspace.utils.getProfileSyncErrorMessage
import com.visionassist.appspace.utils.getProfileSyncedMessage
import com.visionassist.appspace.utils.getQuickAccessType
import com.visionassist.appspace.utils.getQuickActionDisabledMessage
import com.visionassist.appspace.utils.getQuickActionEnabledMessage
import com.visionassist.appspace.utils.getQuickActionInfoMessage
import com.visionassist.appspace.utils.getQuickActionText
import com.visionassist.appspace.utils.getReportsClearedMessage
import com.visionassist.appspace.utils.getSoAInfoMessage
import com.visionassist.appspace.utils.getSoAText
import com.visionassist.appspace.utils.getStorageSectionText
import com.visionassist.appspace.utils.getSyncProfile
import com.visionassist.appspace.utils.getSyncProfileText
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.load_genericErrorDelete
import com.visionassist.appspace.utils.load_genericErrorLogout
import com.visionassist.appspace.utils.load_noInternet
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.robotoLight
import com.visionassist.appspace.utils.vibrate
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.io.File
import java.io.FileNotFoundException
import java.io.IOException

class SettingsActivity : BaseActivity() {
    private val TAG = "SettingsActivity"

    private val mainHandler = Handler(Looper.getMainLooper())
    private val ttsHandler = Handler(Looper.getMainLooper())

    // Managers
    private val ttsManager: TTSManager = PhoneStatusMonitor.getInstance().ttsManager
    private val dbManager: DBManager = PhoneStatusMonitor.getInstance().dbManager
    private lateinit var infoNotificationManager: InfoNotificationManager

    // UI States
    private val showErrorPassword = mutableStateOf(false)
    private val showPasswordDialog = mutableStateOf(false)
    private val passwordValue = mutableStateOf("")
    private val showLoading = mutableStateOf(false)
    private val loadingText = mutableStateOf("")
    private val showNotifDialog = mutableStateOf(false)
    private val notifMessage = mutableStateOf("")
    private val showSlideNotif = mutableStateOf(false)
    private val slideMessage = mutableStateOf("")
    private val currentSection = mutableStateOf(0)

    // Size displays
    private val hashCacheSize = mutableStateOf("")
    private val hashCachePercent = mutableStateOf("")
    private val envReportsSize = mutableStateOf("")

    // Settings states
    private val selectedLanguage = mutableStateOf(AppConfig.mainLanguage)
    private val selectedQuickAction = mutableStateOf(0)
    private val hapticsEnabled = mutableStateOf(AppConfig.haptics)
    private val soAEnabled = mutableStateOf(AppConfig.SoA)

    // Flags
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

        selectedQuickAction.value = getCurrentQuickActionIndex(this)

        // Calculate initial sizes
        calculateSizes()

        // Setup info notification manager
        infoNotificationManager = InfoNotificationManager(this)

        val dbManager = PhoneStatusMonitor.getInstance().dbManager

        setContent {
            SettingsScreen(
                showPasswordDialog = showPasswordDialog.value,
                showLoading = showLoading.value,
                loadingText = loadingText.value,
                showNotification = showNotifDialog.value,
                notificationMessage = notifMessage.value,
                showSlideNotif = showSlideNotif.value,
                slideMessage = slideMessage.value,
                currentSection = currentSection.value,
                selectedLanguage = selectedLanguage.value,
                selectedQuickAction = selectedQuickAction.value,
                hapticsEnabled = hapticsEnabled.value,
                soAEnabled = soAEnabled.value,
                hashCacheSize = hashCacheSize.value,
                hashCachePercent = hashCachePercent.value,
                envReportsSize = envReportsSize.value,
                hasRemoteProfile = dbManager.statusOverview == 1 || dbManager.statusOverview == 2,
                onLanguageChange = ::handleLanguageChange,
                onQuickActionChange = ::handleQuickActionChange,
                onQuickActionInfoClick = ::handleQuickActionInfo,
                onHapticsToggle = ::handleHapticsToggle,
                onSoAToggle = ::handleSoAToggle,
                onSoAInfoClick = ::handleSoAInfo,
                onChangeDetectionColors = ::handleChangeDetectionColors,
                onChangeCaptionColors = ::handleChangeCaptionColors,
                onClearCache = ::handleClearCache,
                onClearReports = ::handleClearReports,
                onSyncProfile = ::handleSyncProfile,
                onLogout = ::handleLogout,
                onDeleteAccount = ::handleDeleteAccount,
                onExportProfile = ::handleExportProfile,
                onNavigateHome = ::handleNavigateHome,
                onNavigateReports = ::handleNavigateReports,
                onNavigateSettings = ::handleNavigateSettings,
                onSectionChange = { section -> currentSection.value = section },
                firstButtonClick = ::handleNotifFirstButton,
                syncStatus = dbManager.statusOverview,
                syncDays = dbManager.diffDays,
                onPasswordChange = { password ->
                    showErrorPassword.value = false
                    passwordValue.value = password
                },
                firstButtonClickPassword = ::handleBackPassword,
                secondButtonClickPassword = ::handleNextPassword,
                showErrorPassword = showErrorPassword.value,
                passwordValue = passwordValue.value
            )
        }
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
        showPasswordDialog.value = false
    }

    private fun handleNextPassword() {
        if (!NetworkUtils.isNetworkConnected(this)) {
            showPasswordDialog.value = false
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
                if (!result) showErrorPassword.value = true
                else {
                    showPasswordDialog.value = false
                    executeDeleteAccount(email)
                }
            }

            override fun onError(e: Exception) {
                slideMessage.value = load_genericErrorDelete(this@SettingsActivity)
                showPasswordDialog.value = false
                showSlideNotif.value = true

                mainHandler.postDelayed({
                    showSlideNotif.value = false
                }, 4500)
            }
        })
    }

    private fun handleLanguageChange(language: Language) {
        vibrateIfNeeded()

        if (language.code == selectedLanguage.value.code) return

        loadingText.value = getApplyingSettingsText(this)
        showLoading.value = true

        mainHandler.postDelayed({
            selectedLanguage.value = language
            AppConfig.mainLanguage = language
            writeWelcomeActivity(false, AppConfig.mainLanguage, false)
            setTTSLanguage()
        }, 1000)
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
                    //slideMessage.value = getLanguageChangedMessage(this@SettingsActivity)

                    showLoading.value = false
                    val intent = Intent(this@SettingsActivity, SettingsActivity::class.java)
                    startActivity(intent)
                    finish()

                    // Show notification
                    //showSlideNotif.value = true

                    // Hide after delay
                    //mainHandler.postDelayed({
                    //    showSlideNotif.value = false
                    // }, 4500)
                } else {
                    Log.w(TAG, "TTS not ready, retrying...")
                    ttsHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS.toLong())
                }
            }
        }
        ttsHandler.post(checkTTS)
    }

    private fun handleQuickActionChange(action: Int) {
        vibrateIfNeeded()

        if (action == selectedQuickAction.value) return

        loadingText.value = getApplyingSettingsText(this)
        showLoading.value = true

        mainHandler.postDelayed({
            selectedQuickAction.value = action

            if (action == 0) {
                LockScreenService.stopService(this)
                slideMessage.value = getQuickActionDisabledMessage(this)
            } else {
                LockScreenService.startService(this, action)
                slideMessage.value = getQuickActionEnabledMessage(this)
            }

            showLoading.value = false
            showSlideNotif.value = true

            mainHandler.postDelayed({
                showSlideNotif.value = false
            }, 4500)
        }, 1000)
    }

    private fun handleQuickActionInfo() {
        vibrateIfNeeded()
        infoNotificationManager.showNotification(
            getQuickActionInfoMessage(this),
            { vibrateIfNeeded(); infoNotificationManager.hideNotification() },
            "OK"
        )
    }

    private fun handleHapticsToggle(enabled: Boolean) {
        if (!AppConfig.haptics) {
            vibrate(haptic_model0())
        }
        hapticsEnabled.value = enabled
        AppConfig.haptics = enabled
        writeUserAccessibility1Caption(
            AppConfig.caption_color, AppConfig.caption_bck_color, AppConfig.haptics
        )
    }

    private fun handleSoAToggle(enabled: Boolean) {
        Log.d(TAG, "SoA toggle: $enabled")
        vibrateIfNeeded()
        soAEnabled.value = enabled
        AppConfig.SoA = enabled
        writeSoA(AppConfig.SoA)
    }

    private fun handleSoAInfo() {
        vibrateIfNeeded()
        infoNotificationManager.showNotification(
            getSoAInfoMessage(this),
            { vibrateIfNeeded(); infoNotificationManager.hideNotification() },
            "OK"
        )
    }

    private fun handleChangeDetectionColors() {
        vibrateIfNeeded()
        val intent = Intent(this, UserAccessibility1Activity::class.java).apply {
            putExtra(Constants.EXTRA_USERACC_OPTION, 1)
            putExtra(Constants.EXTRA_USERACC_OPTION2, true)
        }
        startActivity(intent)
    }

    private fun handleChangeCaptionColors() {
        vibrateIfNeeded()
        val intent = Intent(this, UserAccessibility1Activity::class.java).apply {
            putExtra(Constants.EXTRA_USERACC_OPTION, 2)
            putExtra(Constants.EXTRA_USERACC_OPTION2, true)
        }
        startActivity(intent)
    }

    private fun handleClearCache() {
        vibrateIfNeeded()

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
            { vibrateIfNeeded(); infoNotificationManager.hideNotification() },
            { vibrateIfNeeded(); infoNotificationManager.hideNotification(); executeClearCache() })
    }

    private fun executeClearCache() {
        loadingText.value = getApplyingSettingsText(this)
        showLoading.value = true

        mainHandler.postDelayed({
            try {
                FileUtils.createProfileDirFile(Constants.HASH_CACHE_FILE_NAME)
                calculateSizes()
                slideMessage.value = getCacheClearedMessage(this)
            } catch (e: Exception) {
                Log.e(TAG, "Error clearing cache", e)
                slideMessage.value = if (AppConfig.mainLanguage.code == "en") "Error clearing cache"
                else "Eroare la ștergerea cache-ului"
            }

            showLoading.value = false
            showSlideNotif.value = true

            mainHandler.postDelayed({
                showSlideNotif.value = false
            }, 4500)
        }, 1000)
    }

    private fun handleClearReports() {
        vibrateIfNeeded()

        infoNotificationManager.showNotificationTwoButtons(
            getClearReportsConfirmMessage(this),
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
            { vibrateIfNeeded(); infoNotificationManager.hideNotification() },
            { vibrateIfNeeded(); infoNotificationManager.hideNotification(); executeClearReports() })
    }

    private fun executeClearReports() {
        loadingText.value = getApplyingSettingsText(this)
        showLoading.value = true

        mainHandler.postDelayed({
            try {
                FileUtils.createProfileDirFile(Constants.ENV_REPORTS_FILE_NAME)
                calculateSizes()
                slideMessage.value = getReportsClearedMessage(this)
            } catch (e: Exception) {
                Log.e(TAG, "Error clearing cache", e)
                slideMessage.value = if (AppConfig.mainLanguage.code == "en") "Error clearing cache"
                else "Eroare la ștergerea cache-ului"
            }

            showLoading.value = false
            showSlideNotif.value = true

            mainHandler.postDelayed({
                showSlideNotif.value = false
            }, 4500)
        }, 1000)
    }

    private fun handleSyncProfile() {
        vibrateIfNeeded()

        if (!NetworkUtils.isNetworkConnected(this)) {
            notifMessage.value = load_noInternet(this)
            showNotifDialog.value = true
            return
        }

        loadingText.value = getSyncProfile(this)
        showLoading.value = true

        mainHandler.postDelayed({
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
                        DBConstants.SYNC_ERROR -> getProfileSyncErrorMessage(this@SettingsActivity)
                        DBConstants.SYNC_OK -> getProfileSyncedMessage(this@SettingsActivity)
                        else -> getProfileSyncedMessage(this@SettingsActivity)
                    }
                    showLoading.value = false
                    showSlideNotif.value = true

                    mainHandler.postDelayed({
                        showSlideNotif.value = false
                    }, 4500)
                }

                override fun onError(e: Exception?) {
                    mainHandler.post {
                        slideMessage.value = getProfileSyncErrorMessage(this@SettingsActivity)
                        showLoading.value = false
                        showSlideNotif.value = true

                        mainHandler.postDelayed({
                            showSlideNotif.value = false
                        }, 4500)
                    }
                }
            })
        }, 1000)
    }

    private fun handleLogout() {
        vibrateIfNeeded()

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
            { vibrateIfNeeded(); infoNotificationManager.hideNotification() },
            { vibrateIfNeeded(); infoNotificationManager.hideNotification(); executeLogout() })
    }

    private fun executeLogout() {
        loadingText.value = getLoggingOffText(this)
        showLoading.value = true

        mainHandler.postDelayed({
            try {
                // Delete all local files
                if (!FileUtils.deleteProfileDirFile(Constants.ENV_REPORTS_FILE_NAME)) throw Exception()

                if (!FileUtils.deleteProfileDirFile(Constants.HASH_CACHE_FILE_NAME)) throw Exception()

                if (!FileUtils.deleteProfileDirFile(Constants.PROFILE_FILE_NAME)) throw Exception()

                showLoading.value = false

                // Navigate to configuration
                val intent = Intent(this, MainActivity::class.java)
                startActivity(intent)
                finish()
            } catch (e: Exception) {
                Log.e(TAG, "Error during logout", e)
                slideMessage.value = load_genericErrorLogout(this@SettingsActivity)
                showLoading.value = false
                showSlideNotif.value = true

                mainHandler.postDelayed({
                    showSlideNotif.value = false
                    mainHandler.postDelayed({
                        val intent = Intent(this, MainActivity::class.java)
                        startActivity(intent)
                        finish()
                    }, 1000)
                }, 4500)
            }
        }, 1000)
    }

    private fun handleDeleteAccount() {
        vibrateIfNeeded()

        if (!NetworkUtils.isNetworkConnected(this)) {
            notifMessage.value = load_noInternet(this)
            showNotifDialog.value = true
            return
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
            { vibrateIfNeeded(); infoNotificationManager.hideNotification() },
            {
                vibrateIfNeeded()
                infoNotificationManager.hideNotification()
                showPasswordDialog.value = true
            })
    }

    private fun executeDeleteAccount(email: String) {
        loadingText.value = getDeletingAccountText(this)
        showLoading.value = true

        mainHandler.postDelayed({
            BackgroundTaskExecutor.getInstance().executeAsync({
                // Delete from Firebase
                dbManager.deleteAccount(email)

                // Delete local files first
                if (!FileUtils.deleteProfileDirFile(Constants.ENV_REPORTS_FILE_NAME)) return@executeAsync -1

                if (!FileUtils.deleteProfileDirFile(Constants.HASH_CACHE_FILE_NAME)) return@executeAsync -1

                if (!FileUtils.deleteProfileDirFile(Constants.PROFILE_FILE_NAME)) return@executeAsync -1

                return@executeAsync dbManager.status
            }, object : BackgroundTaskExecutor.TaskCallback<Int> {
                override fun onSuccess(result: Int) {
                    if (result == -1) {
                        Log.e(TAG, "Failed to delete local profile files")
                        slideMessage.value = load_genericErrorDelete(this@SettingsActivity)
                        showLoading.value = false
                        showSlideNotif.value = true

                        mainHandler.postDelayed({
                            showSlideNotif.value = false
                            mainHandler.postDelayed({
                                val intent = Intent(this@SettingsActivity, MainActivity::class.java)
                                startActivity(intent)
                                finish()
                            }, 1000)
                        }, 4500)
                    } else {
                        showLoading.value = false
                        val intent = Intent(this@SettingsActivity, MainActivity::class.java)
                        startActivity(intent)
                        finish()
                    }
                }

                override fun onError(e: Exception?) {
                    slideMessage.value = load_genericErrorDelete(this@SettingsActivity)
                    showLoading.value = false
                    showSlideNotif.value = true

                    mainHandler.postDelayed({
                        showSlideNotif.value = false
                        mainHandler.postDelayed({
                            val intent = Intent(this@SettingsActivity, MainActivity::class.java)
                            startActivity(intent)
                            finish()
                        }, 1000)
                    }, 4500)
                }
            })
        }, 1000)
    }

    private fun handleExportProfile() {
        vibrateIfNeeded()

        infoNotificationManager.showNotification(
            getProfileExportedMessageTutorial(this),
            { vibrateIfNeeded(); infoNotificationManager.hideNotification(); exportProfileLauncher.launch(null) },
            if (AppConfig.mainLanguage.code == "en") "Files"
            else "Fișiere"
        )
    }

    private fun handleExportProfileResult(uri: Uri) {
        loadingText.value = if (AppConfig.mainLanguage.code == "en") {
            "Exporting profile..."
        } else {
            "Se exportă profilul..."
        }
        showLoading.value = true

        mainHandler.postDelayed({
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
                        synchronized(this@SettingsActivity) {
                            exportFilesCompleted++
                        }
                    }

                    override fun onError(e: Exception?) {
                        Log.e(TAG, "Error copying profile.json", e)
                        synchronized(this@SettingsActivity) {
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
                            synchronized(this@SettingsActivity) {
                                exportFilesCompleted++
                            }
                        }

                        override fun onError(e: Exception?) {
                            Log.e(TAG, "Error copying hash_cache", e)
                            synchronized(this@SettingsActivity) {
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
                            synchronized(this@SettingsActivity) {
                                exportFilesCompleted++
                            }
                        }

                        override fun onError(e: Exception?) {
                            Log.e(TAG, "Error copying env_reports", e)
                            synchronized(this@SettingsActivity) {
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
                showLoading.value = false
                slideMessage.value = getProfileExportErrorMessage(this)
                showSlideNotif.value = true

                mainHandler.postDelayed({
                    showSlideNotif.value = false
                }, 4500)
            }
        }, 1000)
    }

    private fun waitForExportCompletion() {
        val checkCompletion = object : Runnable {
            override fun run() {
                synchronized(this@SettingsActivity) {
                    if (exportFilesCompleted >= 3) {
                        if (exportErrorCode == 0) {
                            //  SUCCESS
                            slideMessage.value = getProfileExportedMessage(this@SettingsActivity)
                            showLoading.value = false

                            showSlideNotif.value = true

                            mainHandler.postDelayed({
                                showSlideNotif.value = false
                            }, 4500)

                        } else {
                            //  ERROR - Delete created folder
                            deleteExportedFolder()

                            slideMessage.value = getProfileExportErrorMessage(this@SettingsActivity)
                            showLoading.value = false

                            showSlideNotif.value = true

                            mainHandler.postDelayed({
                                showSlideNotif.value = false
                            }, 4500)
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
        vibrateIfNeeded()
        val intent = Intent(this, HomeActivity::class.java)
        startActivity(intent)
        finish()
    }

    private fun handleNavigateReports() {
        vibrateIfNeeded()
        val intent = Intent(this, EnvironmentReportsActivity::class.java)
        startActivity(intent)
        finish()
    }

    private fun handleNavigateSettings() {
        vibrateIfNeeded()
        // Already in settings, do nothing
    }

    private fun handleNotifFirstButton() {
        vibrateIfNeeded()
        showNotifDialog.value = false
    }

    private fun vibrateIfNeeded() {
        if (AppConfig.haptics) {
            vibrate(haptic_model0())
        }
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
}

@Composable
fun SettingsScreen(
    showPasswordDialog: Boolean,
    showLoading: Boolean,
    loadingText: String,
    showNotification: Boolean,
    notificationMessage: String,
    showSlideNotif: Boolean,
    slideMessage: String,
    currentSection: Int,
    selectedLanguage: Language,
    selectedQuickAction: Int,
    hapticsEnabled: Boolean,
    soAEnabled: Boolean,
    hashCacheSize: String,
    hashCachePercent: String,
    envReportsSize: String,
    hasRemoteProfile: Boolean,
    onLanguageChange: (Language) -> Unit = {},
    onQuickActionChange: (Int) -> Unit = {},
    onQuickActionInfoClick: () -> Unit = {},
    onHapticsToggle: (Boolean) -> Unit = {},
    onSoAToggle: (Boolean) -> Unit = {},
    onSoAInfoClick: () -> Unit = {},
    onChangeDetectionColors: () -> Unit = {},
    onChangeCaptionColors: () -> Unit = {},
    onClearCache: () -> Unit = {},
    onClearReports: () -> Unit = {},
    onSyncProfile: () -> Unit = {},
    onLogout: () -> Unit = {},
    onDeleteAccount: () -> Unit = {},
    onExportProfile: () -> Unit = {},
    onNavigateHome: () -> Unit = {},
    onNavigateReports: () -> Unit = {},
    onNavigateSettings: () -> Unit = {},
    onSectionChange: (Int) -> Unit = {},
    firstButtonClick: () -> Unit = {},
    onPasswordChange: (String) -> Unit,
    firstButtonClickPassword: () -> Unit,
    secondButtonClickPassword: () -> Unit,
    syncStatus: Int,
    syncDays: Long,
    showErrorPassword: Boolean,
    passwordValue: String
) {
    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenWidth = maxWidth
        val navbarHeight = 90.dp / maxHeight
        val sectionMain = 1.0f - navbarHeight
        val context = LocalContext.current

        // Background image
        Image(
            painter = painterResource(R.drawable.app_background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        Box(
            modifier = Modifier.fillMaxSize()
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
                    TopSettingsSection(
                        selectedLanguage = selectedLanguage,
                        selectedQuickAction = selectedQuickAction,
                        hapticsEnabled = hapticsEnabled,
                        soAEnabled = soAEnabled,
                        onLanguageChange = onLanguageChange,
                        onQuickActionChange = onQuickActionChange,
                        onQuickActionInfoClick = onQuickActionInfoClick,
                        onHapticsToggle = onHapticsToggle,
                        onSoAToggle = onSoAToggle,
                        onSoAInfoClick = onSoAInfoClick,
                        context = context
                    )
                }

                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(0.45f),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    SlideableSections(
                        currentSection = currentSection,
                        hashCacheSize = hashCacheSize,
                        hashCachePercent = hashCachePercent,
                        envReportsSize = envReportsSize,
                        hasRemoteProfile = hasRemoteProfile,
                        onChangeDetectionColors = onChangeDetectionColors,
                        onChangeCaptionColors = onChangeCaptionColors,
                        onClearCache = onClearCache,
                        onClearReports = onClearReports,
                        onSyncProfile = onSyncProfile,
                        onLogout = onLogout,
                        onDeleteAccount = onDeleteAccount,
                        onSectionChange = onSectionChange,
                        screenWidth = screenWidth
                    )

                    SectionIndicators(
                        currentSection = currentSection,
                        hasRemoteProfile = hasRemoteProfile,
                        screenWidth = screenWidth
                    )
                }

                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(0.2f),
                    verticalArrangement = Arrangement.Bottom
                ) {
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
                        syncStatus, syncDays.toInt(), false, {}, true
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
                    onNavigateHome, onNavigateReports, onNavigateSettings, AppConfig.env_reports, 2
                )
            }

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

            // Slide Down Notification
            AnimatedVisibility(
                visible = showSlideNotif,
                enter = slideInVertically(initialOffsetY = { -it }),
                exit = slideOutVertically(targetOffsetY = { -it }),
                modifier = Modifier.align(Alignment.TopCenter)
            ) {
                SceneClassifiedNotification(slideMessage)
            }

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
fun PasswordConfirmationDialog(
    showDialog: Boolean,
    title: String,
    passwordValue: String,
    onPasswordChange: (String) -> Unit,
    firstButtonLabel: String,
    secondButtonLabel: String,
    onFirstButtonClick: () -> Unit,
    onSecondButtonClick: () -> Unit,
    showErrorPassword: Boolean
) {
    AnimatedVisibility(
        visible = showDialog, enter = fadeIn(
            initialAlpha = 0f, animationSpec = tween(durationMillis = 0)
        ), exit = fadeOut(
            targetAlpha = 0f, animationSpec = tween(durationMillis = 0)
        )
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Gray.copy(alpha = Constants.BACKGROUND_OPACITY)),
            contentAlignment = Alignment.Center
        ) {
            Card(
                modifier = Modifier.fillMaxWidth(0.8f),
                shape = RoundedCornerShape(28.dp),
                colors = CardDefaults.cardColors(
                    containerColor = colorResource(R.color.notification_white)
                ),
                elevation = CardDefaults.cardElevation(defaultElevation = 3.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    // Icon
                    Icon(
                        imageVector = Icons.Filled.Info,
                        contentDescription = "Password",
                        modifier = Modifier.size(24.dp),
                        tint = colorResource(R.color.std_purple_dark)
                    )

                    // Title
                    Text(
                        text = title,
                        fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                        fontFamily = robotoExtraBold,
                        color = Color.Black,
                        textAlign = TextAlign.Center
                    )

                    // Password Input Field
                    OutlinedTextField(
                        value = passwordValue,
                        onValueChange = onPasswordChange,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(56.dp),
                        singleLine = true,
                        visualTransformation = PasswordVisualTransformation(),
                        keyboardOptions = KeyboardOptions(
                            keyboardType = KeyboardType.Password, imeAction = ImeAction.Done
                        ),
                        shape = RoundedCornerShape(10.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            unfocusedBorderColor = if (showErrorPassword) colorResource(R.color.error_red)
                            else Color.Gray,
                            focusedBorderColor = if (showErrorPassword) colorResource(R.color.error_red)
                            else Color.Gray
                        )
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    // Buttons Row
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        // First Button (Cancel)
                        Button(
                            onClick = onFirstButtonClick,
                            modifier = Modifier
                                .weight(1f)
                                .height(Constants.STD_BUTTON_HEIGHT.dp),
                            shape = RoundedCornerShape(28.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = colorResource(R.color.notification_button_white),
                                contentColor = colorResource(R.color.std_purple)
                            )
                        ) {
                            Text(
                                text = firstButtonLabel,
                                fontSize = Constants.STD_BUTTON_FONT_SIZE.sp
                            )
                        }

                        // Second Button (Right)
                        Button(
                            onClick = onSecondButtonClick,
                            modifier = Modifier
                                .weight(1f)
                                .height(Constants.STD_BUTTON_HEIGHT.dp),
                            shape = RoundedCornerShape(28.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = colorResource(R.color.std_purple),
                                contentColor = Color.White
                            )
                        ) {
                            Text(
                                text = secondButtonLabel,
                                fontSize = Constants.STD_BUTTON_FONT_SIZE.sp
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun TopSettingsSection(
    selectedLanguage: Language,
    selectedQuickAction: Int,
    hapticsEnabled: Boolean,
    soAEnabled: Boolean,
    onLanguageChange: (Language) -> Unit,
    onQuickActionChange: (Int) -> Unit,
    onQuickActionInfoClick: () -> Unit,
    onHapticsToggle: (Boolean) -> Unit,
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

    // Left Column
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(start = 24.dp),
        horizontalAlignment = Alignment.Start
    ) {
        // Language Label
        Text(
            text = getLanguageText(LocalContext.current),
            fontSize = 12.sp,
            color = colorResource(R.color.std_cyan_dark),
            fontFamily = robotoExtraBold
        )

        Spacer(modifier = Modifier.height(3.dp))

        // Language Selector
        LanguageSelector(
            selectedLanguage = selectedLanguage, availableLanguages = listOf(
                Language("en", "English", "US"), Language("ro", "Română", "RO")
            ), onLanguageSelected = onLanguageChange
        )

        Spacer(modifier = Modifier.height(20.dp))

        // Quick Action Label
        Text(
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
                ), availableOptions = listOfQuickAction, onOptionSelected = onQuickActionChange
            )

            Spacer(modifier = Modifier.width(5.dp))

            // Info Button
            IconButton(
                onClick = onQuickActionInfoClick,
                modifier = Modifier.size(Constants.STD_INFO_BUTTON_SIZE.dp)
            ) {
                Icon(
                    imageVector = Icons.Filled.Info,
                    contentDescription = "Info",
                    tint = colorResource(R.color.std_purple),
                    modifier = Modifier.size(Constants.STD_INFO_BUTTON_SIZE.dp)
                )
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(30.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {

            // Haptics Row
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = getHapticsText(LocalContext.current),
                    fontSize = Constants.STD_FONT_SIZE.sp,
                    color = colorResource(R.color.std_cyan_dark),
                    fontFamily = robotoExtraBold
                )

                Spacer(modifier = Modifier.width(9.dp))

                Switch(
                    checked = hapticsEnabled,
                    onCheckedChange = onHapticsToggle,
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = Color.White,
                        checkedTrackColor = colorResource(R.color.std_purple),
                        uncheckedThumbColor = Color.White,
                        uncheckedTrackColor = Color.Gray
                    )
                )
            }

            Spacer(modifier = Modifier.height(20.dp))

            // SoA Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Start
            ) {
                Text(
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
                        modifier = Modifier.size(Constants.STD_INFO_BUTTON_SIZE.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Info,
                            contentDescription = "Info",
                            tint = colorResource(R.color.std_purple),
                            modifier = Modifier.size(Constants.STD_INFO_BUTTON_SIZE.dp)
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun SectionIndicators(
    currentSection: Int, hasRemoteProfile: Boolean, screenWidth: Dp
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

        // Storage indicator
        Box(
            modifier = Modifier
                .size(7.dp)
                .clip(CircleShape)
                .background(if (currentSection == 1) Color.White else Color(0xB3808080))
        )

        // Account indicator (if has profile)
        if (hasRemoteProfile) {
            Box(
                modifier = Modifier
                    .size(7.dp)
                    .clip(CircleShape)
                    .background(if (currentSection == 2) Color.White else Color(0xB3808080))
            )
        }
    }
}

@Composable
fun SlideableSections(
    currentSection: Int,
    hashCacheSize: String,
    hashCachePercent: String,
    envReportsSize: String,
    hasRemoteProfile: Boolean,
    onChangeDetectionColors: () -> Unit,
    onChangeCaptionColors: () -> Unit,
    onClearCache: () -> Unit,
    onClearReports: () -> Unit,
    onSyncProfile: () -> Unit,
    onLogout: () -> Unit,
    onDeleteAccount: () -> Unit,
    onSectionChange: (Int) -> Unit,
    screenWidth: Dp
) {
    var dragOffset by remember { mutableStateOf(0f) }
    val maxSections = if (hasRemoteProfile) 3 else 2

    // Animated offset for smooth transitions
    val animatedOffset = remember { Animatable(0f) }
    val coroutineScope = rememberCoroutineScope()

    // Section offset based on current section + drag
    val sectionOffset = animatedOffset.value + dragOffset

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .fillMaxHeight(0.85f)
            .pointerInput(currentSection, maxSections) {
                detectDragGestures(onDragStart = {
                    // Reset animated offset when starting new drag
                    coroutineScope.launch {
                        animatedOffset.snapTo(0f)
                    }
                }, onDragEnd = {
                    val threshold = size.width * Constants.MIN_HDISTANCE_THRESHOLD

                    when {
                        // Swipe left (next section)
                        dragOffset <= -threshold && currentSection < maxSections - 1 -> {
                            coroutineScope.launch {
                                // Animate rest of the way
                                animatedOffset.animateTo(
                                    targetValue = -size.width.toFloat(), animationSpec = tween(
                                        durationMillis = 250, easing = FastOutSlowInEasing
                                    )
                                )
                                // Change section
                                onSectionChange(currentSection + 1)
                                // Reset offset
                                animatedOffset.snapTo(0f)
                            }
                        }

                        // Swipe right (previous section)
                        dragOffset >= threshold && currentSection > 0 -> {
                            coroutineScope.launch {
                                // Animate rest of the way
                                animatedOffset.animateTo(
                                    targetValue = size.width.toFloat(), animationSpec = tween(
                                        durationMillis = 250, easing = FastOutSlowInEasing
                                    )
                                )
                                // Change section
                                onSectionChange(currentSection - 1)
                                // Reset offset
                                animatedOffset.snapTo(0f)
                            }
                        }

                        // Didn't pass threshold - spring back
                        else -> {
                            coroutineScope.launch {
                                animatedOffset.animateTo(
                                    targetValue = -dragOffset,  // Animate back to 0
                                    animationSpec = spring(
                                        dampingRatio = Spring.DampingRatioMediumBouncy,
                                        stiffness = Spring.StiffnessMedium
                                    )
                                )
                            }
                        }
                    }

                    // Reset drag offset
                    dragOffset = 0f
                }, onDrag = { change, dragAmount ->
                    change.consume()

                    // Update drag offset (live dragging!)
                    val newOffset = dragOffset + dragAmount.x

                    // Constrain dragging at edges
                    dragOffset = when (// At first section - prevent drag right
                        currentSection) {
                        0 if newOffset > 0 -> {
                            newOffset * 0.3f  // Rubber band effect
                        }
                        // At last section - prevent drag left
                        maxSections - 1 if newOffset < 0 -> {
                            newOffset * 0.3f  // Rubber band effect
                        }

                        else -> newOffset
                    }
                })
            }) {
        // Render sections with offset
        Box(
            modifier = Modifier
                .fillMaxSize()
                .graphicsLayer {
                    // Apply horizontal translation based on drag
                    translationX = sectionOffset
                }) {
            when (currentSection) {
                0 -> AppearanceSection(
                    onChangeDetectionColors = onChangeDetectionColors,
                    onChangeCaptionColors = onChangeCaptionColors,
                    screenWidth = screenWidth
                )

                1 -> StorageSection(
                    hashCacheSize = hashCacheSize,
                    hashCachePercent = hashCachePercent,
                    envReportsSize = envReportsSize,
                    onClearCache = onClearCache,
                    onClearReports = onClearReports,
                    screenWidth = screenWidth
                )

                2 -> if (hasRemoteProfile) {
                    AccountSection(
                        onSyncProfile = onSyncProfile,
                        onLogout = onLogout,
                        onDeleteAccount = onDeleteAccount,
                        screenWidth = screenWidth
                    )
                }
            }
        }
    }
}

@Composable
fun AppearanceSection(
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
            text = getAppearanceSectionText(LocalContext.current),
            fontSize = Constants.STD_SUBTITLE_SIZE.sp,
            color = colorResource(R.color.std_cyan_dark),
            fontFamily = robotoExtraBold,
            textAlign = TextAlign.Center
        )

        // Options
        Column(
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Button(
                onClick = onChangeDetectionColors,
                modifier = Modifier
                    .width(screenWidth * 0.5f)
                    .height(Constants.STD_BUTTON_HEIGHT.dp),
                shape = RoundedCornerShape(32.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = colorResource(R.color.std_cyan)
                )
            ) {
                Text(
                    text = getChangeDetectionColorsText(LocalContext.current),
                    fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                    color = Color.White,
                    fontFamily = robotoExtraBold,
                    textAlign = TextAlign.Center
                )
            }

            Button(
                onClick = onChangeCaptionColors,
                modifier = Modifier
                    .width(screenWidth * 0.5f)
                    .height(Constants.STD_BUTTON_HEIGHT.dp),
                shape = RoundedCornerShape(32.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = colorResource(R.color.std_cyan)
                )
            ) {
                Text(
                    text = getChangeCaptionColorsText(LocalContext.current),
                    fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                    color = Color.White,
                    fontFamily = robotoExtraBold,
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

@Composable
fun StorageSection(
    hashCacheSize: String,
    hashCachePercent: String,
    envReportsSize: String,
    onClearCache: () -> Unit,
    onClearReports: () -> Unit,
    screenWidth: Dp
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
            text = getStorageSectionText(LocalContext.current),
            fontSize = Constants.STD_SUBTITLE_SIZE.sp,
            color = colorResource(R.color.std_cyan_dark),
            fontFamily = robotoExtraBold,
            textAlign = TextAlign.Center
        )

        // Options
        Column(
            verticalArrangement = Arrangement.spacedBy(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Clear Cache Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Button(
                    onClick = onClearCache,
                    modifier = Modifier
                        .width(screenWidth * 0.5f)
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

                Spacer(modifier = Modifier.width(8.dp))

                // Size Display
                Card(
                    modifier = Modifier.width(screenWidth * 0.15f),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = colorResource(R.color.std_light_purple)
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(8.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = hashCacheSize,
                            fontSize = Constants.STD_FONT_SIZE.sp,
                            color = colorResource(R.color.std_purple),
                            fontFamily = robotoExtraBold,
                            textAlign = TextAlign.Center
                        )
                        Text(
                            text = "($hashCachePercent)",
                            fontSize = Constants.STD_FONT_SIZE_LW.sp,
                            color = colorResource(R.color.std_purple),
                            fontFamily = robotoLight,
                            textAlign = TextAlign.Center
                        )
                    }
                }
            }

            // Clear Reports Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Button(
                    onClick = onClearReports,
                    modifier = Modifier
                        .width(screenWidth * 0.5f)
                        .height(Constants.STD_BUTTON_HEIGHT.dp),
                    shape = RoundedCornerShape(32.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colorResource(R.color.std_cyan)
                    )
                ) {
                    Text(
                        text = getClearReportsText(LocalContext.current),
                        fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                        color = Color.White,
                        fontFamily = robotoExtraBold,
                        textAlign = TextAlign.Center
                    )
                }

                Spacer(modifier = Modifier.width(8.dp))

                // Size Display
                Card(
                    modifier = Modifier.width(screenWidth * 0.15f),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = colorResource(R.color.std_light_purple)
                    )
                ) {
                    Text(
                        text = envReportsSize,
                        fontSize = Constants.STD_FONT_SIZE.sp,
                        color = colorResource(R.color.std_purple),
                        fontFamily = robotoExtraBold,
                        modifier = Modifier.padding(8.dp),
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
    }
}

@Composable
fun AccountSection(
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
            Button(
                onClick = onSyncProfile,
                modifier = Modifier
                    .width(screenWidth * 0.5f)
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

            Button(
                onClick = onLogout,
                modifier = Modifier
                    .width(screenWidth * 0.5f)
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

            Button(
                onClick = onDeleteAccount,
                modifier = Modifier
                    .width(screenWidth * 0.5f)
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

@Preview(
    name = "Settings Screen - Appearance Section",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
@Composable
fun SettingsScreenPreview1() {
    SettingsScreen(
        showLoading = false,
        loadingText = "",
        showNotification = false,
        notificationMessage = "",
        showSlideNotif = false,
        slideMessage = "Standard notification message",
        currentSection = 0,
        selectedLanguage = Language("en", "English", "US"),
        selectedQuickAction = 0,
        hapticsEnabled = true,
        soAEnabled = true,
        hashCacheSize = "256.5 KB",
        hashCachePercent = "12.3%",
        envReportsSize = "1.2 MB",
        hasRemoteProfile = true,
        onLanguageChange = {},
        onQuickActionChange = {},
        onQuickActionInfoClick = {},
        onHapticsToggle = {},
        onSoAToggle = {},
        onSoAInfoClick = {},
        onChangeDetectionColors = {},
        onChangeCaptionColors = {},
        onClearCache = {},
        onClearReports = {},
        onSyncProfile = {},
        onLogout = {},
        onDeleteAccount = {},
        onExportProfile = {},
        onNavigateHome = {},
        onNavigateReports = {},
        onNavigateSettings = {},
        onSectionChange = {},
        firstButtonClick = {},
        syncStatus = 1,
        syncDays = 3,
        firstButtonClickPassword = {},
        secondButtonClickPassword = {},
        onPasswordChange = {},
        showPasswordDialog = false,
        showErrorPassword = false,
        passwordValue = ""
    )
}

@Preview(
    name = "Settings Screen - Storage Section", showBackground = true, widthDp = 412, heightDp = 917
)
@Composable
fun SettingsScreenPreview2() {
    SettingsScreen(
        showLoading = false,
        loadingText = "",
        showNotification = false,
        notificationMessage = "",
        showSlideNotif = false,
        slideMessage = "",
        currentSection = 1,
        selectedLanguage = Language("en", "English", "US"),
        selectedQuickAction = 0,
        hapticsEnabled = true,
        soAEnabled = true,
        hashCacheSize = "256.5 KB",
        hashCachePercent = "12.3%",
        envReportsSize = "1.2 MB",
        hasRemoteProfile = true,
        onLanguageChange = {},
        onQuickActionChange = {},
        onQuickActionInfoClick = {},
        onHapticsToggle = {},
        onSoAToggle = {},
        onSoAInfoClick = {},
        onChangeDetectionColors = {},
        onChangeCaptionColors = {},
        onClearCache = {},
        onClearReports = {},
        onSyncProfile = {},
        onLogout = {},
        onDeleteAccount = {},
        onExportProfile = {},
        onNavigateHome = {},
        onNavigateReports = {},
        onNavigateSettings = {},
        onSectionChange = {},
        firstButtonClick = {},
        syncStatus = 1,
        syncDays = 3,
        firstButtonClickPassword = {},
        secondButtonClickPassword = {},
        onPasswordChange = {},
        showPasswordDialog = false,
        showErrorPassword = false,
        passwordValue = ""
    )
}

@Preview(
    name = "Settings Screen - Account Section", showBackground = true, widthDp = 412, heightDp = 917
)
@Composable
fun SettingsScreenPreview3() {
    SettingsScreen(
        showLoading = false,
        loadingText = "",
        showNotification = false,
        notificationMessage = "",
        showSlideNotif = false,
        slideMessage = "",
        currentSection = 2,
        selectedLanguage = Language("en", "English", "US"),
        selectedQuickAction = 0,
        hapticsEnabled = true,
        soAEnabled = true,
        hashCacheSize = "256.5 KB",
        hashCachePercent = "12.3%",
        envReportsSize = "1.2 MB",
        hasRemoteProfile = true,
        onLanguageChange = {},
        onQuickActionChange = {},
        onQuickActionInfoClick = {},
        onHapticsToggle = {},
        onSoAToggle = {},
        onSoAInfoClick = {},
        onChangeDetectionColors = {},
        onChangeCaptionColors = {},
        onClearCache = {},
        onClearReports = {},
        onSyncProfile = {},
        onLogout = {},
        onDeleteAccount = {},
        onExportProfile = {},
        onNavigateHome = {},
        onNavigateReports = {},
        onNavigateSettings = {},
        onSectionChange = {},
        firstButtonClick = {},
        syncStatus = 1,
        syncDays = 3,
        firstButtonClickPassword = {},
        secondButtonClickPassword = {},
        onPasswordChange = {},
        showPasswordDialog = false,
        showErrorPassword = false,
        passwordValue = ""
    )
}
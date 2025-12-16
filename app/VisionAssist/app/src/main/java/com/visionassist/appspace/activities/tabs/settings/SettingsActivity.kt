@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.tabs.settings

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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBackIos
import androidx.compose.material.icons.automirrored.filled.ArrowForwardIos
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.BaseActivity
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.main.BottomNavigationBar
import com.visionassist.appspace.activities.main.HomeActivity
import com.visionassist.appspace.activities.main.SyncStatusSection
import com.visionassist.appspace.activities.newprofile.ConfigurationActivity
import com.visionassist.appspace.activities.newprofile.UserAccessibility1Activity
import com.visionassist.appspace.activities.tabs.reports.EnvironmentReportsActivity
import com.visionassist.appspace.database.DBManager
import com.visionassist.appspace.database.NetworkUtils
import com.visionassist.appspace.jetpack.design.LanguageSelector
import com.visionassist.appspace.jetpack.design.LoadingComponent
import com.visionassist.appspace.jetpack.design.QuickActionSelector
import com.visionassist.appspace.jetpack.managers.InfoNotificationManager
import com.visionassist.appspace.models.ttsengine.TTSManager
import com.visionassist.appspace.services.LockScreenService
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
import com.visionassist.appspace.utils.getExportProfileText
import com.visionassist.appspace.utils.getHapticsText
import com.visionassist.appspace.utils.getLanguageChangedMessage
import com.visionassist.appspace.utils.getLanguageText
import com.visionassist.appspace.utils.getLogOutText
import com.visionassist.appspace.utils.getLoggingOffText
import com.visionassist.appspace.utils.getLogoutConfirmMessage
import com.visionassist.appspace.utils.getProfileExportErrorMessage
import com.visionassist.appspace.utils.getProfileExportedMessage
import com.visionassist.appspace.utils.getProfileSyncErrorMessage
import com.visionassist.appspace.utils.getProfileSyncedMessage
import com.visionassist.appspace.utils.getQuickActionDisabledMessage
import com.visionassist.appspace.utils.getQuickActionEnabledMessage
import com.visionassist.appspace.utils.getQuickActionInfoMessage
import com.visionassist.appspace.utils.getQuickActionText
import com.visionassist.appspace.utils.getReportsClearedMessage
import com.visionassist.appspace.utils.getSoAInfoMessage
import com.visionassist.appspace.utils.getSoAText
import com.visionassist.appspace.utils.getStorageSectionText
import com.visionassist.appspace.utils.getSyncProfileText
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.robotoSemibold
import com.visionassist.appspace.utils.vibrate
import org.json.JSONObject
import java.io.File

class SettingsActivity : BaseActivity() {
    private val TAG = "SettingsActivity"

    private val mainHandler = Handler(Looper.getMainLooper())
    private val ttsHandler = Handler(Looper.getMainLooper())

    // Managers
    private lateinit var ttsManager: TTSManager
    private lateinit var dbManager: DBManager
    private lateinit var infoNotificationManager: InfoNotificationManager

    // UI States
    private val showLoading = mutableStateOf(false)
    private val loadingText = mutableStateOf("")
    private val showInfoDialog = mutableStateOf(false)
    private val infoMessage = mutableStateOf("")
    private val showNotifDialog = mutableStateOf(false)
    private val notifMessage = mutableStateOf("")
    private val notifFirstLabel = mutableStateOf("")
    private val notifSecondLabel = mutableStateOf("")
    private val showSlideNotif = mutableStateOf(false)
    private val slideMessage = mutableStateOf("")
    private val currentSection = mutableStateOf(0)

    // Size displays
    private val hashCacheSize = mutableStateOf("")
    private val hashCachePercent = mutableStateOf("")
    private val envReportsSize = mutableStateOf("")

    // Settings states
    private val selectedLanguage = mutableStateOf(AppConfig.mainLanguage)
    private val selectedQuickAction = mutableStateOf("Disabled")
    private val hapticsEnabled = mutableStateOf(AppConfig.haptics)
    private val soAEnabled = mutableStateOf(false)

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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        ttsManager = PhoneStatusMonitor.getInstance().ttsManager
        dbManager = PhoneStatusMonitor.getInstance().dbManager

        // Load profile data
        try {
            val profileFile = File(FileUtils.getProfileDirectory(this), Constants.PROFILE_FILE_NAME)
            if (profileFile.exists()) {
                profileData = JSONObject(profileFile.readText())
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading profile", e)
        }

        // Calculate initial sizes
        calculateSizes()

        // Setup info notification manager
        infoNotificationManager = InfoNotificationManager(this)

        val dbManager = PhoneStatusMonitor.getInstance().dbManager

        setContent {
            SettingsScreen(
                showLoading = showLoading.value,
                loadingText = loadingText.value,
                showInfoDialog = showInfoDialog.value,
                infoMessage = infoMessage.value,
                showNotifDialog = showNotifDialog.value,
                notifMessage = notifMessage.value,
                notifFirstLabel = notifFirstLabel.value,
                notifSecondLabel = notifSecondLabel.value,
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
                hasRemoteProfile = getStatusOverview() == 1 || getStatusOverview() == 2,
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
                onInfoDialogDismiss = { showInfoDialog.value = false },
                onNotifDialogFirstButton = ::handleNotifFirstButton,
                onNotifDialogSecondButton = ::handleNotifSecondButton,
                syncStatus = dbManager.statusOverview,
                syncDays = dbManager.diffDays
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

    private fun handleLanguageChange(language: Language) {
        vibrateIfNeeded()
        showLoading.value = true
        loadingText.value = getApplyingSettingsText(this)

        mainHandler.postDelayed({
            selectedLanguage.value = language
            AppConfig.mainLanguage = language

            setTTSLanguage()
        }, 500)
    }

    private fun setTTSLanguage() {
        if (!AppConfig.mainLanguage.code.equals(
                ttsManager.currentLocale.language,
                ignoreCase = true
            )
        ) {
            Log.d(TAG, "TTS needs language change")
            waitingForTTSLanguage = true
            ttsManager.changeLanguage(AppConfig.mainLanguage, this)
            waitForTTSAndReload()
        } else {
            Log.d(TAG, "TTS already correct language")
            waitForTTSAndReload()
        }
    }

    private fun waitForTTSAndReload() {
        val checkTTS = object : Runnable {
            override fun run() {
                if (ttsManager.isReady) {
                    Log.d(TAG, "TTS ready, reloading UI")
                    waitingForTTSLanguage = false
                    showLoading.value = false

                    // Show notification
                    slideMessage.value = getLanguageChangedMessage(this@SettingsActivity)
                    showSlideNotif.value = true

                    // Hide after delay
                    mainHandler.postDelayed({
                        showSlideNotif.value = false
                    }, 3000)
                } else {
                    Log.w(TAG, "TTS not ready, retrying...")
                    ttsHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS.toLong())
                }
            }
        }
        ttsHandler.post(checkTTS)
    }

    private fun handleQuickActionChange(action: String) {
        vibrateIfNeeded()
        showLoading.value = true
        loadingText.value = getApplyingSettingsText(this)

        mainHandler.postDelayed({
            selectedQuickAction.value = action

            if (action == "Disabled") {
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
            }, 3000)
        }, 500)
    }

    private fun handleQuickActionInfo() {
        vibrateIfNeeded()
        infoMessage.value = getQuickActionInfoMessage(this)
        showInfoDialog.value = true
    }

    private fun handleHapticsToggle(enabled: Boolean) {
        vibrateIfNeeded()
        hapticsEnabled.value = enabled
        AppConfig.haptics = enabled

        showSlideNotif.value = true
        mainHandler.postDelayed({
            showSlideNotif.value = false
        }, 2000)
    }

    private fun handleSoAToggle(enabled: Boolean) {
        vibrateIfNeeded()
        soAEnabled.value = enabled
        AppConfig.SoA = enabled

        showSlideNotif.value = true
        mainHandler.postDelayed({
            showSlideNotif.value = false
        }, 2000)
    }

    private fun handleSoAInfo() {
        vibrateIfNeeded()
        infoMessage.value = getSoAInfoMessage(this)
        showInfoDialog.value = true
    }

    private fun handleChangeDetectionColors() {
        vibrateIfNeeded()
        val intent = Intent(this, UserAccessibility1Activity::class.java).apply {
            putExtra("EXTRA_USERACC_OPTION", 1)
            putExtra("EXTRA_USERACC_OPTION2", true)
        }
        startActivity(intent)
    }

    private fun handleChangeCaptionColors() {
        vibrateIfNeeded()
        val intent = Intent(this, UserAccessibility1Activity::class.java).apply {
            putExtra("EXTRA_USERACC_OPTION", 2)
            putExtra("EXTRA_USERACC_OPTION2", true)
        }
        startActivity(intent)
    }

    private fun handleClearCache() {
        vibrateIfNeeded()
        notifMessage.value = getClearCacheConfirmMessage(this)
        notifFirstLabel.value = if (AppConfig.mainLanguage.code == "en") {
            getString(R.string.dont_button_en)
        } else {
            getString(R.string.dont_button_ro)
        }
        notifSecondLabel.value = if (AppConfig.mainLanguage.code == "en") {
            getString(R.string.clear_button_en)
        } else {
            getString(R.string.clear_button_ro)
        }
        showNotifDialog.value = true
    }

    private fun executeClearCache() {
        showLoading.value = true
        loadingText.value = getApplyingSettingsText(this)

        mainHandler.postDelayed({
            try {
                FileUtils.createProfileDirFile(Constants.HASH_CACHE_FILE_NAME)
                calculateSizes()
                slideMessage.value = getCacheClearedMessage(this)
            } catch (e: Exception) {
                Log.e(TAG, "Error clearing cache", e)
                slideMessage.value = "Error clearing cache"
            }

            showLoading.value = false
            showSlideNotif.value = true

            mainHandler.postDelayed({
                showSlideNotif.value = false
            }, 3000)
        }, 500)
    }

    private fun handleClearReports() {
        vibrateIfNeeded()
        notifMessage.value = getClearReportsConfirmMessage(this)
        notifFirstLabel.value = if (AppConfig.mainLanguage.code == "en") {
            getString(R.string.dont_button_en)
        } else {
            getString(R.string.dont_button_ro)
        }
        notifSecondLabel.value = if (AppConfig.mainLanguage.code == "en") {
            getString(R.string.clear_button_en)
        } else {
            getString(R.string.clear_button_ro)
        }
        showNotifDialog.value = true
    }

    private fun executeClearReports() {
        showLoading.value = true
        loadingText.value = getApplyingSettingsText(this)

        mainHandler.postDelayed({
            try {
                FileUtils.createProfileDirFile(Constants.ENV_REPORTS_FILE_NAME)
                calculateSizes()
                slideMessage.value = getReportsClearedMessage(this)
            } catch (e: Exception) {
                Log.e(TAG, "Error clearing reports", e)
                slideMessage.value = "Error clearing reports"
            }

            showLoading.value = false
            showSlideNotif.value = true

            mainHandler.postDelayed({
                showSlideNotif.value = false
            }, 3000)
        }, 500)
    }

    private fun handleSyncProfile() {
        vibrateIfNeeded()

        if (!NetworkUtils.isNetworkConnected(this)) {
            notifMessage.value = if (AppConfig.mainLanguage.code == "en") {
                "Internet Connection Failed~Please check your internet connection and try again"
            } else {
                "Conexiunea la Internet a Eșuat~Vă rugăm să verificați conexiunea la internet și să încercați din nou"
            }
            notifFirstLabel.value = getString(R.string.ok_button_en)

            showNotifDialog.value = true
            return
        }

        showLoading.value = true
        loadingText.value = getApplyingSettingsText(this)

        mainHandler.postDelayed({
            BackgroundTaskExecutor.getInstance().executeAsync(
                {
                    profileData?.let { dbManager.syncProfile(it) }
                    return@executeAsync 0
                },
                object : BackgroundTaskExecutor.TaskCallback<Int> {
                    override fun onSuccess(result: Int?) {
                        mainHandler.post {
                            showLoading.value = false

                            slideMessage.value = when (getStatusOverview()) {
                                1 -> getProfileSyncedMessage(this@SettingsActivity)
                                2 -> getProfileSyncErrorMessage(this@SettingsActivity)
                                else -> getProfileSyncedMessage(this@SettingsActivity)
                            }

                            showSlideNotif.value = true

                            mainHandler.postDelayed({
                                showSlideNotif.value = false
                            }, 3000)
                        }
                    }

                    override fun onError(e: Exception?) {
                        mainHandler.post {
                            showLoading.value = false
                            slideMessage.value = getProfileSyncErrorMessage(this@SettingsActivity)
                            showSlideNotif.value = true

                            mainHandler.postDelayed({
                                showSlideNotif.value = false
                            }, 3000)
                        }
                    }
                }
            )
        }, 500)
    }

    private fun handleLogout() {
        vibrateIfNeeded()
        notifMessage.value = getLogoutConfirmMessage(this)
        notifFirstLabel.value = if (AppConfig.mainLanguage.code == "en") {
            getString(R.string.dont_button_en)
        } else {
            getString(R.string.dont_button_ro)
        }
        notifSecondLabel.value = if (AppConfig.mainLanguage.code == "en") {
            getString(R.string.logout_button_en)
        } else {
            getString(R.string.logout_button_ro)
        }
        showNotifDialog.value = true
    }

    private fun executeLogout() {
        showLoading.value = true
        loadingText.value = getLoggingOffText(this)

        mainHandler.postDelayed({
            try {
                // Delete all local files
                FileUtils.deleteProfileDirFile(Constants.ENV_REPORTS_FILE_NAME)
                FileUtils.deleteProfileDirFile(Constants.HASH_CACHE_FILE_NAME)
                FileUtils.deleteProfileDirFile(Constants.PROFILE_FILE_NAME)

                showLoading.value = false

                // Navigate to configuration
                val intent = Intent(this, ConfigurationActivity::class.java)
                startActivity(intent)
                finish()
            } catch (e: Exception) {
                Log.e(TAG, "Error during logout", e)
                showLoading.value = false
            }
        }, 500)
    }

    private fun handleDeleteAccount() {
        vibrateIfNeeded()
        notifMessage.value = getDeleteAccountConfirmMessage(this)
        notifFirstLabel.value = if (AppConfig.mainLanguage.code == "en") {
            getString(R.string.dont_button_en)
        } else {
            getString(R.string.dont_button_ro)
        }
        notifSecondLabel.value = if (AppConfig.mainLanguage.code == "en") {
            getString(R.string.delete_button_en)
        } else {
            getString(R.string.delete_button_ro)
        }
        showNotifDialog.value = true
    }

    private fun executeDeleteAccount() {
        showLoading.value = true
        loadingText.value = getLoggingOffText(this)

        mainHandler.postDelayed({
            try {
                // Delete local files first
                FileUtils.deleteProfileDirFile(Constants.ENV_REPORTS_FILE_NAME)
                FileUtils.deleteProfileDirFile(Constants.HASH_CACHE_FILE_NAME)
                FileUtils.deleteProfileDirFile(Constants.PROFILE_FILE_NAME)

                // Delete from Firebase
                dbManager.deleteAccount(
                    this,
                    {
                        mainHandler.post {
                            showLoading.value = false
                            val intent = Intent(this, ConfigurationActivity::class.java)
                            startActivity(intent)
                            finish()
                        }
                    },
                    { e ->
                        mainHandler.post {
                            Log.e(TAG, "Error deleting account", e)
                            showLoading.value = false
                            // Still navigate even if Firebase delete fails
                            val intent = Intent(this, ConfigurationActivity::class.java)
                            startActivity(intent)
                            finish()
                        }
                    }
                )
            } catch (e: Exception) {
                Log.e(TAG, "Error during delete account", e)
                showLoading.value = false
            }
        }, 500)
    }

    private fun handleExportProfile() {
        vibrateIfNeeded()
        exportProfileLauncher.launch(null)
    }

    private fun handleExportProfileResult(uri: Uri) {
        showLoading.value = true
        loadingText.value = getApplyingSettingsText(this)

        mainHandler.postDelayed({
            exportErrorCode = 0
            exportFilesCompleted = 0

            // Copy profile.json
            BackgroundTaskExecutor.getInstance().executeAsync(
                {
                    copyFileToUri(uri, Constants.PROFILE_FILE_NAME)
                    return@executeAsync 0
                },
                object : BackgroundTaskExecutor.TaskCallback<Int> {
                    override fun onSuccess(result: Int?) {
                        synchronized(this@SettingsActivity) {
                            exportFilesCompleted++
                        }
                    }

                    override fun onError(e: Exception?) {
                        synchronized(this@SettingsActivity) {
                            exportErrorCode = 1
                            exportFilesCompleted++
                        }
                    }
                }
            )

            // Copy hash cache
            BackgroundTaskExecutor.getInstance().executeAsync(
                {
                    copyFileToUri(uri, Constants.HASH_CACHE_FILE_NAME)
                    return@executeAsync 0
                },
                object : BackgroundTaskExecutor.TaskCallback<Int> {
                    override fun onSuccess(result: Int?) {
                        synchronized(this@SettingsActivity) {
                            exportFilesCompleted++
                        }
                    }

                    override fun onError(e: Exception?) {
                        synchronized(this@SettingsActivity) {
                            exportErrorCode = 1
                            exportFilesCompleted++
                        }
                    }
                }
            )

            // Copy env reports
            BackgroundTaskExecutor.getInstance().executeAsync(
                {
                    copyFileToUri(uri, Constants.ENV_REPORTS_FILE_NAME)
                    return@executeAsync 0
                },
                object : BackgroundTaskExecutor.TaskCallback<Int> {
                    override fun onSuccess(result: Int?) {
                        synchronized(this@SettingsActivity) {
                            exportFilesCompleted++
                        }
                    }

                    override fun onError(e: Exception?) {
                        synchronized(this@SettingsActivity) {
                            exportErrorCode = 1
                            exportFilesCompleted++
                        }
                    }
                }
            )

            // Wait for all copies to complete
            waitForExportCompletion()
        }, 500)
    }

    private fun waitForExportCompletion() {
        val checkCompletion = object : Runnable {
            override fun run() {
                synchronized(this@SettingsActivity) {
                    if (exportFilesCompleted >= 3) {
                        mainHandler.post {
                            showLoading.value = false

                            slideMessage.value = if (exportErrorCode == 0) {
                                getProfileExportedMessage(this@SettingsActivity)
                            } else {
                                getProfileExportErrorMessage(this@SettingsActivity)
                            }

                            showSlideNotif.value = true

                            mainHandler.postDelayed({
                                showSlideNotif.value = false
                            }, 3000)
                        }
                    } else {
                        mainHandler.postDelayed(this, Constants.WAIT_CHECK)
                    }
                }
            }
        }
        mainHandler.post(checkCompletion)
    }

    private fun copyFileToUri(targetUri: Uri, fileName: String) {
        val sourceFile = File(FileUtils.getProfileDirectory(this), fileName)
        if (!sourceFile.exists()) return

        val resolver = contentResolver
        val folderUri = DocumentsContract.buildDocumentUriUsingTree(
            targetUri,
            DocumentsContract.getTreeDocumentId(targetUri)
        )

        val newFileUri = DocumentsContract.createDocument(
            resolver,
            folderUri,
            "application/octet-stream",
            fileName
        )

        newFileUri?.let { uri ->
            resolver.openOutputStream(uri)?.use { outputStream ->
                sourceFile.inputStream().use { inputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }
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
        // First button is always "Don't" - just hide dialog
        showNotifDialog.value = false
    }

    private fun handleNotifSecondButton() {
        // Second button executes the action
        showNotifDialog.value = false

        // Determine which action based on current dialog message
        when {
            notifMessage.value.contains("cache", ignoreCase = true) -> executeClearCache()
            notifMessage.value.contains("reports", ignoreCase = true) -> executeClearReports()
            notifMessage.value.contains("log out", ignoreCase = true) ||
                    notifMessage.value.contains("deconectare", ignoreCase = true) -> executeLogout()

            notifMessage.value.contains("delete account", ignoreCase = true) ||
                    notifMessage.value.contains(
                        "șterge contul",
                        ignoreCase = true
                    ) -> executeDeleteAccount()
        }
    }

    private fun vibrateIfNeeded() {
        if (AppConfig.haptics) {
            vibrate(haptic_model0())
        }
    }

    private fun getStatusOverview(): Int {
        return try {
            profileData?.let {
                if (it.has("remoteProfile") && it.getBoolean("remoteProfile")) {
                    // Check if synced
                    if (it.has("syncStatus")) {
                        when (it.getString("syncStatus")) {
                            "synced" -> 1
                            "error" -> 2
                            else -> 0
                        }
                    } else {
                        1
                    }
                } else {
                    0
                }
            } ?: 0
        } catch (_: Exception) {
            0
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

    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacksAndMessages(null)
        ttsHandler.removeCallbacksAndMessages(null)
    }
}

@Composable
fun SettingsScreen(
    // Loading state
    showLoading: Boolean = false,
    loadingText: String = "",

    // Info dialog
    showInfoDialog: Boolean = false,
    infoMessage: String = "",

    // Notification dialog (two buttons)
    showNotifDialog: Boolean = false,
    notifMessage: String = "",
    notifFirstLabel: String = "",
    notifSecondLabel: String = "",

    // Slide notification
    showSlideNotif: Boolean = false,
    slideMessage: String = "",

    // Current section (0=Appearance, 1=Storage, 2=Account)
    currentSection: Int = 0,

    // Settings states
    selectedLanguage: Language = Language("en", "English", "US"),
    selectedQuickAction: String = "Disabled",
    hapticsEnabled: Boolean = true,
    soAEnabled: Boolean = false,

    // Size displays
    hashCacheSize: String = "0.0 KB",
    hashCachePercent: String = "0.0%",
    envReportsSize: String = "0.0 KB",

    // Profile status
    hasRemoteProfile: Boolean = false,

    // Handlers
    onLanguageChange: (Language) -> Unit = {},
    onQuickActionChange: (String) -> Unit = {},
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
    onInfoDialogDismiss: () -> Unit = {},
    onNotifDialogFirstButton: () -> Unit = {},
    onNotifDialogSecondButton: () -> Unit = {},
    syncStatus: Int = 0,
    syncDays: Long = 0
) {
    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenHeight = maxHeight
        val navbarHeight = 90.dp / maxHeight
        val sectionMain = 1.0f - navbarHeight

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
        ) {
            // Main content
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .fillMaxHeight(sectionMain)
            ) {
                Spacer(modifier = Modifier.height(16.dp))

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
                    onSoAInfoClick = onSoAInfoClick
                )

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
                    onSectionChange = onSectionChange
                )

                // Export Profile Button
                Button(
                    onClick = onExportProfile,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 24.dp)
                        .height(Constants.STD_BUTTON_HEIGHT.dp),
                    shape = RoundedCornerShape(32.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colorResource(R.color.std_purple)
                    )
                ) {
                    Text(
                        text = getExportProfileText(PhoneStatusMonitor.getInstance().currentContext),
                        fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                        color = Color.White,
                        fontFamily = robotoExtraBold
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                // Sync Status Section
                SyncStatusSection(
                    syncStatus,
                    syncDays.toInt(),
                    false,
                    {},
                    true
                )

                Spacer(modifier = Modifier.height(8.dp))
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
                    onNavigateHome,
                    onNavigateReports,
                    onNavigateSettings,
                    AppConfig.env_reports,
                    2
                )
            }

            // Loading overlay
            LoadingComponent(
                isVisible = showLoading,
                loadingText = loadingText,
                animSpec = Pair(
                    fadeIn(initialAlpha = 0f, animationSpec = tween(durationMillis = 0)),
                    fadeOut(targetAlpha = 0f, animationSpec = tween(durationMillis = 0))
                )
            )

            /*
            // Info Dialog
            InfoNotificationDialog(
                isVisible = showInfoDialog,
                message = infoMessage,
                twoButtons = false,
                firstButtonLabel = if (AppConfig.mainLanguage.code == "en") "OK" else "OK",
                onFirstButtonClick = onInfoDialogDismiss
            )

            // Notification Dialog (Two Buttons)
            InfoNotificationDialog(
                isVisible = showNotifDialog,
                message = notifMessage,
                twoButtons = true,
                firstButtonLabel = notifFirstLabel,
                secondButtonLabel = notifSecondLabel,
                onFirstButtonClick = onNotifDialogFirstButton,
                onSecondButtonClick = onNotifDialogSecondButton
            )
             */

            // Slide Down Notification
            AnimatedVisibility(
                visible = showSlideNotif,
                enter = slideInVertically(
                    initialOffsetY = { -it },
                    animationSpec = spring(
                        dampingRatio = Spring.DampingRatioMediumBouncy,
                        stiffness = Spring.StiffnessLow
                    )
                ),
                exit = slideOutVertically(
                    targetOffsetY = { -it },
                    animationSpec = spring(
                        dampingRatio = Spring.DampingRatioNoBouncy,
                        stiffness = Spring.StiffnessMedium
                    )
                ),
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp, vertical = 16.dp)
            ) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = colorResource(R.color.std_purple)
                    ),
                    elevation = CardDefaults.cardElevation(8.dp)
                ) {
                    Text(
                        text = slideMessage,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        color = Color.White,
                        fontSize = Constants.STD_FONT_SIZE.sp,
                        fontFamily = robotoSemibold
                    )
                }
            }
        }
    }
}

@Composable
fun TopSettingsSection(
    selectedLanguage: Language,
    selectedQuickAction: String,
    hapticsEnabled: Boolean,
    soAEnabled: Boolean,
    onLanguageChange: (Language) -> Unit,
    onQuickActionChange: (String) -> Unit,
    onQuickActionInfoClick: () -> Unit,
    onHapticsToggle: (Boolean) -> Unit,
    onSoAToggle: (Boolean) -> Unit,
    onSoAInfoClick: () -> Unit
) {
    Row(
        modifier = Modifier.padding(horizontal = 24.dp),
        horizontalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Left Column
        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxHeight(),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            // Language Label
            Text(
                text = getLanguageText(PhoneStatusMonitor.getInstance().currentContext),
                fontSize = 12.sp,
                color = colorResource(R.color.std_cyan),
                fontFamily = robotoExtraBold
            )

            // Language Selector
            LanguageSelector(
                selectedLanguage = selectedLanguage,
                availableLanguages = listOf(
                    Language("en", "English", "US"),
                    Language("ro", "Română", "RO")
                ),
                onLanguageSelected = onLanguageChange
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Quick Action Label
            Text(
                text = getQuickActionText(PhoneStatusMonitor.getInstance().currentContext),
                fontSize = 12.sp,
                color = colorResource(R.color.std_cyan),
                fontFamily = robotoExtraBold
            )

            // Quick Action Row (Selector + Info Button)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Quick Action Selector
                Box(modifier = Modifier.weight(1f)) {
                    QuickActionSelector(
                        selectedOption = selectedQuickAction,
                        availableOptions = listOf(
                            "Disabled",
                            "Detection-static",
                            "Detection-dynamic",
                            "Caption"
                        ),
                        onOptionSelected = onQuickActionChange
                    )
                }

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
        }

        // Right Column
        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxHeight(),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Haptics Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = getHapticsText(PhoneStatusMonitor.getInstance().currentContext),
                    fontSize = 14.sp,
                    color = colorResource(R.color.std_cyan),
                    fontFamily = robotoExtraBold
                )

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

            Spacer(modifier = Modifier.height(8.dp))

            // SoA Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = getSoAText(PhoneStatusMonitor.getInstance().currentContext),
                    fontSize = 14.sp,
                    color = colorResource(R.color.std_cyan),
                    fontFamily = robotoExtraBold
                )

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
    onSectionChange: (Int) -> Unit
) {
    var dragOffset by remember { mutableStateOf(0f) }
    val maxSections = if (hasRemoteProfile) 3 else 2

    Box(
        modifier = Modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                detectDragGestures(
                    onDragEnd = {
                        val threshold = size.width * 0.3f
                        when {
                            dragOffset < -threshold && currentSection < maxSections - 1 -> {
                                onSectionChange(currentSection + 1)
                            }

                            dragOffset > threshold && currentSection > 0 -> {
                                onSectionChange(currentSection - 1)
                            }
                        }
                        dragOffset = 0f
                    },
                    onDrag = { change, dragAmount ->
                        change.consume()
                        dragOffset += dragAmount.x
                    }
                )
            }
    ) {
        when (currentSection) {
            0 -> AppearanceSection(
                onChangeDetectionColors = onChangeDetectionColors,
                onChangeCaptionColors = onChangeCaptionColors,
                showRightArrow = true,
                onRightArrowClick = { onSectionChange(1) }
            )

            1 -> StorageSection(
                hashCacheSize = hashCacheSize,
                hashCachePercent = hashCachePercent,
                envReportsSize = envReportsSize,
                onClearCache = onClearCache,
                onClearReports = onClearReports,
                showLeftArrow = true,
                showRightArrow = hasRemoteProfile,
                onLeftArrowClick = { onSectionChange(0) },
                onRightArrowClick = { if (hasRemoteProfile) onSectionChange(2) }
            )

            2 -> if (hasRemoteProfile) {
                AccountSection(
                    onSyncProfile = onSyncProfile,
                    onLogout = onLogout,
                    onDeleteAccount = onDeleteAccount,
                    showLeftArrow = true,
                    onLeftArrowClick = { onSectionChange(1) }
                )
            }
        }
    }
}

@Composable
fun AppearanceSection(
    onChangeDetectionColors: () -> Unit,
    onChangeCaptionColors: () -> Unit,
    showRightArrow: Boolean,
    onRightArrowClick: () -> Unit
) {
    val context = LocalContext.current

    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenWidth = maxWidth

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 24.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            // Title
            Text(
                text = getAppearanceSectionText(context),
                fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                color = colorResource(R.color.std_purple),
                fontFamily = robotoExtraBold
            )

            // Options
            Column(
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Button(
                    onClick = onChangeDetectionColors,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(Constants.STD_BUTTON_HEIGHT.dp),
                    shape = RoundedCornerShape(32.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colorResource(R.color.std_purple)
                    )
                ) {
                    Text(
                        text = getChangeDetectionColorsText(context),
                        fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                        color = Color.White
                    )
                }

                Button(
                    onClick = onChangeCaptionColors,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(Constants.STD_BUTTON_HEIGHT.dp),
                    shape = RoundedCornerShape(32.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colorResource(R.color.std_purple)
                    )
                ) {
                    Text(
                        text = getChangeCaptionColorsText(context),
                        fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                        color = Color.White
                    )
                }
            }
        }

        // Arrow Buttons
        if (showRightArrow) {
            IconButton(
                onClick = onRightArrowClick,
                modifier = Modifier
                    .align(Alignment.CenterEnd)
                    .width((screenWidth * 0.12f))
                    .fillMaxHeight()
                    .background(
                        color = colorResource(R.color.std_light_purple),
                        shape = RoundedCornerShape(
                            topStart = 100.dp,
                            bottomStart = 100.dp
                        )
                    )
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowForwardIos,
                    contentDescription = "Next",
                    tint = colorResource(R.color.std_purple),
                    modifier = Modifier.size(32.dp)
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
    showLeftArrow: Boolean,
    showRightArrow: Boolean,
    onLeftArrowClick: () -> Unit,
    onRightArrowClick: () -> Unit
) {
    val context = LocalContext.current

    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenWidth = maxWidth

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 24.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            // Title
            Text(
                text = getStorageSectionText(context),
                fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                color = colorResource(R.color.std_purple),
                fontFamily = robotoExtraBold
            )

            // Options
            Column(
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // Clear Cache Row
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Button(
                        onClick = onClearCache,
                        modifier = Modifier
                            .weight(1f)
                            .height(Constants.STD_BUTTON_HEIGHT.dp),
                        shape = RoundedCornerShape(32.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = colorResource(R.color.std_purple)
                        )
                    ) {
                        Text(
                            text = getClearCacheText(context),
                            fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                            color = Color.White
                        )
                    }

                    Spacer(modifier = Modifier.width(8.dp))

                    // Size Display
                    Card(
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
                                fontSize = 12.sp,
                                color = colorResource(R.color.std_purple),
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = "($hashCachePercent)",
                                fontSize = 10.sp,
                                color = colorResource(R.color.std_purple)
                            )
                        }
                    }
                }

                // Clear Reports Row
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Button(
                        onClick = onClearReports,
                        modifier = Modifier
                            .weight(1f)
                            .height(Constants.STD_BUTTON_HEIGHT.dp),
                        shape = RoundedCornerShape(32.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = colorResource(R.color.std_purple)
                        )
                    ) {
                        Text(
                            text = getClearReportsText(context),
                            fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                            color = Color.White
                        )
                    }

                    Spacer(modifier = Modifier.width(8.dp))

                    // Size Display
                    Card(
                        shape = RoundedCornerShape(16.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = colorResource(R.color.std_light_purple)
                        )
                    ) {
                        Text(
                            text = envReportsSize,
                            fontSize = 12.sp,
                            color = colorResource(R.color.std_purple),
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(8.dp)
                        )
                    }
                }
            }
        }

        // Arrow Buttons
        if (showLeftArrow) {
            IconButton(
                onClick = onLeftArrowClick,
                modifier = Modifier
                    .align(Alignment.CenterStart)
                    .width((screenWidth * 0.12f))
                    .fillMaxHeight()
                    .background(
                        color = colorResource(R.color.std_light_purple),
                        shape = RoundedCornerShape(
                            topEnd = 100.dp,
                            bottomEnd = 100.dp
                        )
                    )
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowBackIos,
                    contentDescription = "Previous",
                    tint = colorResource(R.color.std_purple),
                    modifier = Modifier.size(32.dp)
                )
            }
        }

        if (showRightArrow) {
            IconButton(
                onClick = onRightArrowClick,
                modifier = Modifier
                    .align(Alignment.CenterEnd)
                    .width((screenWidth * 0.12f))
                    .fillMaxHeight()
                    .background(
                        color = colorResource(R.color.std_light_purple),
                        shape = RoundedCornerShape(
                            topStart = 100.dp,
                            bottomStart = 100.dp
                        )
                    )
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowForwardIos,
                    contentDescription = "Next",
                    tint = colorResource(R.color.std_purple),
                    modifier = Modifier.size(32.dp)
                )
            }
        }
    }
}

@Composable
fun AccountSection(
    onSyncProfile: () -> Unit,
    onLogout: () -> Unit,
    onDeleteAccount: () -> Unit,
    showLeftArrow: Boolean,
    onLeftArrowClick: () -> Unit
) {
    val context = LocalContext.current

    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenWidth = maxWidth

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 24.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            // Title
            Text(
                text = getAccountSectionText(context),
                fontSize = Constants.STD_ERROR_FONT_SIZE.sp,
                color = colorResource(R.color.std_purple),
                fontFamily = robotoExtraBold
            )

            // Options
            Column(
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Button(
                    onClick = onSyncProfile,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(Constants.STD_BUTTON_HEIGHT.dp),
                    shape = RoundedCornerShape(32.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colorResource(R.color.std_purple)
                    )
                ) {
                    Text(
                        text = getSyncProfileText(context),
                        fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                        color = Color.White
                    )
                }

                Button(
                    onClick = onLogout,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(Constants.STD_BUTTON_HEIGHT.dp),
                    shape = RoundedCornerShape(32.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = colorResource(R.color.std_purple)
                    )
                ) {
                    Text(
                        text = getLogOutText(context),
                        fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                        color = Color.White
                    )
                }

                Button(
                    onClick = onDeleteAccount,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(Constants.STD_BUTTON_HEIGHT.dp),
                    shape = RoundedCornerShape(32.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color.Red
                    )
                ) {
                    Text(
                        text = getDeleteAccountText(context),
                        fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                        color = Color.White
                    )
                }
            }
        }

        // Arrow Button
        if (showLeftArrow) {
            IconButton(
                onClick = onLeftArrowClick,
                modifier = Modifier
                    .align(Alignment.CenterStart)
                    .width((screenWidth * 0.12f))
                    .fillMaxHeight()
                    .background(
                        color = colorResource(R.color.std_light_purple),
                        shape = RoundedCornerShape(
                            topEnd = 100.dp,
                            bottomEnd = 100.dp
                        )
                    )
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowBackIos,
                    contentDescription = "Previous",
                    tint = colorResource(R.color.std_purple),
                    modifier = Modifier.size(32.dp)
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
        showInfoDialog = false,
        infoMessage = "",
        showNotifDialog = false,
        notifMessage = "",
        notifFirstLabel = "Cancel",
        notifSecondLabel = "OK",
        showSlideNotif = false,
        slideMessage = "",
        currentSection = 0,
        selectedLanguage = Language("en", "English", "US"),
        selectedQuickAction = "Disabled",
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
        onInfoDialogDismiss = {},
        onNotifDialogFirstButton = {},
        onNotifDialogSecondButton = {}
    )
}

@Preview(
    name = "Settings Screen - Storage Section",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
@Composable
fun SettingsScreenPreview2() {
    SettingsScreen(
        showLoading = false,
        loadingText = "",
        showInfoDialog = false,
        infoMessage = "",
        showNotifDialog = false,
        notifMessage = "",
        notifFirstLabel = "Cancel",
        notifSecondLabel = "OK",
        showSlideNotif = false,
        slideMessage = "",
        currentSection = 1,
        selectedLanguage = Language("en", "English", "US"),
        selectedQuickAction = "Disabled",
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
        onInfoDialogDismiss = {},
        onNotifDialogFirstButton = {},
        onNotifDialogSecondButton = {}
    )
}

@Preview(
    name = "Settings Screen - Account Section",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
@Composable
fun SettingsScreenPreview3() {
    SettingsScreen(
        showLoading = false,
        loadingText = "",
        showInfoDialog = false,
        infoMessage = "",
        showNotifDialog = false,
        notifMessage = "",
        notifFirstLabel = "Cancel",
        notifSecondLabel = "OK",
        showSlideNotif = false,
        slideMessage = "",
        currentSection = 2,
        selectedLanguage = Language("en", "English", "US"),
        selectedQuickAction = "Disabled",
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
        onInfoDialogDismiss = {},
        onNotifDialogFirstButton = {},
        onNotifDialogSecondButton = {}
    )
}
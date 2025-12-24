@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.newprofile

import android.annotation.SuppressLint
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.Pair
import android.view.KeyEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.isImeVisible
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AccountCircle
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CloudOff
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.VisibilityOff
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.view.WindowCompat
import androidx.documentfile.provider.DocumentFile
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.main.BlindHomeActivity
import com.visionassist.appspace.activities.main.HomeActivity
import com.visionassist.appspace.activities.newprofile.LoadProfileActivity.NotificationType
import com.visionassist.appspace.activities.newprofile.jsonCollection.ProfileFileCollection
import com.visionassist.appspace.activities.tabs.settings.BlockingOverlay
import com.visionassist.appspace.database.DBConstants
import com.visionassist.appspace.database.NetworkUtils
import com.visionassist.appspace.jetpack.design.BackArrowLargeFab
import com.visionassist.appspace.jetpack.design.LoadingComponent
import com.visionassist.appspace.jetpack.design.NotificationDialog
import com.visionassist.appspace.jetpack.managers.InfoNotificationManager
import com.visionassist.appspace.models.ttsengine.TTSManager
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.BackgroundTaskExecutor
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.FileUtils
import com.visionassist.appspace.utils.JSONValidation
import com.visionassist.appspace.utils.Utils
import com.visionassist.appspace.utils.load_accountNE
import com.visionassist.appspace.utils.load_errorLocalLoadProfileActivity
import com.visionassist.appspace.utils.load_genericErrorLoad
import com.visionassist.appspace.utils.load_infoLoadProfileActivity
import com.visionassist.appspace.utils.load_loadingText
import com.visionassist.appspace.utils.load_loadingVerifying
import com.visionassist.appspace.utils.load_noInternet
import com.visionassist.appspace.utils.load_passChangedSuccess
import com.visionassist.appspace.utils.load_profileImportedSuccess
import com.visionassist.appspace.utils.load_successLocalLoadProfileActivity
import com.visionassist.appspace.utils.robotoLight
import com.visionassist.appspace.utils.robotoRegular
import com.visionassist.appspace.utils.robotoSemibold
import org.json.JSONObject
import java.io.File

class LoadProfileActivity : ComponentActivity() {
    private val TAG = "LoadProfileActivity"

    private var ttsManager: TTSManager = PhoneStatusMonitor.getInstance().ttsManager
    private val mainHandler = Handler(Looper.getMainLooper())
    private val ttsHandler = Handler(Looper.getMainLooper())
    private val backgroundExecutor: BackgroundTaskExecutor = BackgroundTaskExecutor.getInstance()

    // State management
    private val showProfileSelection = mutableStateOf(true)
    private val showNotification = mutableStateOf(false)
    private val notificationType = mutableStateOf(NotificationType.SUCCESS)
    private val notificationMessage = mutableStateOf("")
    private val showOneButton = mutableStateOf(false)
    private val showTwoButtons = mutableStateOf(false)
    private val showThreeButtons = mutableStateOf(false)
    private val firstButtonLabel = mutableStateOf("OK")
    private val secondButtonLabel = mutableStateOf("Cancel")
    private val thirdButtonLabel = mutableStateOf("Cancel")
    private val firstButtonClick = mutableStateOf({})
    private val secondButtonClick = mutableStateOf({})
    private val thirdButtonClick = mutableStateOf({})
    private val showLoading = mutableStateOf(false)
    private val loadingText = mutableStateOf("")
    private val emailInput = mutableStateOf("example@gmail.com")
    private val passwordInput = mutableStateOf("")
    private val showEmailError = mutableStateOf(false)
    private val showPasswordError = mutableStateOf(false)
    private val fieldTextInteraction = mutableStateOf(false)

    // Loading status members
    private var loadStatus = DBConstants.STATUS_INITIALIZED
    private var loginStatus = DBConstants.STATUS_INITIALIZED
    private var finishedLoading = false

    // Managers
    private val infoNotificationManager = InfoNotificationManager(this)

    // Folder picker launcher
    private lateinit var folderPickerLauncher: ActivityResultLauncher<Uri?>

    // Coming back from TTS language install
    private var waitingForTTSLanguage = false
    private var selectedFolderUri: Uri? = null

    enum class NotificationType {
        SUCCESS, ERROR, NO_INTERNET
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        WindowCompat.setDecorFitsSystemWindows(window, false)

        // Register folder picker
        folderPickerLauncher = registerForActivityResult(
            ActivityResultContracts.OpenDocumentTree()
        ) { uri ->
            uri?.let {
                handleFolderSelection(it)
            }
        }

        setContent {
            LoadProfileScreen(
                infoNotificationManagerValue=infoNotificationManager.isVisibleState.value,
                showProfileSelection = showProfileSelection.value,
                showNotification = showNotification.value,
                notificationType = notificationType.value,
                notificationMessage = notificationMessage.value,
                showOneButton = showOneButton.value,
                showTwoButtons = showTwoButtons.value,
                showThreeButtons = showThreeButtons.value,
                firstButtonLabel = firstButtonLabel.value,
                secondButtonLabel = secondButtonLabel.value,
                thirdButtonLabel = thirdButtonLabel.value,
                firstButtonClick = firstButtonClick.value,
                secondButtonClick = secondButtonClick.value,
                thirdButtonClick = thirdButtonClick.value,
                showLoading = showLoading.value,
                loadingText = loadingText.value,
                emailInput = emailInput.value,
                passwordInput = passwordInput.value,
                showEmailError = showEmailError.value,
                showPasswordError = showPasswordError.value,
                onEmailChange = {
                    emailInput.value = it; showEmailError.value =
                    false; fieldTextInteraction.value = true
                },
                onPasswordChange = { passwordInput.value = it; showPasswordError.value = false },
                onBackClickProfileSelection = ::handleBackFromProfileSelection,
                onLocallyClick = ::handleLocallyClick,
                onHaveAccountClick = ::handleHaveAccountClick,
                onForgotPasswordClick = ::handleForgotPassword,
                onBackClickLoginSection = ::handleBackFromLoginSection,
                onLoginDoneClick = ::handleLoginDone
            )
        }
    }

    override fun onResume() {
        super.onResume()
        // Handle return from TTS language installation
        if (waitingForTTSLanguage) {
            ttsHandler.removeCallbacksAndMessages(null)
            ttsManager.recheckPendingLanguage()
            waitForTTSAndNavigate()
        }
    }

    private fun handleBackFromProfileSelection() {
        // Delete language and new_profile data
        ProfileFileCollection.deleteWelcomeActivity(true)

        // Navigate back to WelcomeActivity with language selection section
        val intent = Intent(this, WelcomeActivity::class.java)
        intent.putExtra(Constants.EXTRA_WELCOME_OPTION, true) // Profile selection
        startActivity(intent)
        finish()
    }

    private fun handleBackFromLoginSection() {
        fieldTextInteraction.value = false
        showNotification.value = false
        mainHandler.postDelayed({
            showProfileSelection.value = true
        }, 100)
    }

    private fun handleLocallyClick() {
        val message = load_infoLoadProfileActivity()
        val butOpt = if (AppConfig.mainLanguage.code == "en") "Browse" else "Răsfoire"
        infoNotificationManager.showNotification(message, {
            infoNotificationManager.hideNotification()
            folderPickerLauncher.launch(null)
        }, butOpt)
    }

    private fun handleHaveAccountClick() {
        // Check network connectivity
        if (!NetworkUtils.isNetworkConnected(this)) {
            showNoInternetNotification()
            return
        }

        showPasswordError.value = false
        showEmailError.value = false
        // Slide to login section
        showProfileSelection.value = false
    }

    private fun handleCreateAccount() {
        Log.d(TAG, "Navigate to create account")

        hideNotification()
        ProfileFileCollection.writeWelcomeActivity(true, null, true)
        val intent = Intent(this, NewProfileActivity::class.java)
        startActivity(intent)
        finish()
    }

    private fun handleForgotPassword() {
        val email = emailInput.value

        if (email.isEmpty() || !fieldTextInteraction.value) {
            showEmailError.value = true
            return
        }

        loadingText.value = load_loadingText(this)
        showLoading.value = true

        // Reset states
        finishedLoading = false
        loginStatus = Constants.LOAD_PROFILE_SUCCESS

        // Launch async task
        backgroundExecutor.executeAsync(
            { performForgotPassword(email) },
            object : BackgroundTaskExecutor.TaskCallback<Int> {
                override fun onSuccess(result: Int) {
                    loginStatus = result
                    finishedLoading = true
                }

                override fun onError(e: Exception) {
                    Log.e(TAG, "Error during forgot password", e)
                    loginStatus = DBConstants.GENERIC_ERROR
                    finishedLoading = true
                }
            }
        )

        // Wait for completion
        waitForForgotPasswordCompletion()
    }

    private fun performForgotPassword(email: String): Int {
        try {
            // Get DB manager
            val dbManager = PhoneStatusMonitor.getInstance().dbManager

            // Reset password (this checks email existence and sends reset)
            return dbManager.resetPassword(email)

        } catch (e: Exception) {
            Log.e(TAG, "Exception in performForgotPassword", e)
            return DBConstants.GENERIC_ERROR
        }
    }

    private fun waitForForgotPasswordCompletion() {
        val checkRunnable = object : Runnable {
            override fun run() {
                if (finishedLoading) {
                    Log.i(TAG, "Load finished, password forgot task")
                    showLoading.value = false
                    handleForgotPasswordResult()
                } else {
                    Log.i(TAG, "Password forgot task isn't finished")
                    mainHandler.postDelayed(this, Constants.LOAD_CHECK_DELAY_MS.toLong())
                }
            }
        }
        mainHandler.post(checkRunnable)
    }

    private fun handleForgotPasswordResult() {
        when (loginStatus) {
            DBConstants.PASSWORD_RESET_SENT -> {
                // Show success notification
                showSuccessNotification(load_passChangedSuccess(emailInput.value))

                // Hide after 5 seconds
                mainHandler.postDelayed({
                    hideNotification()
                }, Constants.SUCCESS_NOTIFICATION_DELAY.toLong())
            }

            DBConstants.INTERNET_CONNECTION_FAILED -> {
                showNoInternetNotification()
                // When OK pressed, navigate to first section
                // This is handled in hideNotification()
            }

            DBConstants.EMAIL_NOT_FOUND -> {
                showLoginErrorNotification(load_accountNE(this))
            }

            else -> {
                // Generic error
                showLoginErrorNotification(load_genericErrorLoad(this))
            }
        }
    }

    private fun handleLoginDone() {
        val email = emailInput.value
        val password = passwordInput.value

        // Reset error states
        showEmailError.value = false
        showPasswordError.value = false

        // Validate fields
        if (email.isEmpty() || !fieldTextInteraction.value) {
            showEmailError.value = true
        }
        if (password.isEmpty()) {
            showPasswordError.value = true
        }

        // If any field is invalid, return
        if (showEmailError.value || showPasswordError.value) {
            return
        }

        // Show loading
        loadingText.value = load_loadingVerifying(this)
        showLoading.value = true

        // Reset states
        finishedLoading = false
        loginStatus = Constants.LOAD_PROFILE_SUCCESS

        // Launch async task
        backgroundExecutor.executeAsync(
            { performLogin(email, password) },
            object : BackgroundTaskExecutor.TaskCallback<Int> {
                override fun onSuccess(result: Int) {
                    loginStatus = result

                    if (result == DBConstants.SYNC_OK) {
                        // Setup TTS on main thread
                        PhoneStatusMonitor.getInstance().modelManager.loadAssets { setTTSLanguage() }
                    } else {
                        // Error case
                        finishedLoading = true
                    }
                }

                override fun onError(e: Exception) {
                    Log.e(TAG, "Error during login", e)
                    loginStatus = DBConstants.GENERIC_ERROR
                    finishedLoading = true
                }
            }
        )

        // Wait for completion
        waitForLoginCompletion()
    }

    private fun performLogin(email: String, password: String): Int {
        try {
            val dbManager = PhoneStatusMonitor.getInstance().dbManager

            // Verify account
            val verifyResult = dbManager.verifyAccount(email, password)
            if (verifyResult != DBConstants.SYNC_OK) {
                return verifyResult  // Return error code
            }

            // Pull profile from server
            val pullResult = dbManager.pullProfile(email)
            if (pullResult.first != DBConstants.SYNC_OK) {
                return pullResult.first  // Return error code
            }

            // Get the pulled profile
            val profileJson = pullResult.second

            // Update last_sync_date to current date
            val currentDate =
                java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault())
                    .format(java.util.Date())
            profileJson.put("last_sync_date", currentDate)

            // Write to profile file
            if (!FileUtils.createProfileDirFile(Constants.PROFILE_FILE_NAME)) {
                return Constants.LOAD_PROFILE_FILE_UPLOAD
            }

            profileJson.put("init", 2)
            if (!ProfileFileCollection.writeProfile(profileJson)) {
                return Constants.LOAD_PROFILE_FILE_UPLOAD
            }

            // Upload profile to AppConfig
            Utils.uploadProfile(profileJson, null)

            return DBConstants.SYNC_OK
        } catch (e: Exception) {
            Log.e(TAG, "Exception in performLogin", e)
            return DBConstants.GENERIC_ERROR
        }
    }

    private fun waitForLoginCompletion() {
        val checkRunnable = object : Runnable {
            override fun run() {
                if (finishedLoading) {
                    Log.i(TAG, "Load finished, login task")
                    showLoading.value = false
                    handleLoginResult()
                } else {
                    Log.i(TAG, "Login task isn't finished")
                    mainHandler.postDelayed(this, Constants.LOAD_CHECK_DELAY_MS.toLong())
                }
            }
        }
        mainHandler.post(checkRunnable)
    }

    private fun handleLoginResult() {
        when (loginStatus) {
            DBConstants.SYNC_OK -> {
                showSuccessNotification(load_profileImportedSuccess())

                // Navigate after 5 seconds
                mainHandler.postDelayed({
                    PhoneStatusMonitor.getInstance().isProfileLoaded(true)

                    val nextActivityClass = if (AppConfig.blindness)
                        BlindHomeActivity::class.java
                    else
                        HomeActivity::class.java

                    val intent = Intent(this, nextActivityClass)
                    startActivity(intent)
                    finish()
                }, Constants.SUCCESS_NOTIFICATION_DELAY.toLong())
            }

            DBConstants.INTERNET_CONNECTION_FAILED -> {
                showNoInternetNotification()
            }

            DBConstants.EMAIL_NOT_FOUND -> {
                showLoginErrorNotification(load_accountNE(this))
            }

            DBConstants.PASSWORD_INCORRECT -> {
                notificationType.value = NotificationType.ERROR
                notificationMessage.value = when (AppConfig.mainLanguage.code) {
                    "en" -> "Incorrect password"
                    "ro" -> "Parola incorectă"
                    else -> "Incorrect password"
                }
                showOneButton.value = true
                showTwoButtons.value = false
                showThreeButtons.value = false
                firstButtonLabel.value = "OK"
                firstButtonClick.value = {
                    hideNotification()
                }
                showNotification.value = true
            }

            DBConstants.DATA_FETCH_ERROR,
            DBConstants.DATA_WRITE_ERROR -> {
                showLoginErrorNotification(load_genericErrorLoad(this))
            }

            else -> {
                // Other errors (asset loading errors, etc.)
                showLoginErrorNotification(load_genericErrorLoad(this))
            }
        }
    }

    private fun handleFolderSelection(uri: Uri) {
        selectedFolderUri = uri

        // Show loading with text
        loadingText.value = load_loadingVerifying(this)
        showLoading.value = true

        // Wait for animation
        mainHandler.postDelayed({
            loadProfileFromFolder(uri)
        }, Constants.ANIMATION_DELAY.toLong())
    }

    private fun loadProfileFromFolder(uri: Uri) {
        finishedLoading = false
        loadStatus = Constants.LOAD_PROFILE_SUCCESS

        // Execute async task
        backgroundExecutor.executeAsync(
            { performProfileLoad(uri) },
            object : BackgroundTaskExecutor.TaskCallback<Int> {
                override fun onSuccess(result: Int) {
                    // OnSuccess - set status
                    loadStatus = result
                    if (result == Constants.LOAD_PROFILE_SUCCESS
                        || result == Constants.LOAD_PROFILE_FILE_HC_UPLOAD_ERROR
                        || result == Constants.LOAD_PROFILE_FILE_ENVR_UPLOAD_ERROR
                    ) {
                        PhoneStatusMonitor.getInstance().modelManager.loadAssets { setTTSLanguage() }
                    } else
                        finishedLoading = true
                }

                override fun onError(e: Exception) {
                    // OnError
                    Log.e(TAG, "Error during profile loading", e)
                    loadStatus = Constants.LOAD_PROFILE_EXCEPTION
                    finishedLoading = true
                }
            }
        )

        // Wait for completion
        waitForLoadingCompletion()
    }

    private fun performProfileLoad(uri: Uri): Int {
        try {
            // Step 1: Check if profile.json exists in selected folder
            val profileFile = findProfileFileInFolder(uri, Constants.PROFILE_FILE_NAME)
            if (profileFile == null) {
                Log.e(TAG, "Profile file not found in selected folder")
                return Constants.LOAD_PROFILE_FILE_MISSING
            }

            // Step 2: Open stream to profile file
            val inputStream = try {
                contentResolver.openInputStream(profileFile)
                    ?: return Constants.LOAD_PROFILE_FILE_STREAMOPEN
            } catch (e: Exception) {
                Log.e(TAG, "Failed to open profile file stream", e)
                return Constants.LOAD_PROFILE_FILE_STREAMOPEN
            }

            // Step 3: Validate profile
            val validationResult: Pair<Int, JSONObject?> = try {
                JSONValidation.validateProfile(inputStream)
            } catch (e: Exception) {
                Log.e(TAG, "Profile validation failed", e)
                return Constants.LOAD_PROFILE_FILE_INVALID
            }

            if (validationResult.first != 0) {
                Log.e(TAG, "Profile validation returned error code: ${validationResult.first}")
                return Constants.LOAD_PROFILE_FILE_INVALID
            }

            val profileJson = validationResult.second
                ?: return Constants.LOAD_PROFILE_FILE_INVALID

            // Step 4: Clear existing profile and write new one
            if (!FileUtils.createProfileDirFile(Constants.PROFILE_FILE_NAME)) {
                return Constants.LOAD_PROFILE_FILE_UPLOAD
            }

            profileJson.put("init", 2)

            if (!ProfileFileCollection.writeProfile(profileJson)) {
                return Constants.LOAD_PROFILE_FILE_UPLOAD
            }

            // Step 5: Handle hash caching if enabled
            if (profileJson.getString("hash_caching") == "heavy" || profileJson.getString("hash_caching") == "light") {
                val hashCacheFile = findProfileFileInFolder(uri, Constants.HASH_CACHE_FILE_NAME)
                if (hashCacheFile != null) {
                    if (!copyFileToAppDirectory(hashCacheFile, Constants.HASH_CACHE_FILE_NAME)) {
                        // Clear hash cache on error
                        if (!FileUtils.createProfileDirFile(Constants.HASH_CACHE_FILE_NAME))
                            return Constants.LOAD_PROFILE_FILE_UPLOAD
                        return Constants.LOAD_PROFILE_FILE_HC_UPLOAD_ERROR
                    }
                }
            }
            if (!profileJson.getBoolean("blindness") && profileJson.getBoolean("env_reports")) {
                val hashCacheFile = findProfileFileInFolder(uri, Constants.ENV_REPORTS_FILE_NAME)
                if (hashCacheFile != null) {
                    if (!copyFileToAppDirectory(hashCacheFile, Constants.ENV_REPORTS_FILE_NAME)) {
                        // Clear hash cache on error
                        if (!FileUtils.createProfileDirFile(Constants.ENV_REPORTS_FILE_NAME))
                            return Constants.LOAD_PROFILE_FILE_UPLOAD
                        return Constants.LOAD_PROFILE_FILE_ENVR_UPLOAD_ERROR
                    }
                }
            }


            // Step 6: Upload phase
            Utils.uploadProfile(profileJson, null)

            return Constants.LOAD_PROFILE_SUCCESS
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error during profile load", e)
            return Constants.LOAD_PROFILE_FILE_UPLOAD
        }
    }

    private fun findProfileFileInFolder(uri: Uri, fileName: String): Uri? {
        try {
            // Use DocumentFile to work with content:// URIs
            val folder = DocumentFile.fromTreeUri(this, uri)

            if (folder == null || !folder.exists() || !folder.isDirectory) {
                Log.e(TAG, "Folder not found or not a directory")
                return null
            }

            // Find profile.json in the folder
            val profileFile = folder.findFile(fileName)

            if (profileFile != null && profileFile.exists() && profileFile.isFile) {
                Log.d(TAG, "Found file \"${fileName}\" in folder: \"${folder.uri}\"")
                return profileFile.uri
            } else {
                Log.d(TAG, "File \"${fileName}\" not found in folder: \"${folder.uri}\"")
                return null
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error finding file \"${fileName}\"", e)
            return null
        }
    }

    private fun copyFileToAppDirectory(sourceUri: Uri, targetFileName: String): Boolean {
        try {
            val targetFile = File(FileUtils.getProfileDirectory(this), targetFileName)

            // Open input stream from URI using ContentResolver
            contentResolver.openInputStream(sourceUri)?.use { inputStream ->
                // Open output stream to target file
                targetFile.outputStream().use { outputStream ->
                    // Copy data
                    inputStream.copyTo(outputStream)
                }
            } ?: run {
                Log.e(TAG, "Failed to open input stream for: $sourceUri")
                return false
            }

            Log.d(TAG, "Successfully copied $targetFileName (${targetFile.length()} bytes)")
            return true

        } catch (e: Exception) {
            Log.e(TAG, "Error copying file: $targetFileName", e)
            return false
        }
    }

    private fun setTTSLanguage() {
        if (AppConfig.mainLanguage.code != ttsManager.currentLocale.language) {
            Log.d(TAG, "TTS is not init on the lang selected")
            waitingForTTSLanguage = true
            ttsManager.changeLanguage(AppConfig.mainLanguage, this)
            waitForTTSAndNavigate()
        } else {
            Log.d(TAG, "TTS is already init, navigating to 2nd section")
            waitForTTSAndNavigate()
        }
    }

    private fun waitForTTSAndNavigate() {
        val checkTTS: Runnable = object : Runnable {
            override fun run() {
                if (ttsManager.isReady) {
                    Log.d(TAG, "TTS is ready, navigating to home")
                    waitingForTTSLanguage = false
                    finishedLoading = true
                } else {
                    Log.w(TAG, "TTS not ready, retrying...")
                    ttsHandler.postDelayed(this, Constants.RETRY_TTS_DELAY_MS.toLong())
                }
            }
        }
        ttsHandler.post(checkTTS)
    }

    private fun waitForLoadingCompletion() {
        val checkRunnable = object : Runnable {
            override fun run() {
                if (finishedLoading) {
                    Log.i(TAG, "Load finished, load locally task")
                    showLoading.value = false
                    handleLoadResult()
                } else {
                    Log.i(TAG, "Main load isn't finished")
                    mainHandler.postDelayed(this, Constants.LOAD_CHECK_DELAY_MS.toLong())
                }
            }
        }
        mainHandler.post(checkRunnable)
    }

    private fun handleLoadResult() {
        when (loadStatus) {
            Constants.LOAD_PROFILE_SUCCESS,
            Constants.LOAD_PROFILE_FILE_ENVR_UPLOAD_ERROR,
            Constants.LOAD_PROFILE_FILE_HC_UPLOAD_ERROR
                -> {
                showSuccessNotification(load_successLocalLoadProfileActivity(loadStatus))

                // Navigate to home after delay
                mainHandler.postDelayed({
                    PhoneStatusMonitor.getInstance().isProfileLoaded(true)

                    val nextActivityClass = if (AppConfig.blindness)
                        BlindHomeActivity::class.java
                    else
                        HomeActivity::class.java
                    val intent = Intent(this, nextActivityClass)
                    startActivity(intent)
                    finish()
                }, Constants.SUCCESS_NOTIFICATION_DELAY.toLong())
            }

            else -> {
                showErrorNotification(loadStatus)
            }
        }
    }

    private fun showSuccessNotification(message: String) {
        notificationType.value = NotificationType.SUCCESS
        notificationMessage.value = message
        showOneButton.value = false
        showTwoButtons.value = false
        showThreeButtons.value = false
        showNotification.value = true
    }

    private fun showErrorNotification(errorCode: Int) {
        notificationType.value = NotificationType.ERROR
        notificationMessage.value = load_errorLocalLoadProfileActivity(errorCode)
        showOneButton.value = false
        showTwoButtons.value = true
        showThreeButtons.value = false
        firstButtonLabel.value = if (AppConfig.mainLanguage.code == "en") "Retry" else "Reîncearcă"
        firstButtonClick.value = { hideNotification() }
        secondButtonLabel.value =
            if (AppConfig.mainLanguage.code == "en") "Create account" else "Creează cont"
        secondButtonClick.value = { handleCreateAccount() }
        showNotification.value = true
    }

    private fun showNoInternetNotification() {
        notificationType.value = NotificationType.NO_INTERNET
        notificationMessage.value = load_noInternet(this)
        showOneButton.value = true
        showTwoButtons.value = false
        showThreeButtons.value = false
        firstButtonLabel.value = "OK"
        firstButtonClick.value = { hideNotification() }
        showNotification.value = true
    }

    private fun showLoginErrorNotification(message: String) {
        notificationType.value = NotificationType.ERROR
        notificationMessage.value = message
        showOneButton.value = false
        showTwoButtons.value = false
        showThreeButtons.value = true
        firstButtonLabel.value = if (AppConfig.mainLanguage.code == "en") "Retry" else "Reîncearcă"
        firstButtonClick.value = { hideNotification() }
        secondButtonLabel.value =
            if (AppConfig.mainLanguage.code == "en") "Create account" else "Creează cont"
        secondButtonClick.value = { handleCreateAccount() }
        thirdButtonLabel.value =
            if (AppConfig.mainLanguage.code == "en") "Load local" else "Încarcă local"
        thirdButtonClick.value = {
            hideNotification()
            mainHandler.postDelayed({
                handleLocallyClick()
            }, 100)
        }
        showNotification.value = true
    }

    private fun hideNotification() {
        showNotification.value = false

        // If we're showing internet error and in login section, navigate to first section
        if (notificationType.value == NotificationType.NO_INTERNET && !showProfileSelection.value) {
            mainHandler.postDelayed({
                showProfileSelection.value = true
            }, 100)
        }
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

    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacksAndMessages(null)
    }
}

@SuppressLint("InflateParams")
@Composable
fun LoadProfileScreen(
    infoNotificationManagerValue: Boolean,
    showProfileSelection: Boolean,
    showNotification: Boolean,
    notificationType: NotificationType,
    notificationMessage: String,
    showOneButton: Boolean,
    showTwoButtons: Boolean,
    showThreeButtons: Boolean,
    firstButtonLabel: String,
    secondButtonLabel: String,
    thirdButtonLabel: String,
    firstButtonClick: () -> Unit,
    secondButtonClick: () -> Unit,
    thirdButtonClick: () -> Unit,
    showLoading: Boolean,
    loadingText: String,
    emailInput: String,
    passwordInput: String,
    showEmailError: Boolean,
    showPasswordError: Boolean,
    onEmailChange: (String) -> Unit,
    onPasswordChange: (String) -> Unit,
    onBackClickProfileSelection: () -> Unit,
    onLocallyClick: () -> Unit,
    onHaveAccountClick: () -> Unit,
    onForgotPasswordClick: () -> Unit,
    onBackClickLoginSection: () -> Unit,
    onLoginDoneClick: () -> Unit
) {
    val blockMainUI=showNotification || showLoading || infoNotificationManagerValue

    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val screenHeight = maxHeight
        // Background image
        Image(
            painter = painterResource(R.drawable.welcome_background),
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
            // Profile Selection Section
            AnimatedVisibility(
                visible = showProfileSelection,
                enter = slideInHorizontally(
                    initialOffsetX = { -it },
                    animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY)
                ),
                exit = slideOutHorizontally(
                    targetOffsetX = { -it },
                    animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY)
                )
            ) {
                ProfileSelectionSection(
                    onLocallyClick = onLocallyClick,
                    onHaveAccountClick = onHaveAccountClick
                )
            }

            // Login Section
            AnimatedVisibility(
                visible = !showProfileSelection,
                enter = slideInHorizontally(
                    initialOffsetX = { it },
                    animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY)
                ),
                exit = slideOutHorizontally(
                    targetOffsetX = { it },
                    animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY)
                )
            ) {
                LoginSection(
                    emailInput = emailInput,
                    passwordInput = passwordInput,
                    showEmailError = showEmailError,
                    showPasswordError = showPasswordError,
                    onEmailChange = onEmailChange,
                    onPasswordChange = onPasswordChange,
                    onForgotPasswordClick = onForgotPasswordClick,
                    onBackClick = onBackClickLoginSection,
                    onDoneClick = onLoginDoneClick
                )
            }
        }

        val bottomSpace = screenHeight * Constants.STD_NAV_MARGIN_BOTTOM
        // Back button for Profile Selection (only visible in that section)
        if (showProfileSelection) {
            Box(
                modifier = Modifier
                    .navigationBarsPadding()
                    .align(Alignment.BottomCenter)
                    .padding(bottom = bottomSpace)
            ) {
                BackArrowLargeFab(onClick = onBackClickProfileSelection)
            }
        }

        BlockingOverlay(blockMainUI)

        // Loading Component Overlay
        LoadingComponent(
            isVisible = showLoading,
            loadingText = loadingText
        )

        // Notification Dialog
        NotificationDialog(
            isVisible = showNotification,
            type = notificationType,
            message = notificationMessage,
            showOneButton = showOneButton,
            showTwoButtons = showTwoButtons,
            showThreeButtons = showThreeButtons,
            firstButtonLabel = firstButtonLabel,
            secondButtonLabel = secondButtonLabel,
            thirdButtonLabel = thirdButtonLabel,
            firstButtonClick = firstButtonClick,
            secondButtonClick = secondButtonClick,
            thirdButtonClick = thirdButtonClick
        )
    }
}

@Composable
fun ProfileSelectionSection(
    onLocallyClick: () -> Unit,
    onHaveAccountClick: () -> Unit
) {
    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenHeight = maxHeight
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.SpaceAround
        ) {
            Box(modifier = Modifier.height(screenHeight * Constants.STD_SUBTITLE_MARGIN_TOP))

            // Title
            Text(
                text = if (AppConfig.mainLanguage.code == "en") "How would you want\nto load the profile?" else "Cum ați vrea\nsă încărcați profilul?",
                fontSize = Constants.STD_SUBTITLE_SIZE.sp,
                color = colorResource(R.color.std_cyan),
                fontFamily = robotoSemibold,
                textAlign = TextAlign.Center,
                lineHeight = 36.sp,
                modifier = Modifier.fillMaxWidth()
            )

            Box(modifier = Modifier.height(screenHeight * Constants.STD_SUBTITLE_BODY_MARGIN_TOP))

            // Buttons row
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.Bottom,
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                ProfileLoadButton(
                    text = if (AppConfig.mainLanguage.code == "en") "Have an account" else "Am un cont",
                    contentDescription = if (AppConfig.mainLanguage.code == "en") "Have an account button" else "Buton am un cont",
                    imageVector = Icons.Filled.AccountCircle, // Replace with actual icon
                    onClick = onHaveAccountClick
                )

                ProfileLoadButton(
                    text = if (AppConfig.mainLanguage.code == "en") "Local profile" else "Profil local",
                    contentDescription = if (AppConfig.mainLanguage.code == "en") "Local profile button" else "Buton profil local",
                    imageVector = Icons.Filled.CloudOff, // Replace with actual icon
                    onClick = onLocallyClick
                )
            }

            Box(modifier = Modifier.height(screenHeight * 0.24f))
        }
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun ProfileLoadButton(
    text: String,
    contentDescription: String,
    imageVector: ImageVector,
    onClick: () -> Unit,
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Button(
            onClick = onClick,
            modifier = Modifier
                .shadow(
                    elevation = 3.dp, shape = MaterialTheme.shapes.large
                )
                .width(144.dp)
                .height(Constants.STD_BUTTON_PAGE_HEIGHT.dp),
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = Color(0xFFEADDFF),        // purple background
                contentColor = Color(0xFF6750A4)
            )
        ) {
            Icon(
                imageVector = imageVector,
                contentDescription = contentDescription,
                tint = Color.Black,
                modifier = Modifier.size(34.dp)
            )
        }

        Text(
            text = text,
            fontSize = Constants.STD_FONT_SIZE.sp,
            color = colorResource(R.color.std_cyan),
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(top = 8.dp),
            fontFamily = robotoSemibold,
        )
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class, ExperimentalLayoutApi::class)
@Composable
fun LoginSection(
    emailInput: String,
    passwordInput: String,
    showEmailError: Boolean,
    showPasswordError: Boolean,
    onEmailChange: (String) -> Unit,
    onPasswordChange: (String) -> Unit,
    onForgotPasswordClick: () -> Unit,
    onBackClick: () -> Unit,
    onDoneClick: () -> Unit
) {

    val keyboardController = LocalSoftwareKeyboardController.current
    val focusManager = LocalFocusManager.current
    val isKeyboardVisible = WindowInsets.isImeVisible

    var fadeComplete by remember { mutableStateOf(!isKeyboardVisible) }

    val titleAlpha by animateFloatAsState(
        targetValue = if (isKeyboardVisible) 0f else 1f,
        animationSpec = tween(
            durationMillis = 150,
            easing = FastOutSlowInEasing
        ),
        finishedListener = { finalValue ->
            // When fade completes, allow slide to start
            if (finalValue == 0f) {
                fadeComplete = true
            }
        },
        label = "Title Fade Animation"
    )

    // Slide animation happens after fade (starts when fadeComplete = true)
    val cardOffsetY by animateDpAsState(
        targetValue = if (isKeyboardVisible && fadeComplete) (-160).dp else 0.dp,
        animationSpec = tween(
            durationMillis = 200,
            delayMillis = if (isKeyboardVisible) 150 else 0, // Delay for fade-out
            easing = FastOutSlowInEasing
        ),
        label = "Card Slide Animation"
    )

    // Reset fadeComplete when keyboard hides
    LaunchedEffect(isKeyboardVisible) {
        if (!isKeyboardVisible) {
            fadeComplete = false
        }
    }

    BoxWithConstraints(
        modifier = Modifier
            .fillMaxSize()
    ) {
        val screenHeight = maxHeight

        Column(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(Unit) {
                    detectTapGestures(onTap = {
                        keyboardController?.hide()
                        focusManager.clearFocus()
                    })
                },
            verticalArrangement = Arrangement.SpaceAround,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(modifier = Modifier.height(screenHeight * Constants.STD_TITLE_MARGIN_TOP))

            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.graphicsLayer {
                    alpha = titleAlpha
                }
            )
            {
                // Title
                Text(
                    text = "VisionAssist\nAccount",
                    fontSize = Constants.STD_TITLE_SIZE.sp,
                    color = colorResource(R.color.std_cyan),
                    fontFamily = robotoLight,
                    letterSpacing = 6.sp,
                    modifier = Modifier.fillMaxWidth(),
                    textAlign = TextAlign.Center,
                    lineHeight = 60.sp
                )

                Box(modifier = Modifier.height(screenHeight * Constants.STD_TITLE_SUBTITLE_MARGIN_TOP))

                // Logo in circle
                Box(
                    modifier = Modifier
                        .size(55.dp)
                        .background(
                            color = colorResource(R.color.std_cyan),
                            shape = RoundedCornerShape(55)
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Image(
                        painter = painterResource(R.drawable.vision_assist_logo),
                        contentDescription = "VisionAssist Logo",
                        modifier = Modifier.size(55.dp)
                    )
                }
                Box(modifier = Modifier.height(screenHeight * 0.01f))
            }

            // Login Card
            Card(
                modifier = Modifier
                    .fillMaxWidth(0.8f)
                    .offset(y = cardOffsetY)
                    .height(Constants.STD_LOGINCARD_WIDTH.dp),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = colorResource(R.color.notification_white)
                ),
                elevation = CardDefaults.cardElevation(defaultElevation = 3.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    // Left section - Email and Password fields (0.7 width)
                    Column(
                        modifier = Modifier
                            .weight(0.75f)
                            .padding(top = 30.dp, start = 20.dp, end = 20.dp)
                    ) {
                        // Email Field
                        Column {
                            Row(
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = "Email",
                                    fontSize = Constants.STD_FONT_SIZE.sp,
                                    color = colorResource(R.color.std_cyan),
                                    fontFamily = robotoSemibold
                                )
                                if (showEmailError) {
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Icon(
                                        imageVector = Icons.Filled.Error,
                                        contentDescription = "Email error",
                                        tint = colorResource(R.color.error_red),
                                        modifier = Modifier.size(16.dp)
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(4.dp))

                            BasicTextField(
                                value = emailInput,
                                onValueChange = onEmailChange,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .background(
                                        color = Color.White,
                                        shape = RoundedCornerShape(10.dp)
                                    )
                                    .border(
                                        width = 1.dp,
                                        color = if (showEmailError)
                                            colorResource(R.color.error_red)
                                        else
                                            Color.LightGray,
                                        shape = RoundedCornerShape(10.dp)
                                    )
                                    .padding(12.dp),
                                textStyle = TextStyle(
                                    fontSize = Constants.STD_FONT_SIZE_LW.sp,
                                    color = colorResource(R.color.std_cyan),
                                    fontFamily = robotoRegular,
                                    letterSpacing = 1.sp
                                ),
                                singleLine = true
                            )
                        }

                        Spacer(modifier = Modifier.height(15.dp))

                        // Password Field
                        Column {
                            Row(
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = if (AppConfig.mainLanguage.code == "en") "Password" else "Parolă",
                                    fontSize = Constants.STD_FONT_SIZE.sp,
                                    color = colorResource(R.color.std_cyan),
                                    fontFamily = robotoSemibold
                                )
                                if (showPasswordError) {
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Icon(
                                        imageVector = Icons.Filled.Error,
                                        contentDescription = "Password error",
                                        tint = colorResource(R.color.error_red),
                                        modifier = Modifier.size(16.dp)
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(4.dp))

                            var passwordVisible by remember { mutableStateOf(false) }

                            BasicTextField(
                                value = passwordInput,
                                onValueChange = onPasswordChange,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .background(
                                        color = Color.White,
                                        shape = RoundedCornerShape(10.dp)
                                    )
                                    .border(
                                        width = 1.dp,
                                        color = if (showEmailError)
                                            colorResource(R.color.error_red)
                                        else
                                            Color.LightGray,
                                        shape = RoundedCornerShape(10.dp)
                                    )
                                    .padding(12.dp),
                                textStyle = TextStyle(
                                    fontSize = Constants.STD_FONT_SIZE_LW.sp,
                                    color = colorResource(R.color.std_cyan),
                                    fontFamily = robotoRegular,
                                    letterSpacing = 1.sp
                                ),
                                visualTransformation = if (passwordVisible)
                                    VisualTransformation.None
                                else
                                    PasswordVisualTransformation(),
                                keyboardActions = KeyboardActions(
                                    onDone = {
                                        keyboardController?.hide()
                                        focusManager.clearFocus()
                                    }
                                ),
                                singleLine = true
                            )

                            IconButton(
                                onClick = { passwordVisible = !passwordVisible },
                                modifier = Modifier.align(Alignment.CenterHorizontally)
                            ) {
                                Icon(
                                    modifier = Modifier.size(35.dp),
                                    imageVector = if (passwordVisible)
                                        Icons.Filled.Visibility
                                    else
                                        Icons.Filled.VisibilityOff,
                                    contentDescription = if (passwordVisible)
                                        if (AppConfig.mainLanguage.code == "en") "Hide password" else "Ascunde parola"
                                    else
                                        if (AppConfig.mainLanguage.code == "en") "Show password" else "Arată parola",
                                    tint = colorResource(R.color.std_purple)
                                )
                            }
                        }

                        // Forgot Password Button
                        TextButton(
                            onClick = onForgotPasswordClick,
                            modifier = Modifier.align(Alignment.CenterHorizontally)
                        ) {
                            Text(
                                text = if (AppConfig.mainLanguage.code == "en") "Forgot password?" else "Ați uitat parola?",
                                fontSize = Constants.STD_FONT_SIZE_LW.sp,
                                color = colorResource(R.color.std_cyan)
                            )
                        }
                    }

                    // Right section - Back and Done buttons (0.3 width)
                    Column(
                        modifier = Modifier
                            .weight(0.25f)
                            .fillMaxHeight()
                    ) {
                        // Back Button (top half)
                        Button(
                            onClick = onBackClick,
                            modifier = Modifier
                                .fillMaxWidth()
                                .weight(0.5f),
                            shape = RoundedCornerShape(
                                topStart = 0.dp,
                                topEnd = 16.dp,
                                bottomEnd = 0.dp,
                                bottomStart = 0.dp
                            ),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = colorResource(R.color.notification_button_white),
                                contentColor = colorResource(R.color.std_cyan)
                            )
                        ) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                contentDescription = "Back",
                                modifier = Modifier.size(26.dp)
                            )
                        }

                        // Done Button (bottom half)
                        Button(
                            onClick = onDoneClick,
                            modifier = Modifier
                                .fillMaxWidth()
                                .weight(0.5f),
                            shape = RoundedCornerShape(
                                topStart = 0.dp,
                                topEnd = 0.dp,
                                bottomEnd = 16.dp,
                                bottomStart = 0.dp
                            ),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = colorResource(R.color.std_cyan),
                                contentColor = colorResource(R.color.notification_button_white)
                            )
                        ) {
                            Icon(
                                imageVector = Icons.Default.Check,
                                contentDescription = "Done",
                                modifier = Modifier.size(24.dp)
                            )
                        }
                    }
                }
            }
            Spacer(modifier = Modifier.weight(1f))
        }
    }
}

@Preview(
    name = "Load Profile Activity/LoadProfileScreenSection",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
@Composable
fun LoadProfileActivity1Preview() {
    MaterialTheme {
        Image(
            painter = painterResource(id = R.drawable.welcome_background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        ProfileSelectionSection(
            onLocallyClick = {},
            onHaveAccountClick = {}
        )
    }
}

@Preview(
    name = "Load Profile Activity/LoginSection",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
@Composable
fun LoadProfileActivity2Preview() {
    MaterialTheme {
        Image(
            painter = painterResource(id = R.drawable.welcome_background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        LoginSection(
            emailInput = "example@gmail.com",
            passwordInput = "",
            showEmailError = false,
            showPasswordError = false,
            onEmailChange = {},
            onPasswordChange = {},
            onForgotPasswordClick = {},
            onBackClick = {},
            onDoneClick = {}
        )
    }
}

@Preview(name = "Load Profile Activity", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun LoadProfileActivityPreview() {
    LoadProfileScreen(
        showProfileSelection = true,
        showNotification = false,
        notificationMessage = "",
        notificationType = NotificationType.SUCCESS,
        showOneButton = true,
        showTwoButtons = false,
        showThreeButtons = false,
        showLoading = false,
        firstButtonLabel = "OK",
        secondButtonLabel = "",
        thirdButtonLabel = "",
        firstButtonClick = {},
        secondButtonClick = {},
        thirdButtonClick = {},
        loadingText = "",
        emailInput = "example@gmail.com",
        passwordInput = "",
        showEmailError = false,
        showPasswordError = false,
        onEmailChange = {},
        onPasswordChange = {},
        onBackClickProfileSelection = {},
        onLocallyClick = {},
        onHaveAccountClick = {},
        onForgotPasswordClick = {},
        onBackClickLoginSection = {},
        onLoginDoneClick = {},
        infoNotificationManagerValue = false
    )
}
/*
Exec flow:
    -main thread:launch async task
        -thread1:check for profile.json existence->open stream to it->run validateProfile
                    ->copy the profile.json->copy the aux files
                    ->uploadProfile->loadModels
        -main thread: changeLanguage of TTS(+handle return from settings)->set task finished
        -main thread: check check if the task is finished
            if it is not valid, display the notification error message, and return to the main page
            if its valid exit code, display the notification success message, after 5 sec
             setProfileLoaded, and navigate
 */
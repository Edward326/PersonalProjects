@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.newprofile

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
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.ExceptionVisionAssist
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.main.BlindHomeActivity
import com.visionassist.appspace.activities.main.HomeActivity
import com.visionassist.appspace.activities.newprofile.jsonCollection.ProfileFileCollection
import com.visionassist.appspace.database.DBConstants
import com.visionassist.appspace.jetpack.design.BackArrowLargeFab
import com.visionassist.appspace.jetpack.design.HashCacheSelector
import com.visionassist.appspace.jetpack.design.LoadingComponent
import com.visionassist.appspace.jetpack.design.NextArrowLargeFab
import com.visionassist.appspace.jetpack.managers.ErrorDialogManager
import com.visionassist.appspace.jetpack.managers.InfoNotificationManager
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.BackgroundTaskExecutor
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.FileUtils
import com.visionassist.appspace.utils.Utils
import com.visionassist.appspace.utils.load_envReportsInfo
import com.visionassist.appspace.utils.load_envReportsTitle
import com.visionassist.appspace.utils.load_hashCacheInfoFirst
import com.visionassist.appspace.utils.load_hashCacheInfoSecond
import com.visionassist.appspace.utils.load_hashCacheTitle
import com.visionassist.appspace.utils.load_loadingUploading
import com.visionassist.appspace.utils.robotoSemibold
import org.json.JSONException
import org.json.JSONObject

class UserHashCachingActivity : ComponentActivity() {
    private val TAG = "UserHashCachingActivity"

    private val monitor = PhoneStatusMonitor.getInstance()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val infoNotificationManager = InfoNotificationManager(this)
    private val backgroundExecutor: BackgroundTaskExecutor = BackgroundTaskExecutor.getInstance()

    // State for sections
    private val currentSection = mutableIntStateOf(1) // 1 or 2

    // Section 1 - Hash Cache
    private val hashCacheOption = mutableStateOf("Don't use")
    private val notificationStep = mutableIntStateOf(0) // 0 = none, 1 = first, 2 = second

    // Section 2 - Environment Reports
    private val envReportsEnabled = mutableStateOf(false)

    // Loading state
    private val isLoading = mutableStateOf(false)
    private val loadingText = mutableStateOf("Please wait")

    // Background task variables
    private var isFinished = false
    private var loadStatus = -1
    private var assetLoadError = -494

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val option = intent.getIntExtra(Constants.EXTRA_HCACHING_OPTION, 1)
        currentSection.intValue = option

        setContent {
            UserHashCachingScreen(
                currentSection = currentSection.intValue,
                // Section 1
                hashCacheOption = hashCacheOption.value,
                onHashCacheOptionSelected = ::handleHashCacheOptionSelected,
                onHashCacheInfoClick = ::handleHashCacheInfoClick,
                // Section 2
                envReportsEnabled = envReportsEnabled.value,
                onEnvReportsToggle = ::handleEnvReportsToggle,
                onEnvReportsInfoClick = ::handleEnvReportsInfoClick,
                // Navigation
                onBackClick = ::handleBackClick,
                onNextClick = ::handleNextClick,
                // Loading
                isLoading = isLoading.value,
                loadingText = loadingText.value
            )
        }
    }

    // Section 1 Handlers
    private fun handleHashCacheOptionSelected(option: String) {
        hashCacheOption.value = option
        Log.d(TAG, "Hash cache option selected: $option")
    }

    private fun handleHashCacheInfoClick() {
        when (notificationStep.intValue) {
            0 -> {
                // Show first notification
                notificationStep.intValue = 1
                val message = load_hashCacheInfoFirst(this)
                infoNotificationManager.showNotification(message, {
                    infoNotificationManager.hideNotification()
                    showSecondNotification()
                }, if (AppConfig.mainLanguage.code == "en") "Next" else "Următorul")
            }
        }
    }

    private fun showSecondNotification() {
        notificationStep.intValue = 2
        val message = load_hashCacheInfoSecond(this)
        infoNotificationManager.showNotification(message, {
            infoNotificationManager.hideNotification()
            notificationStep.intValue = 0 // Reset
        }, "OK")
    }

    // Section 2 Handlers
    private fun handleEnvReportsToggle(enabled: Boolean) {
        envReportsEnabled.value = enabled
        Log.d(TAG, "Environment reports enabled: $enabled")
    }

    private fun handleEnvReportsInfoClick() {
        val message = load_envReportsInfo(this)
        infoNotificationManager.showNotification(message, {
            infoNotificationManager.hideNotification()
        }, "OK")
    }

    // Navigation Handlers
    private fun handleBackClick() {
        when (currentSection.intValue) {
            1 -> {
                // Section 1 - Go back to previous activity
                if (!AppConfig.blindness) {
                    // Not blind - go back to UserAccessibility1Activity section 2
                    ProfileFileCollection.deleteUserAccessibility1Activity(true)
                    val intent = Intent(this, UserAccessibility1Activity::class.java)
                    intent.putExtra(Constants.EXTRA_USERACC_OPTION, 2)
                    startActivity(intent)
                    finish()
                } else {
                    // Blind - go back to UserInfoE3Activity
                    ProfileFileCollection.deleteUserInfoE3Activity()
                    AppConfig.tts_pitch = Constants.TTS_PITCH
                    AppConfig.tts_speech_rate = Constants.TTS_SPEECH_RATE
                    val intent = Intent(this, UserInfoE3Activity::class.java)
                    startActivity(intent)
                    finish()
                }
            }

            2 -> {
                // Section 2 - Go back to section 1
                ProfileFileCollection.deleteUserHashCachingActivity()
                currentSection.intValue = 1
            }
        }
    }

    private fun handleNextClick() {
        when (currentSection.intValue) {
            1 -> {
                // Section 1 - Save hash cache option
                val option = hashCacheOption.value

                // Check if blind user
                if (AppConfig.blindness) {
                    // BLIND USERS: Skip Section 2, load assets directly

                    // Create hash cache file if Light or Heavy
                    if (option == "Light" || option == "Heavy") {
                        FileUtils.createProfileDirFile(Constants.HASH_CACHE_FILE_NAME)
                    }

                    // Write hash_caching to profile
                    ProfileFileCollection.writeUserHashCachingActivity(option, false)
                    AppConfig.hash_caching = option

                    // Start loading assets immediately
                    startLoadingAssets()
                } else {
                    // NON-BLIND USERS: Move to Section 2

                    // Create hash cache file if Light or Heavy
                    if (option == "Light" || option == "Heavy") {
                        FileUtils.createProfileDirFile(Constants.HASH_CACHE_FILE_NAME)
                    }

                    // Write to profile
                    ProfileFileCollection.writeUserHashCachingActivity(option, false)
                    AppConfig.hash_caching = option

                    // Navigate to section 2
                    currentSection.intValue = 2
                }
            }

            2 -> {
                // Section 2 - Save env reports and load assets (NON-BLIND ONLY)
                val enabled = envReportsEnabled.value

                // Create env reports file if enabled
                if (enabled) {
                    FileUtils.createProfileDirFile(Constants.ENV_REPORTS_FILE_NAME)
                }

                ProfileFileCollection.writeUserHashCachingActivity(enabled.toString(), true)
                AppConfig.env_reports = enabled

                // Start loading assets
                startLoadingAssets()
            }
        }
    }

    private fun startLoadingAssets() {
        // Show loading
        loadingText.value = load_loadingUploading(this)
        isLoading.value = true

        // Reset background task variables
        isFinished = false
        loadStatus = -1

        // Start background task
        backgroundExecutor.executeAsync(
            BackgroundTaskExecutor.BackgroundTask {
                // Load profile JSON and update AppConfig
                val profileFile = FileUtils.getProfileFile(this@UserHashCachingActivity)
                val profileContent = profileFile.readText()
                val jsonObject = JSONObject(profileContent)
                Utils.uploadProfile(jsonObject,null)

                val loadModelsBoolean=loadAllAssets()
                if (!loadModelsBoolean) {
                    return@BackgroundTask assetLoadError
                }

                // Sync profile to remote if needed
                if (jsonObject.getBoolean( "remote")) {
                    val dbManager = monitor.dbManager
                    dbManager.syncProfile(jsonObject)
                    return@BackgroundTask dbManager.status
                }

                return@BackgroundTask 0 // Success
            },
            object : BackgroundTaskExecutor.TaskCallback<Int> {
                override fun onSuccess(result: Int) {
                    loadStatus = result
                    isFinished = true
                }

                override fun onError(e: Exception) {
                    if (e is JSONException) {
                        handleProfileError(
                            ExceptionVisionAssist(Constants.JSON_PARSE_ERROR, null)
                        )
                    } else {
                        handleProfileError(e)
                    }
                }
            }
        )

        // Wait for background task to finish
        waitForAssetLoading()
    }

    private fun waitForAssetLoading() {
        mainHandler.postDelayed({
            if (isFinished) {
                // Check load status
                when (loadStatus) {
                    DBConstants.INTERNET_CONNECTION_FAILED, DBConstants.SYNC_OK, DBConstants.SYNC_ERROR , 0 -> {
                        // Success
                        monitor.profileLoaded = true
                        isLoading.value = false

                        // Navigate based on blindness
                        val intent = if (AppConfig.blindness) {
                            Intent(this, BlindHomeActivity::class.java)
                        } else {
                            Intent(this, HomeActivity::class.java)
                        }
                        startActivity(intent)
                        finish()
                    }
                    else -> {
                        // Asset error or other error
                        handleProfileError(
                            ExceptionVisionAssist(loadStatus, null)
                        )
                    }
                }
            } else {
                // Continue waiting
                waitForAssetLoading()
            }
        }, 500)
    }

    private fun loadAllAssets(): Boolean {
        // Similar to MainActivity asset loading
        val modelsLoaded = intArrayOf(0)
        val loadLock = Object()

        try {
            /*
            BackgroundTaskExecutor.getInstance().executeAsync(
                BackgroundTaskExecutor.BackgroundTask {
                    // Load detector
                    return@BackgroundTask 0
                },
                object : BackgroundTaskExecutor.TaskCallback<Int> {
                    override fun onSuccess(result: Int) {
                        if (result == -1) {
                            assetLoadError = Constants.DETECTOR_LOAD_ERROR
                        } else {
                            synchronized(loadLock) {
                                modelsLoaded[0]++
                                loadLock.notifyAll()
                            }
                        }
                    }

                    override fun onError(e: Exception) {
                        assetLoadError = Constants.DETECTOR_LOAD_ERROR
                        synchronized(loadLock) {
                            loadLock.notifyAll()
                        }
                    }
                }
            )

            BackgroundTaskExecutor.getInstance().executeAsync(
                BackgroundTaskExecutor.BackgroundTask {
                    // Load captioner
                    return@BackgroundTask 0
                },
                object : BackgroundTaskExecutor.TaskCallback<Int> {
                    override fun onSuccess(result: Int) {
                        if (result == -1) {
                            assetLoadError = Constants.CAPTIONER_LOAD_ERROR
                        } else {
                            synchronized(loadLock) {
                                modelsLoaded[0]++
                                loadLock.notifyAll()
                            }
                        }
                    }

                    override fun onError(e: Exception) {
                        assetLoadError = Constants.CAPTIONER_LOAD_ERROR
                        synchronized(loadLock) {
                            loadLock.notifyAll()
                        }
                    }
                }
            )

            BackgroundTaskExecutor.getInstance().executeAsync(
                BackgroundTaskExecutor.BackgroundTask {
                    // Load translator
                    return@BackgroundTask 0
                },
                object : BackgroundTaskExecutor.TaskCallback<Int> {
                    override fun onSuccess(result: Int) {
                        if (result == -1) {
                            assetLoadError = Constants.TRANSLATER_LOAD_ERROR
                        } else {
                            synchronized(loadLock) {
                                modelsLoaded[0]++
                                loadLock.notifyAll()
                            }
                        }
                    }

                    override fun onError(e: Exception) {
                        assetLoadError = Constants.TRANSLATER_LOAD_ERROR
                        synchronized(loadLock) {
                            loadLock.notifyAll()
                        }
                    }
                }
            )

            BackgroundTaskExecutor.getInstance().executeAsync(
                BackgroundTaskExecutor.BackgroundTask {
                    // Load classifier
                    return@BackgroundTask 0
                },
                object : BackgroundTaskExecutor.TaskCallback<Int> {
                    override fun onSuccess(result: Int) {
                        if (result == -1) {
                            assetLoadError = Constants.CLASSIFIER_LOAD_ERROR
                        } else {
                            synchronized(loadLock) {
                                modelsLoaded[0]++
                                loadLock.notifyAll()
                            }
                        }
                    }

                    override fun onError(e: Exception) {
                        assetLoadError = Constants.CLASSIFIER_LOAD_ERROR
                        synchronized(loadLock) {
                            loadLock.notifyAll()
                        }
                    }
                }
            )

            BackgroundTaskExecutor.getInstance().executeAsync(
                BackgroundTaskExecutor.BackgroundTask {
                    // Load speech to text
                    return@BackgroundTask 0
                },
                object : BackgroundTaskExecutor.TaskCallback<Int> {
                    override fun onSuccess(result: Int) {
                        if (result == -1) {
                            assetLoadError = Constants.STT_LOAD_ERROR
                        } else {
                            synchronized(loadLock) {
                                modelsLoaded[0]++
                                loadLock.notifyAll()
                            }
                        }
                    }

                    override fun onError(e: Exception) {
                        assetLoadError = Constants.STT_LOAD_ERROR
                        synchronized(loadLock) {
                            loadLock.notifyAll()
                        }
                    }
                }
            )

            BackgroundTaskExecutor.getInstance().executeAsync(
                BackgroundTaskExecutor.BackgroundTask {
                    //load yolo's class names
                    // useless words
                    // synonyms
                    // classifier scene names
                    return@BackgroundTask 0
                },
                object : BackgroundTaskExecutor.TaskCallback<Int> {
                    override fun onSuccess(result: Int) {
                        if (result == -1) {
                            assetLoadError = Constants.ASSETS_ERROR
                        } else {
                            synchronized(loadLock) {
                                modelsLoaded[0]++
                                loadLock.notifyAll()
                            }
                        }
                    }

                    override fun onError(e: Exception) {
                        assetLoadError = Constants.ASSETS_ERROR
                        synchronized(loadLock) {
                            loadLock.notifyAll()
                        }
                    }
                }
            )

            BackgroundTaskExecutor.getInstance().executeAsync(
                BackgroundTaskExecutor.BackgroundTask {
                    //load captioner vocab
                    return@BackgroundTask 0
                },
                object : BackgroundTaskExecutor.TaskCallback<Int> {
                    override fun onSuccess(result: Int) {
                        if (result == -1) {
                            assetLoadError = Constants.ASSETS_ERROR
                        } else {
                            synchronized(loadLock) {
                                modelsLoaded[0]++
                                loadLock.notifyAll()
                            }
                        }
                    }

                    override fun onError(e: Exception) {
                        assetLoadError = Constants.ASSETS_ERROR
                        synchronized(loadLock) {
                            loadLock.notifyAll()
                        }
                    }
                }
            )
             */

            BackgroundTaskExecutor.getInstance().executeAsync(
                BackgroundTaskExecutor.BackgroundTask {
                    // Load translator
                    Log.d(TAG, "Simulate loading models")
                    Thread.sleep(3000)
                    return@BackgroundTask 0
                },
                object : BackgroundTaskExecutor.TaskCallback<Int> {
                    override fun onSuccess(result: Int) {
                        if (result == -1) {
                            assetLoadError = Constants.TRANSLATER_LOAD_ERROR
                        } else {
                            synchronized(loadLock) {
                                modelsLoaded[0]= Constants.MODELS_COUNT + Constants.MODELS_OWN_ASSETS_COUNT
                                loadLock.notifyAll()
                            }
                        }
                    }

                    override fun onError(e: Exception) {
                        assetLoadError = Constants.TRANSLATER_LOAD_ERROR
                        synchronized(loadLock) {
                            loadLock.notifyAll()
                        }
                    }
                }
            )

            // Wait for all models to load (synchronous wait in background thread)
            synchronized(loadLock) {
                while (modelsLoaded[0] < Constants.MODELS_COUNT + Constants.MODELS_OWN_ASSETS_COUNT) {
                    // Check if error occurred
                    if (assetLoadError != -494) {
                        Log.e(TAG, "Asset loading failed with error code: $assetLoadError")
                        return false
                    }

                    // Wait without timeout - this is safe because we're in a background thread
                    loadLock.wait()
                }
            }

            Log.d(TAG, "All assets loaded successfully")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Error loading assets", e)
            assetLoadError = Constants.ASSETS_ERROR
            return false
        }
    }

    private fun handleProfileError(e: Exception) {
        if (e is ExceptionVisionAssist) {
            val errorCode = e.errorCode
            Log.e(TAG, "Thrown special exception, error code: $errorCode")

            val errorDialog = ErrorDialogManager(monitor.currentActivity)
            errorDialog.setupDialog(errorCode)
            isLoading.value = false
            monitor.shutdownApp(errorDialog, monitor.currentContext)
        } else {
            Log.e(TAG, "Thrown exception, explanation: ", e)
            val errorDialog = ErrorDialogManager(monitor.currentActivity)
            errorDialog.setupDialog(Constants.EXCEPTION_CLASS_ERROR)
            monitor.shutdownApp(errorDialog, monitor.currentContext)
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

    override fun onPause() {
        super.onPause()
        mainHandler.removeCallbacksAndMessages(null)
    }

    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacksAndMessages(null)
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun UserHashCachingScreen(
    currentSection: Int,
    hashCacheOption: String,
    onHashCacheOptionSelected: (String) -> Unit,
    onHashCacheInfoClick: () -> Unit,
    envReportsEnabled: Boolean,
    onEnvReportsToggle: (Boolean) -> Unit,
    onEnvReportsInfoClick: () -> Unit,
    onBackClick: () -> Unit,
    onNextClick: () -> Unit,
    isLoading: Boolean,
    loadingText: String
) {
    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenHeight = maxHeight

        // Background image
        Image(
            painter = painterResource(R.drawable.welcome_background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        // Content based on section
        when (currentSection) {
            1 -> Section1Content(
                screenHeight = screenHeight,
                hashCacheOption = hashCacheOption,
                onHashCacheOptionSelected = onHashCacheOptionSelected,
                onHashCacheInfoClick = onHashCacheInfoClick,
            )

            2 -> Section2Content(
                screenHeight = screenHeight,
                envReportsEnabled = envReportsEnabled,
                onEnvReportsToggle = onEnvReportsToggle,
                onEnvReportsInfoClick = onEnvReportsInfoClick,
            )
        }

        // Loading overlay
        LoadingComponent(
            isVisible = isLoading,
            loadingText = loadingText
        )

        val bottomSpace = screenHeight * Constants.STD_NAV_MARGIN_BOTTOM
        // Navigation Buttons
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = bottomSpace),
            horizontalArrangement = Arrangement.spacedBy(30.dp)
        ) {
            BackArrowLargeFab(
                onClick = onBackClick
            )
            NextArrowLargeFab(
                onClick = onNextClick
            )
        }
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun Section1Content(
    screenHeight: Dp,
    hashCacheOption: String,
    onHashCacheOptionSelected: (String) -> Unit,
    onHashCacheInfoClick: () -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(modifier = Modifier.height(screenHeight * Constants.STD_SUBTITLE_MARGIN_TOP))

        // Title
        Text(
            text = load_hashCacheTitle(LocalContext.current),
            fontSize = Constants.STD_SUBTITLE_SIZE.sp,
            color = colorResource(R.color.std_cyan),
            fontFamily = robotoSemibold,
            textAlign = TextAlign.Center,
            lineHeight = 36.sp,
            modifier = Modifier.fillMaxWidth()
        )

        Box(modifier = Modifier.height(screenHeight * Constants.STD_TITLE_SUBTITLE_MARGIN_TOP))

        // Hash Cache Selector with Info Button
        Row(
            modifier = Modifier
                .fillMaxWidth(0.8f).padding(start = 20.dp),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.Bottom
        ) {
            HashCacheSelector(
                selectedOption = hashCacheOption,
                availableOptions = listOf("Don't use", "Light", "Heavy"),
                onOptionSelected = onHashCacheOptionSelected
            )
            Spacer(modifier = Modifier.size(5.dp))

            // Info Button
            IconButton(
                onClick = onHashCacheInfoClick,
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

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun Section2Content(
    screenHeight: Dp,
    envReportsEnabled: Boolean,
    onEnvReportsToggle: (Boolean) -> Unit,
    onEnvReportsInfoClick: () -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(modifier = Modifier.height(screenHeight * 0.68f))

        // Switch with Info Button
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.Bottom
        ) {
            Text(
                text = load_envReportsTitle(LocalContext.current),
                fontSize = Constants.STD_SUBTITLE_SIZE.sp,
                color = colorResource(R.color.std_cyan),
                fontFamily = robotoSemibold,
                textAlign = TextAlign.Start,
                modifier = Modifier
                    .padding(start = 22.dp)
            )

            Switch(
                checked = envReportsEnabled,
                onCheckedChange = onEnvReportsToggle,
                colors = SwitchDefaults.colors(
                    checkedThumbColor = Color.White,
                    checkedTrackColor = colorResource(R.color.std_purple),
                )
            )

            // Info Button
            IconButton(
                onClick = onEnvReportsInfoClick,
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

@Preview(name = "UserHashCaching Section 1", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun UserHashCachingSection1Preview() {
    UserHashCachingScreen(
        currentSection = 1,
        hashCacheOption = "Don't use",
        onHashCacheOptionSelected = {},
        onHashCacheInfoClick = {},
        envReportsEnabled = false,
        onEnvReportsToggle = {},
        onEnvReportsInfoClick = {},
        onBackClick = {},
        onNextClick = {},
        isLoading = false,
        loadingText = "Please wait"
    )
}

@Preview(name = "UserHashCaching Section 2", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun UserHashCachingSection2Preview() {
    UserHashCachingScreen(
        currentSection = 2,
        hashCacheOption = "Light",
        onHashCacheOptionSelected = {},
        onHashCacheInfoClick = {},
        envReportsEnabled = true,
        onEnvReportsToggle = {},
        onEnvReportsInfoClick = {},
        onBackClick = {},
        onNextClick = {},
        isLoading = false,
        loadingText = "Please wait"
    )
}
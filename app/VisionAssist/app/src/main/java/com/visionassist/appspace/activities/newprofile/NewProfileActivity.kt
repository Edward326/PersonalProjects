package com.visionassist.appspace.activities.newprofile

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.KeyEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AccountCircle
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CloudOff
import androidx.compose.material.icons.filled.CloudSync
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.newprofile.jsonCollection.ProfileFileCollection
import com.visionassist.appspace.database.DBConstants
import com.visionassist.appspace.database.NetworkUtils
import com.visionassist.appspace.jetpack.design.LoadingComponent
import com.visionassist.appspace.jetpack.design.NavigationButtons
import com.visionassist.appspace.jetpack.managers.InfoNotificationManager
import com.visionassist.appspace.models.ttsengine.TTSManager
import com.visionassist.appspace.utils.BackgroundTaskExecutor
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.load_loadingVerifying
import com.visionassist.appspace.utils.robotoLight
import com.visionassist.appspace.utils.robotoRegular
import com.visionassist.appspace.utils.robotoSemibold
import java.security.MessageDigest

class NewProfileActivity : ComponentActivity() {
    private val TAG = "NewProfileActivity"

    private var ttsManager: TTSManager = PhoneStatusMonitor.getInstance().ttsManager
    private val mainHandler = Handler(Looper.getMainLooper())

    // State management
    private val showRegisterSection = mutableStateOf(false)
    private val switchEnabled = mutableStateOf(true)
    private val switchChecked = mutableStateOf(false)

    // Email and Password states
    private val emailInput = mutableStateOf("")
    private val passwordInput = mutableStateOf("")
    private val showEmailError = mutableStateOf(false)
    private val showPasswordError = mutableStateOf(false)

    // Loading states
    private val showLoading = mutableStateOf(false)
    private val loadingText = mutableStateOf("")

    // Registration status
    private var registerStatus = DBConstants.STATUS_INITIALIZED
    private var finishedRegistering = false

    // Managers
    private lateinit var infoNotificationManager: InfoNotificationManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize info notification manager
        infoNotificationManager = InfoNotificationManager(this)

        setContent {
            NewProfileScreen(
                showRegisterSection = showRegisterSection.value,
                switchEnabled = switchEnabled.value,
                switchChecked = switchChecked.value,
                emailInput = emailInput.value,
                passwordInput = passwordInput.value,
                showEmailError = showEmailError.value,
                showPasswordError = showPasswordError.value,
                showLoading = showLoading.value,
                loadingText = loadingText.value,
                onSwitchChanged = ::handleSwitchChanged,
                onEmailChange = ::handleEmailChange,
                onPasswordChange = ::handlePasswordChange,
                onBackFromProfileSelection = ::handleBackFromProfileSelection,
                onNextFromProfileSelection = ::handleNextFromProfileSelection,
                onBackFromRegister = ::handleBackFromRegister,
                onDoneFromRegister = ::handleDoneFromRegister
            )
        }
    }

    private fun handleSwitchChanged(checked: Boolean) {
        if (checked) {
            // Check network connectivity
            if (!NetworkUtils.isNetworkConnected(this)) {
                // Show network error notification
                val message = "The device has no access to the internet, try to connect to a network and try again"
                infoNotificationManager.showNotification(message, Runnable {
                    // Reset switch
                    switchChecked.value = false
                }, true)
            } else {
                // Network available, slide to register section
                switchChecked.value = true
                showRegisterSection.value = true
            }
        }
    }

    private fun handleEmailChange(newEmail: String) {
        emailInput.value = newEmail
        showEmailError.value = false
    }

    private fun handlePasswordChange(newPassword: String) {
        passwordInput.value = newPassword
        showPasswordError.value = false
    }

    private fun handleBackFromProfileSelection() {
        // Delete language and new_profile data
        ProfileFileCollection.welcomeActivityDelete(true)

        // Navigate back to WelcomeActivity with profile selection section
        val intent = Intent(this, WelcomeActivity::class.java)
        intent.putExtra(Constants.EXTRA_WELCOME_OPTION, false) // Profile selection
        startActivity(intent)
        finish()
    }

    private fun handleNextFromProfileSelection() {
        cancelAllHandlers()
        // Write to profile with remote = false
        ProfileFileCollection.newProfileActivityWrite(false, null, null)

        // Navigate to UserInfoActivity
        val intent = Intent(this, UserInfoActivity::class.java)
        startActivity(intent)
        finish()
    }

    private fun handleBackFromRegister() {
        // Slide back to profile selection section
        showRegisterSection.value = false
        switchChecked.value = false
    }

    private fun handleDoneFromRegister() {
        // Validate fields
        val email = emailInput.value.trim()
        val password = passwordInput.value.trim()

        if (email.isEmpty() || email == "example@gmail.com") {
            showEmailError.value = true
            return
        }

        if (password.isEmpty()) {
            showPasswordError.value = true
            return
        }

        // Start registration process
        startRegistration(email, password)
    }

    private fun startRegistration(email: String, password: String) {
        // Show loading
        if (PhoneStatusMonitor.getInstance().profileLoaded) {
            loadingText.value = load_loadingVerifying(this)
        } else {
            loadingText.value = "Creating account..."
        }
        showLoading.value = true

        // Reset registration status
        finishedRegistering = false
        registerStatus = DBConstants.STATUS_INITIALIZED

        // Hash the password
        val passwordHash = hashPassword(password)

        // Execute in background
        BackgroundTaskExecutor.getInstance().executeAsync(
            object : BackgroundTaskExecutor.BackgroundTask<Int> {
                override fun execute(): Int {
                    val dbManager = PhoneStatusMonitor.getInstance().dbManager

                    // Check if account exists
                    val accountExists = dbManager.verifyAccount(email, passwordHash)

                    if (accountExists == DBConstants.EMAIL_NOT_FOUND) {
                        // Validate email format
                        val emailValidation = dbManager.validateEmail(email)
                        if (emailValidation == DBConstants.EMAIL_VALID) {
                            // Create account
                            val createResult = dbManager.createAccount(email, passwordHash)
                            return if (createResult == DBConstants.ACCOUNT_CREATED) {
                                Constants.CREATE_PROFILE_SUCCESS
                            } else {
                                createResult
                            }
                        } else {
                            return emailValidation
                        }
                    } else if (accountExists == DBConstants.SYNC_OK) {
                        // Account already exists
                        return DBConstants.SYNC_OK
                    } else {
                        return accountExists
                    }
                }
            },
            object : BackgroundTaskExecutor.TaskCallback<Int> {
                override fun onSuccess(result: Int) {
                    registerStatus = result
                    finishedRegistering = true
                    handleRegistrationResult(email, passwordHash)
                }

                override fun onError(e: Exception) {
                    Log.e(TAG, "Registration error", e)
                    registerStatus = DBConstants.GENERIC_ERROR
                    finishedRegistering = true
                    handleRegistrationResult(email, passwordHash)
                }
            }
        )

        // Wait for registration to complete
        mainHandler.post(object : Runnable {
            override fun run() {
                if (finishedRegistering) {
                    Log.d(TAG, "Registration finished with status: $registerStatus")
                } else {
                    mainHandler.postDelayed(this, 100)
                }
            }
        })
    }

    private fun handleRegistrationResult(email: String, passwordHash: String) {
        showLoading.value = false

        when (registerStatus) {
            DBConstants.INTERNET_CONNECTION_FAILED -> {
                val message = "The device has no access to the internet, try to connect to a network and try again"
                infoNotificationManager.showNotification(message, Runnable {
                    showRegisterSection.value = false
                    switchChecked.value = false
                }, false)
            }

            DBConstants.SYNC_OK -> {
                // Email already exists
                val message = "Email already exists"
                infoNotificationManager.showNotificationTwoButtons(
                    message,
                    "Retry",
                    "Login",
                    Runnable {
                        // Just close notification
                    },
                    Runnable {
                        // Navigate to LoadProfileActivity
                        ProfileFileCollection.welcomeActivityWrite(true, null, true)
                        val intent = Intent(this, LoadProfileActivity::class.java)
                        startActivity(intent)
                        finish()
                    }
                )
            }

            DBConstants.EMAIL_INVALID -> {
                val message = "Invalid mail address"
                infoNotificationManager.showNotification(message, Runnable {}, true)
            }

            DBConstants.ACCOUNT_CREATION_FAILED, DBConstants.GENERIC_ERROR -> {
                val message = "Error was encountered while creating the profile"
                infoNotificationManager.showNotificationTwoButtons(
                    message,
                    "Retry",
                    "Try local",
                    Runnable {
                        // Just close notification
                    },
                    Runnable {
                        // Navigate to NewProfileActivity (local)
                        ProfileFileCollection.welcomeActivityWrite(true, null, false)
                        val intent = Intent(this, NewProfileActivity::class.java)
                        startActivity(intent)
                        finish()
                    }
                )
            }

            Constants.CREATE_PROFILE_SUCCESS -> {
                // Show success notification
                val message = "Account created successfully"
                infoNotificationManager.showNotification(message, Runnable {}, true)

                // Wait 5 seconds then navigate
                mainHandler.postDelayed({
                    infoNotificationManager.hideNotification()
                    // Write profile with remote = true
                    ProfileFileCollection.newProfileActivityWrite(true, email, passwordHash)
                    // Navigate to UserInfoActivity
                    val intent = Intent(this, UserInfoActivity::class.java)
                    startActivity(intent)
                    finish()
                }, Constants.SUCCESS_NOTIFICATION_DELAY.toLong())
            }
        }
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
}

@Composable
fun NewProfileScreen(
    showRegisterSection: Boolean,
    switchEnabled: Boolean,
    switchChecked: Boolean,
    emailInput: String,
    passwordInput: String,
    showEmailError: Boolean,
    showPasswordError: Boolean,
    showLoading: Boolean,
    loadingText: String,
    onSwitchChanged: (Boolean) -> Unit,
    onEmailChange: (String) -> Unit,
    onPasswordChange: (String) -> Unit,
    onBackFromProfileSelection: () -> Unit,
    onNextFromProfileSelection: () -> Unit,
    onBackFromRegister: () -> Unit,
    onDoneFromRegister: () -> Unit
) {
    Box(modifier = Modifier.fillMaxSize()) {
        // Background image
        Image(
            painter = painterResource(R.drawable.welcome_background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        // Animated sections
        AnimatedVisibility(
            visible = !showRegisterSection,
            enter = slideInHorizontally(
                initialOffsetX = { -it },
                animationSpec = tween(Constants.ANIMATION_DELAY)
            ),
            exit = slideOutHorizontally(
                targetOffsetX = { -it },
                animationSpec = tween(Constants.ANIMATION_DELAY)
            )
        ) {
            ProfileSelectionSection(
                switchEnabled = switchEnabled,
                switchChecked = switchChecked,
                onSwitchChanged = onSwitchChanged,
                onBackClick = onBackFromProfileSelection,
                onNextClick = onNextFromProfileSelection
            )
        }

        AnimatedVisibility(
            visible = showRegisterSection,
            enter = slideInHorizontally(
                initialOffsetX = { it },
                animationSpec = tween(Constants.ANIMATION_DELAY)
            ),
            exit = slideOutHorizontally(
                targetOffsetX = { it },
                animationSpec = tween(Constants.ANIMATION_DELAY)
            )
        ) {
            RegisterSection(
                emailInput = emailInput,
                passwordInput = passwordInput,
                showEmailError = showEmailError,
                showPasswordError = showPasswordError,
                onEmailChange = onEmailChange,
                onPasswordChange = onPasswordChange,
                onBackClick = onBackFromRegister,
                onDoneClick = onDoneFromRegister
            )
        }

        // Loading Component
        LoadingComponent(
            isVisible = showLoading,
            loadingText = loadingText
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ProfileSelectionSection(
    switchEnabled: Boolean,
    switchChecked: Boolean,
    onSwitchChanged: (Boolean) -> Unit,
    onBackClick: () -> Unit,
    onNextClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        Spacer(modifier = Modifier.height(40.dp))

        // Logo
        Image(
            painter = painterResource(R.drawable.vision_assist_logo),
            contentDescription = "app logo",
            modifier = Modifier.size(200.dp)
        )

        Spacer(modifier = Modifier.weight(0.5f))

        // Title text
        Text(
            text = "Would you want\nto sync the profile?",
            fontSize = 32.sp,
            color = colorResource(R.color.std_cyan),
            fontFamily = robotoSemibold,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth(),
            lineHeight = 40.sp
        )

        Spacer(modifier = Modifier.height(40.dp))

        // Switch row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            // No option
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(
                    imageVector = Icons.Filled.CloudOff,
                    contentDescription = "No sync",
                    modifier = Modifier.size(40.dp),
                    tint = if (!switchChecked) colorResource(R.color.std_cyan) else Color.Gray
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "No",
                    fontSize = Constants.STD_FONT_SIZE.sp,
                    color = if (!switchChecked) colorResource(R.color.std_cyan) else Color.Gray,
                    fontFamily = robotoSemibold
                )
            }

            // Switch
            Switch(
                checked = switchChecked,
                onCheckedChange = onSwitchChanged,
                enabled = switchEnabled,
                colors = SwitchDefaults.colors(
                    checkedThumbColor = colorResource(R.color.std_purple),
                    checkedTrackColor = colorResource(R.color.std_cyan).copy(alpha = 0.5f),
                    uncheckedThumbColor = Color.Gray,
                    uncheckedTrackColor = Color.LightGray
                )
            )

            // Yes option
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(
                    imageVector = Icons.Filled.CloudSync,
                    contentDescription = "Sync",
                    modifier = Modifier.size(40.dp),
                    tint = if (switchChecked) colorResource(R.color.std_cyan) else Color.Gray
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Yes",
                    fontSize = Constants.STD_FONT_SIZE.sp,
                    color = if (switchChecked) colorResource(R.color.std_cyan) else Color.Gray,
                    fontFamily = robotoSemibold
                )
            }
        }

        Spacer(modifier = Modifier.weight(1f))

        // Navigation buttons
        NavigationButtons(
            onBackClick = onBackClick,
            onNextClick = onNextClick
        )

        Spacer(modifier = Modifier.height(40.dp))
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RegisterSection(
    emailInput: String,
    passwordInput: String,
    showEmailError: Boolean,
    showPasswordError: Boolean,
    onEmailChange: (String) -> Unit,
    onPasswordChange: (String) -> Unit,
    onBackClick: () -> Unit,
    onDoneClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        Spacer(modifier = Modifier.height(80.dp))

        // Title
        Text(
            text = "Account",
            fontSize = 40.sp,
            color = colorResource(R.color.std_cyan),
            fontFamily = robotoLight,
            letterSpacing = 6.sp,
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(20.dp))

        // Account icon
        Icon(
            imageVector = Icons.Filled.AccountCircle,
            contentDescription = "Account",
            modifier = Modifier.size(60.dp),
            tint = colorResource(R.color.std_cyan)
        )

        Spacer(modifier = Modifier.weight(0.5f))

        // Card with email and password - matching LoadProfileActivity design
        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .height(300.dp),
            shape = RoundedCornerShape(16.dp),
            shadowElevation = 4.dp,
            color = colorResource(R.color.notification_white)
        ) {
            Row(modifier = Modifier.fillMaxSize()) {
                // Left section - Email and Password inputs (0.75 width)
                Column(
                    modifier = Modifier
                        .weight(0.75f)
                        .fillMaxHeight()
                        .padding(24.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    // Email field
                    Text(
                        text = "Email",
                        fontSize = Constants.STD_FONT_SIZE.sp,
                        color = if (showEmailError) colorResource(R.color.error_red)
                        else colorResource(R.color.notification_text_gray),
                        fontFamily = robotoSemibold
                    )

                    BasicTextField(
                        value = emailInput,
                        onValueChange = onEmailChange,
                        modifier = Modifier
                            .fillMaxWidth()
                            .shadow(
                                elevation = 2.dp,
                                shape = RoundedCornerShape(8.dp)
                            )
                            .background(
                                color = Color.White,
                                shape = RoundedCornerShape(8.dp)
                            )
                            .border(
                                width = 1.dp,
                                color = if (showEmailError)
                                    colorResource(R.color.error_red)
                                else
                                    Color.LightGray,
                                shape = RoundedCornerShape(8.dp)
                            )
                            .padding(12.dp),
                        textStyle = TextStyle(
                            fontSize = Constants.STD_FONT_SIZE_LW.sp,
                            color = colorResource(R.color.std_cyan),
                            fontFamily = robotoRegular,
                            letterSpacing = 1.sp
                        ),
                        singleLine = true,
                        decorationBox = { innerTextField ->
                            if (emailInput.isEmpty()) {
                                Text(
                                    text = "example@gmail.com",
                                    fontSize = Constants.STD_FONT_SIZE_LW.sp,
                                    color = Color.Gray
                                )
                            }
                            innerTextField()
                        }
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    // Password field
                    Text(
                        text = "Password",
                        fontSize = Constants.STD_FONT_SIZE.sp,
                        color = if (showPasswordError) colorResource(R.color.error_red)
                        else colorResource(R.color.notification_text_gray),
                        fontFamily = robotoSemibold
                    )

                    BasicTextField(
                        value = passwordInput,
                        onValueChange = onPasswordChange,
                        modifier = Modifier
                            .fillMaxWidth()
                            .shadow(
                                elevation = 2.dp,
                                shape = RoundedCornerShape(8.dp)
                            )
                            .background(
                                color = Color.White,
                                shape = RoundedCornerShape(8.dp)
                            )
                            .border(
                                width = 1.dp,
                                color = if (showPasswordError)
                                    colorResource(R.color.error_red)
                                else
                                    Color.LightGray,
                                shape = RoundedCornerShape(8.dp)
                            )
                            .padding(12.dp),
                        textStyle = TextStyle(
                            fontSize = Constants.STD_FONT_SIZE_LW.sp,
                            color = colorResource(R.color.std_cyan),
                            fontFamily = robotoRegular,
                            letterSpacing = 1.sp
                        ),
                        visualTransformation = PasswordVisualTransformation(),
                        singleLine = true
                    )
                }

                // Right section - Back and Done buttons (0.25 width)
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
        Spacer(modifier = Modifier.height(40.dp))
    }
}

@Preview(name = "New Profile Activity", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun NewProfileActivityPreview() {
    NewProfileScreen(
        showRegisterSection = false,
        switchEnabled = true,
        switchChecked = false,
        emailInput = "",
        passwordInput = "",
        showEmailError = false,
        showPasswordError = false,
        showLoading = false,
        loadingText = "",
        onSwitchChanged = {},
        onEmailChange = {},
        onPasswordChange = {},
        onBackFromProfileSelection = {},
        onNextFromProfileSelection = {},
        onBackFromRegister = {},
        onDoneFromRegister = {}
    )
}

@Preview(name = "Register Section", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun RegisterSectionPreview() {
    MaterialTheme {
        Image(
            painter = painterResource(id = R.drawable.welcome_background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        RegisterSection(
            emailInput = "",
            passwordInput = "",
            showEmailError = false,
            showPasswordError = false,
            onEmailChange = {},
            onPasswordChange = {},
            onBackClick = {},
            onDoneClick = {}
        )
    }
}
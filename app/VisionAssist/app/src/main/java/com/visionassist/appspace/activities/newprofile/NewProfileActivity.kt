package com.visionassist.appspace.activities.newprofile

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
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
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AccountCircle
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CloudOff
import androidx.compose.material.icons.filled.Error
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
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
import com.visionassist.appspace.activities.newprofile.LoadProfileActivity.NotificationType
import com.visionassist.appspace.activities.newprofile.jsonCollection.ProfileFileCollection
import com.visionassist.appspace.database.DBConstants
import com.visionassist.appspace.database.DBManager
import com.visionassist.appspace.database.NetworkUtils
import com.visionassist.appspace.jetpack.design.BackArrowLargeFab
import com.visionassist.appspace.jetpack.design.NotificationDialog
import com.visionassist.appspace.jetpack.design.LoadingComponent
import com.visionassist.appspace.jetpack.design.NextArrowLargeFab
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.BackgroundTaskExecutor
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.load_createdAccount
import com.visionassist.appspace.utils.load_creatingAccount
import com.visionassist.appspace.utils.load_emailAlreadyExists
import com.visionassist.appspace.utils.load_genericErrorNew
import com.visionassist.appspace.utils.load_invalidEmail
import com.visionassist.appspace.utils.load_noInternet
import com.visionassist.appspace.utils.robotoLight
import com.visionassist.appspace.utils.robotoRegular
import com.visionassist.appspace.utils.robotoSemibold

class NewProfileActivity : ComponentActivity() {
    private val TAG = "NewProfileActivity"

    private val mainHandler = Handler(Looper.getMainLooper())
    private val backgroundExecutor: BackgroundTaskExecutor = BackgroundTaskExecutor.getInstance()

    // State management
    private val showRegisterSection = mutableStateOf(false)
    private val switchChecked = mutableStateOf(false)
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
    private val emailInput = mutableStateOf("")
    private val passwordInput = mutableStateOf("")
    private val showEmailError = mutableStateOf(false)
    private val showPasswordError = mutableStateOf(false)

    // Registration status members
    private var registerStatus = DBConstants.STATUS_INITIALIZED
    private var finishedRegistering = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            NewProfileScreen(
                showRegisterSection = showRegisterSection.value,
                switchChecked = switchChecked.value,
                emailInput = emailInput.value,
                passwordInput = passwordInput.value,
                showEmailError = showEmailError.value,
                showPasswordError = showPasswordError.value,
                showLoading = showLoading.value,
                loadingText = loadingText.value,
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
                onSwitchChanged = ::handleSwitchChanged,
                onEmailChange = { emailInput.value = it; showEmailError.value = false },
                onPasswordChange = { passwordInput.value = it; showPasswordError.value = false },
                onBackFromProfileSelection = ::handleBackFromProfileSelection,
                onNextFromProfileSelection = ::handleNextFromProfileSelection,
                onBackFromRegister = ::handleBackFromRegister,
                onDoneFromRegister = ::handleDoneFromRegister,
            )
        }
    }

    private fun handleSwitchChanged(checked: Boolean) {
        if (checked) {
            if (!NetworkUtils.isNetworkConnected(this)) {
                showNoInternetNotification()
                switchChecked.value = false
            } else {
                showRegisterSection.value = true
                switchChecked.value = true
            }
        }
    }

    private fun handleBackFromProfileSelection() {
        ProfileFileCollection.welcomeActivityDelete(true)
        val intent = Intent(this, WelcomeActivity::class.java)
        intent.putExtra(Constants.EXTRA_WELCOME_OPTION, false)
        startActivity(intent)
        finish()
    }

    private fun handleNextFromProfileSelection() {
        ProfileFileCollection.newProfileActivityWrite(false, null, null)
        val intent = Intent(this, UserInfoActivity::class.java)
        startActivity(intent)
        finish()
    }

    private fun handleBackFromRegister() {
        switchChecked.value = false
        showRegisterSection.value = false
    }

    private fun handleDoneFromRegister() {
        val email = emailInput.value.trim()
        val password = passwordInput.value.trim()

        // Validate fields
        if (email.isEmpty() || email == "example@gmail.com") {
            showEmailError.value = true
        }

        if (password.isEmpty()) {
            showPasswordError.value = true
        }

        if (showEmailError.value || showPasswordError.value) {
            return
        }

        // Show loading
        loadingText.value = load_creatingAccount(this)
        showLoading.value = true

        // Reset states
        finishedRegistering = false
        registerStatus = DBConstants.ACCOUNT_CREATED

        // Launch async task
        backgroundExecutor.executeAsync(
            { performRegistration(email, password) },
            object : BackgroundTaskExecutor.TaskCallback<Int> {
                override fun onSuccess(result: Int) {
                    registerStatus = result
                    finishedRegistering = true
                }

                override fun onError(e: Exception) {
                    Log.e(TAG, "Error during registration", e)
                    registerStatus = DBConstants.GENERIC_ERROR
                    finishedRegistering = true
                }
            }
        )

        // Wait for completion
        waitForRegistrationCompletion()
    }

    private fun performRegistration(email: String, password: String): Int {
        try {
            val dbManager = PhoneStatusMonitor.getInstance().dbManager

            // Check if account exists
            val accountExists = dbManager.verifyAccount(email, password)

            if (accountExists == DBConstants.EMAIL_NOT_FOUND) {
                // Validate email format
                val emailValidation = dbManager.validateEmail(email)
                return if (emailValidation == DBConstants.EMAIL_VALID) {
                    // Create account
                    dbManager.createAccount(email, password)
                } else
                    emailValidation
            } else
                return accountExists
        } catch (e: Exception) {
            Log.e(TAG, "Exception in performRegistration", e)
            return DBConstants.GENERIC_ERROR
        }
    }

    private fun waitForRegistrationCompletion() {
        val checkRunnable = object : Runnable {
            override fun run() {
                if (finishedRegistering) {
                    showLoading.value = false
                    handleRegistrationResult()
                } else {
                    mainHandler.postDelayed(this, 1000)
                }
            }
        }
        mainHandler.post(checkRunnable)
    }

    private fun handleRegistrationResult() {
        when (registerStatus) {
            DBConstants.ACCOUNT_CREATED -> {
                showSuccessNotification(load_createdAccount(this))

                // Navigate after 5 seconds
                mainHandler.postDelayed({
                    val email = emailInput.value.trim()
                    val password = passwordInput.value.trim()
                    val passwordHash = DBManager.hashPassword(password)

                    ProfileFileCollection.newProfileActivityWrite(true, email, passwordHash)
                    val intent = Intent(this, UserInfoActivity::class.java)
                    startActivity(intent)
                    finish()
                }, Constants.SUCCESS_NOTIFICATION_DELAY.toLong())
            }

            DBConstants.INTERNET_CONNECTION_FAILED -> {
                showNoInternetNotification()
            }

            DBConstants.SYNC_OK, DBConstants.PASSWORD_INCORRECT
                -> {
                showEmailExistsNotification()
            }

            DBConstants.EMAIL_INVALID -> {
                showInvalidEmailNotification()
            }

            DBConstants.ACCOUNT_CREATION_FAILED,
            DBConstants.GENERIC_ERROR -> {
                showCreationErrorNotification()
            }

            else -> {
                showCreationErrorNotification()
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

    private fun showEmailExistsNotification() {
        notificationType.value = NotificationType.ERROR
        notificationMessage.value = load_emailAlreadyExists(this)
        showOneButton.value = false
        showTwoButtons.value = true
        showThreeButtons.value = false
        firstButtonLabel.value = if (AppConfig.mainLanguage.code == "en") "Retry" else "Reîncearcă"
        firstButtonClick.value = { hideNotification() }
        secondButtonLabel.value =
            if (AppConfig.mainLanguage.code == "en") "Login" else "Autentificare"
        firstButtonClick.value = { handleNotificationLogin()}
        showNotification.value = true
    }

    private fun showInvalidEmailNotification() {
        notificationType.value = NotificationType.ERROR
        notificationMessage.value = load_invalidEmail(this)
        showOneButton.value = true
        showTwoButtons.value = false
        showThreeButtons.value = false
        firstButtonLabel.value = "OK"
        firstButtonClick.value = { hideNotification() }
        showNotification.value = true
    }

    private fun showCreationErrorNotification() {
        notificationType.value = NotificationType.ERROR
        notificationMessage.value = load_genericErrorNew(this)
        showOneButton.value = false
        showTwoButtons.value = true
        showThreeButtons.value = false
        firstButtonLabel.value = if (AppConfig.mainLanguage.code == "en") "Retry" else "Reîncearcă"
        firstButtonClick.value = { hideNotification() }
        secondButtonLabel.value =
            if (AppConfig.mainLanguage.code == "en") "Try local" else "Continuați local"
        firstButtonClick.value = { handleNextFromProfileSelection() }
        showNotification.value = true
    }

    private fun hideNotification() {
        showNotification.value = false

        // If we're showing internet error and in login section, navigate to first section
        if (notificationType.value == NotificationType.NO_INTERNET && showRegisterSection.value) {
            mainHandler.postDelayed({
                showRegisterSection.value = false
            }, 100)
        }
    }

    private fun handleNotificationLogin() {
        hideNotification()
        ProfileFileCollection.welcomeActivityWrite(true, null, false)
        val intent = Intent(this, LoadProfileActivity::class.java)
        startActivity(intent)
        finish()
    }
}

@Composable
fun NewProfileScreen(
    showRegisterSection: Boolean,
    switchChecked: Boolean,
    emailInput: String,
    passwordInput: String,
    showEmailError: Boolean,
    showPasswordError: Boolean,
    showLoading: Boolean,
    loadingText: String,
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
    onSwitchChanged: (Boolean) -> Unit,
    onEmailChange: (String) -> Unit,
    onPasswordChange: (String) -> Unit,
    onBackFromProfileSelection: () -> Unit,
    onNextFromProfileSelection: () -> Unit,
    onBackFromRegister: () -> Unit,
    onDoneFromRegister: () -> Unit,
) {
    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val screenHeight = maxHeight
        val screenWidth = maxWidth

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
                switchChecked = switchChecked,
                onSwitchChanged = onSwitchChanged
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

        val bottomSpace=screenHeight * Constants.STD_NAV_MARGIN_BOTTOM
        // Navigation Buttons (only for ProfileSelectionSection)
        if (!showRegisterSection) {
            Row(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = bottomSpace),
                horizontalArrangement = Arrangement.spacedBy(screenWidth * 0.08f)
            ) {
                BackArrowLargeFab(
                    onClick = onBackFromProfileSelection
                )

                NextArrowLargeFab(
                    onClick = onNextFromProfileSelection
                )
            }
        }
    }
}

@Composable
fun ProfileSelectionSection(
    switchChecked: Boolean,
    onSwitchChanged: (Boolean) -> Unit
) {
    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenHeight = maxHeight
        val screenWidth = maxWidth
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.SpaceAround
        ) {
            Box(modifier = Modifier.height(screenHeight * Constants.STD_SUBTITLE_MARGIN_TOP))

            // Title text
            Text(
                text = if (AppConfig.mainLanguage.code == "en")
                    "Would you want\nto sync the profile?"
                else
                    "Ați dori\n sincronizarea profilului?",
                fontSize = 32.sp,
                color = colorResource(R.color.std_cyan),
                fontFamily = robotoSemibold,
                textAlign = TextAlign.Center,
                lineHeight = 36.sp,
                modifier = Modifier.fillMaxWidth()
            )

            Box(modifier = Modifier.height(screenHeight * Constants.STD_SUBTITLE_BODY_MARGIN_TOP))

            // Switch row
            Row(
                modifier = Modifier
                    .fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // No option
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        text = if (AppConfig.mainLanguage.code == "en")
                            "No"
                        else
                            "Nu",
                        fontSize = Constants.STD_FONT_SIZE.sp,
                        color = colorResource(R.color.std_cyan),
                        fontFamily = robotoSemibold
                    )
                    Icon(
                        imageVector = Icons.Filled.CloudOff,
                        contentDescription = if (AppConfig.mainLanguage.code == "en")
                            "No sync icon"
                        else
                            "Iconiță fără sincronizare",
                        modifier = Modifier.size(45.dp),
                        tint = Color.Black
                    )
                }

                // Switch (wider)
                Switch(
                    checked = switchChecked,
                    onCheckedChange = onSwitchChanged,
                    enabled = true,
                    modifier = Modifier.size(screenWidth * 0.05f),
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = colorResource(R.color.std_cyan),
                        checkedTrackColor = colorResource(R.color.std_cyan).copy(alpha = 0.5f),
                        uncheckedThumbColor = Color.Gray,
                        uncheckedTrackColor = Color.LightGray
                    )
                )

                // Yes option
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Icon(
                        imageVector = Icons.Filled.AccountCircle,
                        contentDescription = if (AppConfig.mainLanguage.code == "en")
                            "Sync"
                        else
                            "Iconiță cu sincronizare",
                        modifier = Modifier.size(40.dp),
                        tint = Color.Black
                    )
                    Text(
                        text = if (AppConfig.mainLanguage.code == "en")
                            "Yes"
                        else
                            "Da",
                        fontSize = Constants.STD_FONT_SIZE.sp,
                        color = colorResource(R.color.std_cyan),
                        fontFamily = robotoSemibold
                    )
                }
            }

            Box(modifier = Modifier.height(screenHeight * 0.33f))
        }
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
    BoxWithConstraints(
        modifier = Modifier.fillMaxSize()
    ) {
        val screenHeight = maxHeight

        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.SpaceAround,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(modifier = Modifier.height(screenHeight * Constants.STD_TITLE_MARGIN_TOP))

            // Title
            Text(
                text = "VisionAssist\nAccount",
                fontSize = 40.sp,
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
                    .size(60.dp)
                    .background(
                        color = colorResource(R.color.std_cyan),
                        shape = RoundedCornerShape(55)
                    ),
                contentAlignment = Alignment.Center
            ) {
                Image(
                    painter = painterResource(R.drawable.vision_assist_logo),
                    contentDescription = "VisionAssist Logo",
                    modifier = Modifier.size(60.dp)
                )
            }

            Box(modifier = Modifier.height(screenHeight * 0.01f))

            // Login Card
            Card(
                modifier = Modifier
                    .fillMaxWidth(0.8f)
                    .height(screenHeight * 0.29f),
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
                            .padding(top = 30.dp, start = 20.dp, end = 20.dp),
                        verticalArrangement = Arrangement.spacedBy(30.dp)
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
                                visualTransformation = PasswordVisualTransformation(),
                                singleLine = true
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
    name = "New Profile Activity/CreateProfileScreenSection",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
@Composable
fun ProfileSelectionSectionPreview() {
    MaterialTheme {
        Image(
            painter = painterResource(id = R.drawable.welcome_background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        ProfileSelectionSection(
            switchChecked = false,
            onSwitchChanged = {}
        )
    }
}

@Preview(
    name = "New Profile Activity/RegisterSection",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
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
            emailInput = "example@gmail.com",
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

@Preview(name = "New Profile Activity", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun NewProfileActivityPreview() {
    MaterialTheme {
        NewProfileScreen(
            showRegisterSection = false,
            switchChecked = false,
            emailInput = "example@gmail.com",
            passwordInput = "",
            showEmailError = false,
            showPasswordError = false,
            showLoading = false,
            loadingText = "",
            showNotification = false,
            notificationMessage = "",
            notificationType = NotificationType.SUCCESS,
            showOneButton = true,
            showTwoButtons = false,
            showThreeButtons = false,
            firstButtonLabel = "OK",
            secondButtonLabel = "",
            thirdButtonLabel = "",
            firstButtonClick = {},
            secondButtonClick = {},
            thirdButtonClick = {},
            onSwitchChanged = {},
            onEmailChange = {},
            onPasswordChange = {},
            onBackFromProfileSelection = {},
            onNextFromProfileSelection = {},
            onBackFromRegister = {},
            onDoneFromRegister = {},
        )
    }
}
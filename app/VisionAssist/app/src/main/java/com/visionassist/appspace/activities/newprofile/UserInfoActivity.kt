@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.activities.newprofile

import android.content.Intent
import android.os.Bundle
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
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Error
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.PhoneStatusMonitor
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.newprofile.LoadProfileActivity.NotificationType
import com.visionassist.appspace.activities.newprofile.jsonCollection.ProfileFileCollection
import com.visionassist.appspace.jetpack.design.BackArrowLargeFab
import com.visionassist.appspace.jetpack.design.CustomSlider
import com.visionassist.appspace.jetpack.design.NextArrowLargeFab
import com.visionassist.appspace.jetpack.design.NotificationDialog
import com.visionassist.appspace.jetpack.design.ThumbStyle
import com.visionassist.appspace.jetpack.managers.InfoNotificationManager
import com.visionassist.appspace.models.ttsengine.TTSManager
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.haptic_model0
import com.visionassist.appspace.utils.load_aboutSubtitle
import com.visionassist.appspace.utils.load_agreeButton
import com.visionassist.appspace.utils.load_contributeResearch
import com.visionassist.appspace.utils.load_disagreeButton
import com.visionassist.appspace.utils.load_howOldAreYou
import com.visionassist.appspace.utils.load_invalidCombination
import com.visionassist.appspace.utils.load_whatTypeOfVision
import com.visionassist.appspace.utils.load_whatsYourName
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.robotoRegular
import com.visionassist.appspace.utils.robotoSemibold
import java.util.regex.Pattern

class UserInfoActivity : ComponentActivity() {
    private val TAG = "UserInfoActivity"

    // Section management
    private val currentSection = mutableIntStateOf(1) // 1=name, 2=age, 3=vision
    private val lastSection = mutableIntStateOf(1)

    // Section 1 states (Name)
    private val nameInput = mutableStateOf(
        if (AppConfig.user_name != null) AppConfig.user_name
        else
            "Eduard"
    )
    private val showNameError = mutableStateOf(false)

    // Section 2 states (Age)
    private val ageValue = mutableIntStateOf(
        if (AppConfig.age != 0) AppConfig.age
        else
            55
    )

    // Section 3 states (Vision condition)
    private val visionInput = mutableStateOf(
        if (AppConfig.visual_condition != null) AppConfig.visual_condition
        else
            "Myopia"
    )
    private val showVisionError = mutableStateOf(false)

    // Notification states
    private val showNotification = mutableStateOf(false)
    private val notificationMessage = mutableStateOf("")
    private val notificationType = mutableStateOf(NotificationType.SUCCESS)
    private val showOneButton = mutableStateOf(true)
    private val showTwoButtons = mutableStateOf(false)
    private val showThreeButtons = mutableStateOf(false)
    private val firstButtonLabel = mutableStateOf("OK")
    private val secondButtonLabel = mutableStateOf("Cancel")
    private val thirdButtonLabel = mutableStateOf("Cancel")
    private val firstButtonClick = mutableStateOf({})
    private val secondButtonClick = mutableStateOf({})
    private val thirdButtonClick = mutableStateOf({})
    private val fieldTextInteraction = mutableStateOf(false)

    // Info notification manager
    private lateinit var infoNotificationManager: InfoNotificationManager
    private var ttsManager: TTSManager = PhoneStatusMonitor.getInstance().ttsManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize info notification manager
        infoNotificationManager = InfoNotificationManager(this)

        // Get section from intent extras
        val sectionParam = intent.getIntExtra(Constants.EXTRA_USERINFO_OPTION, 1)
        currentSection.intValue = sectionParam

        //if(AppConfig.user_name!=null)fieldTextInteraction.value=true

        setContent {
            UserInfoScreen(
                lastSection = lastSection.intValue,
                currentSection = currentSection.intValue,
                nameInput = nameInput.value,
                showNameError = showNameError.value,
                ageValue = ageValue.intValue,
                visionInput = visionInput.value,
                showVisionError = showVisionError.value,
                showNotification = showNotification.value,
                notificationMessage = notificationMessage.value,
                notificationType = notificationType.value,
                showOneButton = showOneButton.value,
                showTwoButtons = showTwoButtons.value,
                showThreeButtons = showThreeButtons.value,
                firstButtonLabel = firstButtonLabel.value,
                secondButtonLabel = secondButtonLabel.value,
                thirdButtonLabel = thirdButtonLabel.value,
                firstButtonClick = firstButtonClick.value,
                secondButtonClick = secondButtonClick.value,
                thirdButtonClick = thirdButtonClick.value,
                onNameChange = ::handleNameChange,
                onAgeChange = ::handleAgeChange,
                onVisionChange = ::handleVisionChange,
                onBackClick = ::handleBackClick,
                onNextClick = ::handleNextClick
            )
        }
    }

    private fun handleNameChange(newName: String) {
        nameInput.value = newName
        showNameError.value = false
        fieldTextInteraction.value = true
    }

    private fun handleAgeChange(newAge: Int) {
        ageValue.intValue = newAge
        if (AppConfig.blindness) {
            ttsManager.stopSpeaking()
            ttsManager.speak(
                newAge.toString(),
                Constants.TTS_PITCH,
                Constants.TTS_SPEECH_RATE,
                false,
                haptic_model0()
            )
        }
    }

    private fun handleVisionChange(newVision: String) {
        visionInput.value = newVision
        showVisionError.value = false
        fieldTextInteraction.value = true
    }

    private fun handleBackClick() {
        when (currentSection.intValue) {
            2 -> {
                // Delete section 1 data and go to section 1
                ProfileFileCollection.deleteUserInfoActivity(0)
                currentSection.intValue = 1
                fieldTextInteraction.value = false
            }

            3 -> {
                // Delete section 2 data and go to section 2
                ProfileFileCollection.deleteUserInfoActivity(1)
                lastSection.intValue=3
                currentSection.intValue = 2
                fieldTextInteraction.value = false
            }
        }
    }

    private fun handleNextClick() {
        when (currentSection.intValue) {
            1 -> handleSection1Next()
            2 -> handleSection2Next()
            3 -> handleSection3Next()
        }
    }

    private fun handleSection1Next() {
        val name = nameInput.value.trim()
        if (AppConfig.user_name != null) {
            if (name.isEmpty()) {
                showNameError.value = true
                return
            }
            val namePattern = Pattern.compile("^[A-Z][a-z]+$")
            if (!namePattern.matcher(name).matches()) {
                showInvalidCombinationNotification()
                return
            }
            AppConfig.user_name = name
            showContributionDialog()
            return
        }

        // Check if field is empty
        if (name.isEmpty() || !fieldTextInteraction.value) {
            showNameError.value = true
            return
        }

        // Validate name format [A-Z][a-z]+
        val namePattern = Pattern.compile("^[A-Z][a-z]+$")
        if (!namePattern.matcher(name).matches()) {
            showInvalidCombinationNotification()
            return
        }

        fieldTextInteraction.value = false
        AppConfig.user_name = name

        // Show contribution dialog
        showContributionDialog()
    }

    private fun handleSection2Next() {
        // Write age and navigate
        ProfileFileCollection.writeUserInfoActivity(1, false, "", ageValue.intValue, "")
        AppConfig.age = ageValue.intValue

        if (AppConfig.blindness) {
            val intent = Intent(this, UserInfoE3Activity::class.java)
            startActivity(intent)
            finish()
        } else {
            currentSection.intValue = 3
        }
    }

    private fun handleSection3Next() {
        val vision = visionInput.value.trim()
        if (AppConfig.visual_condition != null) {
            if (vision.isEmpty()) {
                showNameError.value = true
                return
            }
            val namePattern = Pattern.compile("^[A-Z][a-z]+$")
            if (!namePattern.matcher(vision).matches()) {
                showInvalidCombinationNotification()
                return
            }
            ProfileFileCollection.writeUserInfoActivity(2, false, "", 0, vision)
            AppConfig.visual_condition = vision

            val intent = Intent(this, UserAccessibility1Activity::class.java)
            intent.putExtra(Constants.EXTRA_USERACC_OPTION, 1)
            startActivity(intent)
            finish()
            return
        }

        // Check if field is empty
        if (vision.isEmpty() || !fieldTextInteraction.value) {
            showVisionError.value = true
            return
        }

        // Check if contains digits
        val namePattern = Pattern.compile("^[A-Z][a-z]+$")
        if (!namePattern.matcher(vision).matches()) {
            showInvalidCombinationNotification()
            return
        }

        // Write vision condition and navigate
        ProfileFileCollection.writeUserInfoActivity(2, false, "", 0, vision)
        AppConfig.visual_condition = vision

        val intent = Intent(this, UserAccessibility1Activity::class.java)
        intent.putExtra(Constants.EXTRA_USERACC_OPTION, 1)
        startActivity(intent)
        finish()
    }

    private fun showInvalidCombinationNotification() {
        notificationType.value = NotificationType.ERROR
        notificationMessage.value = load_invalidCombination(this)
        showOneButton.value = true
        showTwoButtons.value = false
        showThreeButtons.value = false
        firstButtonLabel.value = "OK"
        firstButtonClick.value = { hideNotification() }
        showNotification.value = true
    }

    private fun showContributionDialog() {
        infoNotificationManager.showNotificationTwoButtons(
            load_contributeResearch(this),
            load_agreeButton(),
            load_disagreeButton(),
            { handleAgreeClick() },
            { handleDisagreeClick() }
        )
    }

    private fun handleAgreeClick() {
        infoNotificationManager.hideNotification()
        AppConfig.isContributor = true
        ProfileFileCollection.writeUserInfoActivity(0, true, nameInput.value.trim(), 0, "")
        navigateToSection2()
    }

    private fun handleDisagreeClick() {
        infoNotificationManager.hideNotification()
        AppConfig.isContributor = false
        ProfileFileCollection.writeUserInfoActivity(0, false, nameInput.value.trim(), 0, "")

        if (AppConfig.blindness) {
            val intent = Intent(this, UserInfoE3Activity::class.java)
            intent.putExtra(Constants.EXTRA_HCACHING_OPTION, 1)
            startActivity(intent)
            finish()
        } else {
            val intent = Intent(this, UserAccessibility1Activity::class.java)
            intent.putExtra(Constants.EXTRA_USERINFO_OPTION, 1)
            startActivity(intent)
            finish()
        }
    }

    private fun navigateToSection2() {
        lastSection.intValue=1
        currentSection.intValue = 2
    }

    private fun hideNotification() {
        showNotification.value = false
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

@Composable
fun UserInfoScreen(
    lastSection: Int,
    currentSection: Int,
    nameInput: String,
    showNameError: Boolean,
    ageValue: Int,
    visionInput: String,
    showVisionError: Boolean,
    showNotification: Boolean,
    notificationMessage: String,
    notificationType: NotificationType,
    showOneButton: Boolean,
    showTwoButtons: Boolean,
    showThreeButtons: Boolean,
    firstButtonLabel: String,
    secondButtonLabel: String,
    thirdButtonLabel: String,
    firstButtonClick: () -> Unit,
    secondButtonClick: () -> Unit,
    thirdButtonClick: () -> Unit,
    onNameChange: (String) -> Unit,
    onAgeChange: (Int) -> Unit,
    onVisionChange: (String) -> Unit,
    onBackClick: () -> Unit,
    onNextClick: () -> Unit
) {
    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val screenHeight = maxHeight

        // Background image
        Image(
            painter = painterResource(R.drawable.welcome_background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        // Animated sections
        AnimatedVisibility(
            visible = currentSection == 1,
            enter = slideInHorizontally(
                initialOffsetX = { if (currentSection < 1) it else -it },
                animationSpec = tween(Constants.ANIMATION_DELAY)
            ),
            exit = slideOutHorizontally(
                targetOffsetX = { if (currentSection > 1) -it else it },
                animationSpec = tween(Constants.ANIMATION_DELAY)
            )
        ) {
            NameSection(
                screenHeight = screenHeight,
                nameInput = nameInput,
                showNameError = showNameError,
                onNameChange = onNameChange
            )
        }

        AnimatedVisibility(
            visible = currentSection == 2,
            enter = slideInHorizontally(
                initialOffsetX = { if(lastSection==1) it else -it },
                animationSpec = tween(Constants.ANIMATION_DELAY)
            ),
            exit = slideOutHorizontally(
                targetOffsetX = { if (currentSection > 2) -it else it },
                animationSpec = tween(Constants.ANIMATION_DELAY)
            )
        ) {
            AgeSection(
                screenHeight = screenHeight,
                ageValue = ageValue,
                onAgeChange = onAgeChange
            )
        }

        AnimatedVisibility(
            visible = currentSection == 3,
            enter = slideInHorizontally(
                initialOffsetX = { it },
                animationSpec = tween(Constants.ANIMATION_DELAY)
            ),
            exit = slideOutHorizontally(
                targetOffsetX = { it },
                animationSpec = tween(Constants.ANIMATION_DELAY)
            )
        ) {
            VisionSection(
                screenHeight = screenHeight,
                visionInput = visionInput,
                showVisionError = showVisionError,
                onVisionChange = onVisionChange
            )
        }

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

        val bottomSpace = screenHeight * Constants.STD_NAV_MARGIN_BOTTOM
        // Navigation Buttons
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = bottomSpace),
            horizontalArrangement = Arrangement.spacedBy(30.dp)
        ) {
            if (currentSection > 1) {
                BackArrowLargeFab(
                    onClick = onBackClick
                )
            }

            NextArrowLargeFab(
                onClick = onNextClick
            )
        }
    }
}

@Composable
fun NameSection(
    screenHeight: androidx.compose.ui.unit.Dp,
    nameInput: String,
    showNameError: Boolean,
    onNameChange: (String) -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Title positioned using Constants
        Box(modifier = Modifier.height(screenHeight * Constants.STD_SUBTITLE_MARGIN_TOP))

        Text(
            text = load_whatsYourName(androidx.compose.ui.platform.LocalContext.current),
            fontSize = Constants.STD_SUBTITLE_SIZE.sp,
            color = colorResource(R.color.std_cyan),
            fontFamily = robotoSemibold,
            textAlign = TextAlign.Center,
            lineHeight = 36.sp,
            modifier = Modifier.fillMaxWidth()
        )

        Box(modifier = Modifier.height(screenHeight * Constants.STD_TITLE_SUBTITLE_MARGIN_TOP))

        // Name input field
        Row(
            modifier = Modifier
                .fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.Center
        ) {
            BasicTextField(
                value = nameInput,
                onValueChange = onNameChange,
                modifier = Modifier
                    .fillMaxWidth(0.4f)
                    .shadow(
                        elevation = 3.dp,
                        shape = RoundedCornerShape(10.dp)
                    )
                    .background(
                        color = Color.White,
                        shape = RoundedCornerShape(10.dp)
                    )
                    .border(
                        width = 1.dp,
                        color = if (showNameError)
                            colorResource(R.color.error_red)
                        else
                            Color.LightGray,
                        shape = RoundedCornerShape(10.dp)
                    )
                    .padding(8.dp),
                textStyle = TextStyle(
                    fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                    color = colorResource(R.color.std_cyan),
                    fontFamily = robotoRegular,
                    letterSpacing = 1.sp
                ),
                singleLine = true,
            )

            if (showNameError) {
                Icon(
                    imageVector = Icons.Filled.Error,
                    contentDescription = "Error",
                    modifier = Modifier
                        .size(24.dp)
                        .padding(start = 8.dp),
                    tint = colorResource(R.color.error_red)
                )
            }
        }
    }
}

@Composable
fun AgeSection(
    screenHeight: androidx.compose.ui.unit.Dp,
    ageValue: Int,
    onAgeChange: (Int) -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Title positioned using Constants
        Box(modifier = Modifier.height(screenHeight * Constants.STD_SUBTITLE2_MARGIN_TOP))

        Text(
            text = load_aboutSubtitle(androidx.compose.ui.platform.LocalContext.current),
            fontSize = Constants.STD_SUBTITLE_SIZE.sp,
            color = colorResource(R.color.std_cyan),
            fontFamily = robotoSemibold,
            textAlign = TextAlign.Start,
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 22.dp)
        )

        Box(modifier = Modifier.height(screenHeight * Constants.STD_TITLE_SUBTITLE_MARGIN_TOP))

        // "How old are you?" text
        Text(
            text = load_howOldAreYou(androidx.compose.ui.platform.LocalContext.current),
            fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
            color = colorResource(R.color.std_cyan),
            fontFamily = robotoExtraBold,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )

        Box(modifier = Modifier.height(screenHeight * 0.025f))

        // Age slider
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.fillMaxWidth()
        ) {
            CustomSlider(
                value = ageValue.toFloat(),
                onValueChange = { onAgeChange(it.toInt()) },
                valueRange = 0f..100f,
                steps = 0,
                thumbStyle = ThumbStyle.BAR,  // ROUND, BAR, or DOUBLE_BAR
                thumbColor = colorResource(R.color.std_purple),
                thumbWidth = 8.dp,
                thumbHeight = 55.dp,
                trackHeight = 20.dp,
                activeTrackColor = Color.White,
                inactiveTrackColor = Color.White,
                trackShadow = 5.dp,
                modifier = Modifier.fillMaxWidth(0.75f)
            )

            Box(modifier = Modifier.height(screenHeight * 0.012f))

            Text(
                text = ageValue.toString(),
                fontSize = Constants.STD_SLIDER_INFO_SIZE.sp,
                color = colorResource(R.color.std_purple),
                fontFamily = robotoSemibold
            )
        }
    }
}

@Composable
fun VisionSection(
    screenHeight: androidx.compose.ui.unit.Dp,
    visionInput: String,
    showVisionError: Boolean,
    onVisionChange: (String) -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Title positioned using Constants
        Box(modifier = Modifier.height(screenHeight * Constants.STD_SUBTITLE2_MARGIN_TOP))

        Text(
            text = load_aboutSubtitle(androidx.compose.ui.platform.LocalContext.current),
            fontSize = Constants.STD_SUBTITLE_SIZE.sp,
            color = colorResource(R.color.std_cyan),
            fontFamily = robotoSemibold,
            textAlign = TextAlign.Start,
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 22.dp)
        )

        Box(modifier = Modifier.height(screenHeight * Constants.STD_TITLE_SUBTITLE_MARGIN_TOP))

        // "What type of vision difficulty" text
        Text(
            text = load_whatTypeOfVision(androidx.compose.ui.platform.LocalContext.current),
            fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
            color = colorResource(R.color.std_cyan),
            fontFamily = robotoExtraBold,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(24.dp))

        // Vision input field
        Row(
            modifier = Modifier
                .fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.Center
        ) {
            BasicTextField(
                value = visionInput,
                onValueChange = onVisionChange,
                modifier = Modifier
                    .fillMaxWidth(0.6f)
                    .shadow(
                        elevation = 3.dp,
                        shape = RoundedCornerShape(10.dp)
                    )
                    .background(
                        color = Color.White,
                        shape = RoundedCornerShape(10.dp)
                    )
                    .border(
                        width = 1.dp,
                        color = if (showVisionError)
                            colorResource(R.color.error_red)
                        else
                            Color.LightGray,
                        shape = RoundedCornerShape(10.dp)
                    )
                    .padding(8.dp),
                textStyle = TextStyle(
                    fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                    color = colorResource(R.color.std_cyan),
                    fontFamily = robotoRegular,
                    letterSpacing = 1.sp
                ),
                singleLine = true,
            )

            if (showVisionError) {
                Icon(
                    imageVector = Icons.Filled.Error,
                    contentDescription = "Error",
                    modifier = Modifier
                        .size(24.dp)
                        .padding(start = 8.dp),
                    tint = colorResource(R.color.error_red)
                )
            }
        }
    }
}

@Preview(name = "UserInfo - Name Section", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun UserInfoNameSectionPreview() {
    UserInfoScreen(
        lastSection = 1,
        currentSection = 1,
        nameInput = "Eduard",
        showNameError = false,
        ageValue = 55,
        visionInput = "Myopia",
        showVisionError = false,
        showNotification = false,
        notificationMessage = "",
        notificationType = NotificationType.SUCCESS,
        showOneButton = true,
        showTwoButtons = false,
        showThreeButtons = false,
        firstButtonLabel = "OK",
        secondButtonLabel = "Cancel",
        thirdButtonLabel = "Cancel",
        firstButtonClick = {},
        secondButtonClick = {},
        thirdButtonClick = {},
        onNameChange = {},
        onAgeChange = {},
        onVisionChange = {},
        onBackClick = {},
        onNextClick = {}
    )
}

@Preview(name = "UserInfo - Age Section", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun UserInfoAgeSectionPreview() {
    UserInfoScreen(
        lastSection = 1,
        currentSection = 2,
        nameInput = "Eduard",
        showNameError = false,
        ageValue = 55,
        visionInput = "Myopia",
        showVisionError = false,
        showNotification = false,
        notificationMessage = "",
        notificationType = NotificationType.SUCCESS,
        showOneButton = true,
        showTwoButtons = false,
        showThreeButtons = false,
        firstButtonLabel = "OK",
        secondButtonLabel = "Cancel",
        thirdButtonLabel = "Cancel",
        firstButtonClick = {},
        secondButtonClick = {},
        thirdButtonClick = {},
        onNameChange = {},
        onAgeChange = {},
        onVisionChange = {},
        onBackClick = {},
        onNextClick = {}
    )
}

@Preview(name = "UserInfo - Vision Section", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun UserInfoVisionSectionPreview() {
    UserInfoScreen(
        lastSection = 1,
        currentSection = 3,
        nameInput = "Eduard",
        showNameError = false,
        ageValue = 55,
        visionInput = "Myopia",
        showVisionError = false,
        showNotification = false,
        notificationMessage = "",
        notificationType = NotificationType.SUCCESS,
        showOneButton = true,
        showTwoButtons = false,
        showThreeButtons = false,
        firstButtonLabel = "OK",
        secondButtonLabel = "Cancel",
        thirdButtonLabel = "Cancel",
        firstButtonClick = {},
        secondButtonClick = {},
        thirdButtonClick = {},
        onNameChange = {},
        onAgeChange = {},
        onVisionChange = {},
        onBackClick = {},
        onNextClick = {}
    )
}
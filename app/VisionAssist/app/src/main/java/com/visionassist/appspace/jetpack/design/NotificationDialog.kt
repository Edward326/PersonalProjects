@file:OptIn(ExperimentalMaterial3Api::class)

package com.visionassist.appspace.jetpack.design

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.SignalCellularConnectedNoInternet0Bar
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.R
import com.visionassist.appspace.activities.newprofile.LoadProfileActivity
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.robotoRegular
import com.visionassist.appspace.utils.robotoSemibold

@Composable
fun NotificationDialog(
    modifier: Modifier = Modifier,
    isVisible: Boolean = false,
    type: LoadProfileActivity.NotificationType,
    message: String,
    showOneButton: Boolean=true,
    showTwoButtons: Boolean = false,
    showThreeButtons: Boolean = false,
    firstButtonLabel: String = "OK",
    secondButtonLabel: String = "OK",
    thirdButtonLabel: String = "OK",
    firstButtonClick: () -> Unit = {},
    secondButtonClick: () -> Unit = {},
    thirdButtonClick: () -> Unit = {},
) {
    AnimatedVisibility(
        visible = isVisible,
        enter = fadeIn(
            initialAlpha = 0f,
            animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY)
        )
    ) {
        Box(
            modifier = modifier
                .fillMaxSize()
                .background(Color.Gray.copy(alpha = 0.5f)),
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
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Icon based on type
                    when (type) {
                        LoadProfileActivity.NotificationType.SUCCESS -> {
                            Icon(
                                imageVector = Icons.Filled.CheckCircle,
                                contentDescription = if (AppConfig.mainLanguage.code == "en") "Success" else "Succes",
                                modifier = Modifier.size(24.dp),
                                tint = Color(0xFF4CAF50) // Green
                            )
                        }

                        LoadProfileActivity.NotificationType.ERROR -> {
                            Icon(
                                imageVector = Icons.Filled.Error,
                                contentDescription = if (AppConfig.mainLanguage.code == "en") "Error" else "Eroare",
                                modifier = Modifier.size(24.dp),
                                tint = colorResource(R.color.error_red)
                            )
                        }

                        LoadProfileActivity.NotificationType.NO_INTERNET -> {
                            Icon(
                                imageVector = Icons.Filled.SignalCellularConnectedNoInternet0Bar,
                                contentDescription = if (AppConfig.mainLanguage.code == "en") "Warning" else "Atenționare",
                                modifier = Modifier.size(24.dp),
                                tint = Color(0xFFFF9800) // Orange
                            )
                        }
                    }

                    // Message with bold formatting for codes
                    Text(
                        text = buildAnnotatedString {
                            val parts = message.split("@")
                            withStyle(
                                style = SpanStyle(
                                    fontSize = Constants.STD_FONT_SIZE.sp
                                )
                            ) {
                                append(parts[0])
                            }

                            if (parts.size > 1) {
                                withStyle(
                                    style = SpanStyle(
                                        fontWeight = FontWeight.Bold,
                                        fontSize = Constants.STD_ERROR_FONT_SIZE.sp
                                    )
                                ) {
                                    append(parts[1])
                                }
                            }
                        },
                        fontSize = Constants.STD_FONT_SIZE.sp,
                        color = colorResource(R.color.notification_text_gray),
                        textAlign = TextAlign.Center,
                        lineHeight = 20.sp
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    // Buttons based on configuration
                    when {
                        // THREE BUTTONS
                        showThreeButtons -> {
                            Column(
                                modifier = Modifier.fillMaxWidth(),
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                // Retry button (full width)
                                Button(
                                    onClick = firstButtonClick,
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(Constants.STD_BUTTON_HEIGHT.dp),
                                    shape = RoundedCornerShape(28.dp),
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = colorResource(R.color.notification_button_white),
                                        contentColor = colorResource(R.color.std_purple)
                                    )
                                ) {
                                    Text(
                                        text = firstButtonLabel,
                                        fontSize = Constants.STD_BUTTON_FONT_SIZE.sp)
                                }

                                // Create account + Load local (row)
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    Button(
                                        onClick = secondButtonClick,
                                        modifier = Modifier
                                            .weight(1f)
                                            .height(Constants.STD_BUTTON_HEIGHT.dp),
                                        shape = RoundedCornerShape(28.dp),
                                        colors = ButtonDefaults.buttonColors(
                                            containerColor = colorResource(R.color.std_purple),
                                            contentColor = Color.White,
                                        )
                                    ) {
                                        Text(
                                            text = secondButtonLabel,
                                            fontSize = Constants.STD_BUTTON_FONT_SIZE.sp
                                        )
                                    }

                                    Button(
                                        onClick = thirdButtonClick,
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
                                            text = thirdButtonLabel,
                                            fontSize = Constants.STD_BUTTON_FONT_SIZE.sp
                                        )
                                    }
                                }
                            }
                        }

                        // TWO BUTTONS: Retry + Create account
                        type == LoadProfileActivity.NotificationType.ERROR && showTwoButtons -> {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                Button(
                                    onClick = firstButtonClick,
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

                                Button(
                                    onClick = secondButtonClick,
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

                        // ONE BUTTON: OK only
                        showOneButton -> {
                            Button(
                                onClick = firstButtonClick,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(Constants.STD_BUTTON_HEIGHT.dp),
                                shape = RoundedCornerShape(28.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = colorResource(R.color.notification_button_white),
                                    contentColor = colorResource(R.color.std_purple)
                                )
                            ) {
                                Text(text = firstButtonLabel,
                                    fontSize = Constants.STD_BUTTON_FONT_SIZE.sp
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Preview(name = "Success Notification", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun LoadProfileSuccessPreview() {
    NotificationDialog(
        isVisible = true,
        message = "Profile imported successfully\n\n@(Exit code: 0)",
        type = LoadProfileActivity.NotificationType.SUCCESS,
        firstButtonLabel = "OK",
        firstButtonClick = {}
    )
}

@Preview(
    name = "Error Notification - 2 Buttons",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
@Composable
fun LoadProfileErrorPreview() {
    NotificationDialog(
        isVisible = true,
        message = "Error was encountered while fetching the profile\n\n@(Error code: 9)",
        type = LoadProfileActivity.NotificationType.ERROR,
        showTwoButtons = true,
        firstButtonLabel = "Retry",
        firstButtonClick = {},
        secondButtonLabel = "Create account",
        secondButtonClick = {},
    )
}

@Preview(
    name = "Error Notification - 3 Buttons",
    showBackground = true,
    widthDp = 412,
    heightDp = 917
)
@Composable
fun LoadProfileError3ButtonsPreview() {
    NotificationDialog(
        isVisible = true,
        message = "The email is not registered in the database",
        type = LoadProfileActivity.NotificationType.ERROR,
        showThreeButtons = true,
        firstButtonLabel = "Retry",
        firstButtonClick = {},
        secondButtonLabel = "Create account",
        secondButtonClick = {},
        thirdButtonLabel = "Load local",
        thirdButtonClick ={}
    )
}

@Preview(name = "No Internet Notification", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun LoadProfileNoInternetPreview() {
    NotificationDialog(
        isVisible = true,
        message = "The device has no access to the internet, try to connect to a network and try again",
        type = LoadProfileActivity.NotificationType.NO_INTERNET,
        firstButtonLabel = "OK",
        firstButtonClick = {}
    )
}
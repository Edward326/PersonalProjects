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

@Composable
fun LoadProfileNotificationDialog(
    modifier: Modifier = Modifier,
    isVisible: Boolean = false,
    message: String,
    type: LoadProfileActivity.NotificationType,
    showTwoButtons: Boolean = false,
    showThreeButtons: Boolean = false,
    onRetryClick: () -> Unit = {},
    onCreateAccountClick: () -> Unit = {},
    onLoadLocalClick: () -> Unit = {},
    onOkClick: () -> Unit = {},
    newprofile:Boolean=false
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
                            append(parts[0])

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
                        // THREE BUTTONS: Retry (full width) + Create account + Load local (row)
                        showThreeButtons -> {
                            Column(
                                modifier = Modifier.fillMaxWidth(),
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                // Retry button (full width)
                                Button(
                                    onClick = onRetryClick,
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
                                        text = if (AppConfig.mainLanguage.code == "en") "Retry" else "Reîncearcă",
                                        fontSize = Constants.STD_BUTTON_FONT_SIZE.sp
                                    )
                                }

                                // Create account + Load local (row)
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    Button(
                                        onClick = onCreateAccountClick,
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
                                            text = if (AppConfig.mainLanguage.code == "en") "Create account" else "Creează cont",
                                            fontSize = Constants.STD_BUTTON_FONT_SIZE.sp
                                        )
                                    }

                                    Button(
                                        onClick = onLoadLocalClick,
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
                                            text = if (AppConfig.mainLanguage.code == "en") "Load local" else "Încarcă local",
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
                                    onClick = onRetryClick,
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
                                        text = if (AppConfig.mainLanguage.code == "en") "Retry" else "Reîncearcă",
                                        fontSize = Constants.STD_BUTTON_FONT_SIZE.sp
                                    )
                                }

                                Button(
                                    onClick = onCreateAccountClick,
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
                                        text = if (!newprofile) {
                                            if (AppConfig.mainLanguage.code == "en") "Create account" else "Creează cont"
                                        } else {
                                            if (AppConfig.mainLanguage.code == "en") "Load local" else "Încarcă local"
                                        },
                                    fontSize = Constants.STD_BUTTON_FONT_SIZE.sp
                                    )
                                }
                            }
                        }

                        // ONE BUTTON: OK only
                        else -> {
                            Button(
                                onClick = onOkClick,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(Constants.STD_BUTTON_HEIGHT.dp),
                                shape = RoundedCornerShape(28.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = colorResource(R.color.notification_button_white),
                                    contentColor = colorResource(R.color.std_purple)
                                )
                            ) {
                                Text(text = "OK", fontSize = Constants.STD_BUTTON_FONT_SIZE.sp)
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
    LoadProfileNotificationDialog(
        isVisible = true,
        message = "Profile imported successfully\n\n@(Exit code: 0)",
        type = LoadProfileActivity.NotificationType.SUCCESS
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
    LoadProfileNotificationDialog(
        isVisible = true,
        message = "Error was encountered while fetching the profile\n\n@(Error code: 9)",
        type = LoadProfileActivity.NotificationType.ERROR,
        showTwoButtons = true,
        onRetryClick = {},
        onCreateAccountClick = {}
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
    LoadProfileNotificationDialog(
        isVisible = true,
        message = "The email is not registered in the database",
        type = LoadProfileActivity.NotificationType.ERROR,
        showThreeButtons = true,
        onRetryClick = {},
        onCreateAccountClick = {},
        onLoadLocalClick = {}
    )
}

@Preview(name = "No Internet Notification", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun LoadProfileNoInternetPreview() {
    LoadProfileNotificationDialog(
        isVisible = true,
        message = "The device has no access to the internet, try to connect to a network and try again",
        type = LoadProfileActivity.NotificationType.NO_INTERNET,
        onOkClick = {}
    )
}
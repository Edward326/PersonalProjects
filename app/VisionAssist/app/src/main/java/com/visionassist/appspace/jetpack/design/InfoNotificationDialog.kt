@file:OptIn(ExperimentalMaterial3Api::class)

package com.visionassist.appspace.jetpack.design

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
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
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
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
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.robotoSemibold

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun InfoNotificationDialog(
    modifier: Modifier = Modifier,
    isVisible: Boolean = false,
    message: String = "",
    twoButtons: Boolean = false,
    firstButtonLabel: String = "OK",
    secondButtonLabel: String = "Cancel",
    onFirstButtonClick: () -> Unit = {},
    onSecondButtonClick: () -> Unit = {}
) {
    // AnimatedVisibility with fade in/out animation
    AnimatedVisibility(
        visible = isVisible,
        enter = fadeIn(
            initialAlpha = 0f,
            animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY)
        ),
        exit = fadeOut(
            targetAlpha = 0f,
            animationSpec = tween(durationMillis = 0)  // ← Instant exit, no glitch!
        )
    ) {
        // Full screen overlay with semi-transparent background
        Box(
            modifier = modifier
                .fillMaxSize()
                .background(Color.Gray.copy(alpha = Constants.BACKGROUND_OPACITY)),
            contentAlignment = Alignment.Center
        ) {
            // Dialog Card
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
                    // Info Icon
                    Icon(
                        imageVector = Icons.Filled.Info,
                        contentDescription = "Information",
                        modifier = Modifier.size(24.dp),
                        tint = colorResource(R.color.std_purple)
                    )

                    // Message Text
                    Text(
                        text = buildAnnotatedString {
                            val parts = message.split("~")

                            when {
                                parts.size==1 -> {
                                    // Format: TitleContent
                                    withStyle(
                                        style = SpanStyle(
                                            fontSize = Constants.STD_FONT_SIZE.sp
                                        )
                                    ) {
                                        append(parts[0])
                                    }
                                }
                                parts.size > 1 -> {
                                    // Format: Title~Content
                                    withStyle(
                                        style = SpanStyle(
                                            fontWeight = FontWeight.Bold,
                                            fontSize = Constants.STD_ERROR_FONT_SIZE.sp

                                        )
                                    ) {
                                        append(parts[0])
                                    }
                                    append("\n\n")
                                    withStyle(
                                        style = SpanStyle(
                                            fontSize = Constants.STD_FONT_SIZE.sp
                                        )
                                    ) {
                                        append(parts[1])
                                    }
                                }
                            }
                        },
                        fontSize = Constants.STD_FONT_SIZE.sp,
                        color = colorResource(R.color.notification_text_gray),
                        textAlign = TextAlign.Center,
                        lineHeight = 20.sp
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    // Buttons - Single or Two based on twoButtons parameter
                    if (twoButtons) {
                        // Two buttons side by side
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            // First Button (Left)
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
                    } else {
                        // Single OK Button
                        Button(
                            onClick = onFirstButtonClick,
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
                                fontSize = Constants.STD_BUTTON_FONT_SIZE.sp
                            )
                        }
                    }
                }
            }
        }
    }
}

@Preview(name = "Info Notification Dialog - Single Button", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun InfoNotificationDialogPreview() {
    InfoNotificationDialog(
        isVisible = true,
        message = "Hello?~This is sample information text that explains what this option does in the application.",
        onFirstButtonClick = {}
    )
}

@Preview(name = "Info Notification Dialog - Two Buttons", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun InfoNotificationDialogTwoButtonsPreview() {
    InfoNotificationDialog(
        isVisible = true,
        message = "The microphone permission is required for the Find My Object feature. Would you like to grant access?",
        twoButtons = true,
        firstButtonLabel = "Give Access",
        secondButtonLabel = "Don't Give",
        onFirstButtonClick = {},
        onSecondButtonClick = {}
    )
}
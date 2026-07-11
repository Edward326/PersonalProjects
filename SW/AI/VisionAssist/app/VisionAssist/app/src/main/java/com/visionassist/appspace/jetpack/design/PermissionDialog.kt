@file:OptIn(ExperimentalMaterial3Api::class)

package com.visionassist.appspace.jetpack.design

import android.content.Context
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Error
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
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.R
import com.visionassist.appspace.utils.Constants

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun PermissionDialog(
    context: Context,
    modifier: Modifier = Modifier,
    isVisible: Boolean = true,
    message: String,
    buttonText: String,
    onOkClick: () -> Unit = {},
) {
    // AnimatedVisibility with fade animation (same as LoadingBox)
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
        // Full screen white overlay with 50% opacity
        Box(
            modifier = modifier
                .fillMaxSize()
                .background(Color.Gray.copy(alpha = Constants.BACKGROUND_OPACITY)),
            contentAlignment = Alignment.Center
        ) {
            // Dialog Card
            Card(
                modifier = Modifier
                    .fillMaxWidth(0.80f),
                shape = RoundedCornerShape(28.dp),
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFFECE6F0)
                ),
                elevation = CardDefaults.cardElevation(defaultElevation = 3.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(25.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,

                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    // Warning Icon
                    Icon(
                        imageVector = Icons.Filled.Error,
                        contentDescription = "Error",
                        modifier = Modifier.size(24.dp),
                        tint = Color(0xFFB3261E)
                    )

                    // Message
                    Text(
                        text = message,
                        fontSize = Constants.STD_FONT_SIZE.sp,
                        color = Color(0xFF49454F),
                        textAlign = TextAlign.Center,
                        lineHeight = 23.sp
                    )

                    Spacer(modifier = Modifier.height(2.dp))

                    // OK Button
                    Button(
                        onClick = onOkClick,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(Constants.STD_BUTTON_HEIGHT.dp),
                        shape = RoundedCornerShape(28.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = colorResource(R.color.notification_button_white),
                            contentColor = colorResource(R.color.std_purple),
                        )
                    ) {
                        Text(
                            text = buttonText,
                            fontSize = Constants.STD_BUTTON_FONT_SIZE.sp,
                        )
                    }
                }
            }
        }
    }
}

@Preview(name = "Permission Dialog", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun PermissionDialogPreview() {
    PermissionDialog(
        context = androidx.compose.ui.platform.LocalContext.current,
        message = stringResource(R.string.always_allow_en),
        buttonText = stringResource(R.string.ok),
        onOkClick = {}
    )
}
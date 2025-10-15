@file:OptIn(ExperimentalMaterial3Api::class)

package com.visionassist.appspace.jetpack.design

import android.content.Context
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
import com.visionassist.appspace.utils.load_errorText
import com.visionassist.appspace.utils.robotoSemibold

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun ErrorDialog(
    context: Context,
    modifier: Modifier = Modifier,
    isVisible: Boolean = true,
    message: String,
    errorCode: Int
) {
    // AnimatedVisibility with fade animation (same as LoadingBox)
    if(isVisible) {
        // Full screen white overlay with 50% opacity
        Box(
            modifier = modifier
                .fillMaxSize()
                .background(Color.Gray.copy(alpha = 0.5f)),
            contentAlignment = Alignment.Center
        ) {
            // Dialog Card
            Card(
                modifier = Modifier
                    .fillMaxWidth(0.80f),
                shape = RoundedCornerShape(28.dp),
                colors = CardDefaults.cardColors(
                    containerColor = colorResource(R.color.notification_white)
                )
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
                        tint = colorResource(R.color.error_red)
                    )

                    // Message
                    Text(
                        text = if(message=="_undefined_") load_errorText(context) else message,
                        fontSize = 14.sp,
                        color = colorResource(R.color.notification_text_gray),
                        textAlign = TextAlign.Center,
                        lineHeight = 23.sp
                    )

                    Spacer(modifier=Modifier.height(3.dp))

                    // Message
                    Text(
                        text = "(Error code: $errorCode)",
                        fontSize = 16.sp,
                        fontFamily = robotoSemibold,
                        color = Color(0xFF000000),
                        textAlign = TextAlign.Center,
                        lineHeight = 23.sp
                    )
                }
            }
        }
    }
}

@Preview(name = "Error Dialog", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun ErrorDialogPreview() {
    ErrorDialog(
        context = androidx.compose.ui.platform.LocalContext.current,
        message= stringResource(R.string.exit_error_en),
        errorCode = 34
    )
}
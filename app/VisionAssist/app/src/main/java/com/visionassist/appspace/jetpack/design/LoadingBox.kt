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
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.R
import com.visionassist.appspace.utils.load_loadingText

val robotoRegular = FontFamily(
    Font(R.font.roboto_regular_ttf, weight = FontWeight.Medium)
)

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun LoadingComponent(
    context: Context,
    modifier: Modifier = Modifier,
    isVisible: Boolean = true,
    loadingText: String = "_undefined_",
) {
    // AnimatedVisibility with fade animation
    AnimatedVisibility(
        visible = isVisible,
        enter = fadeIn(
            // Duration of fade in animation (in milliseconds)
            initialAlpha = 0f,
            animationSpec = tween(durationMillis = 1500)
        ),
        exit = fadeOut(
            // Duration of fade out animation
            targetAlpha = 0f,
            animationSpec = tween(durationMillis = 1500)
        )
    ) {
        // Full screen white overlay with 50% opacity
        Box(
            modifier = modifier
                .fillMaxSize()
                .background(Color.White.copy(alpha = 0.5f)),
            contentAlignment = Alignment.Center
        ) {
            // Loading content
            Column(
                modifier = Modifier.padding(32.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                // Loading Text
                Text(
                    text = if((loadingText)=="_undefined_") load_loadingText(context) else loadingText,
                    fontSize = 12.sp,
                    fontFamily = robotoRegular,
                    fontWeight = FontWeight.Medium,
                    fontStyle = FontStyle.Normal,
                    color = Color(0xFF6096BA),
                    letterSpacing = 1.sp
                )

                Spacer(modifier = Modifier.height(8.dp))

                // Linear Progress Indicator
                LinearProgressIndicator(
                    modifier = Modifier
                        .width(200.dp)
                        .height(4.dp)
                        .clip(RoundedCornerShape(1000.dp)),
                    color = colorResource(R.color.std_purple),
                    trackColor = Color(0xFFE8DEF8),
                    gapSize = 10.dp
                )
            }
        }
    }
}

@Preview(name = "Loading Linear", showBackground = true, widthDp = 500, heightDp = 500)
@Composable
fun LoadingComponentPreview() {
    LoadingComponent(
        isVisible = true,
        loadingText = "Please wait",
        context = androidx.compose.ui.platform.LocalContext.current
    )
}
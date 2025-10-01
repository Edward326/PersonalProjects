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
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.utils.Language
import com.visionassist.appspace.utils.AppConfig
import com.visionassist.appspace.R
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle

val robotoRegular = FontFamily(
    Font(R.font.roboto_regular_ttf, weight = FontWeight.Medium)
)

fun loadingTextFor(language: Language, context: Context): String {
    return when (language.code) {
        "en" -> context.getString(R.string.wait_en)
        "ro" -> context.getString(R.string.wait_ro)
        else -> context.getString(R.string.wait_en)
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun LoadingComponent(
    isVisible: Boolean = true,
    loadingText: String = loadingTextFor(AppConfig.mainLanguage, LocalContext.current),
    modifier: Modifier = Modifier
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
                    text = loadingText,
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
                    color = Color(0xFF6750A4),
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
        loadingText = "Please wait"
    )
}
@file:OptIn(ExperimentalMaterial3Api::class)

package com.visionassist.appspace.jetpack.design

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.R
import com.visionassist.appspace.utils.Constants
import com.visionassist.appspace.utils.robotoSemibold

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun LoadingComponent(
    modifier: Modifier = Modifier,
    isVisible: Boolean = true,
    loadingText: String,
    animSpec: Pair<EnterTransition, ExitTransition> =
        Pair(
            fadeIn(
                initialAlpha = 0f,
                animationSpec = tween(durationMillis = Constants.ANIMATION_DELAY)
            ),
            fadeOut(
                targetAlpha = 0f,
                animationSpec = tween(durationMillis = 0)  // ← Instant exit, no glitch!
            )
        )
) {
    // AnimatedVisibility with fade animation
    AnimatedVisibility(
        visible = isVisible,
        enter = animSpec.component1(),
        exit = animSpec.component2()
    ) {
        // Full screen white overlay with 50% opacity
        Box(
            modifier = modifier
                .fillMaxSize()
                .background(Color.Gray.copy(alpha = Constants.BACKGROUND_OPACITY)),
            contentAlignment = Alignment.Center
        ) {
            // Loading content
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center,
                modifier = Modifier.fillMaxWidth(0.8f)
            ) {
                // Loading Text
                Text(
                    text = loadingText,
                    fontSize = Constants.STD_FONT_SIZE.sp,
                    fontFamily = robotoSemibold,
                    color = colorResource(R.color.std_purple),
                    textAlign = TextAlign.Center,
                    letterSpacing = 1.sp
                )

                Spacer(modifier = Modifier.height(10.dp))

                // Linear Progress Indicator
                LinearProgressIndicator(
                    modifier = Modifier
                        .width(230.dp)
                        .height(8.dp),
                    color = colorResource(R.color.std_purple),
                    trackColor = Color(0xFFE8DEF8),
                    gapSize = 10.dp
                )
            }
        }
    }
}

@Preview(name = "Loading Linear", showBackground = true, widthDp = 412, heightDp = 917)
@Composable
fun LoadingComponentPreview() {

    MaterialTheme {
        Image(
            painter = painterResource(id = R.drawable.welcome_background),
            contentDescription = null,
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )
        LoadingComponent(
            isVisible = true,
            loadingText = "Please wait"
        )
    }
}
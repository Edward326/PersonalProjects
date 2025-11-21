@file:Suppress("COMPOSE_APPLIER_CALL_MISMATCH")

package com.visionassist.appspace.jetpack.design

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionassist.appspace.R
import com.visionassist.appspace.utils.robotoExtraBold
import com.visionassist.appspace.utils.robotoSemibold

/**
 * Java-friendly helper to setup FPS slider in ComposeView
 * Call from Java: FindMyObjectUIKt.setupFPSSlider(composeView, defaultFPS, callback);
 */
fun setupFPSSlider(
    composeView: ComposeView,
    defaultFPS: Int,
    onFPSChange: (Int) -> Unit
) {
    composeView.setContent {
        FPSSliderUI(
            defaultFPS = defaultFPS,
            onFPSChange = onFPSChange
        )
    }
}

/**
 * Java-friendly helper to setup navigation UI in ComposeView
 * Call from Java: FindMyObjectUIKt.setupNavigationUI(composeView, hasMore, backCallback, nextCallback);
 */
fun setupNavigationUI(
    composeView: ComposeView,
    hasMoreObjects: Boolean,
    onBackClick: () -> Unit,
    onNextClick: (() -> Unit)?
) {
    composeView.setContent {
        NavigationUI(
            hasMoreObjects = hasMoreObjects,
            onBackClick = onBackClick,
            onNextClick = onNextClick
        )
    }
}

/**
 * FPS Slider UI for FindMyObjectActivity
 * Allows user to control detection frame rate from 1-60 FPS
 */
@Composable
fun FPSSliderUI(
    defaultFPS: Int,
    onFPSChange: (Int) -> Unit
) {
    var currentFPS by remember { mutableIntStateOf(defaultFPS) }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // FPS Value Display
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "$currentFPS",
                fontSize = 32.sp,
                color = colorResource(R.color.std_purple),
                fontFamily = robotoExtraBold
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = "FPS",
                fontSize = 24.sp,
                color = colorResource(R.color.std_purple),
                fontFamily = robotoSemibold
            )
        }

        Spacer(modifier = Modifier.height(12.dp))

        // FPS Slider (1-60)
        CustomSlider(
            value = currentFPS.toFloat(),
            onValueChange = { newValue ->
                currentFPS = newValue.toInt()
                onFPSChange(currentFPS)
            },
            valueRange = 1f..60f,
            steps = 0,  // Continuous slider
            thumbStyle = ThumbStyle.DOUBLE_BAR,
            thumbColor = colorResource(R.color.std_purple),
            thumbWidth = 8.dp,
            thumbHeight = 55.dp,
            thumbBarSpacing = 4.dp,
            trackHeight = 25.dp,
            activeTrackColor = Color.White,
            inactiveTrackColor = Color.White,
            trackShadow = 5.dp,
            modifier = Modifier.fillMaxWidth(0.85f),
            stepsColor = colorResource(R.color.purple_light)
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Description
        Text(
            text = "Detection Speed",
            fontSize = 16.sp,
            color = colorResource(R.color.std_cyan),
            fontFamily = robotoSemibold,
            textAlign = TextAlign.Center
        )
    }
}

/**
 * Navigation UI for FindMyObjectActivity
 * Shows Back button always, Next button only if more objects to find
 */
@Composable
fun NavigationUI(
    hasMoreObjects: Boolean,
    onBackClick: () -> Unit,
    onNextClick: (() -> Unit)?
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(bottom = 32.dp),
        horizontalArrangement = if (hasMoreObjects) {
            Arrangement.SpaceEvenly
        } else {
            Arrangement.Center
        },
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Back button (always shown)
        BackArrowLargeFab(
            onClick = onBackClick
        )

        // Next button (only if more objects to find)
        if (hasMoreObjects && onNextClick != null) {
            NextArrowLargeFab(
                onClick = onNextClick
            )
        }
    }
}

/**
 * Full screen layout for detection phase
 * Shows camera preview with FPS slider at bottom
 */
@Composable
fun DetectionPhaseUI(
    defaultFPS: Int,
    onFPSChange: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier.fillMaxSize()
    ) {
        // FPS Slider at bottom
        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(bottom = 16.dp)
        ) {
            FPSSliderUI(
                defaultFPS = defaultFPS,
                onFPSChange = onFPSChange
            )
        }
    }
}

/**
 * Result screen layout
 * Shows detection result image with navigation buttons
 */
@Composable
fun ResultPhaseUI(
    hasMoreObjects: Boolean,
    onBackClick: () -> Unit,
    onNextClick: (() -> Unit)?,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier.fillMaxSize()
    ) {
        // Navigation buttons at bottom
        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
        ) {
            NavigationUI(
                hasMoreObjects = hasMoreObjects,
                onBackClick = onBackClick,
                onNextClick = onNextClick
            )
        }
    }
}
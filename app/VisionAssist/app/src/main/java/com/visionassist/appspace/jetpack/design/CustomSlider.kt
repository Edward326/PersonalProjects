package com.visionassist.appspace.jetpack.design

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import com.visionassist.appspace.R
import kotlin.math.roundToInt

/**
 * Thumb style options for CustomSlider
 */
enum class ThumbStyle {
    ROUND,      // Circular thumb (default Material3 style)
    BAR,        // Single vertical/horizontal bar
    DOUBLE_BAR  // Two parallel bars (like in screenshot)
}

/**
 * Orientation options for CustomSlider
 */
enum class SliderOrientation {
    HORIZONTAL,
    VERTICAL
}

/**
 * Fully customizable slider component
 *
 * @param value Current value of the slider
 * @param onValueChange Callback when value changes
 * @param modifier Modifier for the slider
 * @param valueRange Range of values (e.g., 0f..100f, 1f..10f, etc.)
 * @param steps Number of discrete steps (0 for continuous sliding)
 * @param onSliderMove Optional callback triggered while slider is being moved
 * @param orientation Horizontal or Vertical (default HORIZONTAL)
 * @param thumbStyle Style of thumb: ROUND, BAR, or DOUBLE_BAR (default DOUBLE_BAR)
 * @param thumbColor Color of the thumb
 * @param thumbWidth Width of thumb (for BAR/DOUBLE_BAR) or diameter (for ROUND)
 * @param thumbHeight Height of thumb (for BAR/DOUBLE_BAR) or diameter (for ROUND)
 * @param thumbBarSpacing Spacing between bars (only for DOUBLE_BAR)
 * @param trackHeight Height/thickness of the track
 * @param trackWidth Width of the track (for VERTICAL orientation)
 * @param trackShadow Elevation/shadow around the track (default 0.dp for no shadow)
 * @param activeTrackColor Color of the filled portion of track
 * @param inactiveTrackColor Color of the unfilled portion of track
 * @param showSteps Whether to show step indicators (default true if steps > 0)
 * @param enabled Whether the slider is interactive
 */
@Composable
fun CustomSlider(
    value: Float,
    onValueChange: (Float) -> Unit,
    modifier: Modifier = Modifier,
    valueRange: ClosedFloatingPointRange<Float> = 0f..100f,
    steps: Int = 0,
    onSliderMove: ((Float) -> Unit)? = null,
    orientation: SliderOrientation = SliderOrientation.HORIZONTAL,
    thumbStyle: ThumbStyle = ThumbStyle.DOUBLE_BAR,
    thumbColor: Color = colorResource(R.color.std_purple),
    thumbWidth: Dp = 8.dp,
    thumbHeight: Dp = 40.dp,
    thumbBarSpacing: Dp = 4.dp,
    trackHeight: Dp = 6.dp,
    trackWidth: Dp = 6.dp,
    trackShadow: Dp = 0.dp,
    activeTrackColor: Color = colorResource(R.color.std_cyan),
    inactiveTrackColor: Color = Color.LightGray,
    showSteps: Boolean = steps > 0,
    enabled: Boolean = true,
    stepsColor: Color = colorResource(R.color.std_cyan),
    activeTrackSpacingMultiplier: Float = 20f
) {
    when (orientation) {
        SliderOrientation.HORIZONTAL -> HorizontalCustomSlider(
            value = value,
            onValueChange = onValueChange,
            modifier = modifier,
            valueRange = valueRange,
            steps = steps,
            onSliderMove = onSliderMove,
            thumbStyle = thumbStyle,
            thumbColor = thumbColor,
            thumbWidth = thumbWidth,
            thumbHeight = thumbHeight,
            thumbBarSpacing = thumbBarSpacing,
            trackHeight = trackHeight,
            trackShadow = trackShadow,
            activeTrackColor = activeTrackColor,
            inactiveTrackColor = inactiveTrackColor,
            showSteps = showSteps,
            stepsColor = stepsColor,
            enabled = enabled,
            activeTrackSpacingMultiplier = activeTrackSpacingMultiplier
        )

        SliderOrientation.VERTICAL -> VerticalCustomSlider(
            value = value,
            onValueChange = onValueChange,
            modifier = modifier,
            valueRange = valueRange,
            steps = steps,
            onSliderMove = onSliderMove,
            thumbStyle = thumbStyle,
            thumbColor = thumbColor,
            thumbWidth = thumbWidth,
            thumbHeight = thumbHeight,
            thumbBarSpacing = thumbBarSpacing,
            trackWidth = trackWidth,
            trackShadow = trackShadow,
            activeTrackColor = activeTrackColor,
            inactiveTrackColor = inactiveTrackColor,
            showSteps = showSteps,
            stepsColor = stepsColor,
            enabled = enabled,
            activeTrackSpacingMultiplier = activeTrackSpacingMultiplier
        )
    }
}

/**
 * Calculate the actual width of the thumb based on its style
 */
private fun calculateThumbWidth(
    style: ThumbStyle,
    thumbWidth: Dp,
    thumbHeight: Dp,
    thumbBarSpacing: Dp,
    orientation: SliderOrientation
): Dp {
    return when (style) {
        ThumbStyle.ROUND -> if (orientation == SliderOrientation.HORIZONTAL) thumbHeight else thumbWidth
        ThumbStyle.BAR -> thumbWidth
        ThumbStyle.DOUBLE_BAR -> thumbWidth * 2 + thumbBarSpacing
    }
}

/**
 * Horizontal slider implementation
 */
@Composable
private fun HorizontalCustomSlider(
    value: Float,
    onValueChange: (Float) -> Unit,
    modifier: Modifier,
    valueRange: ClosedFloatingPointRange<Float>,
    steps: Int,
    onSliderMove: ((Float) -> Unit)?,
    thumbStyle: ThumbStyle,
    thumbColor: Color,
    thumbWidth: Dp,
    thumbHeight: Dp,
    thumbBarSpacing: Dp,
    trackHeight: Dp,
    trackShadow: Dp,
    activeTrackColor: Color,
    inactiveTrackColor: Color,
    showSteps: Boolean,
    stepsColor: Color,
    enabled: Boolean,
    activeTrackSpacingMultiplier: Float
) {
    var sliderWidth by remember { mutableFloatStateOf(0f) }

    // Calculate the normalized value (0f to 1f)
    val normalizedValue = (value - valueRange.start) / (valueRange.endInclusive - valueRange.start)

    Box(
        modifier = modifier
            .fillMaxWidth()
            .height(thumbHeight + 20.dp),
        contentAlignment = Alignment.CenterStart
    ) {
        // Track (background + active portion)
        var trackPadding by remember { mutableFloatStateOf(0f) }

        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .shadow(trackShadow, shape = RoundedCornerShape(100.dp))
                .height(trackHeight)
                .pointerInput(enabled) {
                    if (!enabled) return@pointerInput

                    detectTapGestures { offset ->
                        if (sliderWidth > 0f) {
                            val newValue = calculateNewValue(
                                offset.x,
                                sliderWidth,
                                valueRange,
                                steps,
                                trackPadding  // Pass padding
                            )
                            onValueChange(newValue)
                            onSliderMove?.invoke(newValue)
                        }
                    }

                    detectDragGestures { change, _ ->
                        change.consume()
                        if (sliderWidth > 0f) {
                            val newValue = calculateNewValue(
                                change.position.x,
                                sliderWidth,
                                valueRange,
                                steps,
                                trackPadding  // Pass padding
                            )
                            onValueChange(newValue)
                            onSliderMove?.invoke(newValue)
                        }
                    }
                }
        ) {
            sliderWidth = size.width

            // Calculate padding for alignment with steps and thumb
            trackPadding = size.height * 0.3f // Same as circle radius
            val availableTrackWidth = size.width - (trackPadding * 2)

            // Draw inactive track (with padding)
            drawLine(
                color = inactiveTrackColor,
                start = Offset(trackPadding, size.height / 2),
                end = Offset(size.width - trackPadding, size.height / 2),
                strokeWidth = size.height,
                cap = StrokeCap.Round
            )

            // Draw active track (from start to thumb position, with padding)
            val activeTrackSpacing =
                activeTrackSpacingMultiplier  // Fixed spacing in pixels (e.g., 20f)
            val thumbPositionX = trackPadding + (availableTrackWidth * normalizedValue)
            val activeEndX = thumbPositionX - activeTrackSpacing  // Always stop BEFORE thumb

// Only draw active track if there's actually space for it
            if (activeEndX > trackPadding) {
                val roundedStartLength = size.height / 2f  // Length of rounded cap

                // Calculate where rounded start ends
                val roundedEndX = trackPadding + roundedStartLength

                if (activeEndX <= roundedEndX) {
                    // Case 1: activeEndX is within the rounded start portion
                    // Only draw a partial rounded line (no separate flat section)
                    drawLine(
                        color = activeTrackColor,
                        start = Offset(trackPadding, size.height / 2),
                        end = Offset(activeEndX, size.height / 2),
                        strokeWidth = size.height,
                        cap = StrokeCap.Round  // Rounded start (naturally ends before reaching full circle)
                    )
                } else {
                    // Case 2: activeEndX extends beyond rounded start
                    // Draw rounded start + flat section

                    // Draw rounded start
                    drawLine(
                        color = activeTrackColor,
                        start = Offset(trackPadding, size.height / 2),
                        end = Offset(roundedEndX, size.height / 2),
                        strokeWidth = size.height,
                        cap = StrokeCap.Round  // Creates rounded left cap
                    )

                    // Draw flat section from end of rounded start to activeEndX
                    drawLine(
                        color = activeTrackColor,
                        start = Offset(roundedEndX, size.height / 2),
                        end = Offset(activeEndX, size.height / 2),
                        strokeWidth = size.height,
                        cap = StrokeCap.Butt  // Flat end (no cap)
                    )
                }
            }

            // Draw step indicators if enabled
            if (showSteps && steps > 0) {
                // Calculate circle radius and padding to prevent clipping
                val circleRadius = size.height * 0.3f

                // Calculate available width for steps (inset from edges)
                val availableWidth = size.width - (circleRadius * 2)
                val stepWidth = availableWidth / steps

                for (i in 0..steps) {
                    val x = circleRadius + (stepWidth * i)
                    drawCircle(
                        color = if (x <= size.width * normalizedValue)
                            stepsColor
                        else
                            stepsColor,
                        radius = circleRadius,
                        center = Offset(x, size.height / 2)
                    )
                }
            }
        }

        // Calculate actual thumb width for centering
        val thumbActualWidth = calculateThumbWidth(
            style = thumbStyle,
            thumbWidth = thumbWidth,
            thumbHeight = thumbHeight,
            thumbBarSpacing = thumbBarSpacing,
            orientation = SliderOrientation.HORIZONTAL
        )

        // Calculate padding to match step indicators
        val thumbPadding = with(LocalDensity.current) {
            (trackHeight.toPx() * 0.3f) // Same as circle radius
        }

        // Calculate available width for thumb movement (same as steps)
        val availableThumbWidth = sliderWidth - (thumbPadding * 2)

        // Calculate centered thumb offset with padding
        val thumbOffset =
            (thumbPadding + (availableThumbWidth * normalizedValue) - with(LocalDensity.current) { thumbActualWidth.toPx() } / 2f).roundToInt()

        // Custom thumb
        CustomThumb(
            style = thumbStyle,
            offset = thumbOffset,
            thumbWidth = thumbWidth,
            thumbHeight = thumbHeight,
            thumbBarSpacing = thumbBarSpacing,
            thumbColor = thumbColor,
            orientation = SliderOrientation.HORIZONTAL
        )
    }
}

/**
 * Vertical slider implementation
 */
@Composable
private fun VerticalCustomSlider(
    value: Float,
    onValueChange: (Float) -> Unit,
    modifier: Modifier,
    valueRange: ClosedFloatingPointRange<Float>,
    steps: Int,
    onSliderMove: ((Float) -> Unit)?,
    thumbStyle: ThumbStyle,
    thumbColor: Color,
    thumbWidth: Dp,
    thumbHeight: Dp,
    thumbBarSpacing: Dp,
    trackWidth: Dp,
    trackShadow: Dp,
    activeTrackColor: Color,
    inactiveTrackColor: Color,
    showSteps: Boolean,
    stepsColor: Color,
    enabled: Boolean,
    activeTrackSpacingMultiplier: Float
) {
    var sliderHeight by remember { mutableFloatStateOf(0f) }

    // Calculate the normalized value (0f to 1f)
    val normalizedValue = (value - valueRange.start) / (valueRange.endInclusive - valueRange.start)

    Box(
        modifier = modifier
            .fillMaxHeight()
            .width(thumbWidth + 20.dp),
        contentAlignment = Alignment.BottomCenter
    ) {
        // Track (background + active portion)
        var trackPadding by remember { mutableFloatStateOf(0f) }

        Canvas(
            modifier = Modifier
                .fillMaxHeight()
                .width(trackWidth)
                .shadow(trackShadow, shape = RoundedCornerShape(100.dp))
                .pointerInput(enabled) {
                    if (!enabled) return@pointerInput

                    detectTapGestures { offset ->
                        if (sliderHeight > 0f) {
                            // Invert Y coordinate (bottom = min, top = max)
                            val invertedY = sliderHeight - offset.y
                            val newValue = calculateNewValue(
                                invertedY,
                                sliderHeight,
                                valueRange,
                                steps,
                                trackPadding  // Pass padding
                            )
                            onValueChange(newValue)
                            onSliderMove?.invoke(newValue)
                        }
                    }

                    detectDragGestures { change, _ ->
                        change.consume()
                        if (sliderHeight > 0f) {
                            val invertedY = sliderHeight - change.position.y
                            val newValue = calculateNewValue(
                                invertedY,
                                sliderHeight,
                                valueRange,
                                steps,
                                trackPadding  // Pass padding
                            )
                            onValueChange(newValue)
                            onSliderMove?.invoke(newValue)
                        }
                    }
                }
        ) {
            sliderHeight = size.height

            // Calculate padding for alignment with steps and thumb
            trackPadding = size.width * 0.3f // Same as circle radius

            // Draw inactive track (full height)
            drawLine(
                color = inactiveTrackColor,
                start = Offset(size.width / 2, 0f),
                end = Offset(size.width / 2, size.height),
                strokeWidth = size.width,
                cap = StrokeCap.Round
            )

            // Draw active track (from bottom to thumb position)
            val activeTrackSpacing = activeTrackSpacingMultiplier
            val activeEndY = size.height * (1 - normalizedValue) + activeTrackSpacing

            if (activeEndY < size.height) {
                // Draw rounded start (bottom cap as circle)
                drawCircle(
                    color = activeTrackColor,
                    radius = size.width / 2,
                    center = Offset(size.width / 2, size.height)
                )

                // Draw flat rectangle for middle/end (no cap at top = flat)
                if (activeEndY < size.height - size.width / 2) {
                    drawLine(
                        color = activeTrackColor,
                        start = Offset(size.width / 2, size.height - size.width / 2),
                        end = Offset(size.width / 2, activeEndY),
                        strokeWidth = size.width,
                        cap = StrokeCap.Butt  // Flat end
                    )
                }
            }

            // Draw step indicators if enabled
            if (showSteps && steps > 0) {
                // Calculate circle radius and padding to prevent clipping
                val circleRadius = size.width * 0.3f

                // Calculate available height for steps (inset from edges)
                val availableHeight = size.height - (circleRadius * 2)
                val stepHeight = availableHeight / steps

                for (i in 0..steps) {
                    val y =
                        size.height - circleRadius - (stepHeight * i)  // From bottom up with padding
                    drawCircle(
                        color = if (y >= size.height * (1 - normalizedValue))
                            stepsColor
                        else
                            stepsColor,
                        radius = circleRadius,
                        center = Offset(size.width / 2, y)
                    )
                }
            }
        }

        // Calculate actual thumb height for centering
        val thumbActualHeight = calculateThumbWidth(
            style = thumbStyle,
            thumbWidth = thumbWidth,
            thumbHeight = thumbHeight,
            thumbBarSpacing = thumbBarSpacing,
            orientation = SliderOrientation.VERTICAL
        )

        // Calculate padding to match step indicators
        val thumbPaddingVertical = with(LocalDensity.current) {
            (trackWidth.toPx() * 0.3f) // Same as circle radius
        }

        // Calculate available height for thumb movement (same as steps)
        val availableThumbHeight = sliderHeight - (thumbPaddingVertical * 2)

        // Calculate centered thumb offset with padding
        val thumbOffsetVertical =
            -(thumbPaddingVertical + (availableThumbHeight * normalizedValue) - with(LocalDensity.current) { thumbActualHeight.toPx() } / 2f).roundToInt()

        // Custom thumb (vertical orientation)
        Box(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .offset {
                    IntOffset(
                        0,
                        thumbOffsetVertical
                    )
                }
        ) {
            CustomThumb(
                style = thumbStyle,
                offset = 0,
                thumbWidth = thumbWidth,
                thumbHeight = thumbHeight,
                thumbBarSpacing = thumbBarSpacing,
                thumbColor = thumbColor,
                orientation = SliderOrientation.VERTICAL
            )
        }
    }
}

/**
 * Calculate new value based on position, considering steps
 */
private fun calculateNewValue(
    position: Float,
    trackSize: Float,
    valueRange: ClosedFloatingPointRange<Float>,
    steps: Int,
    padding: Float = 0f  // Add padding parameter
): Float {
    // Adjust position to account for padding
    val adjustedPosition = (position - padding).coerceIn(0f, trackSize - (padding * 2))
    val adjustedTrackSize = trackSize - (padding * 2)

    // Calculate normalized position (0f to 1f)
    val normalizedPosition = if (adjustedTrackSize > 0f) {
        (adjustedPosition / adjustedTrackSize).coerceIn(0f, 1f)
    } else {
        0f
    }

    // If steps are defined, snap to nearest step
    val snappedPosition = if (steps > 0) {
        val stepSize = 1f / steps
        (normalizedPosition / stepSize).roundToInt() * stepSize
    } else {
        normalizedPosition
    }

    // Convert back to actual value in range
    return valueRange.start + (snappedPosition * (valueRange.endInclusive - valueRange.start))
}

/**
 * Custom thumb component that supports different styles
 */
@Composable
private fun CustomThumb(
    style: ThumbStyle,
    offset: Int,
    thumbWidth: Dp,
    thumbHeight: Dp,
    thumbBarSpacing: Dp,
    thumbColor: Color,
    orientation: SliderOrientation
) {
    when (style) {
        ThumbStyle.ROUND -> RoundThumb(
            offset = offset,
            size = if (orientation == SliderOrientation.HORIZONTAL) thumbHeight else thumbWidth,
            thumbColor = thumbColor,
            orientation = orientation
        )

        ThumbStyle.BAR -> BarThumb(
            offset = offset,
            width = thumbWidth,
            height = thumbHeight,
            thumbColor = thumbColor,
            orientation = orientation
        )

        ThumbStyle.DOUBLE_BAR -> DoubleBarThumb(
            offset = offset,
            width = thumbWidth,
            height = thumbHeight,
            spacing = thumbBarSpacing,
            thumbColor = thumbColor,
            orientation = orientation
        )
    }
}

/**
 * Round circular thumb
 */
@Composable
private fun RoundThumb(
    offset: Int,
    size: Dp,
    thumbColor: Color,
    orientation: SliderOrientation
) {
    Box(
        modifier = Modifier
            .offset {
                if (orientation == SliderOrientation.HORIZONTAL)
                    IntOffset(offset, 0)
                else
                    IntOffset(0, offset)
            }
            .size(size)
            .background(
                color = thumbColor,
                shape = CircleShape
            )
    )
}

/**
 * Single bar thumb
 */
@Composable
private fun BarThumb(
    offset: Int,
    width: Dp,
    height: Dp,
    thumbColor: Color,
    orientation: SliderOrientation
) {
    Box(
        modifier = Modifier
            .offset {
                if (orientation == SliderOrientation.HORIZONTAL)
                    IntOffset(offset, 0)
                else
                    IntOffset(0, offset)
            }
            .size(
                width = if (orientation == SliderOrientation.HORIZONTAL) width else height,
                height = if (orientation == SliderOrientation.HORIZONTAL) height else width
            )
            .background(
                color = thumbColor,
                shape = RoundedCornerShape(100.dp)
            )
    )
}

/**
 * Double bar thumb (two parallel bars)
 */
@Composable
private fun DoubleBarThumb(
    offset: Int,
    width: Dp,
    height: Dp,
    spacing: Dp,
    thumbColor: Color,
    orientation: SliderOrientation
) {
    if (orientation == SliderOrientation.HORIZONTAL) {
        // Horizontal: two vertical bars side by side
        Box(
            modifier = Modifier
                .offset { IntOffset(offset, 0) }
                .size(width = width * 2 + spacing, height = height)
        ) {
            // Left bar
            Box(
                modifier = Modifier
                    .align(Alignment.CenterStart)
                    .size(width = width, height = height)
                    .background(
                        color = thumbColor,
                        shape = RoundedCornerShape(100.dp)
                    )
            )

            // Right bar
            Box(
                modifier = Modifier
                    .align(Alignment.CenterEnd)
                    .size(width = width, height = height)
                    .background(
                        color = thumbColor,
                        shape = RoundedCornerShape(100.dp)
                    )
            )
        }
    } else {
        // Vertical: two horizontal bars stacked
        Box(
            modifier = Modifier
                .offset { IntOffset(0, offset) }
                .size(width = height, height = width * 2 + spacing)
        ) {
            // Top bar
            Box(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .size(width = height, height = width)
                    .background(
                        color = thumbColor,
                        shape = RoundedCornerShape(100.dp)
                    )
            )

            // Bottom bar
            Box(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .size(width = height, height = width)
                    .background(
                        color = thumbColor,
                        shape = RoundedCornerShape(100.dp)
                    )
            )
        }
    }
}

// ==================== PREVIEWS ====================

@Preview(name = "Horizontal - Double Bar", showBackground = true, widthDp = 412, heightDp = 200)
@Composable
fun CustomSliderPreview1() {
    var value by remember { mutableFloatStateOf(56f) }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(200.dp),
        contentAlignment = Alignment.Center
    ) {
        CustomSlider(
            value = value,
            onValueChange = { value = it },
            valueRange = 0f..100f,
            thumbStyle = ThumbStyle.DOUBLE_BAR,
            modifier = Modifier.fillMaxWidth(0.8f)
        )
    }
}

@Preview(name = "Horizontal - Round Thumb", showBackground = true, widthDp = 412, heightDp = 200)
@Composable
fun CustomSliderPreview2() {
    var value by remember { mutableFloatStateOf(30f) }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(200.dp),
        contentAlignment = Alignment.Center
    ) {
        CustomSlider(
            value = value,
            onValueChange = { value = it },
            valueRange = 0f..100f,
            thumbStyle = ThumbStyle.ROUND,
            thumbWidth = 24.dp,
            thumbHeight = 24.dp,
            modifier = Modifier.fillMaxWidth(0.8f)
        )
    }
}

@Preview(name = "Horizontal - Bar Thumb", showBackground = true, widthDp = 412, heightDp = 200)
@Composable
fun CustomSliderPreview3() {
    var value by remember { mutableFloatStateOf(75f) }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(200.dp),
        contentAlignment = Alignment.Center
    ) {
        CustomSlider(
            value = value,
            onValueChange = { value = it },
            valueRange = 0f..100f,
            thumbStyle = ThumbStyle.BAR,
            thumbWidth = 10.dp,
            thumbHeight = 50.dp,
            modifier = Modifier.fillMaxWidth(0.8f),
        )
    }
}

@Preview(name = "Horizontal - With Steps", showBackground = true, widthDp = 412, heightDp = 200)
@Composable
fun CustomSliderPreview4() {
    var value by remember { mutableFloatStateOf(60f) }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(200.dp),
        contentAlignment = Alignment.Center
    ) {
        CustomSlider(
            value = value,
            trackHeight = 20.dp,
            onValueChange = { value = it },
            valueRange = 0f..100f,
            steps = 5,
            thumbStyle = ThumbStyle.DOUBLE_BAR,
            modifier = Modifier.fillMaxWidth(0.8f),
            inactiveTrackColor = Color.LightGray,
            activeTrackColor = Color.LightGray
        )
    }
}

@Preview(name = "Vertical - Double Bar", showBackground = true, widthDp = 200, heightDp = 600)
@Composable
fun CustomSliderPreview5() {
    var value by remember { mutableFloatStateOf(60f) }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .fillMaxHeight(),
        contentAlignment = Alignment.Center
    ) {
        CustomSlider(
            value = value,
            onValueChange = { value = it },
            valueRange = 0f..100f,
            orientation = SliderOrientation.VERTICAL,
            thumbStyle = ThumbStyle.DOUBLE_BAR,
            modifier = Modifier.fillMaxHeight(0.8f)
        )
    }
}

@Preview(name = "Custom Range (1-10)", showBackground = true, widthDp = 412, heightDp = 200)
@Composable
fun CustomSliderPreview6() {
    var value by remember { mutableFloatStateOf(5.5f) }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(200.dp),
        contentAlignment = Alignment.Center
    ) {
        CustomSlider(
            value = value,
            onValueChange = { value = it },
            valueRange = 1f..10f,  // Custom range!
            thumbStyle = ThumbStyle.ROUND,
            trackShadow = 3.dp,
            thumbWidth = 30.dp,
            thumbHeight = 30.dp,
            trackHeight = 10.dp,
            modifier = Modifier.fillMaxWidth(0.8f)
        )
    }
}